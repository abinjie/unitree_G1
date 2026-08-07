"""Evaluate trained velocity policies: deterministic play metrics across checkpoints.

Unlike ``play`` (an interactive viewer), this runs headless over many parallel
environments and reports quantitative deployment metrics per checkpoint:
linear/angular velocity tracking error, fall rate, episode length, action
smoothness. The policy runs deterministically (mean action, no exploration).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.metrics_manager import MetricsManager
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class EvaluateConfig:
  """Configuration for velocity policy evaluation."""

  checkpoint_files: list[str]
  """Checkpoint paths to evaluate, in order."""

  num_envs: int = 128
  """Number of parallel environments (= episodes per checkpoint)."""

  episode_length_s: float = 20.0
  """Episode length override. Play default is infinite; finite needed to measure
  fall rate and episode length."""

  lin_vel_mag: float | None = None
  """Override linear velocity command range to (-mag, mag) m/s for both x and y.
  Default: use the task's play config range."""

  ang_vel_mag: float | None = None
  """Override angular velocity command range to (-mag, mag) rad/s.
  Default: use the task's play config range."""

  device: str | None = None
  """Device to run on. Defaults to CUDA if available."""

  output_file: str | None = None
  """Optional path to save metrics as JSON."""


def _evaluate_checkpoint(
  runner: MjlabOnPolicyRunner,
  env: RslRlVecEnvWrapper,
  policy,
  ckpt_path: str,
  num_envs: int,
  max_steps: int,
  device: str,
) -> dict[str, Any]:
  """Run deterministic evaluation for one checkpoint. Returns aggregated metrics."""
  runner.load(
    ckpt_path,
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)

  unwrapped = env.unwrapped
  rm = unwrapped.reward_manager
  mm = unwrapped.metrics_manager
  assert isinstance(mm, MetricsManager), "metrics manager is required for evaluation"
  term_mgr = unwrapped.termination_manager
  cmd_term = unwrapped.command_manager.get_term("twist")
  assert cmd_term is not None, "twist command not found"
  robot = unwrapped.scene["robot"]

  rew_idx = {name: i for i, name in enumerate(rm.active_terms)}
  met_idx = {name: i for i, name in enumerate(mm.active_terms)}

  track_lin_i = rew_idx["track_linear_velocity"]
  track_ang_i = rew_idx["track_angular_velocity"]
  upright_i = rew_idx["upright"]
  action_acc_i = met_idx["mean_action_acc"]

  n = num_envs
  track_lin_sum = torch.zeros(n, device=device)
  track_ang_sum = torch.zeros(n, device=device)
  upright_sum = torch.zeros(n, device=device)
  action_acc_sum = torch.zeros(n, device=device)
  lin_err_sum = torch.zeros(n, device=device)
  ang_err_sum = torch.zeros(n, device=device)
  active_steps = torch.zeros(n, device=device)
  ep_len = torch.zeros(n, device=device)

  fell = torch.zeros(n, dtype=torch.bool, device=device)
  done_envs = torch.zeros(n, dtype=torch.bool, device=device)

  obs = env.get_observations()

  step = 0
  while not done_envs.all() and step < max_steps:
    with torch.no_grad():
      actions = policy(obs)
    obs, _, dones, _ = env.step(actions)

    active = ~done_envs

    track_lin_sum[active] += rm._step_reward[active, track_lin_i]
    track_ang_sum[active] += rm._step_reward[active, track_ang_i]
    upright_sum[active] += rm._step_reward[active, upright_i]
    action_acc_sum[active] += mm._step_values[active, action_acc_i]

    cmd = cmd_term.command
    lin_err = torch.norm(cmd[:, :2] - robot.data.root_link_lin_vel_b[:, :2], dim=-1)
    ang_err = torch.abs(cmd[:, 2] - robot.data.root_link_ang_vel_b[:, 2])
    lin_err_sum[active] += lin_err[active]
    ang_err_sum[active] += ang_err[active]

    active_steps[active] += 1
    ep_len[active] += 1

    newly_done = dones.bool() & ~done_envs
    if newly_done.any():
      fell[newly_done] = term_mgr.terminated[newly_done]
      done_envs = done_envs | newly_done
    step += 1

  safe = active_steps.clamp(min=1)
  finished = done_envs.sum().item()
  return {
    "checkpoint": Path(ckpt_path).name,
    "lin_vel_err": (lin_err_sum / safe).mean().item(),
    "ang_vel_err": (ang_err_sum / safe).mean().item(),
    "track_lin_vel": (track_lin_sum / safe).mean().item(),
    "track_ang_vel": (track_ang_sum / safe).mean().item(),
    "upright": (upright_sum / safe).mean().item(),
    "mean_action_acc": (action_acc_sum / safe).mean().item(),
    "fall_rate": fell.float().mean().item(),
    "episode_length": (
      ep_len[done_envs].mean().item() if finished > 0 else float(ep_len.mean().item())
    ),
    "num_episodes": finished,
  }


def run_evaluate(task_id: str, cfg: EvaluateConfig) -> list[dict[str, Any]]:
  """Run evaluation across all checkpoints and return per-checkpoint metrics."""
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  env_cfg.episode_length_s = cfg.episode_length_s
  env_cfg.scene.num_envs = cfg.num_envs

  if cfg.lin_vel_mag is not None or cfg.ang_vel_mag is not None:
    twist = cast(UniformVelocityCommandCfg, env_cfg.commands["twist"])
    if cfg.lin_vel_mag is not None:
      twist.ranges.lin_vel_x = (-cfg.lin_vel_mag, cfg.lin_vel_mag)
      twist.ranges.lin_vel_y = (-cfg.lin_vel_mag, cfg.lin_vel_mag)
    if cfg.ang_vel_mag is not None:
      twist.ranges.ang_vel_z = (-cfg.ang_vel_mag, cfg.ang_vel_mag)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)

  step_dt = env.unwrapped.step_dt
  max_steps = int(cfg.episode_length_s / step_dt) + 50
  print(
    f"[INFO] task={task_id} num_envs={cfg.num_envs} "
    f"step_dt={step_dt:.4f} max_steps={max_steps} "
    f"lin_vel_mag={cfg.lin_vel_mag} ang_vel_mag={cfg.ang_vel_mag}"
  )

  results: list[dict[str, Any]] = []
  for ckpt_path in cfg.checkpoint_files:
    path = str(Path(ckpt_path).resolve())
    if not Path(path).exists():
      print(f"[WARN] Checkpoint not found, skipping: {path}")
      continue
    print(f"[INFO] Evaluating {Path(path).name} ...")
    res = _evaluate_checkpoint(runner, env, None, path, cfg.num_envs, max_steps, device)
    results.append(res)
    print(
      f"  fall={res['fall_rate']:.3f} lin_err={res['lin_vel_err']:.3f} "
      f"ang_err={res['ang_vel_err']:.3f} trk_lin={res['track_lin_vel']:.3f} "
      f"trk_ang={res['track_ang_vel']:.3f} ep_len={res['episode_length']:.0f} "
      f"act_acc={res['mean_action_acc']:.3f}"
    )

  env.close()
  return results


def _print_table(results: list[dict[str, Any]]) -> None:
  if not results:
    print("[INFO] No results to display.")
    return

  header = (
    f"{'checkpoint':>16} {'lin_err':>8} {'ang_err':>8} "
    f"{'trk_lin':>8} {'trk_ang':>8} {'upright':>8} "
    f"{'fall%':>6} {'ep_len':>7} {'act_acc':>8}"
  )
  print("\n" + "=" * 90)
  print(header)
  print("-" * 90)
  for r in results:
    print(
      f"{r['checkpoint']:>16} {r['lin_vel_err']:>8.3f} {r['ang_vel_err']:>8.3f} "
      f"{r['track_lin_vel']:>8.3f} {r['track_ang_vel']:>8.3f} "
      f"{r['upright']:>8.3f} {r['fall_rate'] * 100:>5.1f}% "
      f"{r['episode_length']:>7.0f} {r['mean_action_acc']:>8.3f}"
    )
  print("=" * 90)
  print("lin_err/ang_err: lower is better | trk_*/upright: higher is better")
  print("fall%: lower is better | ep_len: higher is better | act_acc: lower is better")


def main():
  import mjlab  # noqa: F401  (registers tasks, exposes TYRO_FLAGS)

  velocity_tasks = [t for t in list_tasks() if "Velocity" in t]
  if not velocity_tasks:
    print("No velocity tasks found.")
    sys.exit(1)

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(velocity_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    EvaluateConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )

  results = run_evaluate(chosen_task, args)
  _print_table(results)

  if args.output_file and results:
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
      json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to {output_path}")


if __name__ == "__main__":
  main()
