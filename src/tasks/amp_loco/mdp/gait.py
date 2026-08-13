from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def gait_phase(
  env: ManagerBasedRlEnv,
  gait_cycle: float,
  phase_offsets: tuple[float, float],
) -> torch.Tensor:
  """Return normalized left/right gait phases in [0, 1)."""
  time = env.episode_length_buf.to(torch.float32) * env.step_dt
  offsets = torch.tensor(
    phase_offsets,
    device=env.device,
    dtype=torch.float32,
  )
  return torch.remainder(
    time[:, None] / gait_cycle + offsets[None, :],
    1.0,
  )


def gait_clock_obs(
  env: ManagerBasedRlEnv,
  gait_cycle: float,
  phase_offsets: tuple[float, float],
  air_ratios: tuple[float, float],
) -> torch.Tensor:
  """Clock observation: sin phase, cos phase and swing ratios."""
  phase = gait_phase(env, gait_cycle, phase_offsets)
  ratio = torch.tensor(
    air_ratios,
    device=env.device,
    dtype=torch.float32,
  ).expand(env.num_envs, -1)

  return torch.cat(
    (
      torch.sin(2.0 * torch.pi * phase),
      torch.cos(2.0 * torch.pi * phase),
      ratio,
    ),
    dim=-1,
  )

def gait_clock_obs_masked(
  env: ManagerBasedRlEnv,
  gait_cycle: float,
  phase_offsets: tuple[float, float],
  air_ratios: tuple[float, float],
  command_name: str = "twist",
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Clock observation; fixed when standing."""
  phase = gait_phase(env, gait_cycle, phase_offsets)

  moving = _moving_mask(env, command_name, command_threshold).unsqueeze(-1)

  ratio = torch.tensor(
    air_ratios,
    device=env.device,
    dtype=torch.float32,
  ).expand(env.num_envs, -1)

  clock = torch.cat(
    (
      torch.sin(2.0 * torch.pi * phase),
      torch.cos(2.0 * torch.pi * phase),
      ratio,
    ),
    dim=-1,
  )

  standing_clock = torch.tensor(
    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    device=env.device,
    dtype=torch.float32,
  ).expand(env.num_envs, -1)

  return moving * clock + (1.0 - moving) * standing_clock


def _gait_masks(
  phase: torch.Tensor,
  air_ratio: torch.Tensor,
  transition: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return smooth swing and stance masks."""
  swing = (phase >= transition) & (
    phase <= air_ratio - transition
  )
  stance = (phase >= air_ratio + transition) & (
    phase <= 1.0 - transition
  )

  transition_start = phase < transition
  transition_air = (
    (phase > air_ratio - transition)
    & (phase < air_ratio + transition)
  )
  transition_end = phase > 1.0 - transition

  swing_mask = (
    swing.float()
    + (0.5 + phase / (2.0 * transition))
    * transition_start.float()
    - (
      (phase - air_ratio - transition)
      / (2.0 * transition)
    )
    * transition_air.float()
    + 0.0 * stance.float()
    + (
      (phase - 1.0 + transition)
      / (2.0 * transition)
    )
    * transition_end.float()
  )

  return swing_mask, 1.0 - swing_mask


def _clock_masks(
  env: ManagerBasedRlEnv,
  gait_cycle: float,
  phase_offsets: tuple[float, float],
  air_ratios: tuple[float, float],
  transition: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  phase = gait_phase(env, gait_cycle, phase_offsets)
  ratios = torch.tensor(
    air_ratios,
    device=env.device,
    dtype=torch.float32,
  ).expand(env.num_envs, -1)

  return _gait_masks(phase, ratios, transition)


def _moving_mask(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
) -> torch.Tensor:
  """Disable periodic rewards for standing commands."""
  command = env.command_manager.get_command(command_name)
  assert command is not None

  command_norm = (
    torch.linalg.norm(command[:, :2], dim=-1)
    + torch.abs(command[:, 2])
  )
  return (command_norm > command_threshold).float()


def gait_feet_force_swing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  gait_cycle: float,
  phase_offsets: tuple[float, float],
  air_ratios: tuple[float, float],
  force_std: float = 20.0,
  transition: float = 0.02,
  command_name: str = "twist",
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward low foot force during swing."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None

  force = torch.linalg.norm(sensor.data.force, dim=-1)
  if force.shape[1] != 2:
    raise RuntimeError(
      f"Expected two foot contacts, got {force.shape[1]}."
    )

  swing_mask, _ = _clock_masks(
    env,
    gait_cycle,
    phase_offsets,
    air_ratios,
    transition,
  )
  score = torch.exp(-torch.square(force / force_std))
  moving = _moving_mask(env, command_name, command_threshold)

  return torch.sum(swing_mask * score, dim=-1) * moving


def gait_feet_speed_stance(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  gait_cycle: float,
  phase_offsets: tuple[float, float],
  air_ratios: tuple[float, float],
  speed_std: float = 0.15,
  transition: float = 0.02,
  command_name: str = "twist",
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward low foot speed during stance."""
  asset: Entity = env.scene[asset_cfg.name]
  speed = torch.linalg.norm(
    asset.data.site_lin_vel_w[:, asset_cfg.site_ids],
    dim=-1,
  )

  _, stance_mask = _clock_masks(
    env,
    gait_cycle,
    phase_offsets,
    air_ratios,
    transition,
  )
  score = torch.exp(-torch.square(speed / speed_std))
  moving = _moving_mask(env, command_name, command_threshold)

  return torch.sum(stance_mask * score, dim=-1) * moving


def gait_feet_force_stance(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  gait_cycle: float,
  phase_offsets: tuple[float, float],
  air_ratios: tuple[float, float],
  force_std: float = 50.0,
  transition: float = 0.02,
  command_name: str = "twist",
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward nonzero supporting force during stance."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None

  force = torch.linalg.norm(sensor.data.force, dim=-1)
  _, stance_mask = _clock_masks(
    env,
    gait_cycle,
    phase_offsets,
    air_ratios,
    transition,
  )

  score = 1.0 - torch.exp(-torch.square(force / force_std))
  moving = _moving_mask(env, command_name, command_threshold)

  return torch.sum(stance_mask * score, dim=-1) * moving