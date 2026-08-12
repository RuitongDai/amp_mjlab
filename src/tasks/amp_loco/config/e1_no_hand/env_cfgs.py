"""E1 AMP Locomotion environment configurations."""

import os

from src.assets.robots import (
  E1_NO_HAND_ACTION_SCALE,
  get_e1_no_hand_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.amp_loco.amp_env_cfg import make_amp_env_cfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg


def e1_no_hand_amp_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create E1 rough terrain velocity configuration."""
  cfg = make_amp_env_cfg()

  # Keep CCD high enough for stability but avoid Warp OOM from excessive EPA buffers.
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_e1_no_hand_robot_cfg()}

  # Set raycast sensor frame to e1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )
  body_names = ("pelvis",
                "left_hip_roll_link",
                "left_knee_link",
                "left_ankle_roll_link",
                "right_hip_roll_link",
                "right_knee_link",
                "right_ankle_roll_link",
                )
  anchor_name = "torso_link"
  root_name = "pelvis"

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = E1_NO_HAND_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"


  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15
  # Train direct velocity commands consistently.
  twist_cmd.heading_command = False
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.ranges.heading = None
  # Keep commands close to the motion-data distribution.
  twist_cmd.ranges.lin_vel_x = (-0.3, 0.8)
  twist_cmd.ranges.lin_vel_y = (-0.2, 0.2)
  twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)


  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Configure motion reset to sample from the entire motion with a delay.
  cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.0
  cfg.events["init_motion_loader"].params["max_delay_steps"] = 250

  # Set motion data path for startup loader and reset.
  _motion_base = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "assets", "motions", "e1_no_hand", "amp"
  )
  _motion_dir = os.path.abspath(os.path.join(_motion_base, "WalkandRun"))
  _recovery_dir = os.path.abspath(os.path.join(_motion_base, "Recovery"))

  cfg.events["init_motion_loader"].params["motion_dir"] = _motion_dir
  cfg.events["init_motion_loader"].params["recovery_dir"] = None
  cfg.events["reset_from_motion"].params["motion_dir"] = _motion_dir

  cfg.rewards["track_anchor_linear_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.rewards["track_anchor_angular_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards.pop("self_collisions", None)
  cfg.rewards["body_ang_vel_xy_l2"].params["body_cfg"].body_names = (root_name,)
  cfg.rewards["track_anchor_linear_velocity"].params["std"] = 0.5
  cfg.rewards["track_anchor_angular_velocity"].params["std"] = 0.6

  # # gait周期步态
  # gait_cycle = 1.2
  # phase_offsets = (0.38, 0.88)
  # air_ratios = (0.38, 0.38)
  # gait_obs_params = {
  #   "gait_cycle": gait_cycle,
  #   "phase_offsets": phase_offsets,
  #   "air_ratio": air_ratios,
  # }
  # cfg.observations["actor"].terms["gait_clock"] = ObservationTermCfg(
  #   func=mdp.gait_clock_obs,
  #   params=gait_obs_params,
  # )
  # cfg.observations["critic"].terms["gait_clock"] = ObservationTermCfg(
  #   func=mdp.gait_clock_obs,
  #   params=gait_obs_params,
  # )
  # gait_common_params = {
  #   "gait_cycle": gait_cycle,
  #   "phase_offsets": phase_offsets,
  #   "air_ratios": air_ratios,
  #   "transition": 0.02,
  #   "command_name": "twist",
  #   "command_threshold": 0.1,
  # }
  #
  # cfg.rewards["gait_feet_force_swing"] = RewardTermCfg(
  #   func=mdp.gait_feet_force_swing,
  #   weight=0.15,
  #   params={
  #     **gait_common_params,
  #     "sensor_name": "feet_ground_contact",
  #     "force_std": 20.0,
  #   },
  # )
  #
  # cfg.rewards["gait_feet_speed_stance"] = RewardTermCfg(
  #   func=mdp.gait_feet_speed_stance,
  #   weight=0.15,
  #   params={
  #     **gait_common_params,
  #     "asset_cfg": SceneEntityCfg(
  #       "robot",
  #       site_names=site_names,
  #     ),
  #     "speed_std": 0.15,
  #   },
  # )
  #
  # cfg.rewards["gait_feet_force_stance"] = RewardTermCfg(
  #   func=mdp.gait_feet_force_stance,
  #   weight=0.10,
  #   params={
  #     **gait_common_params,
  #     "sensor_name": "feet_ground_contact",
  #     "force_std": 50.0,
  #   },
  # )

  cfg.observations["critic"].terms["body_pos_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["critic"].terms["body_pos_b"].params["body_cfg"].body_names = body_names
 
  cfg.observations["critic"].terms["body_ori_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["critic"].terms["body_ori_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_pos_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_pos_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_ori_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_ori_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_lin_vel_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_lin_vel_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_ang_vel_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_ang_vel_b"].params["body_cfg"].body_names = body_names

  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {
      "step": 2000 * 24,
      "lin_vel_x": (-0.4, 0.9),
      "lin_vel_y": (-0.35, 0.35),
      "ang_vel_z": (-1.0, 1.0),
    },
    {
      "step": 5000 * 24,
      "lin_vel_x": (-0.6, 1.1),
      "lin_vel_y": (-0.5, 0.5),
      "ang_vel_z": (-1.5, 1.5),
    },
  ]

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 1.0

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def e1_no_hand_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create E1 flat terrain velocity configuration."""
  cfg = e1_no_hand_amp_rough_env_cfg(play=play)

  cfg.sim.njmax = 640
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 256
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (0,0)
    twist_cmd.ranges.lin_vel_y = (0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (0,0)

  return cfg
