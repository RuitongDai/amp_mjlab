"""F3 constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

F3_XML: Path = (
  SRC_PATH / "assets" / "robots" / "f3" / "xmls" / "f3.xml"
)
assert F3_XML.exists()

def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(F3_XML))

##
# Actuator config.
##

F3_ACTUATOR_HIPS = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint"),
  stiffness=150.0,
  damping=2.0,
  effort_limit=80.0,
  armature=0.01,
)

F3_ACTUATOR_KNEES = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=150.0,
  damping=2.0,
  effort_limit=120.0,
  armature=0.01,
)

F3_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",),
  stiffness=30.0,
  damping=2.0,
  effort_limit=89.0,
  armature=0.01,
)

F3_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll_joint",),
  stiffness=30.0,
  damping=2.0,
  effort_limit=12.0,
  armature=0.01,
)

F3_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"),
  stiffness=150.0,
  damping=2.0,
  effort_limit=80.0,
  armature=0.01,
)

F3_ACTUATOR_SHOULDER_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_pitch_joint",),
  stiffness=30.0,
  damping=2.0,
  effort_limit=48.0,
  armature=0.008,
)

F3_ACTUATOR_SHOULDER_ROLL_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_roll_joint", ".*_shoulder_yaw_joint"),
  stiffness=30.0,
  damping=2.0,
  effort_limit=84.0,
  armature=0.008,
)

F3_ACTUATOR_FORE_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_elbow_joint", ".*_wrist_roll_joint"),
  stiffness=30.0,
  damping=2.0,
  effort_limit=37.0,
  armature=0.005,
)

F3_ACTUATOR_WRIST_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_yaw_joint",),
  stiffness=20.0,
  damping=2.0,
  effort_limit=20.0,
  armature=0.005,
)

F3_ACTUATOR_WRIST_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint",),
  stiffness=20.0,
  damping=2.0,
  effort_limit=25.0,
  armature=0.005,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.86),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.2,
    ".*_ankle_pitch_joint": -0.1,
    ".*_shoulder_pitch_joint": 0.0,
    ".*_elbow_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "right_shoulder_roll_joint": -0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

F3_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    F3_ACTUATOR_HIPS,
    F3_ACTUATOR_KNEES,
    F3_ACTUATOR_ANKLE_PITCH,
    F3_ACTUATOR_ANKLE_ROLL,
    F3_ACTUATOR_WAIST,
    F3_ACTUATOR_SHOULDER_PITCH,
    F3_ACTUATOR_SHOULDER_ROLL_YAW,
    F3_ACTUATOR_FORE_ARM,
    F3_ACTUATOR_WRIST_YAW,
    F3_ACTUATOR_WRIST_PITCH,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_f3_robot_cfg() -> EntityCfg:
  """Get a fresh f3 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=F3_ARTICULATION,
  )


F3_ACTION_SCALE: dict[str, float] = {}
for a in F3_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    F3_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_f3_robot_cfg())

  viewer.launch(robot.spec.compile())
