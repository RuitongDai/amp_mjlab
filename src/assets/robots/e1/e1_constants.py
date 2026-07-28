"""E1 constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

E1_XML: Path = (
  SRC_PATH / "assets" / "robots" / "e1" / "xmls" / "E1_25dof.xml"
)
assert E1_XML.exists()

def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(E1_XML))

##
# Actuator config.
##

E1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  stiffness=150.0,
  damping=5.0,
  effort_limit=120.0,
  armature=0.01,
)

E1_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint",),
  stiffness=80.0,
  damping=3.0,
  effort_limit=36.0,
  armature=0.01,
)

E1_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint",),
  stiffness=100.0,
  damping=3.0,
  effort_limit=60.0,
  armature=0.01,
)

E1_ACTUATOR_KNEES = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=150.0,
  damping=5.0,
  effort_limit=120.0,
  armature=0.01,
)

E1_ACTUATOR_ANKLES = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=30.0,
  damping=2.0,
  effort_limit=30.0,
  armature=0.01,
)

E1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint",),
  stiffness=100.0,
  damping=5.0,
  effort_limit=60.0,
  armature=0.01,
)

E1_ACTUATOR_SHOULDER_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_pitch_joint",),
  stiffness=30.0,
  damping=2.0,
  effort_limit=60.0,
  armature=0.01,
)

E1_ACTUATOR_SHOULDER_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_roll_joint",),
  stiffness=30.0,
  damping=2.0,
  effort_limit=36.0,
  armature=0.01,
)

E1_ACTUATOR_SHOULDER_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_yaw_joint",),
  stiffness=30.0,
  damping=2.0,
  effort_limit=15.0,
  armature=0.01,
)

E1_ACTUATOR_ELBOWS = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_elbow_joint",),
  stiffness=30.0,
  damping=2.0,
  effort_limit=60.0,
  armature=0.01,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.67),
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

E1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    E1_ACTUATOR_HIP_PITCH,
    E1_ACTUATOR_HIP_YAW,
    E1_ACTUATOR_HIP_ROLL,
    E1_ACTUATOR_KNEES,
    E1_ACTUATOR_ANKLES,
    E1_ACTUATOR_WAIST,
    E1_ACTUATOR_SHOULDER_PITCH,
    E1_ACTUATOR_SHOULDER_ROLL,
    E1_ACTUATOR_SHOULDER_YAW,
    E1_ACTUATOR_ELBOWS,
  ),
  soft_joint_pos_limit_factor=0.9,
)

def get_e1_robot_cfg() -> EntityCfg:
  """Get a fresh e1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=E1_ARTICULATION,
  )


E1_ACTION_SCALE: dict[str, float] = {}
for a in E1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    E1_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_e1_robot_cfg())

  viewer.launch(robot.spec.compile())
