''' 
this is supose to hold all variables that are common to all controlers
to keep it more organized

like:
    the diferent stats of all robot variants
'''

# --- LIBS --- #
import math
from dataclasses import dataclass, field


#------------------#
# CONSTANTS ACROSS #
#------------------#

TIME_STEP = 64 # ms — deve coincidir com WorldInfo.basicTimeStep
EPISODE_DURATION = 60

FIELD_X_HALF = 1.5
FIELD_Z_HALF = 1.0
GOAL_Z_HALF  = 0.35
BALL_RADIUS  = 0.043
BALL_SPAWN   = (0.0, BALL_RADIUS, 0.0)

# parking spot for unused robot slots (outside field)
PARK_POSITION = (-3.0, 0.1, 0.0)

MAX_ROBOT_PER_TYPE=1



#-----------------#
# --- CLASSES --- #
#-----------------#

#--------------------------#
# --- ROBO-CLASS STATS --- #
#--------------------------#
@dataclass(frozen=True) 
# dataclass - automatically generates common methods for a class
# frozen = T - makes the instance immutable after creation
class RobotConfig:
    name: str # if you wanna identify it, not needed for the code
    def_prefix: str # matches DEF name in .wbt: ROBOT_BLUE, ROBOT_RED
    msg_prefix: str # the colours -> "blue", "red"
    max_speed: float  = 14.0 # rad/s 
    wheel_radius: float = 0.030# m
    wheel_center_dist: float = 0.080 # m // distância do centro geométrico ao ponto de contacto de cada roda
    wheel_angles: list[float] = field(default_factory=lambda:[
        math.radians(90),   # W1 — frente/front
        math.radians(210),  # W2 — trás-esquerda/back-left 
        math.radians(330),  # W3 — trás-direita/back-right 
    ])


# --- SUB CLASSES --- #
# notes:
# Per-robot speed ceilings (max_speed must match maxVelocity in soccer.wbt)

# TITAN: grande, defensivo
TITAN = RobotConfig(
    name = 'TITAN',
    def_prefix = 'ROBOT_BLUE',
    msg_prefix = 'blue',
    max_speed = 5.0,
    wheel_radius = 0.032,
    wheel_center_dist = 0.090,
)

# VIPER: compacto, ofensivo
VIPER = RobotConfig(
    name ='VIPER',
    def_prefix = 'ROBOT_RED',
    msg_prefix = 'red',
    max_speed = 14.0,
    wheel_radius = 0.026,
    wheel_center_dist = 0.070,
)

ALL_ROBOTS = [TITAN, VIPER]


#---------------------------#
# --- TEAM-CONFIG STATS --- #
#---------------------------#

@dataclass
class TeamConfig:
    robot_type: RobotConfig
    faces_right: bool
    n_robots: int = 1 # should have the same size as spawn
    spawn: list[tuple[float, float]] | None = None # per robot get (x,z) coords 

    # default spawn positions if none provided
    _DEFAULT_SPAWNS_LEFT  = [(-0.75, 0.0), (-0.50, 0.3), (-0.50, -0.3)]
    _DEFAULT_SPAWNS_RIGHT = [( 0.75, 0.0), ( 0.50, 0.3), ( 0.50, -0.3)]

    def __post_init__(self):
        self.n_robots = min(abs(self.n_robots), MAX_ROBOT_PER_TYPE) #for now have max 3 just in case of hardcore diff reached

        if self.spawn is None:
            defaults = self._DEFAULT_SPAWNS_RIGHT if self.faces_right \
                       else self._DEFAULT_SPAWNS_LEFT
            self.spawn = defaults[:self.n_robots]

        assert len(self.spawn) == self.n_robots, \
            f"spawn list has {len(self.spawn)} entries but n_robots={self.n_robots}"

    @property
    def rotation(self) -> list:
        angle = 0 if self.faces_right else math.pi
        return [0, 1, 0, angle]
    
# for now assume all robots alredy exist on soccer.wbt and depending on n_robots, n_robots will be realocated to the inside of field and used

#ex: 
#   t1 = TeamConfig(
#           robot_type=TITAN, 
#           n_robots=1
#           spawn=[(0.75,0.0)], 
#           faces_right=True)


