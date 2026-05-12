"""
supervisor_controller.py
------------------------
Webots Supervisor controller — loop de treino RL para robôs F180 omni.
---
    Can see everything (all node positions, velocities) and can teleport objects, reset positions, etc. 
    Where RL is used/decision making
    Can't directly set motor speeds on the robots, only communicate with them via the Emitter/Receiver channel.
---

Fluxo:
  1. Reset  – reposicionar bola e robôs nas posições iniciais.
  2. Observe – ler posições/velocidades da simulação.
  3. Step   – enviar comandos omni (vx, vz, omega) a cada robô e avançar.
  4. Reward – detectar golos e calcular recompensa escalar.
  5. Done   – sinalizar fim de episódio por golo ou timeout.

Constantes do campo (de soccer.wbt):
  Área de jogo: X ∈ [-1.5, +1.5],  Z ∈ [-1.0, +1.0]
  Boca da baliza esquerda:  x = -1.5,  postes em z = ±0.35
  Boca da baliza direita:   x = +1.5,  postes em z = ±0.35
  Raio da bola = 0.043 m

Limites de velocidade dos robôs (devem coincidir com maxVelocity no soccer.wbt):
  TITAN (ROBOT_BLUE) – grande, defensivo  →  5.0 rad/s nas rodas
  VIPER (ROBOT_RED)  – compacto, ofensivo → 14.0 rad/s nas rodas

Formato do comando enviado ao robot_controller:
  "<prefix> <vx> <vz> <omega>"
  vx    = velocidade linear X do robô (m/s)
  vz    = velocidade linear Z do robô (m/s)
  omega = velocidade angular em Y (rad/s)

Espaço de acções (por robô):
  (vx, vz, omega) — 3 floats contínuos
"""

import math
from controller import Supervisor


import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # go up one level
from shared_configs import TeamConfig, RobotConfig, TITAN, VIPER,MAX_ROBOT_PER_TYPE # RobotConfig, (TITAN, VIPER,)

#-----------#
# CONSTANTS #
#-----------#

TIME_STEP = 64
EPISODE_DURATION = 60

FIELD_X_HALF = 1.5
FIELD_Z_HALF = 1.0
GOAL_Z_HALF  = 0.35
BALL_RADIUS  = 0.043
BALL_SPAWN   = (0.0, BALL_RADIUS, 0.0)

# parking spot for unused robot slots (outside field)
PARK_POSITION = (-3.0, 0.1, 0.0)




class RobotUnit:
    """
    Supervisor-side representation of one robot node
    Knows its node handle, config, team, and instance index
    Handles its own placement and command sending
    """

    def __init__(self, 
            sv: Supervisor, 
            cfg: RobotConfig,
            team_id: int, 
            instance_id: int, 
            emitter
        ):

        self.cfg = cfg
        self.team_id = team_id
        self.instance_id = instance_id
        self.emitter = emitter

        # msg_prefix is per-robot so multiple robots on same team
        # can be addressed individually: "blue_0", "blue_1", ...
        self.msg_prefix = f"{cfg.msg_prefix}_{instance_id}"

        # DEF name in .wbt: e.g. ROBOT_BLUE_0, ROBOT_BLUE_1
        def_name   = f"{cfg.def_prefix}_{instance_id}"
        self.node  = sv.getFromDef(def_name)
        assert self.node, f"DEF {def_name} not found in .wbt"

        self.trans = self.node.getField("translation")
        self.rot = self.node.getField("rotation")   

    
    def place(self, 
            x: float, 
            z: float, 
            faces_right: bool
        ) -> None:
        """ 
        Place robot in map at location x,z
        assumes robot will alyas be on the floor
        """
        y = self.cfg.wheel_radius
        self.trans.setSFVec3f([x, y, z])
        angle = 0 if faces_right else math.pi
        self.rot.setSFRotation([0, 1, 0, angle])
        return

    def park(self) -> None:
        """
        Move robot off-field if not active this episode
        """
        self.trans.setSFVec3f(list(PARK_POSITION))
        return
        
    def send_command(self, 
            vx: float, 
            vz: float, 
            omega: float
        ) -> None:
        """
        Send velocity and rotation commands to the robot, respecting its max speed limits
        """

        if self.emitter is None:
            return
        cfg = self.cfg
        # derive linear and angular limits from wheel physics
        max_linear = cfg.wheel_center_dist * cfg.max_speed
        max_omega  = cfg.max_speed

        vx = max(-max_linear, min(max_linear, vx))
        vz = max(-max_linear, min(max_linear, vz))
        omega = max(-max_omega,  min(max_omega,  omega))

        msg = f"{self.msg_prefix} {vx:.4f} {vz:.4f} {omega:.4f}"
        self.emitter.send(msg.encode())
        return

    def get_position(self) -> list[float]:
        """
        Return the robot's current [x, y, z] position in the world
        """
        return self.node.getPosition()

    def get_yaw(self) -> float:
        """
        Return the robot's current yaw (rotation around the vertical axis) in radians
        """
        ax, ay, az, angle = self.rot.getSFRotation()
        return angle * (1 if ay > 0 else -1)
    

class TeamController:
    """
    Owns all RobotUnits for one team
    Handles reset positioning and dispatching actions to each robot
    """

    def __init__(self, 
            sv: Supervisor, 
            team_cfg: TeamConfig,
            team_id: int, emitter
        ):

        self.team_cfg = team_cfg
        self.team_id = team_id

        # build one RobotUnit per active robot slot (up to max 3 in .wbt trust that are 3 robots in env alredy)
        self.units: list[RobotUnit] = [
            RobotUnit(sv, team_cfg.robot_type, team_id, i, emitter)
            for i in range(MAX_ROBOT_PER_TYPE)  # max slots pre-populated in .wbt
        ]
        self.score = 0  

    def reset(self) -> None:
        """
        Place active robots at spawn positions, park the rest
        """

        n = self.team_cfg.n_robots
        for i, unit in enumerate(self.units):
            if i < n:
                x, z = self.team_cfg.spawn[i]
                unit.place(x, z, self.team_cfg.faces_right)
            else:
                unit.park()
        return

    def send_actions(self, 
            actions: list[tuple]
        ) -> None:
        """
        actions -> list of (vx, vz, omega) tuples, one per active robot
        Extra actions are ignored; 
        missing ones leave motors unchanged
        """
        for unit, action in zip(self.units[:self.team_cfg.n_robots], actions):
            unit.send_command(*action)
        return
    
    def get_observations(self) -> list:
        """
        Returns flat observation for all active robots:
            [x, z, yaw] per robot, normalised
        """
        obs = []
        for unit in self.units[:self.team_cfg.n_robots]:
            pos = unit.get_position()
            obs += [
                pos[0] / FIELD_X_HALF,
                pos[2] / FIELD_Z_HALF,
                unit.get_yaw() / math.pi,
            ]
        return obs


class SoccerSupervisor:
    """
    God-view supervisor 
    Owns both TeamControllers and the ball
    Exposes reset() / step() for the RL training loop
    """

    def __init__(self, 
            team1_cfg: TeamConfig, #TeamConfig(robot_type, faces_right, n_robots, spawn)
            team2_cfg: TeamConfig
        ):
        self.sv = Supervisor()

        self.ball_node = self.sv.getFromDef("BALL")
        assert self.ball_node, "DEF BALL not found in .wbt"
        self.ball_trans = self.ball_node.getField("translation")

        emitter1 = self.sv.getDevice("emitter_blue")
        emitter2 = self.sv.getDevice("emitter_red")

        self.team1 = TeamController(self.sv, team1_cfg, team_id=1, emitter=emitter1)
        self.team2 = TeamController(self.sv, team2_cfg, team_id=2, emitter=emitter2)

        self.episode_steps = 0
        self.max_steps = int(EPISODE_DURATION * 1000 / TIME_STEP)

    def reset(self):

        self.ball_trans.setSFVec3f(list(BALL_SPAWN))
        self.ball_node.setVelocity([0, 0, 0, 0, 0, 0])
        self.team1.reset()
        self.team2.reset()
        self.episode_steps = 0
        self.sv.step(TIME_STEP)
        return self._get_observation()


    def step(self, 
            actions1: list[tuple], 
            actions2: list[tuple]
        ):
        """
        actions1 / actions2 -> list of (vx, vz, omega) per robot in each team.
        Returns (obs, reward, done, info).
        """
        self.team1.send_actions(actions1)
        self.team2.send_actions(actions2)
        self.sv.step(TIME_STEP)
        self.episode_steps += 1
        self._constrain_ball_to_floor()

        obs  = self._get_observation()
        reward, info = self._compute_reward()
        done = info["goal"] or (self.episode_steps >= self.max_steps)

        if info["goal"]:
            if info["scorer"] == "team1":
                self.team1.score += 1
            else:
                self.team2.score += 1
            print(f"GOAL! {info['scorer']}  "
                  f"Score: {self.team1.score} - {self.team2.score}")

        return obs, reward, done, info
    
    def _get_observation(self):
        bpos = self.ball_node.getPosition()
        bvel = self.ball_node.getVelocity()
        ball_obs = [
            bpos[0] / FIELD_X_HALF,
            bpos[2] / FIELD_Z_HALF,
            bvel[0] / FIELD_X_HALF,
            bvel[2] / FIELD_Z_HALF,
        ]
        return ball_obs + self.team1.get_observations() + self.team2.get_observations()

    def _compute_reward(self):
        bpos = self.ball_node.getPosition()
        bx, bz = bpos[0], bpos[2]
        info   = {"goal": False, "scorer": None}
        reward = -0.01

        if bx > FIELD_X_HALF and abs(bz) < GOAL_Z_HALF:
            reward += 10.0
            info = {"goal": True, "scorer": "team1"}
            return reward, info

        if bx < -FIELD_X_HALF and abs(bz) < GOAL_Z_HALF:
            reward -= 10.0
            info = {"goal": True, "scorer": "team2"}
            return reward, info

        bvel = self.ball_node.getVelocity()
        if bvel[0] > 0:
            reward += 0.1 * bvel[0] / FIELD_X_HALF

        return reward, info
    
    def _constrain_ball_to_floor(self):
        pos = self.ball_node.getPosition()
        vel = self.ball_node.getVelocity()
        if abs(pos[1] - BALL_RADIUS) > 0.001:
            self.ball_trans.setSFVec3f([pos[0], BALL_RADIUS, pos[2]])
        if abs(vel[1]) > 0.001:
            self.ball_node.setVelocity(
                [
                    vel[0], 0.0, vel[2],
                    vel[3], vel[4], vel[5]
                    ]
            )
        return





#----------#
# RUN/MAIN #
#----------#

N_SIM = 5

if __name__ == "__main__":
    import random

    # team config
    team1_cfg = TeamConfig(robot_type=TITAN, faces_right=True,
                           n_robots=1, spawn=[(-0.75, 0.0)])
    team2_cfg = TeamConfig(robot_type=VIPER, faces_right=False,
                           n_robots=1, spawn=[(0.75, 0.0)])
    
    # create env
    env = SoccerSupervisor(team1_cfg, team2_cfg)

    # the simulations 
    for episode in range(N_SIM):
        obs  = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            cfg1 = team1_cfg.robot_type
            cfg2 = team2_cfg.robot_type
            max_lin1 = cfg1.wheel_center_dist * cfg1.max_speed
            max_lin2 = cfg2.wheel_center_dist * cfg2.max_speed

            actions1 = [(random.uniform(-max_lin1, max_lin1),
                         random.uniform(-max_lin1, max_lin1),
                         random.uniform(-cfg1.max_speed, cfg1.max_speed))
                        for _ in range(team1_cfg.n_robots)]
            actions2 = [(random.uniform(-max_lin2, max_lin2),
                         random.uniform(-max_lin2, max_lin2),
                         random.uniform(-cfg2.max_speed, cfg2.max_speed))
                        for _ in range(team2_cfg.n_robots)]

            obs, reward, done, info = env.step(actions1, actions2)
            total_reward += reward

        print(f"Episode {episode + 1}  total reward: {total_reward:.2f}")
