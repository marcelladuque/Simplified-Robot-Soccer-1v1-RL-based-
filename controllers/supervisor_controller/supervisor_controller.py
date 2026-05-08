"""
supervisor_controller.py
------------------------
Webots Supervisor controller.  This is the brain of the RL training loop:

  1. Reset  – relocate the ball and both robots to their starting positions.
  2. Observe – read positions / velocities from the simulation.
  3. Step    – send wheel commands to each robot and advance one time-step.
  4. Reward  – detect goal events and assign a scalar reward.
  5. Done    – signal episode end on goal or timeout.

Field constants (from soccer.wbt)
  Playing area: X ∈ [-1.5, +1.5],  Z ∈ [-1.0, +1.0]
  Left  goal mouth at x = -1.5,  posts at z = ±0.35
  Right goal mouth at x = +1.5,  posts at z = ±0.35
  Ball radius = 0.043 m
"""

import math
import struct
from controller import Supervisor

# ── Simulation constants ──────────────────────────────────────────────────────
TIME_STEP   = 64       # ms – must match WorldInfo.basicTimeStep
MAX_SPEED   = 6.28     # rad/s
EPISODE_DURATION = 60  # seconds per episode  (60 s × 1000 ms / 64 ms ≈ 937 steps)

# ── Field geometry ────────────────────────────────────────────────────────────
FIELD_X_HALF = 1.5     # half-length
FIELD_Z_HALF = 1.0     # half-width
GOAL_Z_HALF  = 0.35    # half-width of each goal mouth
BALL_RADIUS  = 0.043


# ── Spawn positions ───────────────────────────────────────────────────────────
BLUE_SPAWN  = (-0.75, 0.0,  0.0)
RED_SPAWN   = ( 0.75, 0.0,  0.0)
BALL_SPAWN  = ( 0.0,  BALL_RADIUS, 0.0)

# ── Rotation helpers (axis-angle around Y for robots) ────────────────────────
ROT_FACING_RIGHT = [0, 1, 0, 0]                # blue robot: faces +X
ROT_FACING_LEFT  = [0, 1, 0, math.pi]          # red  robot: faces -X


class SoccerSupervisor:
    def __init__(self):
        self.sv = Supervisor()

        # ── Grab node handles ─────────────────────────────────────────────────
        self.ball_node  = self.sv.getFromDef("BALL")
        self.blue_node  = self.sv.getFromDef("ROBOT_BLUE")
        self.red_node   = self.sv.getFromDef("ROBOT_RED")

        assert self.ball_node,  "DEF BALL not found in .wbt"
        assert self.blue_node,  "DEF ROBOT_BLUE not found in .wbt"
        assert self.red_node,   "DEF ROBOT_RED not found in .wbt"

        # Translation / rotation fields used for resets
        self.ball_trans  = self.ball_node.getField("translation")
        self.blue_trans  = self.blue_node.getField("translation")
        self.blue_rot    = self.blue_node.getField("rotation")
        self.red_trans   = self.red_node.getField("translation")
        self.red_rot     = self.red_node.getField("rotation")

        # Emitters – supervisor → robots
        self.emitter_blue = self.sv.getDevice("emitter_blue")
        self.emitter_red  = self.sv.getDevice("emitter_red")

        self.episode_steps   = 0
        self.max_steps       = int(EPISODE_DURATION * 1000 / TIME_STEP)
        self.score_blue      = 0
        self.score_red       = 0

    # ── Public API (call from the RL agent) ───────────────────────────────────

    def reset(self):
        """Move all objects back to their start positions, return initial obs."""
        self._place(self.ball_trans, BALL_SPAWN)
        self._place(self.blue_trans, BLUE_SPAWN)
        self._place(self.red_trans,  RED_SPAWN)
        self.blue_rot.setSFRotation(ROT_FACING_RIGHT)
        self.red_rot.setSFRotation(ROT_FACING_LEFT)

        # Zero velocities so the ball does not carry momentum across episodes.
        self.ball_node.setVelocity([0, 0, 0, 0, 0, 0])

        self.episode_steps = 0
        self.sv.step(TIME_STEP)
        return self._get_observation()

    def step(self, blue_action, red_action):
        """
        Apply actions, advance the simulation, return (obs, reward, done, info).

        Actions are (left_wheel_speed, right_wheel_speed) tuples in rad/s,
        clipped to ±MAX_SPEED.
        """
        self._send_command(self.emitter_blue, "blue", blue_action)
        self._send_command(self.emitter_red,  "red",  red_action)

        self.sv.step(TIME_STEP)
        self.episode_steps += 1

        obs    = self._get_observation()
        reward, goal_info = self._compute_reward()
        done   = goal_info["goal"] or (self.episode_steps >= self.max_steps)
        info   = goal_info

        if goal_info["goal"]:
            if goal_info["scorer"] == "blue":
                self.score_blue += 1
            else:
                self.score_red += 1
            print(f"GOAL! Scorer: {goal_info['scorer']}  "
                  f"Score: Blue {self.score_blue} – {self.score_red} Red")

        return obs, reward, done, info

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_observation(self):
        """
        Returns a flat list of 10 values:
          [ball_x, ball_z,
           blue_x, blue_z, blue_heading,
           red_x,  red_z,  red_heading,
           ball_vx, ball_vz]

        All positions are normalised to [-1, 1] w.r.t. half-field dimensions.
        """
        bpos = self.ball_node.getPosition()   # [x, y, z]
        blpos = self.blue_node.getPosition()
        rpos  = self.red_node.getPosition()

        bvel = self.ball_node.getVelocity()   # [vx, vy, vz, wx, wy, wz]

        blue_heading = self._get_yaw(self.blue_node)
        red_heading  = self._get_yaw(self.red_node)

        obs = [
            bpos[0]  / FIELD_X_HALF,
            bpos[2]  / FIELD_Z_HALF,
            blpos[0] / FIELD_X_HALF,
            blpos[2] / FIELD_Z_HALF,
            blue_heading / math.pi,
            rpos[0]  / FIELD_X_HALF,
            rpos[2]  / FIELD_Z_HALF,
            red_heading  / math.pi,
            bvel[0]  / MAX_SPEED,
            bvel[2]  / MAX_SPEED,
        ]
        return obs

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self):
        """
        Dense + sparse reward for the BLUE robot (agent).

        Positive events:
          +10  scoring in the RIGHT goal
          +0.1 per step that the ball is moving towards the right goal

        Negative events:
          -10  conceding into the LEFT goal
          -0.01 per step (time penalty to encourage fast play)
        """
        bpos = self.ball_node.getPosition()
        bx, bz = bpos[0], bpos[2]

        info = {"goal": False, "scorer": None}
        reward = -0.01  # time penalty

        # Goal in the RIGHT net  →  blue scores
        if bx > FIELD_X_HALF and abs(bz) < GOAL_Z_HALF:
            reward += 10.0
            info = {"goal": True, "scorer": "blue"}
            return reward, info

        # Goal in the LEFT net  →  red scores (blue concedes)
        if bx < -FIELD_X_HALF and abs(bz) < GOAL_Z_HALF:
            reward -= 10.0
            info = {"goal": True, "scorer": "red"}
            return reward, info

        # Dense: reward ball moving toward the right goal
        bvel = self.ball_node.getVelocity()
        if bvel[0] > 0:
            reward += 0.1 * bvel[0] / MAX_SPEED

        return reward, info

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _place(self, trans_field, xyz):
        trans_field.setSFVec3f(list(xyz))

    def _send_command(self, emitter, robot_prefix, action):
        """Send 'prefix left_v right_v' so each robot can self-filter."""
        if emitter is None:
            return
        left_v  = max(-MAX_SPEED, min(MAX_SPEED, action[0]))
        right_v = max(-MAX_SPEED, min(MAX_SPEED, action[1]))
        message = f"{robot_prefix} {left_v:.4f} {right_v:.4f}"
        emitter.send(message.encode())

    def _get_yaw(self, node):
        """Extract yaw angle (rotation around Y-axis) from a robot node."""
        rot = node.getField("rotation").getSFRotation()
        # rot = [ax, ay, az, angle]
        # For pure Y-axis rotation, yaw = angle if ay > 0 else -angle
        ax, ay, az, angle = rot
        return angle * (1 if ay > 0 else -1)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point – simple "random agent" test to verify the environment works
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random

    env = SoccerSupervisor()

    for episode in range(5):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Random actions: (left_speed, right_speed) in [-MAX_SPEED, MAX_SPEED]
            blue_action = (
                random.uniform(-MAX_SPEED, MAX_SPEED),
                random.uniform(-MAX_SPEED, MAX_SPEED),
            )
            red_action = (
                random.uniform(-MAX_SPEED, MAX_SPEED),
                random.uniform(-MAX_SPEED, MAX_SPEED),
            )
            obs, reward, done, info = env.step(blue_action, red_action)
            total_reward += reward

        print(f"Episode {episode + 1}  total reward: {total_reward:.2f}")
