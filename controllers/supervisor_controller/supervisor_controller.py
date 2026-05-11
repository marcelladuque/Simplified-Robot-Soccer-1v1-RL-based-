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

Robot speed limits (must match maxVelocity in soccer.wbt)
  TITAN  (ROBOT_BLUE) – large defensive  →   5.0 rad/s
  VIPER  (ROBOT_RED)  – compact offensive → 14.0 rad/s
"""

import math
from controller import Supervisor

# ── Simulation constants ──────────────────────────────────────────────────────
TIME_STEP        = 64       # ms – controller step (must be a multiple of basicTimeStep=16)
EPISODE_DURATION = 60       # seconds per episode

# Per-robot speed ceilings (must match maxVelocity in soccer.wbt)
BLUE_MAX_SPEED =  5.0       # rad/s – TITAN: large defensive
RED_MAX_SPEED  = 14.0       # rad/s – VIPER: compact offensive

# Common normalisation denominator for observations (use the faster robot's max)
NORM_SPEED = RED_MAX_SPEED

# ── Field geometry ────────────────────────────────────────────────────────────
FIELD_X_HALF = 1.5
FIELD_Z_HALF = 1.0
GOAL_Z_HALF  = 0.35
BALL_RADIUS  = 0.043

# ── Ball stillness guard ──────────────────────────────────────────────────────
# Distances (centre-to-centre) at which each robot is considered to be
# touching the ball  (robot_body_radius + ball_radius + 15 mm margin).
TITAN_CONTACT_DIST = 0.090 + 0.043 + 0.015   # 0.148 m
VIPER_CONTACT_DIST = 0.070 + 0.043 + 0.015   # 0.128 m
# Ball speed above which it is considered "in play" after a kick.
BALL_MOVING_SPEED  = 0.05                     # m/s

# Out-of-bounds thresholds – trigger is the field boundary line, not the wall.
# The ball respawns as soon as it crosses a white line, just like a real match.
#   Sideline:  |z| > FIELD_Z_HALF  (1.0 m)
#   End line:  |x| > FIELD_X_HALF  (1.5 m) AND outside the goal mouth
#   Goal mouth is handled by _compute_reward (not here).
# A small slack (half a ball diameter) avoids false triggers while the ball
# is still rolling along the line.
OOB_SLACK = BALL_RADIUS          # 0.043 m ≈ one ball radius of tolerance

# ── Spawn positions ───────────────────────────────────────────────────────────
# Robot y = wheel_radius so the wheel Sphere bounding objects rest at y = 0.
BLUE_WHEEL_RADIUS = 0.032   # m – TITAN
RED_WHEEL_RADIUS  = 0.026   # m – VIPER

BLUE_SPAWN = (-0.75, BLUE_WHEEL_RADIUS, 0.0)
RED_SPAWN  = ( 0.75, RED_WHEEL_RADIUS,  0.0)
BALL_SPAWN = ( 0.0,  BALL_RADIUS,       0.0)

# ── Rotation helpers (axis-angle around Y) ────────────────────────────────────
ROT_FACING_RIGHT = [0, 1, 0, 0]          # blue robot: faces +X
ROT_FACING_LEFT  = [0, 1, 0, math.pi]   # red  robot: faces -X


class SoccerSupervisor:
    def __init__(self):
        self.sv = Supervisor()

        # ── Grab node handles ─────────────────────────────────────────────────
        self.ball_node = self.sv.getFromDef("BALL")
        self.blue_node = self.sv.getFromDef("ROBOT_BLUE")
        self.red_node  = self.sv.getFromDef("ROBOT_RED")

        assert self.ball_node, "DEF BALL not found in .wbt"
        assert self.blue_node, "DEF ROBOT_BLUE not found in .wbt"
        assert self.red_node,  "DEF ROBOT_RED not found in .wbt"

        # Translation / rotation fields used for resets
        self.ball_trans = self.ball_node.getField("translation")
        self.blue_trans = self.blue_node.getField("translation")
        self.blue_rot   = self.blue_node.getField("rotation")
        self.red_trans  = self.red_node.getField("translation")
        self.red_rot    = self.red_node.getField("rotation")

        # Emitters – supervisor → robots
        self.emitter_blue = self.sv.getDevice("emitter_blue")
        self.emitter_red  = self.sv.getDevice("emitter_red")

        self.episode_steps = 0
        self.max_steps     = int(EPISODE_DURATION * 1000 / TIME_STEP)
        self.score_blue    = 0
        self.score_red     = 0

    # ── Public API (call from the RL agent) ───────────────────────────────────

    def reset(self):
        """Move all objects back to their start positions; return initial obs."""
        self._place(self.ball_trans, BALL_SPAWN)
        self._place(self.blue_trans, BLUE_SPAWN)
        self._place(self.red_trans,  RED_SPAWN)
        self.blue_rot.setSFRotation(ROT_FACING_RIGHT)
        self.red_rot.setSFRotation(ROT_FACING_LEFT)

        # Zero ball momentum so it does not carry across episodes.
        self.ball_node.setVelocity([0, 0, 0, 0, 0, 0])

        self.episode_steps = 0
        self.sv.step(TIME_STEP)
        return self._get_observation()

    def step(self, blue_action, red_action):
        """
        Apply actions, advance the simulation, return (obs, reward, done, info).

        Actions are (left_wheel_speed, right_wheel_speed) tuples in rad/s.
        Blue  is clipped to ± BLUE_MAX_SPEED.
        Red   is clipped to ± RED_MAX_SPEED.
        """
        self._send_command(self.emitter_blue, "blue", blue_action,  BLUE_MAX_SPEED)
        self._send_command(self.emitter_red,  "red",  red_action,   RED_MAX_SPEED)

        self.sv.step(TIME_STEP)
        self.episode_steps += 1

        # Keep ball in the XZ floor plane: zero any vertical velocity/spin
        # so it never bounces up or rolls off the ground.
        self._constrain_ball_to_floor()

        obs              = self._get_observation()
        reward, goal_info = self._compute_reward()
        done             = goal_info["goal"] or (self.episode_steps >= self.max_steps)
        info             = goal_info

        if goal_info["goal"]:
            if goal_info["scorer"] == "blue":
                self.score_blue += 1
            else:
                self.score_red += 1
            print(f"GOAL! Scorer: {goal_info['scorer']}  "
                  f"Score: Blue {self.score_blue} – {self.score_red} Red")
            # Respawn ball at centre circle immediately so the scene looks
            # correct in the frame before the RL agent calls reset().
            self._respawn_ball()
        else:
            # Safety net: teleport ball back to centre if it somehow escapes
            # the arena walls due to a physics glitch.
            self._check_ball_bounds()

        return obs, reward, done, info

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_observation(self):
        """
        Returns a flat list of 10 values (all in [-1, 1]):
          [ball_x, ball_z,
           blue_x, blue_z, blue_heading,
           red_x,  red_z,  red_heading,
           ball_vx, ball_vz]

        Positions are normalised by half-field dimensions.
        Velocities are normalised by NORM_SPEED (= BLUE_MAX_SPEED = 12 rad/s).
        """
        bpos  = self.ball_node.getPosition()   # [x, y, z]
        blpos = self.blue_node.getPosition()
        rpos  = self.red_node.getPosition()
        bvel  = self.ball_node.getVelocity()   # [vx, vy, vz, wx, wy, wz]

        blue_heading = self._get_yaw(self.blue_node)
        red_heading  = self._get_yaw(self.red_node)

        return [
            bpos[0]  / FIELD_X_HALF,
            bpos[2]  / FIELD_Z_HALF,
            blpos[0] / FIELD_X_HALF,
            blpos[2] / FIELD_Z_HALF,
            blue_heading / math.pi,
            rpos[0]  / FIELD_X_HALF,
            rpos[2]  / FIELD_Z_HALF,
            red_heading  / math.pi,
            bvel[0]  / NORM_SPEED,
            bvel[2]  / NORM_SPEED,
        ]

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self):
        """
        Dense + sparse reward for the BLUE robot (agent).

        Positive events:
          +10  scoring in the RIGHT goal
          +0.1 per step the ball moves toward the right goal (dense shaping)

        Negative events:
          -10  conceding into the LEFT goal
          -0.01 per step (time penalty to encourage fast play)
        """
        bpos = self.ball_node.getPosition()
        bx, bz = bpos[0], bpos[2]

        info   = {"goal": False, "scorer": None}
        reward = -0.01  # time penalty

        # Ball crosses RIGHT goal line → blue scores
        if bx > FIELD_X_HALF and abs(bz) < GOAL_Z_HALF:
            reward += 10.0
            info = {"goal": True, "scorer": "blue"}
            return reward, info

        # Ball crosses LEFT goal line → red scores (blue concedes)
        if bx < -FIELD_X_HALF and abs(bz) < GOAL_Z_HALF:
            reward -= 10.0
            info = {"goal": True, "scorer": "red"}
            return reward, info

        # Dense shaping: reward ball moving toward the right goal
        bvel = self.ball_node.getVelocity()
        if bvel[0] > 0:
            reward += 0.1 * bvel[0] / NORM_SPEED

        return reward, info

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _respawn_ball(self):
        """Teleport ball to the centre spot and zero all linear/angular momentum."""
        self._place(self.ball_trans, BALL_SPAWN)
        self.ball_node.setVelocity([0, 0, 0, 0, 0, 0])

    def _check_ball_bounds(self):
        """
        Respawn the ball whenever it leaves the field, just like a real match.

        Triggers:
          |z| > FIELD_Z_HALF + OOB_SLACK       → crossed a sideline
          |x| > FIELD_X_HALF + OOB_SLACK
            AND |z| > GOAL_Z_HALF + OOB_SLACK  → crossed an end line outside
                                                  the goal mouth

        Goal-mouth crossings (|x| > 1.5, |z| < GOAL_Z_HALF) are handled
        separately by _compute_reward and never reach this method.
        """
        pos = self.ball_node.getPosition()
        bx, bz = pos[0], pos[2]

        crossed_sideline = abs(bz) > FIELD_Z_HALF + OOB_SLACK
        crossed_endline  = (abs(bx) > FIELD_X_HALF + OOB_SLACK
                            and abs(bz) > GOAL_Z_HALF + OOB_SLACK)

        if crossed_sideline or crossed_endline:
            print(f"[supervisor] Ball out of bounds at "
                  f"({bx:.3f}, {bz:.3f}) – respawning at centre.")
            self._respawn_ball()

    def _constrain_ball_to_floor(self):
        """
        Keep the ball in the XZ floor plane and enforce stillness when nothing
        is kicking it.

        Logic every step
        ────────────────
        1. Correct Y drift (ball must rest on the floor).
        2. Always zero the vertical (Y) velocity – no bouncing.
        3. Compute whether a robot is currently touching the ball
           (centre-to-centre distance ≤ contact threshold).
        4. If NO robot is touching AND the ball's horizontal speed is below
           BALL_MOVING_SPEED → zero ALL remaining velocity components.
           This makes physics-engine noise, gravity micro-slopes, and
           constraint-force drift completely harmless.
        5. If a robot IS touching OR the ball is already rolling fast (was
           kicked), leave the velocity untouched so physics handles it.
        """
        pos = self.ball_node.getPosition()
        vel = self.ball_node.getVelocity()   # [vx, vy, vz, wx, wy, wz]
        vx, vy, vz, wx, wy, wz = vel
        bx, bz = pos[0], pos[2]

        # 1. Fix Y position
        if abs(pos[1] - BALL_RADIUS) > 0.001:
            self.ball_trans.setSFVec3f([bx, BALL_RADIUS, bz])

        # 2. Zero vertical velocity
        vy = 0.0

        # 3. Is a robot touching the ball?
        bp = self.blue_node.getPosition()
        rp = self.red_node.getPosition()
        d_blue = math.sqrt((bx - bp[0]) ** 2 + (bz - bp[2]) ** 2)
        d_red  = math.sqrt((bx - rp[0]) ** 2 + (bz - rp[2]) ** 2)
        robot_touching = (d_blue < TITAN_CONTACT_DIST or
                          d_red  < VIPER_CONTACT_DIST)

        # 4. Freeze ball when idle (no robot contact AND slow)
        speed = math.sqrt(vx * vx + vz * vz)
        if not robot_touching and speed < BALL_MOVING_SPEED:
            vx = vz = wx = wy = wz = 0.0

        self.ball_node.setVelocity([vx, vy, vz, wx, wy, wz])

    def _place(self, trans_field, xyz):
        trans_field.setSFVec3f(list(xyz))

    def _send_command(self, emitter, robot_prefix, action, max_speed):
        """Send 'prefix left_v right_v' so each robot can self-filter."""
        if emitter is None:
            return
        left_v  = max(-max_speed, min(max_speed, action[0]))
        right_v = max(-max_speed, min(max_speed, action[1]))
        emitter.send(f"{robot_prefix} {left_v:.4f} {right_v:.4f}".encode())

    def _get_yaw(self, node):
        """Extract yaw (rotation around Y) from axis-angle rotation field."""
        ax, ay, az, angle = node.getField("rotation").getSFRotation()
        return angle * (1 if ay > 0 else -1)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point – random-agent smoke test (5 episodes)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random

    env = SoccerSupervisor()

    for episode in range(5):
        obs  = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            blue_action = (
                random.uniform(-BLUE_MAX_SPEED, BLUE_MAX_SPEED),
                random.uniform(-BLUE_MAX_SPEED, BLUE_MAX_SPEED),
            )
            red_action = (
                random.uniform(-RED_MAX_SPEED, RED_MAX_SPEED),
                random.uniform(-RED_MAX_SPEED, RED_MAX_SPEED),
            )
            obs, reward, done, info = env.step(blue_action, red_action)
            total_reward += reward

        print(f"Episode {episode + 1}  total reward: {total_reward:.2f}")
