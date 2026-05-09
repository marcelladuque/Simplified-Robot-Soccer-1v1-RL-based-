"""
robot_controller.py
-------------------
Webots controller that runs inside each F180-style soccer robot.

The supervisor sends wheel-speed commands via a broadcast Emitter (channel -1).
Every message is prefixed with the target robot name so each robot only acts
on commands that belong to it.

Message format (UTF-8 string):   "<prefix> <left_v> <right_v>"
  e.g.  "blue 6.2832 -3.1416"

The robot's prefix is derived from its Webots name field:
  name "robot_blue"  →  prefix "blue"
  name "robot_red"   →  prefix "red"

Motor convention (both wheels share axis 0 0 -1):
  Positive velocity  →  robot moves forward  (+local X)
  Both motors equal  →  straight line
  Left > Right       →  turns right
  Left < Right       →  turns left

Per-robot speed limits (must match maxVelocity in soccer.wbt):
  blue  →   5.0 rad/s  (TITAN: large defensive)
  red   →  14.0 rad/s  (VIPER: compact offensive)
"""

from controller import Robot

# Per-robot physical speed ceilings — must match the VRML maxVelocity values.
MAX_SPEED_BY_ROBOT = {
    "blue":  5.0,   # TITAN
    "red":  14.0,   # VIPER
}
DEFAULT_MAX_SPEED = 14.0   # fallback for any unknown name

TIME_STEP = 64             # ms – must match WorldInfo.basicTimeStep


def run():
    robot = Robot()

    # Derive prefix from the robot's name: "robot_blue" → "blue"
    full_name = robot.getName()
    prefix    = full_name.split("_", 1)[-1]
    max_speed = MAX_SPEED_BY_ROBOT.get(prefix, DEFAULT_MAX_SPEED)

    # ── Motors ────────────────────────────────────────────────────────────────
    left_motor  = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float("inf"))    # velocity-control mode
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # ── Receiver – listens for supervisor commands (broadcast channel -1) ─────
    receiver = robot.getDevice("receiver")
    if receiver:
        receiver.enable(TIME_STEP)

    # ── Main loop ─────────────────────────────────────────────────────────────
    while robot.step(TIME_STEP) != -1:
        if receiver is None:
            continue

        # Drain the queue; act only on the most recent command for this robot.
        left_v, right_v = None, None
        while receiver.getQueueLength() > 0:
            raw = receiver.getString()
            receiver.nextPacket()
            parts = raw.strip().split()
            if len(parts) == 3 and parts[0] == prefix:
                try:
                    left_v  = float(parts[1])
                    right_v = float(parts[2])
                except ValueError:
                    pass

        if left_v is not None:
            left_motor.setVelocity(max(-max_speed, min(max_speed, left_v)))
            right_motor.setVelocity(max(-max_speed, min(max_speed, right_v)))
        # If no command arrived this step, keep the previous wheel velocities.


if __name__ == "__main__":
    run()
