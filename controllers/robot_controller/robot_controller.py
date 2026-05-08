"""
robot_controller.py
-------------------
Webots controller that runs inside each E-puck robot.

The supervisor sends wheel-speed commands via a broadcast Emitter (channel -1).
Every message is prefixed with the target robot name so each robot only acts
on commands that belong to it.

Message format (UTF-8 string):   "<prefix> <left_v> <right_v>"
  e.g.  "blue 3.1416 -1.5708"

The robot's prefix is derived from its Webots name field:
  name "robot_blue"  →  prefix "blue"
  name "robot_red"   →  prefix "red"

Coordinate system reminder
  X – long axis of the field (left goal < 0 < right goal)
  Y – up
  Z – short axis of the field

E-puck motor API
  Each motor is a RotationalMotor; set velocity with setVelocity().
  MAX_SPEED ≈ 6.28 rad/s  (one full wheel revolution per second)
"""

from controller import Robot

MAX_SPEED = 6.28   # rad/s – physical limit of E-puck motors
TIME_STEP = 64     # ms – must match WorldInfo.basicTimeStep


def run():
    robot = Robot()

    # Derive prefix from the robot's name: "robot_blue" → "blue"
    full_name = robot.getName()                    # e.g. "robot_blue"
    prefix    = full_name.split("_", 1)[-1]       # e.g. "blue"

    # ── Motors ───────────────────────────────────────────────────────────────
    left_motor  = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float("inf"))   # velocity-control mode
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0)
    right_motor.setVelocity(0)

    # ── Proximity sensors (8 IR sensors) ─────────────────────────────────────
    ps = [robot.getDevice(f"ps{i}") for i in range(8)]
    for sensor in ps:
        sensor.enable(TIME_STEP)

    # ── Receiver – listens for supervisor commands (broadcast channel -1) ────
    receiver = robot.getDevice("receiver")
    if receiver:
        receiver.enable(TIME_STEP)

    # ── Main loop ─────────────────────────────────────────────────────────────
    while robot.step(TIME_STEP) != -1:
        if receiver is None:
            continue

        # Drain the queue; act only on the most recent command for this robot
        left_v, right_v = None, None
        while receiver.getQueueLength() > 0:
            raw = receiver.getString()
            receiver.nextPacket()
            parts = raw.strip().split()
            # Expected format: "<prefix> <left_v> <right_v>"
            if len(parts) == 3 and parts[0] == prefix:
                try:
                    left_v  = float(parts[1])
                    right_v = float(parts[2])
                except ValueError:
                    pass

        if left_v is not None:
            left_v  = max(-MAX_SPEED, min(MAX_SPEED, left_v))
            right_v = max(-MAX_SPEED, min(MAX_SPEED, right_v))
            left_motor.setVelocity(left_v)
            right_motor.setVelocity(right_v)
        # If no command arrived this step, keep the previous wheel velocities.


if __name__ == "__main__":
    run()
