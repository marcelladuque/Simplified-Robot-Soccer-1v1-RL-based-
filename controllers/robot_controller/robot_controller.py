"""
robot_controller.py
-------------------
Webots controller para robôs F180 omnidirecionais com 3 rodas.
---
    receive motion commands from a supervisor and convert them into wheel motor velocities for robot
    it can only see and control the hardware of the robot it's running inside: 
        its motors, 
        its sensors
    It has no view of the field, other robots, or the ball
---

O supervisor envia comandos via Emitter (canal -1) no formato:
    "<prefix> <vx> <vz> <omega>"
onde:
    vx    – velocidade linear no eixo X do robô (m/s)
    vz    – velocidade linear no eixo Z do robô (m/s)
    omega – velocidade angular em torno de Y (rad/s), positivo = anti-horário

Cinemática inversa omni 3 rodas (θ₁=90°, θ₂=210°, θ₃=330°):
    ω_i = ( -sin(θ_i)*vx + cos(θ_i)*vz + L*omega ) / r_roda

onde L é a distância do centro às rodas (m) e r_roda é o raio da roda (m).

Parâmetros por robô (devem corresponder aos valores no soccer.wbt):
    TITAN (blue): L=0.090m, r_roda=0.032m, maxVelocity=5 rad/s
    VIPER  (red): L=0.070m, r_roda=0.026m, maxVelocity=14 rad/s

Prefixo derivado do nome Webots:
    "robot_blue" → "blue"
    "robot_red"  → "red"
"""

import math
from controller import Robot

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # go up one level
from shared_configs import ALL_ROBOTS # TeamConfig, RobotConfig, (TITAN, VIPER,)


TIME_STEP = 64  # ms — deve coincidir com WorldInfo.basicTimeStep


class MoveRobot:
    def __init__(self, robot: Robot):
        self.robot = robot

        # Derive identity from Webots node name: "robot_blue" -> "blue"
        # change this, each robot should have 
        full_name  = robot.getName()
        self.prefix = full_name.split("_", 1)[-1]

        # Look up physical config from shared_configs
        cfg_lookup = {r.msg_prefix: r for r in ALL_ROBOTS}
        self.cfg = cfg_lookup.get(self.prefix)
        if self.cfg is None:
            raise RuntimeError(
                f"No RobotConfig found for prefix '{self.prefix}'. "
                f"Known prefixes: {list(cfg_lookup.keys())}"
            )

        # --- Motors --- #
        # named "wheel1 motor", "wheel2 motor", "wheel3 motor" in .wbt
        self.motors = []
        for i in range(1, 4):
            m = robot.getDevice(f"wheel{i} motor")
            if m is None:
                raise RuntimeError(
                    f"Motor 'wheel{i} motor' not found in robot '{full_name}'"
                )
            m.setPosition(float("inf"))  # velocity control mode
            m.setVelocity(0.0)
            self.motors.append(m)

        # --- Receiver --- # 
        # listens on broadcast channel -1
        self.receiver = robot.getDevice("receiver")
        if self.receiver:
            self.receiver.enable(TIME_STEP)
        else:
            print(f"[{self.prefix}] WARNING: no receiver device found")



    
    def inverse_kinematics(self, 
            vx: float, 
            vz: float, 
            omega: float
        ) -> list[float]:
        """
        Converts (vx, vz, omega) to individual wheel angular velocities.

        3-wheel omni layout (equilateral triangle):
            wheel 1 — front        (90°)
            wheel 2 — back-left   (210°)
            wheel 3 — back-right  (330°)

        Formula per wheel i:
            w_i = ( -sin(θ_i)*vx + cos(θ_i)*vz + L*omega ) / r
        """

        L = self.cfg.wheel_center_dist
        r = self.cfg.wheel_radius

        return [
            (-math.sin(theta) * vx + math.cos(theta) * vz + L * omega) / r
            for theta in self.cfg.wheel_angles
        ]
    
    def read_command(self):
        """
        Drains the receiver queue, returns the most recent (vx, vz, omega)
        tuple addressed to this robot, or None if no valid command arrived.
        """
        if not self.receiver:
            return None

        result = None
        while self.receiver.getQueueLength() > 0:
            raw = self.receiver.getString()
            self.receiver.nextPacket()

            parts = raw.strip().split()
            if len(parts) == 4 and parts[0] == self.prefix:
                try:
                    result = (float(parts[1]), float(parts[2]), float(parts[3]))
                except ValueError:
                    pass  # malformed packet -> ignore, keep previous

        return result
    
    def step(self):
        """
        Called once per simulation step.
        Reads the latest command and updates motor velocities if one arrived.
        If no command arrived this step, previous velocities are kept.
        """
        command = self.read_command()
        if command is None:
            return

        vx, vz, omega = command
        wheel_speeds   = self.inverse_kinematics(vx, vz, omega)

        for motor, speed in zip(self.motors, wheel_speeds):
            clamped = max(-self.cfg.max_speed, min(self.cfg.max_speed, speed))
            motor.setVelocity(clamped)



def run():
    robot = Robot()
    mover = MoveRobot(robot)

    while robot.step(TIME_STEP) != -1:
        mover.step()

if __name__ == "__main__":
    run()

