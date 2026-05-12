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




# clean assumig no errors

'''
# ── Parâmetros físicos por robô ───────────────────────────────────────────────
# L   = distância do centro geométrico ao ponto de contacto de cada roda (m)
# r   = raio da roda (m)
# max = velocidade angular máxima das rodas (rad/s) — deve coincidir com soccer.wbt
ROBOT_PARAMS = {
    "blue": {"L": 0.090, "r": 0.032, "max_vel": 5.0},   # TITAN: grande, defensivo
    "red":  {"L": 0.070, "r": 0.026, "max_vel": 14.0},  # VIPER: compacto, ofensivo
}
DEFAULT_PARAMS = {"L": 0.080, "r": 0.030, "max_vel": 14.0}

# Ângulos das rodas (rad) — mesmo para os dois robôs (triângulo equilátero)
WHEEL_ANGLES = [
    math.radians(90),   # W1 — frente
    math.radians(210),  # W2 — trás-esquerda
    math.radians(330),  # W3 — trás-direita
]
'''


'''


def omni_inverse_kinematics(vx, vz, omega, L, r):
    """
    Calcula as velocidades angulares de cada roda (rad/s) para um robô
    omnidirecional com 3 rodas a 120°.

    Referencial do robô:
        X — frente do robô
        Z — lateral (esquerda positiva)
        Y — cima

    Args:
        vx    (float): velocidade linear em X (m/s)
        vz    (float): velocidade linear em Z (m/s)
        omega (float): velocidade angular em Y (rad/s)
        L     (float): distância centro→roda (m)
        r     (float): raio da roda (m)

    Returns:
        list[float]: [w1, w2, w3] velocidades angulares em rad/s
    """
    speeds = []
    for theta in WHEEL_ANGLES:
        # Componente de translação que esta roda deve fornecer
        v_wheel = -math.sin(theta) * vx + math.cos(theta) * vz + L * omega
        # Converter velocidade linear na roda para velocidade angular
        speeds.append(v_wheel / r)
    return speeds


def run():
    robot = Robot()

    # Derivar prefixo do nome Webots: "robot_blue" → "blue"
    full_name = robot.getName()
    prefix = full_name.split("_", 1)[-1]
    params = ROBOT_PARAMS.get(prefix, DEFAULT_PARAMS)
    L       = params["L"]
    r       = params["r"]
    max_vel = params["max_vel"]

    # ── Motores das 3 rodas ───────────────────────────────────────────────────
    motors = []
    for i in range(1, 4):
        m = robot.getDevice(f"wheel{i} motor")
        if m is None:
            raise RuntimeError(f"Motor 'wheel{i} motor' não encontrado no robô '{full_name}'")
        m.setPosition(float("inf"))   # modo velocidade
        m.setVelocity(0.0)
        motors.append(m)

    # ── Receptor — ouve comandos do supervisor (canal -1) ────────────────────
    receiver = robot.getDevice("receiver")
    if receiver:
        receiver.enable(TIME_STEP)

    # ── Loop principal ────────────────────────────────────────────────────────
    while robot.step(TIME_STEP) != -1:
        if receiver is None:
            continue

        # Esvaziar fila; actuar apenas no comando mais recente para este robô.
        vx, vz, omega = None, None, None
        while receiver.getQueueLength() > 0:
            raw = receiver.getString()
            receiver.nextPacket()
            parts = raw.strip().split()
            # Formato esperado: "<prefix> <vx> <vz> <omega>"
            if len(parts) == 4 and parts[0] == prefix:
                try:
                    vx    = float(parts[1])
                    vz    = float(parts[2])
                    omega = float(parts[3])
                except ValueError:
                    pass

        if vx is not None:
            wheel_speeds = omni_inverse_kinematics(vx, vz, omega, L, r)
            for motor, speed in zip(motors, wheel_speeds):
                clamped = max(-max_vel, min(max_vel, speed))
                motor.setVelocity(clamped)
        # Sem comando neste step → mantém velocidades anteriores

'''