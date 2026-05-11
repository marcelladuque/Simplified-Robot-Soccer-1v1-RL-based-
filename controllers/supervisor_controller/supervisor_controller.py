"""
supervisor_controller.py
------------------------
Webots Supervisor controller — loop de treino RL para robôs F180 omni.

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

# ── Constantes de simulação ───────────────────────────────────────────────────
TIME_STEP        = 64       # ms — deve coincidir com WorldInfo.basicTimeStep
EPISODE_DURATION = 60       # segundos por episódio

# ── Parâmetros físicos por robô ───────────────────────────────────────────────
# L_roda × max_vel_roda ≈ velocidade linear máxima do robô
# TITAN:  L=0.090 × 5.0  ≈ 0.45 m/s
# VIPER:  L=0.070 × 14.0 ≈ 0.98 m/s
BLUE_WHEEL_MAX = 5.0    # rad/s  — TITAN
RED_WHEEL_MAX  = 14.0   # rad/s  — VIPER

BLUE_L = 0.090          # m — distância centro→roda TITAN
RED_L  = 0.070          # m — distância centro→roda VIPER
BLUE_R = 0.032          # m — raio da roda TITAN
RED_R  = 0.026          # m — raio da roda VIPER

# Velocidade linear máxima aproximada (usada para normalizar observações)
BLUE_MAX_LINEAR = BLUE_L * BLUE_WHEEL_MAX   # ~0.45 m/s
RED_MAX_LINEAR  = RED_L  * RED_WHEEL_MAX    # ~0.98 m/s
NORM_SPEED = RED_MAX_LINEAR                 # denominador comum de normalização

# ── Geometria do campo ────────────────────────────────────────────────────────
FIELD_X_HALF = 1.5
FIELD_Z_HALF = 1.0
GOAL_Z_HALF  = 0.35
BALL_RADIUS  = 0.043

# ── Posições de spawn ─────────────────────────────────────────────────────────
# Y de spawn = raio da roda para que a roda repouse exactamente no chão (y=0)
BLUE_SPAWN = (-0.75, BLUE_R, 0.0)
RED_SPAWN  = ( 0.75, RED_R,  0.0)
BALL_SPAWN = ( 0.0,  BALL_RADIUS, 0.0)

# ── Rotações iniciais (axis-angle em torno de Y) ──────────────────────────────
ROT_FACING_RIGHT = [0, 1, 0, 0]           # robô azul: frente em +X
ROT_FACING_LEFT  = [0, 1, 0, math.pi]    # robô vermelho: frente em -X


class SoccerSupervisor:
    def __init__(self):
        self.sv = Supervisor()

        # ── Nós do mundo ──────────────────────────────────────────────────────
        self.ball_node = self.sv.getFromDef("BALL")
        self.blue_node = self.sv.getFromDef("ROBOT_BLUE")
        self.red_node  = self.sv.getFromDef("ROBOT_RED")

        assert self.ball_node, "DEF BALL não encontrado no .wbt"
        assert self.blue_node, "DEF ROBOT_BLUE não encontrado no .wbt"
        assert self.red_node,  "DEF ROBOT_RED não encontrado no .wbt"

        # Campos de translação/rotação usados nos resets
        self.ball_trans = self.ball_node.getField("translation")
        self.blue_trans = self.blue_node.getField("translation")
        self.blue_rot   = self.blue_node.getField("rotation")
        self.red_trans  = self.red_node.getField("translation")
        self.red_rot    = self.red_node.getField("rotation")

        # Emissores — supervisor → robôs
        self.emitter_blue = self.sv.getDevice("emitter_blue")
        self.emitter_red  = self.sv.getDevice("emitter_red")

        self.episode_steps = 0
        self.max_steps     = int(EPISODE_DURATION * 1000 / TIME_STEP)
        self.score_blue    = 0
        self.score_red     = 0

    # ── API pública (chamada pelo agente RL) ──────────────────────────────────

    def reset(self):
        """Reposiciona todos os objectos; devolve observação inicial."""
        self._place(self.ball_trans, BALL_SPAWN)
        self._place(self.blue_trans, BLUE_SPAWN)
        self._place(self.red_trans,  RED_SPAWN)
        self.blue_rot.setSFRotation(ROT_FACING_RIGHT)
        self.red_rot.setSFRotation(ROT_FACING_LEFT)

        # Zerar momento da bola entre episódios
        self.ball_node.setVelocity([0, 0, 0, 0, 0, 0])

        self.episode_steps = 0
        self.sv.step(TIME_STEP)
        return self._get_observation()

    def step(self, blue_action, red_action):
        """
        Aplica acções, avança a simulação, devolve (obs, reward, done, info).

        Acções são tuplos (vx, vz, omega):
            vx    – velocidade linear X  (m/s)
            vz    – velocidade linear Z  (m/s)
            omega – velocidade angular Y (rad/s)

        Os limites máximos são impostos antes de enviar ao controlador:
            Blue (TITAN): v_max_linear ≈ 0.45 m/s, omega_max ≈ 5.0 rad/s
            Red  (VIPER): v_max_linear ≈ 0.98 m/s, omega_max ≈ 14.0 rad/s
        """
        self._send_omni_command(self.emitter_blue, "blue", blue_action,
                                BLUE_MAX_LINEAR, BLUE_WHEEL_MAX)
        self._send_omni_command(self.emitter_red,  "red",  red_action,
                                RED_MAX_LINEAR,  RED_WHEEL_MAX)

        self.sv.step(TIME_STEP)
        self.episode_steps += 1

        # Manter bola no plano XZ do chão
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
            print(f"GOLO! Marcador: {goal_info['scorer']}  "
                  f"Placar: Blue {self.score_blue} – {self.score_red} Red")

        return obs, reward, done, info

    # ── Observação ────────────────────────────────────────────────────────────

    def _get_observation(self):
        """
        Devolve lista plana de 10 valores em [-1, 1]:
          [ball_x, ball_z,
           blue_x, blue_z, blue_heading,
           red_x,  red_z,  red_heading,
           ball_vx, ball_vz]

        Posições normalizadas pelas semi-dimensões do campo.
        Velocidades normalizadas por NORM_SPEED.
        """
        bpos  = self.ball_node.getPosition()
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

    # ── Recompensa ────────────────────────────────────────────────────────────

    def _compute_reward(self):
        """
        Recompensa densa + esparsa para o robô BLUE (agente).

        Positivo:
          +10   golo na baliza DIREITA
          +0.1  por step que a bola se move em direcção à baliza direita

        Negativo:
          -10   golo na baliza ESQUERDA
          -0.01 por step (penalidade de tempo)
        """
        bpos = self.ball_node.getPosition()
        bx, bz = bpos[0], bpos[2]

        info   = {"goal": False, "scorer": None}
        reward = -0.01  # penalidade de tempo

        # Bola ultrapassa a linha direita → blue marca
        if bx > FIELD_X_HALF and abs(bz) < GOAL_Z_HALF:
            reward += 10.0
            info = {"goal": True, "scorer": "blue"}
            return reward, info

        # Bola ultrapassa a linha esquerda → red marca (blue sofre)
        if bx < -FIELD_X_HALF and abs(bz) < GOAL_Z_HALF:
            reward -= 10.0
            info = {"goal": True, "scorer": "red"}
            return reward, info

        # Shaping denso: bola a mover-se em direcção à baliza direita
        bvel = self.ball_node.getVelocity()
        if bvel[0] > 0:
            reward += 0.1 * bvel[0] / NORM_SPEED

        return reward, info

    # ── Auxiliares ────────────────────────────────────────────────────────────

    def _constrain_ball_to_floor(self):
        """Força a bola a permanecer no plano XZ do chão após cada step."""
        pos = self.ball_node.getPosition()
        vel = self.ball_node.getVelocity()

        if abs(pos[1] - BALL_RADIUS) > 0.001:
            self.ball_trans.setSFVec3f([pos[0], BALL_RADIUS, pos[2]])
        if abs(vel[1]) > 0.001:
            self.ball_node.setVelocity([vel[0], 0.0, vel[2],
                                        vel[3], vel[4], vel[5]])

    def _place(self, trans_field, xyz):
        trans_field.setSFVec3f(list(xyz))

    def _send_omni_command(self, emitter, prefix, action,
                           max_linear, max_omega):
        """
        Envia comando omni ao robô no formato '<prefix> <vx> <vz> <omega>'.

        Limita vx e vz a ±max_linear, e omega a ±max_omega.
        """
        if emitter is None:
            return
        vx    = max(-max_linear, min(max_linear, float(action[0])))
        vz    = max(-max_linear, min(max_linear, float(action[1])))
        omega = max(-max_omega,  min(max_omega,  float(action[2])))
        emitter.send(f"{prefix} {vx:.4f} {vz:.4f} {omega:.4f}".encode())

    def _get_yaw(self, node):
        """Extrai yaw (rotação em Y) do campo rotation em axis-angle."""
        ax, ay, az, angle = node.getField("rotation").getSFRotation()
        return angle * (1 if ay > 0 else -1)


# ─────────────────────────────────────────────────────────────────────────────
#  Ponto de entrada — teste com agente aleatório (5 episódios)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random

    env = SoccerSupervisor()

    for episode in range(5):
        obs  = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Acção aleatória: (vx, vz, omega)
            blue_action = (
                random.uniform(-BLUE_MAX_LINEAR, BLUE_MAX_LINEAR),
                random.uniform(-BLUE_MAX_LINEAR, BLUE_MAX_LINEAR),
                random.uniform(-BLUE_WHEEL_MAX,  BLUE_WHEEL_MAX),
            )
            red_action = (
                random.uniform(-RED_MAX_LINEAR, RED_MAX_LINEAR),
                random.uniform(-RED_MAX_LINEAR, RED_MAX_LINEAR),
                random.uniform(-RED_WHEEL_MAX,  RED_WHEEL_MAX),
            )
            obs, reward, done, info = env.step(blue_action, red_action)
            total_reward += reward

        print(f"Episódio {episode + 1}  recompensa total: {total_reward:.2f}")
