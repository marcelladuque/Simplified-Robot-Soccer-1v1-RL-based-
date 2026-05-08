# Simplified Robot Soccer – 1v1 RL-based

Intelligent Robotics course final project.  
A 1v1 robot soccer simulation built in **Webots R2023b** with
a **Reinforcement Learning** training loop.

---

## Project structure

```
.
├── worlds/
│   └── soccer.wbt                   ← Webots world (soccer pitch)
├── controllers/
│   ├── robot_controller/
│   │   └── robot_controller.py      ← Runs inside each E-puck robot
│   └── supervisor_controller/
│       └── supervisor_controller.py ← RL env (reset / step / reward)
├── requirements.txt
└── README.md
```

---

## Field dimensions

| Parameter | Value |
|---|---|
| Playing area | 3.0 m × 2.0 m |
| Goal width | 0.70 m |
| Goal height | 0.40 m |
| Goal depth | 0.25 m |
| Ball radius | 0.043 m (~size-1) |
| Robot | E-puck (Webots built-in) |

Coordinate system: **X** = long axis (left goal < 0 < right goal),  
**Y** = up, **Z** = short axis.

---

## Quick start

1. **Install Webots R2023b** – https://cyberbotics.com  
2. Open `worlds/soccer.wbt` in Webots.  
3. Press **Play** – the random-agent test in `supervisor_controller.py`
   runs five episodes automatically.

---

## Observation space (10 floats, all in [-1, 1])

| Index | Description |
|---|---|
| 0 | Ball X (normalised) |
| 1 | Ball Z (normalised) |
| 2 | Blue robot X |
| 3 | Blue robot Z |
| 4 | Blue robot heading / π |
| 5 | Red robot X |
| 6 | Red robot Z |
| 7 | Red robot heading / π |
| 8 | Ball velocity X (normalised) |
| 9 | Ball velocity Z (normalised) |

## Action space (per robot)

`(left_wheel_speed, right_wheel_speed)` in rad/s, clipped to ±6.28.

## Reward function (for the Blue agent)

| Event | Reward |
|---|---|
| Score in right goal | +10 |
| Concede in left goal | −10 |
| Ball moving toward right goal | +0.1 × speed |
| Every step | −0.01 (time penalty) |

---

## Roadmap

- [ ] Implement PPO / SAC agent with `stable-baselines3`
- [ ] Log training curves (TensorBoard)
- [ ] Evaluate against random baseline
- [ ] Extra: compare reward strategies
- [ ] Extra: extend to 2v2
- [ ] Extra: dynamic ball (random initial velocity)
