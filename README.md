# microduck-jump — a MicroDuck robot learns to jump, in RL

A periodic **hop-in-place** skill for the tiny MicroDuck robot, trained purely
with PPO in simulation (mjlab, MuJoCo × GPU). No motion capture, no hand-written
control — only a **reward ladder**: *crouch → launch → airtime*.

**Result (final policy, 256 parallel envs, measured step-by-step):**

| Metric | Value |
|---|---|
| Envs with both feet airborne (≥ 5 steps after spawn) | **256 / 256** |
| Longest air time | **0.21 s** |
| Root height at apex | **0.187 m** (standing: 0.120 m, +55%) |
| Posture | upright (0° tilt), continuous crouch–hop cycles |
| Reward params | 600 iterations, 4096 envs, 50 Hz |

<video src="media/jump_closeup.mp4" controls width="640"></video>

---

## Why this is not trivial

A naive "reward +X if airborne during phase-window + ascent" is hopeless: for a
policy that starts standing, *windows ∧ air ∧ ascent* is a probability-≈0 event,
so the gradient is 0 and the robot stays a statue forever. The fix is a
**graded ladder** where every rung is reachable from the previous one:

```
phase 0.10–0.30  crouch    wt 10   z → 0.106 m        (get low)
phase 0.28–0.38  launch    wt 10   vz / 0.12           (push off)
phase 0.26–0.55  airtime   wt 75   air × vz-impulse    (stay up)
```

Several traps cost real debugging time (all documented in the code comments):

1. **Spawn artifact** — `foot_air_time` is nonzero for the first 1–2 steps of
   every episode; "every env is airborne" on an instant snapshot is not a jump.
   The stress test starts at step ≥ 5.
2. **`fell_over` resets the phase clock** — any episode that tilts > 70° is
   terminated and the periodic command restarts from φ=0, so windowed rewards
   never fire for open-loop probes.
3. **Joint-role asymmetry** — pushing all 14 joints at +1.0 makes the duck
   **dive forward head-first** (a roll), not jump; pushing the 10 leg joints
   from a deep crouch is what produces an upright hop.
4. **Real launch speeds** — measured hops leave the ground at vz ≈ 0.11–0.2 m/s,
   not the 0.5 m/s one expects from ballistic spec. Thresholds must be
   calibrated against the actual physics.

## Files

```
configs/microduck_jump_env_cfg.py   # env + the reward ladder (heavily commented)
configs/agent.yaml                  # RL hyper-parameters (PPO, 600 iters)
configs/env.yaml                    # full serialized env config
policies/policy.onnx                # exported inference graph (775 KB, metadata embedded)
policies/model_599.pt               # training checkpoint (4.8 MB, actor+critic)
scripts/export_policy_to_onnx.py    # re-export + self-check (torch↔onnxrt)
scripts/verify_jumps.py             # step-by-step jump stress test (256 envs)
media/jump_closeup.mp4              # close-up render of the final policy
```

## Reproduce

Dependencies: `mjlab==1.3.0`, `rsl-rl`, the `mjlab-microduck` task package
(entry point `mjlab.tasks → mjlab_microduck`, so tasks register by import),
CUDA GPU (trained on a Jetson THOR devkit, 4096 envs live).

```bash
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 600

# render a video of the final policy (64 envs, world camera):
MUJOCO_GL=egl RENDER_W=640 RENDER_H=480 uv run python rl-console/render_one.py \
  Mjlab-Jump-Flat-MicroDuck --ckpt logs/rsl_rl/jump/*_jump-main/model_599.pt \
  --frames 200 --num-envs 64 --extra 30 --origin world --distance 1.3 \
  --elevation -12 --out render.raw
```

Verification (the honest one — per-step, physical criterion):

```bash
uv run python scripts/verify_jumps.py policies/model_599.pt
# expect: EVEN-ONCE air>=0.02 (steps>=5): 256/256, MAX air_time ~0.21 s
```

ONNX re-export:

```bash
uv run python scripts/export_policy_to_onnx.py policies/model_599.pt --out policies/policy.onnx
```

## ONNX inference

`policy.onnx` — input `obs` float32 `[1, 61]` (concatenated observation terms,
order in the model metadata `observation_names`, 50 Hz), output `actions`
float32 `[1, 14]` (joint targets **relative to home pose**, clip to `[-1, 1]`).

```python
import numpy as np, onnxruntime as ort
sess = ort.InferenceSession("policies/policy.onnx")         # CPU is fine
obs = np.zeros((1, 61), dtype=np.float32)                   # replace with real obs
action = sess.run(None, {"obs": obs})[0][0]                 # (14,), home-relative
```

Joint order and metadata (names, stiffness/damping, default pose) are embedded
in the ONNX file — read them with `onnx.load("policy.onnx").metadata_props`.

## 中文说明

MicroDuck(豌豆鸭): 一只超小型四足机器人, 在 mjlab(MuJoCo × GPU)仿真中仅靠
PPO + **阶梯奖励**学会了原地连续弹跳——下蹲(相位 0.10-0.30)→ 蹬地(vz 推力)
→ 腾空(双足离地)。600 轮、4096 环境并行训练, 最终策略 256/256 环境双足离地,
最长腾空 0.21s, 重心高度 0.120→0.187m(站立时保持直身不倒)。

**为什么难**: 直接奖励"相位窗口内腾空且上升"对从站立开始的策略是概率≈0 事件
→ 梯度为 0 → 策略永远站着。解法是阶梯化: 每一级必须从上一级够得着。
其它坑: 出生伪影(前 1-2 步脚部传感器即显示"离地")、倾角>70° 终止会重置相位钟、
全关节 +1.0 是头朝前滚翻而腿关节深蹲才是直立跳、真实离地 vz≈0.11-0.2 m/s
(阈值必须按实测标定, 不能按弹道理论猜)。

视频: `media/jump_closeup.mp4`。训练/复现命令见上文 Reproduce 一节。

## License

Apache-2.0, matching the upstream `mjlab-microduck` package.
The policy weights are training output of the same license.

MicroDuck is the open-source biped robot by Pollen Robotics × Hugging Face
(≈800 g, 15 motors, camera + ToF depth + IMUs). Training runs on
[mjlab](https://github.com/pollen-robotics/microduck_rl) (MuJoCo × Warp, PPO,
50 Hz); policies export to ONNX and deploy to the Pollen Robotics runtime under
the shared 61-dim observation contract.
