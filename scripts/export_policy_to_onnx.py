#!/usr/bin/env python3
"""microduck-jump: 把训练好的策略 checkpoint 导出为 ONNX 并自检。

用法 (Thor / 有 mjlab 环境的机器):
    MUJOCO_GL=egl uv run --no-sync python export_policy_to_onnx.py <model_XXX.pt> [--out policy.onnx]

流程:
  1. 按任务注册表加载 Mjlab-Jump-Flat-MicroDuck 环境 (取观测/动作维度, 无需跑环境)。
  2. runner.load(load_cfg={"actor": True}) 只载入 actor + obs_normalizer。
  3. rsl_rl 的 MLPModel.as_onnx() 导出纯推理图 (obs [1,61] -> actions [1,14], opset 18):
     输入 = 归一化后的 61 维观测向量, 输出 = 14 关节目标增量 (相对 home 位, scale=1.0)。
  4. exporter_utils 附元数据: 关节名/默认位/刚度阻尼/观测词表/动作 scale。
  5. onnxruntime CPU 回读, 与 torch 侧 wrapped 模型数值对比 (max abs diff < 1e-5) 才算成功。
"""
import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
from dataclasses import asdict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=str, help="训练产物, 如 model_599.pt")
    ap.add_argument("--task", default="Mjlab-Jump-Flat-MicroDuck")
    ap.add_argument("--out", default="policy.onnx")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--dev", default="cpu")
    args = ap.parse_args()

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.envs
    env = ManagerBasedRlEnv(env_cfg, device=args.dev)
    env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(args.task).clip_actions)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(load_rl_cfg(args.task)), device=args.dev)
    runner.load(args.ckpt, load_cfg={"actor": True}, strict=True,
                map_location=args.dev)

    # exporter_utils 需要自然关节序的元数据
    md = get_base_metadata(env.unwrapped, "botversex/microduck-jump")

    out_dir, fname = os.path.dirname(args.out) or ".", os.path.basename(args.out)
    runner.export_policy_to_onnx(out_dir, filename=fname, verbose=False)
    attach_metadata_to_onnx(args.out, md)
    print(f"[export] ONNX written: {args.out} "
          f"({os.path.getsize(args.out) / 1024:.0f} KB)")

    # ---- 自检: torch wrapped 模型 vs onnxruntime ----
    torch_mod = runner.alg.get_policy().as_onnx(verbose=False).to("cpu").eval()
    x = torch.randn(4, 61, dtype=torch.float32)
    with torch.no_grad():
        ref = torch_mod(x).numpy()
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"obs": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    assert ref.shape == (4, 14) and got.shape == (4, 14), (ref.shape, got.shape)
    assert diff < 1e-5, f"onnx/torch mismatch: {diff}"
    print(f"[export] VERIFY OK: shape={got.shape} max|torch-onnx|={diff:.2e}")
    mdl = onnx.load(args.out)
    mds = {mp.key: mp.value for mp in mdl.metadata_props}
    print("[export] joints:", mds.get("joint_names", "")[:120])
    print("[export] obs   :", mds.get("observation_names", "")[:120])


if __name__ == "__main__":
    main()
