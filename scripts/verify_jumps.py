#!/usr/bin/env python3
"""microduck-jump 验收: 逐帧 240 步, 物理判据(两脚离地), 排除出生伪影(step>=5).

用法:
    uv run python verify_jumps.py <model_XXX.pt>  [--envs 256] [--steps 240]

判据输出 (与 README 结果表一致):
    EVEN-ONCE air>=0.02 (steps>=5): 256/256    <- 真跳: 每环境至少一次两脚离地
    MAX air_time: ~0.21 s | air steps total  / air 步占比
"""
import argparse
import sys
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_microduck.tasks import mdp as microduck_mdp  # noqa: hook scene sensors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=str)
    ap.add_argument("--task", default="Mjlab-Jump-Flat-MicroDuck")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--dev", default="cuda:0")
    args = ap.parse_args()

    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = args.envs
    env = ManagerBasedRlEnv(cfg, device=args.dev)
    env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(args.task).clip_actions)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(load_rl_cfg(args.task)), device=args.dev)
    runner.load(args.ckpt, load_cfg={"actor": True}, strict=True,
                map_location=args.dev)
    policy = runner.get_inference_policy(device=args.dev)
    obs, _ = env.reset()
    raw = env.env
    asset = raw.scene["robot"]

    with torch.no_grad():
        airs = []
        rows = []
        for i in range(args.steps):
            obs, rew, dones, info = env.step(policy(obs))
            a = microduck_mdp.foot_air_time_safe(
                raw, "feet_ground_contact").detach().min(dim=1).values
            z = (asset.data.root_link_pos_w[:, 2]
                 - raw.scene.terrain.env_origins[:, 2]).detach()
            cmd = raw.command_manager.get_command("twist")
            ph = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0
            airs.append(a)
            if i < 5 or (a >= 0.02).any() or (z.max() > 0.125):
                rows.append("i%3d ph%.3f z%.4f/%.4f air%.3f n%d" % (
                    i, float(ph.mean().item()), float(z.mean().item()),
                    float(z.max().item()), float(a.mean().item()),
                    int((a >= 0.02).sum().item())))
    a = torch.stack(airs)
    mask = torch.zeros(args.steps, dtype=torch.bool, device=a.device)
    mask[5:] = True  # 出生 1-2 步的"离地"伪影不计
    ever = ((a >= 0.02) & mask[:, None]).any(dim=0)
    print("\n".join(rows))
    print("== VERDICT ==")
    print("EVEN-ONCE air>=0.02 (steps>=5): %d/%d"
          % (int(ever.sum().item()), args.envs))
    print("MAX air_time: %.3f s | air steps total: %d" % (
        float(a[mask].max().item()),
        int(((a[mask] >= 0.02).sum().item()))))
    ok = bool(ever.all().item())
    print("JUMP-VERDICT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
