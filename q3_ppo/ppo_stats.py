import json, sys, numpy as np, torch, torch.nn as nn
ns = {}
for c in json.load(open(sys.argv[1]))["cells"]:
    s = "".join(c["source"])
    if c["cell_type"] != "code": continue
    if "all_runs" in s: break
    exec(s, ns)
LOG = []
def ppo_update(agent, opt, b, cfg):
    n = len(b["obs"]); mb_size = n // cfg["num_minibatches"]; dev, vl = [], []
    raw = b["adv"]
    for _ in range(cfg["update_epochs"]):
        idx = torch.randperm(n)
        for start in range(0, n, mb_size):
            mb = idx[start:start + mb_size]
            _, logp, ent, v = agent.get_action_and_value(b["obs"][mb], b["actions"][mb])
            ratio = (logp - b["logp"][mb]).exp()
            adv = b["adv"][mb]
            if cfg["norm_adv"]: adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg = torch.max(-adv * ratio, -adv * ratio.clamp(1 - cfg["clip_coef"], 1 + cfg["clip_coef"])).mean()
            v_loss = 0.5 * ((v - b["returns"][mb]) ** 2).mean()
            loss = pg - cfg["ent_coef"] * ent.mean() + cfg["vf_coef"] * v_loss
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(), cfg["max_grad_norm"]); opt.step()
            d = (ratio - 1).abs().detach(); dev.append(((d > 0.2).float().mean().item(), d.max().item())); vl.append(v_loss.item())
    # relative std of the raw advantages (std / mean |return|) as a noise proxy
    LOG.append((np.mean([a for a, _ in dev]), np.max([m for _, m in dev]), raw.std().item(), np.mean(vl)))
    return {"clip_frac": LOG[-1][0]}
ns["ppo_update"] = ppo_update
for name, cfg in [("PPO", {}), ("no clip", {"clip_coef": 1e9}), ("no GAE", {"gae_lambda": 1.0})]:
    for seed in [0, 1]:
        LOG.clear(); ns["train_ppo"](seed, cfg)
        L = np.array(LOG); h = len(L) // 2
        print(f"{name:8s} seed {seed}: frac |r-1|>0.2  first half {L[:h,0].mean():.3f} second half {L[h:,0].mean():.3f}; "
              f"max |r-1| {L[:,1].max():.2f}; raw adv std first/second half {L[:h,2].mean():6.2f}/{L[h:,2].mean():6.2f}; "
              f"value loss {L[:,3].mean():7.1f}")
