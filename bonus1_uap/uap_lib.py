# %% [setup]
import itertools
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.optimize import Bounds, LinearConstraint, milp

torch.set_num_threads(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS = 0.1          # l_inf radius of the universal perturbation (CITRUS, MNIST setting)
LAM = 0.5          # tau = LAM * eps: radius of the small propagation boxes
GROUP = 5          # images per group: CITRUS training batch size and size of each certification set
BATCH = 500        # 100 groups of 5 for each optimizer step
WARMUP, RAMP, EPOCHS = 1, 10, 20
LR = 4e-3
KAPPA = 1.0        # margin saturation of the group-UAP objective
CKPT_DIR = "checkpoints"


# %% [data]
def load_mnist(train):
    ds = torchvision.datasets.MNIST("data", train=train, download=True)
    x = ds.data.float().div(255).unsqueeze(1)
    return x.to(device), ds.targets.to(device)


x_train, y_train = load_mnist(train=True)
x_test, y_test = load_mnist(train=False)


def make_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 50), nn.ReLU(),
        nn.Linear(50, 50), nn.ReLU(),
        nn.Linear(50, 50), nn.ReLU(),
        nn.Linear(50, 10),
    ).to(device)


def linear_layers(model):
    return [m for m in model if isinstance(m, nn.Linear)]


def margins(logits, y):
    # z_y - max_{j != y} z_j  (negative = misclassified)
    zy = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    other = logits.scatter(-1, y.unsqueeze(-1), -float("inf")).max(-1).values
    return zy - other


# %% [ibp]
def ibp_margin_lower(model, c, r, y):
    # Lower bounds of z_y - z_j on the box [c - r, c + r] (the last layer is merged into the spec).
    # No clipping to [0, 1]: the UAP threat model is x + u with ||u||_inf <= eps (as in Zeng et al.).
    layers = linear_layers(model)
    c = c.flatten(1)
    first = layers[0]
    mid, rad = first(c), r * first.weight.abs().sum(1)
    l, u = mid - rad, mid + rad
    for layer in layers[1:-1]:
        l, u = F.relu(l), F.relu(u)
        mid, rad = layer((u + l) / 2), ((u - l) / 2) @ layer.weight.abs().t()
        l, u = mid - rad, mid + rad
    l, u = F.relu(l), F.relu(u)
    last = layers[-1]
    w = last.weight[y].unsqueeze(1) - last.weight.unsqueeze(0)
    b = last.bias[y].unsqueeze(1) - last.bias.unsqueeze(0)
    mid, rad = (u + l) / 2, (u - l) / 2
    return (w @ mid.unsqueeze(2)).squeeze(2) - (w.abs() @ rad.unsqueeze(2)).squeeze(2) + b


def ibp_loss(model, c, r, y):
    # L_IBP of Eq. 2 of CITRUS: CE of the negative margin lower bounds (entry y is 0).
    return F.cross_entropy(-ibp_margin_lower(model, c, r, y), y)


# %% [attacks]
def pgd_delta(model, x, y, eps, steps):
    # Per-input PGD in the l_inf ball of radius eps (SABR / CITRUS centers).
    alpha = 2.5 * eps / steps
    delta = (torch.rand_like(x) * 2 - 1) * eps
    for _ in range(steps):
        delta.requires_grad_(True)
        g, = torch.autograd.grad(F.cross_entropy(model(x + delta), y), delta)
        delta = (delta.detach() + alpha * g.sign()).clamp(-eps, eps)
    return delta.detach()


def group_uap(model, x, y, eps, steps, objective="sat"):
    # One shared perturbation per group. x: (G, N, 1, 28, 28), y: (G, N). Returns u: (G, 1, 1, 28, 28).
    # objective "sat": maximize sum_i -max(m_i, -KAPPA): once an image is fooled by KAPPA its term is saturated,
    # so the perturbation moves on to fool more images of the group (a k-common perturbation).
    # objective "ce": maximize the summed cross-entropy (the easiest image dominates).
    G, N = x.shape[:2]
    alpha = 2.5 * eps / steps
    u = (torch.rand(G, 1, *x.shape[2:], device=x.device) * 2 - 1) * eps
    for _ in range(steps):
        u.requires_grad_(True)
        logits = model((x + u).flatten(0, 1)).view(G, N, -1)
        if objective == "sat":
            loss = -margins(logits, y).clamp(min=-KAPPA).sum()
        else:
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum")
        g, = torch.autograd.grad(loss, u)
        u = (u.detach() + alpha * g.sign()).clamp(-eps, eps)
    return u.detach()


# %% [losses]
def eps_scale(t):
    return 0.0 if t <= WARMUP else min((t - WARMUP) / RAMP, 1.0)


def standard_loss(model, x, y, t):
    return F.cross_entropy(model(x), y)


def ibp_train_loss(model, x, y, t):
    eps = EPS * eps_scale(t)
    if eps == 0:
        return standard_loss(model, x, y, t)
    return ibp_loss(model, x, eps, y)


def sabr_loss(model, x, y, t, lam=LAM):
    eps = EPS * eps_scale(t)
    if eps == 0:
        return standard_loss(model, x, y, t)
    tau = lam * eps
    delta = pgd_delta(model, x, y, eps - tau, steps=8)
    return ibp_loss(model, x + delta, tau, y)


def citrus_loss(model, x, y, t, lam=LAM, n=GROUP):
    # CITRUS (Xu and Singh, 2024): for each image i of a group, the IBP loss on the boxes B(x_i + v_j, tau), j != i,
    # where v_j is the PGD perturbation of image j (20 steps, in the ball of radius eps - tau).
    # Averaged over the N(N-1) pairs (CITRUS sums them; this only rescales the loss).
    eps = EPS * eps_scale(t)
    if eps == 0:
        return standard_loss(model, x, y, t)
    tau = lam * eps
    v = pgd_delta(model, x, y, eps - tau, steps=20)
    G = len(x) // n
    xg, vg, yg = x.view(G, n, *x.shape[1:]), v.view(G, n, *x.shape[1:]), y.view(G, n)
    pairs = torch.tensor([(i, j) for i in range(n) for j in range(n) if i != j], device=x.device)
    centers = xg[:, pairs[:, 0]] + vg[:, pairs[:, 1]]           # (G, n(n-1), 1, 28, 28)
    labels = yg[:, pairs[:, 0]]
    return ibp_loss(model, centers.flatten(0, 1), tau, labels.flatten())


def group_box_loss(model, x, y, t, lam=LAM, n=GROUP, objective="sat"):
    # Our method (G-CITRUS): one shared group-UAP u_G per group of n images, found by PGD on the saturated
    # group objective in the ball of radius eps - tau; then the IBP loss on B(x_i + u_G, tau) for every image.
    # With lam = 0 this is UAT-style group adversarial training (no boxes).
    eps = EPS * eps_scale(t)
    if eps == 0:
        return standard_loss(model, x, y, t)
    tau = lam * eps
    G = len(x) // n
    xg, yg = x.view(G, n, *x.shape[1:]), y.view(G, n)
    u = group_uap(model, xg, yg, eps - tau, steps=20, objective=objective)
    centers = (xg + u).flatten(0, 1)
    if tau == 0:
        return F.cross_entropy(model(centers), y)
    return ibp_loss(model, centers, tau, y)


# %% [train]
def train(model, loss_fn, epochs=EPOCHS, lr=LR, batch_size=BATCH, seed=0, verbose=True):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[15], gamma=0.1)
    n_batches = len(x_train) // batch_size
    epoch_times = []
    for epoch in range(epochs):
        model.train()
        start, total = time.time(), 0.0
        idx = torch.randperm(len(x_train), device=device)
        for i in range(n_batches):
            j = idx[i * batch_size:(i + 1) * batch_size]
            loss = loss_fn(model, x_train[j], y_train[j], epoch + i / n_batches)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()
        epoch_times.append(time.time() - start)
        if verbose:
            print(f"epoch {epoch + 1:2d}  loss {total / n_batches:.4f}  time {epoch_times[-1]:.1f} s", flush=True)
    return epoch_times


LOSSES = {
    "standard": standard_loss,
    "ibp": ibp_train_loss,
    "sabr": sabr_loss,
    "citrus": citrus_loss,
    "gcitrus": group_box_loss,                                                  # our method
    "gcitrus_ce": lambda m, x, y, t: group_box_loss(m, x, y, t, objective="ce"),  # ablation: CE group objective
    "gcitrus_tau0": lambda m, x, y, t: group_box_loss(m, x, y, t, lam=0.0),       # ablation: no boxes
    "gcitrus_n2": lambda m, x, y, t: group_box_loss(m, x, y, t, n=2),             # ablation: groups of 2
}
for _lam in [0.3, 0.7]:                                                           # trade-off curves (lambda sweep)
    LOSSES[f"citrus_l{_lam}"] = lambda m, x, y, t, lam=_lam: citrus_loss(m, x, y, t, lam=lam)
    LOSSES[f"gcitrus_l{_lam}"] = lambda m, x, y, t, lam=_lam: group_box_loss(m, x, y, t, lam=lam)


def get_model(name, seed=0, verbose=True):
    # Trains the model, or loads it from CKPT_DIR if it was trained before.
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"{name}_s{seed}.pt")
    model = make_model()
    if os.path.exists(path):
        ck = torch.load(path, map_location=device)
        model.load_state_dict(ck["state"])
        return model, ck["epoch_times"]
    torch.manual_seed(seed)
    model = make_model()
    times = train(model, LOSSES[name], seed=seed, verbose=verbose)
    torch.save({"state": model.state_dict(), "epoch_times": times}, path)
    return model, times


# %% [crown]
def _relu_relaxation(l, u):
    active, unstable = l >= 0, (l < 0) & (u > 0)
    s = torch.where(active, torch.ones_like(l), torch.zeros_like(l))
    s = torch.where(unstable, u / (u - l).clamp(min=1e-12), s)
    c = torch.where(unstable, -s * l, torch.zeros_like(l))
    alpha = torch.where(active | (unstable & (u > -l)), torch.ones_like(l), torch.zeros_like(l))
    return alpha, s, c


def _backward(A, layers, bounds):
    # Linear lower bound A' x' + const of A z (z = output of layers[-1]) that holds for all x' with the given
    # intermediate bounds. Returns (A' (B, m, 784), const (B, m)).
    const = torch.zeros(A.shape[:2], device=A.device)
    for i in range(len(layers) - 1, -1, -1):
        const = const + A @ layers[i].bias
        A = A @ layers[i].weight
        if i > 0:
            alpha, s, c = _relu_relaxation(*bounds[i - 1])
            a_pos, a_neg = A.clamp(min=0), A.clamp(max=0)
            const = const + (a_neg * c.unsqueeze(1)).sum(2)
            A = a_pos * alpha.unsqueeze(1) + a_neg * s.unsqueeze(1)
    return A, const


def _concretize(A, const, x, eps):
    return const + (A @ x.unsqueeze(2)).squeeze(2) - eps * A.abs().sum(2)


@torch.no_grad()
def linear_margin_bounds(model, x, y, eps):
    # CROWN linear bounds of the margins on the box [x - eps, x + eps] (no clipping).
    # Returns A (B, 10, 784), c (B, 10) with z_y - z_j >= A_j x' + c_j for all x' in the box, and the
    # concrete lower bounds max(CROWN, IBP) (B, 10). The intermediate bounds are the tighter of IBP and CROWN.
    layers = linear_layers(model)
    x = x.flatten(1)
    bounds, l, u = [], None, None
    for k in range(len(layers) - 1):
        lay = layers[k]
        if k == 0:
            mid, rad = lay(x), eps * lay.weight.abs().sum(1)
            il, iu = mid - rad, mid + rad
        else:
            pl, pu = F.relu(l), F.relu(u)
            mid, rad = lay((pu + pl) / 2), ((pu - pl) / 2) @ lay.weight.abs().t()
            il, iu = mid - rad, mid + rad
        eye = torch.eye(lay.out_features, device=x.device).expand(len(x), -1, -1)
        lo = _concretize(*_backward(eye, layers[:k + 1], bounds), x, eps)
        hi = -_concretize(*_backward(-eye, layers[:k + 1], bounds), x, eps)
        l, u = torch.maximum(lo, il), torch.minimum(hi, iu)
        bounds.append((l, u))
    eye = torch.eye(10, device=x.device)
    spec = eye[y].unsqueeze(1) - eye.unsqueeze(0)
    A, c = _backward(spec, layers, bounds)
    lower = torch.maximum(_concretize(A, c, x, eps), ibp_margin_lower(model, x, eps, y))
    return A, c, lower


# %% [certify]
def max_fooled_milp(A, c, lower, x, eps, time_limit=30.0):
    # Upper bound on the number of images of one set that one shared u (||u||_inf <= eps) can misclassify.
    # A: (k, 10, 784), c, lower: (k, 10), x: (k, 784) for the k images of the set that are not certified alone.
    # Image i is misclassified only if some margin j is <= 0, which needs A_ij (x_i + u) + c_ij <= 0.
    # MILP: binaries s_ij (image i is fooled through class j), z_i <= sum_j s_ij, maximize sum_i z_i.
    k, d = len(A), x.shape[1]
    rows, cols = [], []
    for i in range(k):
        for j in range(10):
            if lower[i, j] <= 0:          # class j is still possible for image i
                rows.append(i); cols.append(j)
    m = len(rows)
    nv = d + m + k                        # u, s, z
    Acon = np.zeros((m + k, nv)); lb = np.full(m + k, -np.inf); ub = np.zeros(m + k)
    for r, (i, j) in enumerate(zip(rows, cols)):
        a = A[i, j]
        b = float(a @ x[i] + c[i, j])
        M = max(b + eps * float(a.abs().sum()), 0.0)          # max of a (x_i + u) + c over the box
        Acon[r, :d] = a.numpy()
        Acon[r, d + r] = M                                    # a u + M s_ij <= M - b
        ub[r] = M - b
    for i in range(k):
        Acon[m + i, d + m + i] = 1.0                          # z_i - sum_j s_ij <= 0
        for r in range(m):
            if rows[r] == i:
                Acon[m + i, d + r] = -1.0
    obj = np.zeros(nv); obj[d + m:] = -1.0
    integrality = np.zeros(nv); integrality[d:] = 1
    lo = np.concatenate([np.full(d, -eps), np.zeros(m + k)])
    hi = np.concatenate([np.full(d, eps), np.ones(m + k)])
    res = milp(obj, constraints=LinearConstraint(Acon, lb, ub), integrality=integrality, bounds=Bounds(lo, hi),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if res.status == 0:
        return int(round(-res.fun)), torch.tensor(res.x[:d], dtype=torch.float32), True
    dual = getattr(res, "mip_dual_bound", None)                 # time limit: use the solver's bound (sound)
    bound = k if dual is None or not np.isfinite(dual) else min(k, int(np.floor(-dual + 1e-6)))
    u = torch.tensor(res.x[:d], dtype=torch.float32) if res.x is not None else None
    return bound, u, False


@torch.no_grad()
def certify_sets(model, x, y, eps=EPS, n=GROUP, chunk=500):
    # Certified worst-case UAP accuracy, averaged over consecutive sets of n test images (CITRUS, App. B).
    # For each set: images that are misclassified count as wrong, images that CROWN/IBP certify alone count as
    # correct, and the MILP bounds how many of the other images one shared perturbation can misclassify.
    model.eval()
    correct_lb, indiv, n_milp, n_timeout, milp_us = 0, 0, 0, 0, {}
    for s0 in range(0, len(x), chunk):
        xb, yb = x[s0:s0 + chunk], y[s0:s0 + chunk]
        clean = model(xb).argmax(1) == yb
        A, c, lower = linear_margin_bounds(model, xb, yb, eps)
        lower[torch.arange(len(yb)), yb] = float("inf")
        cert = clean & (lower.min(1).values > 0)
        indiv += cert.sum().item()
        xf = xb.flatten(1)
        for g in range(0, len(xb), n):
            idx = torch.arange(g, min(g + n, len(xb)))
            unc = idx[clean[idx] & ~cert[idx]]
            fooled = len(unc)
            if len(unc) >= 2:
                fooled, u, ok = max_fooled_milp(A[unc], c[unc], lower[unc], xf[unc], eps)
                n_milp += 1; n_timeout += not ok
                if u is not None:
                    milp_us[s0 + g] = u
            correct_lb += clean[idx].sum().item() - fooled
    N = len(x)
    return {"cert_uap": 100 * correct_lb / N, "cert_indiv": 100 * indiv / N, "n_milp": n_milp,
            "n_timeout": n_timeout}, milp_us


# %% [uap_attacks]
SUBSETS = torch.tensor([m for m in itertools.product([0, 1], repeat=GROUP) if any(m)], dtype=torch.float32)


def attack_sets(model, x, y, eps=EPS, n=GROUP, steps=50, restarts=2, chunk=100, extra_us=None):
    # Empirical worst-case UAP accuracy for the same sets of n images (an upper bound of the true value).
    # For each set, PGD on one shared u for each of the 2^n - 1 subsets (saturated margin loss on the subset),
    # with random restarts; the MILP solutions (extra_us) are also tried. A set's count is its best attack.
    model.eval()
    masks = SUBSETS.to(x.device)
    K = len(masks)
    S = len(x) // n
    xs, ys = x[:S * n].view(S, n, -1), y[:S * n].view(S, n)
    best = torch.zeros(S, device=x.device)
    alpha = 2.5 * eps / steps
    for s0 in range(0, S, chunk):
        xc, yc = xs[s0:s0 + chunk], ys[s0:s0 + chunk]
        C = len(xc)
        yk = yc[:, None].expand(C, K, n)
        for _ in range(restarts):
            u = (torch.rand(C, K, 1, xc.shape[-1], device=x.device) * 2 - 1) * eps
            for _ in range(steps):
                u.requires_grad_(True)
                logits = model((xc[:, None] + u).reshape(-1, xc.shape[-1])).view(C, K, n, -1)
                loss = -(margins(logits, yk).clamp(min=-KAPPA) * masks).sum()
                g, = torch.autograd.grad(loss, u)
                u = (u.detach() + alpha * g.sign()).clamp(-eps, eps)
            with torch.no_grad():
                logits = model((xc[:, None] + u).reshape(-1, xc.shape[-1])).view(C, K, n, -1)
                fooled = (logits.argmax(-1) != yk).sum(-1).float().max(1).values
                best[s0:s0 + C] = torch.maximum(best[s0:s0 + C], fooled)
    with torch.no_grad():
        for i, u in (extra_us or {}).items():
            s = i // n
            if s < S:
                fooled = (model(xs[s] + u.to(x.device)).argmax(-1) != ys[s]).sum().float()
                best[s] = torch.maximum(best[s], fooled)
    return 100 * (1 - best.sum().item() / (S * n))


def dataset_uap(model, x, y, eps=EPS, epochs=5, batch_size=1000, steps_per_batch=5):
    # One universal perturbation for the whole test set (sign-gradient ascent on the saturated loss over
    # mini-batches, 3 random restarts). Returns the accuracy under the best perturbation found.
    model.eval()
    best_acc, best_u = 100.0, None
    for r in range(3):
        u = ((torch.rand(1, *x.shape[1:], device=x.device) * 2 - 1) * eps) if r else torch.zeros(1, *x.shape[1:], device=x.device)
        for _ in range(epochs):
            for s0 in range(0, len(x), batch_size):
                xb, yb = x[s0:s0 + batch_size], y[s0:s0 + batch_size]
                for _ in range(steps_per_batch):
                    u.requires_grad_(True)
                    loss = -margins(model(xb + u), yb).clamp(min=-KAPPA).sum()
                    g, = torch.autograd.grad(loss, u)
                    u = (u.detach() + 0.005 * g.sign()).clamp(-eps, eps)
        with torch.no_grad():
            acc = 100 * (model(x + u).argmax(1) == y).float().mean().item()
        if acc < best_acc:
            best_acc, best_u = acc, u
    return best_acc, best_u


# %% [evaluate]
RESULTS_PATH = "results.json"


def evaluate(model, name, n_test=10000, use_cache=True):
    cache = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}
    if use_cache and name in cache:
        return cache[name]
    x, y = x_test[:n_test], y_test[:n_test]
    t0 = time.time()
    with torch.no_grad():
        clean = 100 * (model(x).argmax(1) == y).float().mean().item()
    cert, milp_us = certify_sets(model, x, y)
    t1 = time.time()
    att = attack_sets(model, x, y, extra_us=milp_us)
    t2 = time.time()
    uap, _ = dataset_uap(model, x, y)
    res = {"name": name, "clean": clean, **cert, "attack_uap_sets": att, "attack_uap_global": uap,
           "cert_time": t1 - t0, "attack_time": t2 - t1}
    cache = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}
    cache[name] = res
    json.dump(cache, open(RESULTS_PATH, "w"), indent=1)
    return res


# %% [diagnostics]
def box_diagnostics(model, n_img=2000, lam=LAM, n=GROUP, seed=0):
    # For training images and the full eps: the IBP loss of each kind of propagation box, and how often its center
    # is adversarial (at the point). SI = same-input SABR box, CI = CITRUS cross-input boxes, G = group-UAP box.
    torch.manual_seed(seed)
    model.eval()
    x, y = x_train[:n_img], y_train[:n_img]
    eps, tau = EPS, lam * EPS
    G = n_img // n
    xg, yg = x.view(G, n, *x.shape[1:]), y.view(G, n)
    v = pgd_delta(model, x, y, eps - tau, steps=20)
    vg = v.view(G, n, *x.shape[1:])
    u = group_uap(model, xg, yg, eps - tau, steps=20)
    pairs = torch.tensor([(i, j) for i in range(n) for j in range(n) if i != j])
    out = {}
    with torch.no_grad():
        def box_loss(c, lab):
            return F.cross_entropy(-ibp_margin_lower(model, c, tau, lab), lab, reduction="none")
        def adv(c, lab):
            return (model(c).argmax(1) != lab).float()
        si = box_loss(x + v, y)
        ci = box_loss((xg[:, pairs[:, 0]] + vg[:, pairs[:, 1]]).flatten(0, 1), yg[:, pairs[:, 0]].flatten())
        ci = ci.view(G, n, n - 1)
        gb = box_loss((xg + u).flatten(0, 1), y)
        out["loss_SI"] = si.mean().item()
        out["loss_CI_mean"] = ci.mean().item()
        out["loss_CI_max_over_j"] = ci.max(2).values.mean().item()
        out["frac_CI_boxes_loss<0.01"] = (ci < 0.01).float().mean().item()
        out["loss_G"] = gb.mean().item()
        out["frac_G_boxes_loss<0.01"] = (gb < 0.01).float().mean().item()
        out["adv_SI_center"] = adv(x + v, y).mean().item()
        out["adv_CI_center"] = adv((xg[:, pairs[:, 0]] + vg[:, pairs[:, 1]]).flatten(0, 1),
                                   yg[:, pairs[:, 0]].flatten()).mean().item()
        out["adv_G_center"] = adv((xg + u).flatten(0, 1), y).mean().item()
        fooled = adv((xg + u).flatten(0, 1), y).view(G, n).sum(1)
        out["groups_uG_fools>=2"] = (fooled >= 2).float().mean().item()
    return out
