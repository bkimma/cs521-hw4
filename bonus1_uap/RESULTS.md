# Bonus Research Question 1: G-CITRUS (group-UAP centered propagation boxes)

Notebook: [uap_certified_training.ipynb](uap_certified_training.ipynb) (executed, with outputs). The code is in
[uap_lib.py](uap_lib.py); the notebook contains the same code as cells. The drivers `run_all.py`, `run_sweep.py` and
`run_extra.py` produced `checkpoints/` (not committed), `results.json` and `diagnostics.json`.

## Setting

* MNIST, the 3x50 network of Problems 1 and 2, ℓ∞ UAP with ε = 0.1 (the CITRUS MNIST setting). x + u is not clipped (as in Zeng et al.).
* 20 epochs (1 warm-up, ε ramp over 10), Adam 4e-3, batches of 500 = 100 groups of 5, τ = λε with λ = 0.5. CPU only.
* Metric (CITRUS, App. B): certified worst-case UAP accuracy averaged over sets of 5 test images (all 10,000).
  * Sound lower bound: CROWN linear bounds for each image, plus an exact HiGHS MILP over one shared u (a Zeng et al. variant).
  * Upper bound: PGD on a shared u for each of the 31 subsets of each set, plus the MILP solutions.

## Method

For each group of 5 images, find one shared perturbation u_G by PGD (20 steps, ball of radius ε − τ). The objective is the
saturated sum Σ_i −max(margin_i, −κ). Then train on the IBP loss of B(x_i + u_G, τ) for every image of the group.
CITRUS instead uses the other images' per-input PGD perturbations: N(N−1) boxes and N PGD runs per group. G-CITRUS
uses N boxes and one shared PGD run. Related work: CITRUS (Xu and Singh 2024), SABR (Müller et al. 2023),
UAT (Shafahi et al. 2020), C&W confidence loss.

## Results (λ = 0.5; mean of 2 seeds except standard)

| method | clean | cert. indiv. | cert. UAP | attack UAP (sets) | attack UAP (global) | s/epoch |
|---|---|---|---|---|---|---|
| standard | 97.50 | 0.00 | 0.00 | 0.31 | 17.68 | 0.3 |
| IBP | 87.67 | 74.18 | 74.22 | 76.59 | 85.64 | 1.0 |
| SABR | 91.99 | 67.58 | 67.72 | 79.78 | 90.19 | 4.2 |
| CITRUS | 94.17 | 59.04 | 59.55 | 77.20 | 92.07 | 8.2 |
| G-CITRUS N=5 (ours) | 93.60 | 64.00 | 64.44 | 78.81 | 92.30 | 4.1 |
| G-CITRUS N=2 (ours) | 93.19 | 66.62 | 66.82 | 79.50 | 91.73 | 2.4 |

λ sweep (clean / cert. UAP; seed 0, except λ = 0.5, which is the mean of 2 seeds):

| λ | CITRUS | G-CITRUS |
|---|---|---|
| 0.3 | 95.68 / 38.61 | 94.59 / 46.80 |
| 0.5 | 94.17 / 59.55 | 93.60 / 64.44 |
| 0.7 | 92.26 / 69.69 | 91.81 / 72.83 |

Ablations (seed 0; G-CITRUS itself gives 93.75 / 64.01):

| ablation | clean / cert. UAP |
|---|---|
| summed CE instead of the saturated objective | 92.68 / 66.35 (moves toward SABR) |
| no boxes (τ = 0, UAT-like) | 97.85 / **0.01** (but 76.1% under the set attack) |
| groups of 2 | 93.33 / 68.27 (seed 1: 93.05 / 65.36) |

## Assessment

* **No dominance.** At the same λ, G-CITRUS trades about 0.5–1 point of clean accuracy for 3–8 points of certified UAP accuracy. It does not beat CITRUS on both metrics at once.
* **Trade-off curve.** At the same clean accuracy (interpolating CITRUS's λ curve), G-CITRUS is +1.8 (λ = 0.5), +2.0 (N = 2) and +0.7 (λ = 0.7) points better. It is −6.8 worse at λ = 0.3.
* **Speed.** It trains about 1.5–2x faster than CITRUS.
* **Uncertainty.** The gains are about the size of the seed spread (2 seeds at most).
* **Diagnostics.** The group UAP fools ≥ 2 images in only 7–9% of the groups. In practice it is one image's adversarial perturbation shared with the others, so G-CITRUS sits between SABR and CITRUS. The "k-common perturbation" motivation holds only for a small fraction of groups.
* **Single-input baselines.** G-CITRUS does not dominate SABR either: SABR reaches 91.99 / 67.72 (clean / certified UAP) and G-CITRUS with N = 2 reaches 93.19 / 66.82, so the two lie on the same trade-off curve. Plain IBP has the highest certified UAP accuracy of all methods (74.22) at a much lower clean accuracy (87.67). With this small network and verifier, the cross-input structure of the training does not give a clear gain over single-input certified training; its benefit over CITRUS is mainly speed and a better trade-off at λ ≥ 0.5.
* **Relational MILP.** It adds ≤ 1.5 points over per-image certification. For this network and verifier, the certified UAP accuracy is mostly per-image certified accuracy.
* **Certified–attack gap.** It is large for the cross-input methods (14–18 points) and small for IBP (2.4), so a tighter relational verifier (RACoon) would likely favor CITRUS and G-CITRUS.
* **Not comparable to the paper.** The network is small and the training is short on CPU, so the absolute numbers are not comparable to the CITRUS paper's CNN7 results.
