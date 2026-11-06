# CS 521 HW4, Problem 2: Certified training for multiple norms (CURE)

Brian Kim (bmkim2)

Notebook: [multi_norm_certified_training_mnist.ipynb](multi_norm_certified_training_mnist.ipynb) (executed; raw numbers in [q2_results.json](q2_results.json)).

**Hardware.** All numbers and times come from one execution of the notebook on one NVIDIA A100-PCIE-40GB GPU
(the course cluster). The pilots of part (c) ran on the same type of GPU. The data, model, attacks, verifier,
`evaluate`, `train` and the default hyperparameters are unchanged; the only additions to the training code are the
loss functions, a helper `run` that sets the seed before `make_model` and averages over seeds, and `time_epoch`
(a controlled timing of one epoch with the same steps as `train`).

**Implementation notes.**
* `propagation_region`: $\tau = \lambda\epsilon$; the center is the starter's `pgd` (10 steps, random start) in the ball of radius $\epsilon-\tau$, which also keeps it in $[0,1]$.
* `certified_loss`: cross-entropy of $\hat z = -$`ibp_margin_bounds` on the small ball ($\hat z_y = 0$).
* `single_norm_loss`: standard loss while `eps_scale(t) = 0` (first epoch), then `certified_loss` with radius $\epsilon \cdot$ `eps_scale(t)`.
* `cure_loss`: per-image max of the two certified losses, plus $\eta$ times the mean of $\mathrm{KL}(d^2\|d^\infty)$ over $\gamma$ (images whose $\ell_2$ small ball is certified: all $\hat z^2_j < 0$, $j \neq y$); the KL term is skipped if $\gamma$ is empty.
* `gradient_projection`: RAMP Eq. 4-7: $GP(g_n, g_c) = \cos(g_n, g_c)\, g_n$ if the cosine is positive, else 0; the result is $\beta\, GP + (1-\beta)\, g_c$, per parameter tensor.

### (a) Results and answers

All networks: one seed (seed 0), 20 epochs, A100 GPU. `gap` = PGD accuracy minus certified accuracy (a large gap means loose bounds).

| network | clean | cert $\ell_\infty$ | cert $\ell_2$ | cert union | PGD $\ell_\infty$ | PGD $\ell_2$ | PGD union | time/epoch (s) |
|---|---|---|---|---|---|---|---|---|
| standard | 97.54 | 0.00 | 0.00 | 0.00 | 1.42 | 10.56 | 1.41 | 0.20 |
| $\ell_\infty$ certified | 93.56 | **80.36** | 3.72 | 3.72 | 84.71 | 70.69 | 70.41 | 1.41 |
| $\ell_2$ certified | 93.09 | 54.62 | **57.84** | **51.84** | 71.33 | 78.10 | 70.47 | 1.71 |

Time per epoch for each epoch: certified training uses the full time after the first epoch (the first epoch is standard training). The controlled measurement at full radius (section (c)) gives 0.19 s (standard), 1.41 s ($\ell_\infty$) and 1.59 s ($\ell_2$).

**(i) Does certified robustness transfer?** Only partly, and much less than empirical robustness.
* The $\ell_\infty$ network is certified for $\ell_\infty$ on 80.4% of the images, but for $\ell_2$ on only 3.7%. Its $\ell_2$ PGD accuracy is 70.7% (standard network: 10.6%), so the network *is* empirically robust to $\ell_2$, but the verifier cannot prove it: the $\ell_2$ gap is 67 points. Empirical robustness transfers; certified robustness does not.
* The $\ell_2$ network transfers better: 54.6% certified for $\ell_\infty$ (PGD 71.3%, gap 16.7 points).
* Why: the bound on the first layer is $0.1\,\|w_j\|_1$ for $\ell_\infty$ and $1.0\,\|w_j\|_2$ for $\ell_2$. The $\ell_\infty$ training makes the rows of the first layer sparse: on the (CPU) replica of this network the median $\|w_j\|_1/\|w_j\|_2$ is 6.0 (it is 16.2 for the $\ell_2$ network and at most $\sqrt{784}=28$). For sparse rows, $\|w_j\|_2$ is large compared to $0.1\|w_j\|_1$: median radius 3.2 for $\ell_2$ vs 2.1 for $\ell_\infty$, so the $\ell_2$ bounds of the $\ell_\infty$ network are loose. Also, the $\ell_\infty$ region is intersected with $[0,1]^{784}$ (on the black background a pixel can only increase), but the $\ell_2$ ball is not; this helps the transfer from $\ell_2$ to $\ell_\infty$.

**(ii) Which norm limits the union certified accuracy?**
* standard: neither norm is certified (0% for both).
* $\ell_\infty$ network: $\ell_2$ (3.7% vs 80.4%; union = 3.7%).
* $\ell_2$ network: $\ell_\infty$ (54.6% vs 57.8%; union = 51.8%). The two norms are close, and the union is lower than both, so the two certified sets are different.

### (b) Results and answers

(a): seed 0. (b): mean of seeds 0 and 1. A100 GPU. Time per epoch is the mean over the 20 epochs in `train`; the controlled time at full radius is in section (c).

| network | clean | cert $\ell_\infty$ | cert $\ell_2$ | cert union | PGD $\ell_\infty$ | PGD $\ell_2$ | PGD union | gap $\ell_\infty$ | gap $\ell_2$ | time/epoch (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| standard | **97.54** | 0.00 | 0.00 | 0.00 | 1.42 | 10.56 | 1.41 | 1.42 | 10.56 | 0.20 |
| $\ell_\infty$ certified | 93.56 | **80.36** | 3.72 | 3.72 | 84.71 | 70.69 | 70.41 | 4.35 | 66.97 | 1.41 |
| $\ell_2$ certified | 93.09 | 54.62 | **57.84** | 51.84 | 71.33 | 78.10 | 70.47 | 16.71 | 20.26 | 1.71 |
| CURE-Max | 91.82 | 65.60 | 55.76 | 55.65 | 76.19 | 77.65 | 75.44 | 10.59 | 21.89 | 3.17 |
| CURE-Max + BA | 92.08 | 65.92 | 56.16 | **56.10** | 76.32 | 78.50 | 75.97 | 10.40 | 22.34 | 3.23 |
| CURE-Max + BA + GP | 92.36 | 65.74 | 55.84 | 55.79 | 76.42 | 78.54 | 76.02 | 10.68 | 22.70 | 3.39 |

Per seed, union certified: CURE-Max 55.46 / 55.84, +BA 56.16 / 56.03, +BA+GP 55.91 / 55.67.

**(i) CURE vs. single-norm networks.** All CURE networks have a much higher union certified accuracy than the single-norm networks (55.7-56.1% vs 51.8% for $\ell_2$ and 3.7% for $\ell_\infty$). The highest union certified accuracy is **CURE-Max + BA (56.10%)**. CURE gives up single-norm certified accuracy for this: 65.9% for $\ell_\infty$ (80.4% for the $\ell_\infty$ network) and 56.2% for $\ell_2$ (57.8% for the $\ell_2$ network). The highest clean accuracy is the **standard network (97.54%)**; among the certified networks it is the $\ell_\infty$ network (93.56%), and among the CURE networks it is CURE-Max + BA + GP (92.36%). For all CURE networks, union $\approx$ $\ell_2$ certified accuracy, so $\ell_2$ limits the union, and the $\ell_2$ gap (about 22 points) is twice the $\ell_\infty$ gap (about 10.5 points).

**(ii) Do BA and GP help?** Only a little, and not more than the seed noise.
* BA: +0.45 points of union (55.65 → 56.10), and +0.40 on $\ell_2$. The difference between seeds is about 0.4 points, so this gain is small. BA makes the distribution of the $\ell_2$ bounds similar to the $\ell_\infty$ bounds, but only on the images that are already certified for $\ell_2$ on the small ball ($\gamma$). The union is limited by the images that are *not* certified for $\ell_2$, and BA does not change their loss. Also, for this small network the union is almost equal to the $\ell_2$ certified accuracy, so there are few images that are certified for $\ell_2$ but not for $\ell_\infty$ (the case that BA is designed for).
* GP: −0.31 union and +0.28 clean compared with CURE-Max + BA. This is the expected direction: GP adds the part of the clean gradient that agrees with the certified gradient, which helps the clean accuracy and the PGD accuracy, but not the bounds. In `train`, GP is used only after the radius schedule ends ($t \ge 11$), so it changes 9 of the 20 epochs.

**(iii) Time.** CURE uses about 3.2-3.4 s per epoch, versus 1.4-1.7 s for the single-norm networks: about **2× more** (1.9-2.6× in the controlled measurement: 3.04-3.65 s vs 1.41-1.59 s). For each batch, CURE does the SABR step for both norms: two PGD attacks (10 forward and backward passes each) and two IBP passes. PGD is most of the cost. GP adds two more gradient calculations per batch (a separate clean forward/backward and a separate gradient of the CURE loss), so CURE + GP is the slowest (3.65 s in the controlled measurement). Standard training is about 16× faster (0.2 s).

## (c)(i) Idea: feasibility-gated CURE (CURE-Gate)

**Observation.** In (b), $\ell_2$ limits the union for every CURE network: union $\approx$ certified $\ell_2$ (56.1 vs 56.2), and $\ell_\infty$ is certified on about 10 points more images. CURE-Max puts the *same* certified loss on every image, for both norms. But SABR already computes a PGD center $x^*_p$ inside the $\epsilon_p$-ball for each norm. If $f(x^*_p) \neq y$, the image has a counterexample inside the ball, so **no** loss can make it certified for $p$; its certified loss only takes capacity from the other images. About 6-12% of each batch is like this (plot at the end).

**Method.** For each image $i$ and each *gated* norm $p$, replace the certified loss by the clean cross-entropy if the PGD center is misclassified. We gate only $\ell_\infty$, the norm that does not limit the union, and always keep the certified loss of $\ell_2$:

$$
\ell^p_i = \begin{cases} \mathcal{L}_{CE}(f(x_i), y_i) & \text{if } p \in G \text{ and } \arg\max f(x^*_{p,i}) \neq y_i \\ \mathcal{L}_{CE}(\hat z^p_i, y_i) & \text{otherwise}\end{cases}
\qquad
\mathcal{L} = \frac1n\sum_i \max(\ell^\infty_i, \ell^2_i) + \eta\,\frac{1}{|\gamma|}\sum_{i\in\gamma}\mathrm{KL}(d^2_i \,\|\, d^\infty_i),
\qquad G = \{\ell_\infty\}.
$$

```
for each batch (x, y), radius scale s:
    for p in {linf, l2}:
        x*_p, tau_p = SABR PGD center in the (s*eps_p - tau_p)-ball, tau_p = lam * s * eps_p
        zhat_p     = -IBP margin bounds on the ball B_p(x*_p, tau_p)
        hopeless_p = argmax f(x*_p) != y          # no gradient; one extra forward pass
        l_p = CE(f(x), y) if (p in G and hopeless_p) else CE(zhat_p, y)
    loss = mean(max(l_linf, l_l2)) + eta * KL term of CURE (unchanged)
```

**Why it should help.** When the $\ell_\infty$ loss of a hopeless image is replaced by the (small) clean loss, the per-image max picks the $\ell_2$ certified loss for this image. So the gradient for this image goes to the norm that limits the union, instead of to an $\ell_\infty$ certificate that cannot exist. Since $\ell_2$ is never gated, we do not give up $\ell_2$ certificates. The gate is free: it reuses the PGD center that SABR computes anyway (one extra forward pass).

**Related work and what is new.** MART [Wang et al., ICLR 2020] treats misclassified examples differently in adversarial training, by reweighting the loss. Expressive losses / CC-IBP / MTL-IBP [De Palma et al., ICLR 2024] mix the adversarial and the IBP loss with a fixed weight for all images, and SABR [Müller et al., ICLR 2023] is the small-box loss we use. CURE [Jiang et al., 2024] and RAMP [Jiang and Singh, NeurIPS 2024] combine the losses of several norms with the same rule for all images. New here: (1) a per-image, *per-norm* hard gate of the certified loss, based on a feasibility test (the SABR PGD center) that costs nothing extra; (2) applying it *asymmetrically*: only to the norm that does not limit the union, so the freed loss goes to the limiting norm through CURE's max.

## (c)(iii) Results

Mean of seeds 0 and 1, A100 GPU. "controlled" = time of one epoch at full radius, median of 3 runs (`time_epoch`).

| method | clean | cert $\ell_\infty$ | cert $\ell_2$ | cert union | PGD $\ell_\infty$ | PGD $\ell_2$ | PGD union | gap $\ell_2$ | time/epoch (s) | controlled (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| CURE-Max + BA (best of (b)) | 92.08 | 65.92 | 56.16 | 56.10 | 76.32 | 78.50 | 75.97 | 22.34 | 3.23 | 3.22 |
| CURE-Max + BA, 23 epochs (same time) | 92.04 | 65.79 | 56.07 | 56.00 | 76.38 | 78.42 | 75.98 | 22.35 | 3.27 | 3.22 |
| **CURE-Gate (ours)** | 92.01 | **66.34** | **56.74** | **56.65** | 77.22 | 78.94 | 76.66 | 22.20 | 3.47 | 3.63 |

Per seed, union certified: CURE-Max + BA 56.16 / 56.03, ours **57.06 / 56.24**. Our method uses 13% more time per epoch (one extra forward pass of the clean image and of each PGD center), so we also trained CURE-Max + BA for $\lceil 20 \cdot 1.127\rceil = 23$ epochs: it does not improve (56.00%).

## (c)(iv) Ablation: which norms are gated

| method | clean | cert $\ell_\infty$ | cert $\ell_2$ | cert union | PGD union | gap $\ell_2$ | final training loss (seed 0) |
|---|---|---|---|---|---|---|---|
| CURE-Max + BA | 92.08 | 65.92 | 56.16 | 56.10 | 75.97 | 22.34 | 1.166 |
| ours: gate $\ell_\infty$ only | 92.01 | 66.34 | 56.74 | **56.65** | 76.66 | 22.20 | 1.163 |
| ablation: gate both norms | **93.91** | 67.07 | 55.46 | 55.42 | **78.19** | 25.02 | 0.934 |

Per seed, union certified of the ablation: 56.05 / 54.78.

## (c)(v) Analysis

**Does it work?** Partly. Gating only $\ell_\infty$ increases the union certified accuracy by +0.55 points (56.10 → 56.65) at the same clean accuracy (92.01 vs 92.08), with the same gain in both seeds (+0.90 and +0.21), and it is better than CURE with the same training time (56.00). The gain is small, and with two seeds we cannot say it is larger than the noise with high confidence.

**Why it works (evidence).**
* The gain is in $\ell_2$, the norm that limits the union: certified $\ell_2$ +0.58 (56.16 → 56.74), and the union follows it (+0.55). Certified $\ell_\infty$ also increases (+0.42), so removing the impossible $\ell_\infty$ targets does not hurt the $\ell_\infty$ certificates of the other images.
* The training loss is almost the same as CURE-Max + BA (1.163 vs 1.166). This agrees with the mechanism: for a gated image the max of the two losses moves from the $\ell_\infty$ loss to the $\ell_2$ certified loss, so the total loss changes little, but the gradient of 6-12% of the batch (plot) goes to $\ell_2$.
* The $\ell_2$ gap gets a little smaller (22.20 vs 22.34): the extra certificates come from tighter bounds, not only from more empirical robustness.

**Why gating both norms is different (ablation).** If $\ell_2$ is gated too, the images that are hopeless for both norms are trained only with the clean loss. The training loss drops (0.934), the clean accuracy increases by +1.8 points (93.91) and the PGD union by +2.2 points, but the certified $\ell_2$ drops by 0.7 and the $\ell_2$ gap grows from 22.3 to 25.0 points. The freed capacity goes to clean and empirical robustness, not to certificates of the limiting norm. So the asymmetry is the important part of the method: gate the norm that does not limit the union, keep the certified loss of the norm that does. The ablation is still useful if clean accuracy matters more: +1.8 clean for −0.7 union.

**Why the gain is small.** The bottleneck is the looseness of the $\ell_2$ bounds: the gap is about 22 points for every method, and the $\ell_2$ network of (a) reaches only 57.8%. Gating changes *which* images get the $\ell_2$ loss, not how tight the $\ell_2$ bounds can be for this network, so it cannot remove most of the gap. Attempts to tighten the $\ell_2$ bounds directly did not work (next section).

## Process

1. Implemented and checked (a) and (b) first (also on the Mac CPU: the numbers were within 2 points of the GPU numbers; the GPU and CPU random streams differ).
2. Diagnosis: union $\approx$ certified $\ell_2$, and the $\ell_2$ gap is 2x the $\ell_\infty$ gap, so $\ell_2$ bounds are the bottleneck. On the trained networks, the verifier certifies almost all images with IBP; CROWN adds about 1% (measured on 2,000 test images).
3. Pilots (seed 0 unless noted, A100, union / clean). Same-GPU CURE-Max + BA reference: 56.16 / 91.83.

| pilot | idea | union | clean |
|---|---|---|---|
| verifier-aligned $\ell_2$ bound | $\ell_2$ loss on $\max(\text{IBP}, \text{CROWN})$, the verifier's bound (CROWN intermediate bounds without gradient) | 53.42 | 91.30 |
| CROWN-IBP $\ell_2$ bound [Zhang et al., ICLR 2020] | $\ell_2$ loss on $\max(\text{IBP}, \text{CROWN-IBP})$ | 55.51 | 92.63 |
| union logits | CE of the elementwise max $\max(\hat z^\infty, \hat z^2)$ | 56.14 | 91.66 |
| focus | extra $\ell_2$ loss on images certified for $\ell_\infty$ but not $\ell_2$ | 56.41 | 91.73 |
| ReLU stability [Xiao et al., ICLR 2019], weight 0.01 / 0.1 | regularizer on the $\ell_2$ ball | 56.01 / 51.26 | 89.51 / 83.75 |
| full $\ell_2$ ball ($\lambda_2 = 1$, diagnostic only) | IBP on the full $\ell_2$ ball | 56.26 | 85.22 |
| gate both, adversarial fallback (2 seeds) | CE at $x^*_p$ instead of clean CE | 54.82 | 93.58 |
| gate both, soft 0.5 (2 seeds) | average of clean and certified loss | 55.60 | 93.17 |
| **gate both, clean fallback (2 seeds)** | became the ablation | 55.42 | 93.91 |
| **gate $\ell_\infty$ only (2 seeds)** | final method | **56.65** | 92.01 |

The tighter bounds made the union *worse* (53.4% with the verifier's own CROWN bound). This agrees with "the paradox of certified training" [Jovanović et al., TMLR 2022]: tighter relaxations such as CROWN give less smooth losses and worse certified training than IBP. After these negative results we changed the direction from the bounds to *which images* get the certified loss, which led to the gating.

## References

* [GDS+19] Gowal et al. On the effectiveness of interval bound propagation for training verifiably robust models, 2019.
* [JCS24] Jiang, Cheung, Singh. Towards universal certified robustness with multi-norm training (CURE), 2024.
* Jiang and Singh. RAMP: Boosting adversarial robustness against multiple $l_p$ perturbations for universal robustness. NeurIPS 2024 (gradient projection, Eq. 4-7).
* Müller, Eckert, Fischer, Vechev. Certified training: Small boxes are all you need (SABR). ICLR 2023.
* Wang, Zou, Yi, Bailey, Ma, Gu. Improving adversarial robustness requires revisiting misclassified examples (MART). ICLR 2020.
* De Palma, Bunel, Dvijotham, Kumar, Stanforth, Lomuscio. Expressive losses for verified robustness via convex combinations. ICLR 2024.
* Jovanović, Balunović, Baader, Vechev. On the paradox of certified training. TMLR 2022.
* Zhang et al. Towards stable and efficient training of verifiably robust neural networks (CROWN-IBP). ICLR 2020.
* Xiao, Tjeng, Shafiullah, Madry. Training for faster adversarial robustness verification via inducing ReLU stability. ICLR 2019.
