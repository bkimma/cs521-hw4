# CS 521 HW4, Problem 1: IBP training — results

Notebook: [q1_ibp_training.ipynb](q1_ibp_training.ipynb) (executed, outputs saved).

Setup: network 784-50-50-50-10 (ReLU), no normalization. Adam lr 1e-3, batch 128, 20 epochs, lr /10 after epoch 15. Schedule: epoch 0-1 warm-up (kappa=1, eps=0); epochs 1-11 linear ramp kappa 1->0.5, eps_train 0->0.1 (per batch); then constant. Last layer merged with the specification (Eq. 9). PGD: l_inf, 50 steps, step 2.5*eps/50, 1 random start. CPU (M2, 2 threads).

## Results of (a)

| model | clean accuracy | PGD accuracy ($\epsilon = 0.1$, 50 steps) | training time (20 epochs) | time / epoch |
|---|---|---|---|---|
| standard | 97.57% | 0.64% | 54.5 s | 2.7 s |
| IBP ($\kappa: 1 \to 0.5$, $\epsilon_{train}: 0 \to 0.1$) | 94.38% | 83.76% | 120.2 s | 6.0 s |

* **Robustness.** Standard training gives a network that PGD breaks almost always at $\epsilon = 0.1$ (0.64%).
  IBP training raises the PGD accuracy to 83.8%, at a cost of 3.2 points of clean accuracy.
  The cost comes from the robust term $(1-\kappa)\,CE(\hat z_K)$: it pushes the network to keep the worst-case
  margin over the whole box positive, which limits how sharp the decision boundary can be near the data.
* **Training time.** IBP training is about **2.2x** slower than standard training (CPU, same epochs, batch size and data loader).
  For each batch, IBP does the normal forward pass and one interval pass. The interval pass costs about two forward
  passes (one matrix product for the center and one with $|W|$ for the radius), and the merged last layer is a
  per-example $10 \times 50$ product. The backward pass doubles accordingly. This is the main advantage of IBP: it costs only
  a small constant factor more than standard training, whereas tighter certified training (for example CROWN-IBP or
  linear-relaxation methods) costs much more. The training loss of IBP increases during the ramp (epochs 2–11), because
  $\epsilon_{train}$ grows and $\kappa$ falls, and then decreases when the schedule is constant.
* The times are wall-clock times on a MacBook (M2, CPU, 2 threads) while other jobs were running, so they are a little noisy; the ratio is the meaningful number.

## Adversarial search on uncertified images (IBP network, 100-step PGD, 5 restarts)

| eps | certified | correct but not certified | attacked | undecided |
|---|---|---|---|---|
| 0.01 | 9349 | 89 | 72 | 17 |
| 0.02 | 9271 | 167 | 140 | 27 |
| 0.03 | 9181 | 257 | 216 | 41 |
| 0.04 | 9049 | 389 | 313 | 76 |
| 0.05 | 8912 | 526 | 427 | 99 |
| 0.06 | 8777 | 661 | 542 | 119 |
| 0.07 | 8619 | 819 | 640 | 179 |
| 0.08 | 8436 | 1002 | 777 | 225 |
| 0.09 | 8193 | 1245 | 926 | 319 |
| 0.10 | 7876 | 1562 | 1104 | 458 |

(562 test images are misclassified by the IBP network.)

## Results and analysis of (b)

| $\epsilon_{test}$ | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 | 0.06 | 0.07 | 0.08 | 0.09 | 0.10 |
|---|---|---|---|---|---|---|---|---|---|---|
| IBP network: verified accuracy (%) | 93.49 | 92.71 | 91.81 | 90.49 | 89.12 | 87.77 | 86.19 | 84.36 | 81.93 | 78.76 |
| IBP network: PGD accuracy (%) | 93.66 | 93.00 | 92.24 | 91.32 | 90.17 | 89.06 | 88.14 | 86.87 | 85.35 | 83.67 |
| standard network: verified accuracy (%) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| standard network: PGD accuracy (%) | 93.00 | 81.68 | 59.33 | 37.94 | 22.68 | 12.23 | 6.00 | 2.66 | 1.16 | 0.61 |

**Verified accuracy.** The box verifier certifies 78.8% of the test set for the IBP network at $\epsilon_{test} = 0.1$
and 93.5% at $\epsilon_{test} = 0.01$. It certifies nothing for the standard network, even at $\epsilon_{test} = 0.01$:
without IBP training, the intervals grow with every layer (the $|W| r$ term) and become too wide, even though PGD only
breaks 1.6% of the correct images at 0.01. IBP training makes the network *verifiable by the box domain*: the weights
learn to keep the intervals small. The gap between the PGD accuracy and the verified accuracy of the IBP network is small
(0.2 points at 0.01, 4.9 points at 0.1). The true robust accuracy is between the two curves.

**Where the network is not certified.** Of the 9,438 correctly classified test images, the stronger attack (100 steps, 5
restarts) finds a real adversarial example for most of the uncertified images: 72 of 89 at $\epsilon_{test} = 0.01$ and
1,104 of 1,562 at 0.1. For these images, the network is really not robust, so no verifier could certify them. The remaining
17 (at 0.01) to 458 (at 0.1) images are *undecided*. They are either robust but not provable with boxes (the box domain
ignores the dependence between neurons, so it is incomplete), or robust only against our attack. The number of undecided images grows
with $\epsilon$, because the box over-approximation grows with the size of the input region.

**The adversarial examples** (figures above):

* *Small $\epsilon$ (0.01).* The attacked images are the ambiguous ones: the network gives the true class only
  0.28–0.63 probability on the clean image (an open 9 that looks like a 4, a 7 with a bar that looks like a 2, a thin 7
  that looks like a 1). These images are already close to a decision boundary, so a perturbation of 2.5 gray levels is enough.
* *Large $\epsilon$ (0.1).* The attack also breaks images that the network classifies with high confidence (for example
  0.95 for a 9 and 0.99 for a 1). To a human, all the adversarial images are still clearly the true digit. The perturbation
  adds a gray haze to the background and dims the strokes in selected places. It is a structured pattern: it brightens
  pixels where the target digit would have a stroke (for example it closes the top of a 4-like 9 and turns a 5 into a 3 by
  brightening the region where a 3 has its upper-left curve). At $\epsilon = 0.1$, 49% of the pixels are at $\pm\epsilon$,
  as expected from the sign steps of $\ell_\infty$ PGD, and 36% are unchanged. These are mostly background pixels that the attack wants to darken,
  but they are already 0 and the $[0,1]$ clamp stops them.
* *Confusions.* The most frequent (true → adversarial) pairs are 5→3, 4→9, 2→8, 5→8, 3→5 and 9→4. These are pairs of digits
  that differ by a few strokes, so a bounded perturbation can add or remove the difference. The digits 5, 2, 3 and 8
  are the least robust (14–22% of them get an adversarial example at some $\epsilon \le 0.1$), and the simple shapes 0 and
  1 are the most robust (4–5%).
