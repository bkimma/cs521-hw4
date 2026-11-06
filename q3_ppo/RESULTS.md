# Problem 3(a): PPO on CartPole-v1

All numbers come from `summarize` and `plot_curves` in `ppo_cartpole.ipynb` (5 seeds, 200,000 environment steps per run).

| Method | auc (mean ± std) | steps_to_475 (mean) | solved runs | final (greedy, mean ± std) | time / run |
|---|---|---|---|---|---|
| PPO (default) | 323.3 ± 37.3 | 68,270 | 5/5 | 409.1 ± 108.5 | 13.9 s |
| PPO without clipping (`clip_coef = 1e9`) | 341.1 ± 47.3 | 81,654 | 5/5 | 464.7 ± 61.5 | 16.4 s |
| PPO without GAE (`gae_lambda = 1.0`) | 337.0 ± 25.7 | 119,546 | 5/5 | 496.0 ± 5.9 | 19.4 s |

Diagnostics from an instrumented copy of `ppo_update` (seeds 0 and 1; not part of the notebook):

| Method | fraction of samples with \|r − 1\| > 0.2 | max \|r − 1\| | std of raw advantages | mean value loss |
|---|---|---|---|---|
| PPO | 0.000–0.001 | 0.60–0.64 | 5.8–9.1 | 47–51 |
| PPO without clipping | 0.000–0.002 | 0.71–0.80 | 6.1–10.3 | 47–57 |
| PPO without GAE | 0.000–0.001 | 0.32–0.58 | 15.3–17.0 | 276–280 |

## (i) How does the clipping change learning? Why?

In these runs, removing the clipping changes almost nothing. The differences in auc (323 vs. 341) and in
steps_to_475 (68k vs. 82k) are smaller than the spread between seeds (std ≈ 40), and both versions solve 5/5 runs.

The reason is that the clipping is almost never active with the default settings. With lr = 2.5e-4 (annealed to 0),
only 4 epochs on each rollout, and gradient-norm clipping at 0.5, the new policy stays close to the old one:
fewer than 0.2% of the samples have a probability ratio outside [0.8, 1.2], with or without the clip. When the ratio
stays inside the trust region, the clipped and the unclipped objectives (and their gradients) are the same. Without
clipping, the ratios still move a little further (max |r − 1| of 0.71–0.80 instead of 0.60–0.64), and the
no-clip curves vary more between seeds: one seed collapsed to about 200 and stayed there. So the clip is a safety
mechanism that limits rare large updates; it does not speed up learning. It matters more with larger learning rates,
more update epochs or larger minibatches, where the policy can move far from the policy that collected the data.

## (ii) How does GAE change learning? (bias and variance)

GAE with λ = 0.95 learns about 1.75× faster at the start: it reaches a mean return of 475 after 68k steps,
compared with 120k steps for λ = 1. The λ = 1 curve is visibly lower for the first 50k steps.

* **Variance.** With λ = 1, the advantage is (almost) the Monte Carlo return minus V(s). It sums the noise of every
  future reward and action until the episode ends, or until the end of the 128-step rollout, where it bootstraps
  once. The raw advantages of λ = 1 have a standard deviation 2–3× larger (15–17 vs. 6–9), and the value loss is
  about 5× larger (≈ 278 vs. ≈ 50). Noisier advantages give noisier policy gradients, so learning is slower.
* **Bias.** GAE with λ < 1 weights short TD errors more, so it uses the critic V more. This lowers the variance,
  but it adds bias when V is wrong. The critic is wrong early in training, and also later, for a specific reason: the
  starter treats the time limit at 500 steps (truncation) as a terminal state, but the observation does not contain
  the time. Thus V cannot predict a "terminal" value of 0 near step 500, which biases the TD targets. λ = 1 depends
  less on V, so it has less bias.

In our runs, the variance reduction of GAE wins early: faster learning and a 5/5 solve rate at about 68k steps. Later,
the lower bias of λ = 1 seems to help stability: its greedy final return is 496 ± 6, compared with 409 ± 108, and its
learning curve dips less after it reaches its maximum. This is the usual bias–variance trade-off that λ controls.

## (iii) Does PPO keep the maximum return after it gets there?

No, not reliably. All 5 default PPO seeds reach a mean return of ≥ 475 (between 62k and 90k steps). But afterwards,
the per-seed curves often fall back:

| Seed | first ≥ 475 at | minimum after | fraction of later points ≥ 475 | value at 200k |
|---|---|---|---|---|
| 0 | 90k | 177 | 0.05 | 249 |
| 1 | 74k | 222 | 0.20 | 483 |
| 2 | 82k | 415 | 0.53 | 500 |
| 3 | 72k | 152 | 0.26 | 244 |
| 4 | 62k | 137 | 0.11 | 180 |

Seeds 0, 3 and 4 collapse to about 200 after about 125k steps and do not recover. Thus the mean curve falls from
about 440 to about 330, and the final greedy return has a large spread (214 to 500). Only seed 2 stays near 500.
The ablations show the same effect, but weaker: without clipping, 2 of 5 seeds end low; without GAE, the dips
are smaller.

Reasons:

* **No signal at the maximum.** When almost all episodes reach 500, the true advantages are close to 0. With
  `norm_adv = True`, each minibatch is normalized to unit variance anyway, so the update follows noise (and the
  wrong values near truncation, see (ii)) with the full step size, and it can move a good policy away from the optimum.
* **Truncation as termination.** The value targets near step 500 are wrong, so the advantages near the end of an
  episode are biased.
* **Learning-rate annealing** decreases the learning rate linearly to 0. This reduces the size of the late updates
  and limits the damage (the collapses occur between 100k and 150k, when the learning rate is still 25–50% of the
  start), but it cannot undo a collapse: at the end, the learning rate is too small to recover, so collapsed seeds
  stay at about 200.

_The clip-ratio, advantage-std and value-loss statistics quoted above come from `ppo_stats.py` (run: `../.venv/bin/python ppo_stats.py ppo_cartpole.ipynb`), which re-runs seeds 0 and 1 with an instrumented `ppo_update`._
