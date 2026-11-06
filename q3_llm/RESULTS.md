# Problem 3(b): OLMo 2 1B checkpoints

Run on one free A100 of the department cluster (`run_eval.sh`, `probes.py`); full tables, examples and analysis are in
[olmo_checkpoints_eval.ipynb](olmo_checkpoints_eval.ipynb). 200 examples per benchmark, greedy decoding,
chat template for SFT/DPO/Instruct. The base checkpoint's probes were rerun separately (`rerun_base_probes.sh`) after
the first attempt used AdvBench, which is gated; all four checkpoints use JailbreakBench.

| checkpoint | GSM8K (flex) | IFEval prompt strict | TruthfulQA MC1 / MC2 | refuse harmful | refuse benign | pairwise sample sim. | GSM8K words |
|---|---|---|---|---|---|---|---|
| base | 35.5 | 8.5 | 22.0 / 38.9 | 0 | 0 | 0.009 | 51 |
| SFT | 41.0 | 43.0 | 26.0 / 44.8 | 98 | 42 | 0.014 | 63 |
| DPO | 50.5 | 58.0 | 28.0 / 46.6 | 91 | 15 | 0.017 | 74 |
| Instruct (RLVR) | 62.5 | 58.5 | 31.0 / 48.7 | 91 | 15 | 0.021 | 95 |

* SFT: chat format, instruction following (+34.5 IFEval) and refusals, with heavy over-refusal (42% of benign requests).
* DPO: more instruction following (+15), less over-refusal (42 → 15), slightly less safe (98 → 91), longer answers.
* RLVR: math (+12 GSM8K), the rewarded skill; other behavior almost unchanged.
* Worse after a step: over-refusal (SFT), harmful refusal (DPO), verbosity (each step), a slight trend toward less diverse samples (weak evidence), and confidence rises without better calibration (MC1 ECE 56–59% throughout).
