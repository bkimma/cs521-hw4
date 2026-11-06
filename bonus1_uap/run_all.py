# Trains and evaluates every configuration (cached in checkpoints/ and results.json).
from uap_lib import *

RUNS = [("standard", 0), ("ibp", 0), ("ibp", 1), ("sabr", 0), ("sabr", 1), ("citrus", 0), ("citrus", 1),
        ("gcitrus", 0), ("gcitrus", 1), ("gcitrus_ce", 0), ("gcitrus_tau0", 0), ("gcitrus_n2", 0)]
models = {}
for name, seed in RUNS:
    print(f"=== train {name} seed {seed}", flush=True)
    models[(name, seed)] = get_model(name, seed)
for name, seed in RUNS:
    m, times = models[(name, seed)]
    r = evaluate(m, f"{name}_s{seed}")
    print(f"{name}_s{seed}: {r}  time/epoch {np.mean(times):.1f}", flush=True)
