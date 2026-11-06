# lambda sweep for CITRUS and G-CITRUS (seed 0), cached like run_all.py.
from uap_lib import *

for name in ["gcitrus_l0.3", "citrus_l0.3", "gcitrus_l0.7", "citrus_l0.7"]:
    m, times = get_model(name, 0)
    print(f"{name}_s0: {evaluate(m, f'{name}_s0')}  time/epoch {np.mean(times):.1f}", flush=True)
