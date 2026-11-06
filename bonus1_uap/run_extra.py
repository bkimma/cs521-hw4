# Second seed of the groups-of-2 variant, and the box diagnostics.
from uap_lib import *

m, times = get_model("gcitrus_n2", 1)
print(f"gcitrus_n2_s1: {evaluate(m, 'gcitrus_n2_s1')}  time/epoch {np.mean(times):.1f}", flush=True)
diag = {}
for name in ["sabr", "citrus", "gcitrus", "gcitrus_n2"]:
    m, _ = get_model(name, 0)
    diag[name] = box_diagnostics(m)
    print(name, diag[name], flush=True)
json.dump(diag, open("diagnostics.json", "w"), indent=1)
