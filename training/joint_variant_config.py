"""Round-trip the registered G3 source ordering through canonical JSON metadata."""
import json
from pathlib import Path


def read_variant_config(path, primary):
    cfg = json.loads(Path(path).read_bytes())
    registered = primary["families"]
    if cfg.get("training_family_order") != list(registered) or cfg.get("families") != registered:
        raise ValueError("Derived run must use the ordered source mixture from the primary registration")
    # The metadata writer sorts mapping keys. The sampler uses insertion order for
    # both its shuffled cycle and the per-family permutation seed.
    cfg["families"] = {family: cfg["families"][family] for family in cfg["training_family_order"]}
    return cfg
