"""Write a predeclared experiment manifest without importing a compute runtime."""
import argparse
import json
from pathlib import Path


def manifest():
    base = {"channels": 64, "blocks": 4, "global_every": 0, "effect_channels": 0,
            "curriculum": 0.0, "restart_fraction": 0.0, "ema_decay": 0.0,
            "workers": 1, "inference_cache_mb": 0}
    arms = [("reference-96x6", {"channels": 96, "blocks": 6}), ("compact-64x4", {}),
            ("global-64x4", {"global_every": 2}), ("effects-64x4", {"effect_channels": 16}),
            ("global-effects-64x4", {"global_every": 2, "effect_channels": 16})]
    return {"format": 1, "status": "planned-not-run", "runtime": {"tinygrad": "0.14.0", "device": "METAL"},
            "architecture_arms": [{"name": n, "config": {**base, **c}} for n, c in arms],
            "training_seeds": [20260906, 20260907, 20260908], "minutes_per_arm_seed": 60,
            "systems_sweeps": {"actors": [1, 8, 32, 64], "workers": [1, 2, 4, 8, 0],
                               "cache_mb": [0, 64], "tail_buckets": [False, True]},
            "post_architecture_ablations": [{"restart_fraction": .25}, {"fast_explore": False},
                                           {"ema_decay": .995}, {"reanalysis": 128}],
            "development_evaluation": {"seed": 918273, "pairs": 32, "move_ms": 50},
            "final_evaluation": {"seed": 718291, "pairs": 256, "move_ms": 50, "fixed_sample_size": True},
            "promotion": {"automatic": False, "require": ["correctness battery passes", "bounded memory and safe stop",
                "paired lower confidence bound above 0.5 versus incumbent", "no material held-out tactical regression"],
                "note": "Final seeds must not tune architecture; inconclusive evidence is not a pass."},
            "measurement": "Interleave A/B/identity controls; separate cold/warm, latency/throughput, profiling/absolute timing."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(manifest(), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__": main()
