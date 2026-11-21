#!/usr/bin/env python3
"""Merge several benchmark runs into one report.

A full WAI evaluation is easiest to run as one job per model, in parallel.
This puts their ``results.json`` back together into a single ``report.md`` /
``results.csv`` / ``results.json``, exactly as if one run had produced them.

A run that was killed before the end has no results.json; its per-set log
(records.jsonl + run_config.json) is used instead, so the sets it finished
still count. Passing a single such run reports on it alone.

Runs are only merged when they can be compared: every setting that changes
the numbers (code revision, dataset, image size, sets, thresholds, ...) must
agree, and no (model, scene) may appear twice. ``models`` and ``timestamp``
are the only settings allowed to differ. Usage::

    python scripts/merge_results.py runs/123 runs/124 runs/125 --output-dir runs/merged
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointmap_bench.report import RECORDS_LOG, load_partial_run, write_reports  # noqa: E402

# Settings that may legitimately differ between the runs being merged.
PER_RUN_SETTINGS = ("models", "timestamp")


def merge(run_dirs):
    """Return ``(records, config)`` for the given run directories."""
    records, configs, partial = [], [], []
    for run_dir in run_dirs:
        results = os.path.join(run_dir, "results.json")
        if os.path.exists(results):
            with open(results, encoding="utf-8") as fh:
                data = json.load(fh)
            configs.append((run_dir, data.get("config", {})))
            records.extend(data["records"])
        elif os.path.exists(os.path.join(run_dir, RECORDS_LOG)):
            # A run that never reached the end: use what it saved per set.
            config, run_records = load_partial_run(run_dir)
            configs.append((run_dir, config))
            records.extend(run_records)
            partial.append(run_dir)
        else:
            raise SystemExit(f"{run_dir} has neither results.json nor {RECORDS_LOG}")

    reference_dir, reference = configs[0]
    for run_dir, config in configs[1:]:
        keys = (set(reference) | set(config)) - set(PER_RUN_SETTINGS)
        differing = {
            key: (reference.get(key), config.get(key))
            for key in sorted(keys)
            if reference.get(key) != config.get(key)
        }
        if differing:
            detail = "; ".join(f"{k}: {a!r} vs {b!r}" for k, (a, b) in differing.items())
            raise SystemExit(
                f"{run_dir} is not comparable with {reference_dir}: {detail}"
            )

    seen = set()
    for record in records:
        key = (record.get("model"), record.get("scene"))
        if key in seen:
            raise SystemExit(f"{key} appears in more than one run")
        seen.add(key)

    models = []
    for _, config in configs:
        models.extend(m for m in config.get("models", []) if m not in models)
    config = {k: v for k, v in reference.items() if k not in PER_RUN_SETTINGS}
    config["models"] = models
    config["timestamp"] = max(str(c.get("timestamp", "")) for _, c in configs)
    config["merged_from"] = [os.path.abspath(d) for d in run_dirs]
    if partial:
        config["incomplete_runs"] = [os.path.abspath(d) for d in partial]
    return records, config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", help="Run directories holding results.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    records, config = merge(args.runs)
    notes = [
        f"Merged from {len(args.runs)} runs with identical settings; see merged_from.",
    ]
    if config.get("incomplete_runs"):
        notes.append(
            "Some runs stopped before the end and contribute only the sets they "
            "finished (see incomplete_runs); check num_scenes per model."
        )
    if config.get("image_size") == "native":
        notes.append(
            "Each model ran at its own native resolution; compare runtimes with that in mind."
        )
    paths = write_reports(records, args.output_dir, extra_notes=notes, config=config)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
