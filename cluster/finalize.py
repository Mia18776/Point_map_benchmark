#!/usr/bin/env python3
"""One command from finished jobs to updated reports (steps 3 + 4 of the workflow).

    python ~/pmb_jobs/finalize.py --logs full-scannetpp
    python ~/pmb_jobs/finalize.py 1863300 1863301 1863302 ...

It does, in order:

1. Find the result folders: either the job ids you give, or every job whose
   log is ~/pmb_jobs/logs/<prefix>-<jobid>.out (the --job-name you submitted
   with, e.g. full-scannetpp-pi3). A job that was killed still counts with the
   sets it finished (records.jsonl).
2. Merge them into one folder, pmb_runs/<name>, with the repository's own
   merge_results.py - which refuses runs whose code or settings differ.
   <name> defaults to <dataset>_<N>sets_<image size>, read from the runs.
3. Link that folder into pmb_runs/final/<section>/<name> (section "main" by
   default) so the report scripts pick it up.
4. Regenerate final/RESULTS.md, final/DETAILS.md and final/report.html, and
   the download archive pmb_runs/final_results.tar.gz.

Publishing report.html as a web page is done from Claude Code, not here.
"""

import argparse
import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pmb_final import DATASETS, FINAL, RUNS, ok_rows, run_config  # noqa: E402

LOGS = os.path.join(HERE, "logs")
DEFAULT_CODE = os.path.expanduser("~/Point_map_benchmark_sets")


def runs_from_logs(prefix):
    """Run folders of every job whose log is logs/<prefix>-<jobid>.out."""
    found = []
    for log in sorted(glob.glob(os.path.join(LOGS, f"{prefix}-*.out"))):
        match = re.search(r"-(\d+)\.out$", log)
        if not match:
            continue
        run = os.path.join(RUNS, match.group(1))
        if os.path.exists(os.path.join(run, "results.json")) or os.path.exists(
            os.path.join(run, "records.jsonl")
        ):
            found.append(run)
        else:
            print(f"  skip {os.path.basename(log)}: job {match.group(1)} left no results")
    return found


def resolve(run):
    """A job id or a path -> the run folder."""
    path = run if os.sep in run else os.path.join(RUNS, run)
    if not os.path.isdir(path):
        raise SystemExit(f"No result folder for {run} (looked at {path})")
    return path


def default_name(runs):
    cfg = run_config(runs[0])
    dataset = cfg.get("wai_dataset") or "folder"
    short = DATASETS.get(dataset, (dataset,))[0]
    return short, cfg.get("image_size", "native")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", nargs="*", help="Job ids (or result folders) to merge")
    parser.add_argument("--logs", metavar="PREFIX",
                        help="Take every job whose log is logs/PREFIX-<jobid>.out")
    parser.add_argument("--name", help="Folder name for the merged result (default: from the runs)")
    parser.add_argument("--section", default="main",
                        help="Where under final/ to link it: main (default), mapanything_nomask, ...")
    parser.add_argument("--code", default=DEFAULT_CODE,
                        help=f"Benchmark checkout whose merge_results.py to use (default {DEFAULT_CODE})")
    parser.add_argument("--replace", action="store_true",
                        help="Overwrite an existing merged folder / final link of the same name")
    args = parser.parse_args()

    runs = [resolve(r) for r in args.runs]
    if args.logs:
        runs += runs_from_logs(args.logs)
    if not runs:
        raise SystemExit("No runs given: pass job ids, or --logs PREFIX")
    print(f"Runs ({len(runs)}): " + " ".join(os.path.basename(r) for r in runs))

    # 1-2. Merge with the repository's own script (it checks comparability).
    sys.path.insert(0, os.path.join(args.code, "scripts"))
    sys.path.insert(0, args.code)
    import merge_results
    from pointmap_bench.report import write_reports

    records, config = merge_results.merge(runs)
    if args.name:
        name = args.name
    else:
        short, size = default_name(runs)
        sets = len({r["scene"] for r in records if r.get("status") == "ok"})
        name = f"{short}_{sets}sets_{size}"
    merged = os.path.join(RUNS, name)
    link = os.path.join(FINAL, args.section, name)
    if not args.replace:
        for path in (merged, link):
            if os.path.lexists(path):
                raise SystemExit(f"{path} already exists; pass --replace to overwrite it, or --name")

    notes = [f"Merged from {len(runs)} runs with identical settings; see merged_from."]
    if config.get("incomplete_runs"):
        notes.append("Some runs stopped before the end and contribute only the sets they finished "
                     "(see incomplete_runs); check num_scenes per model.")
    write_reports(records, merged, extra_notes=notes, config=config)

    ok = ok_rows(merged)
    per_model = ok.groupby("model").scene.nunique().to_dict()
    print(f"Merged into {merged}: " + ", ".join(f"{m} {n}" for m, n in sorted(per_model.items())))
    if len(set(per_model.values())) > 1:
        print("  WARNING: the models completed different numbers of sets - see report.md")

    # 3. Link into final/.
    os.makedirs(os.path.dirname(link), exist_ok=True)
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.relpath(merged, os.path.dirname(link)), link)
    print(f"Linked {link}")

    # 4. Regenerate every report and the download archive.
    for script in ("summarize_results.py", "make_report_html.py"):
        subprocess.run([sys.executable, os.path.join(HERE, script)], check=True)
    archive = os.path.join(RUNS, "final_results.tar.gz")
    subprocess.run(["tar", "-czhf", archive, "--exclude=*.ply", "--exclude=*.glb",
                    "--exclude=*.npz", "--exclude=*.log", "-C", RUNS, "final"], check=True)
    print(f"Archive {archive}")
    print("Done. The hand-written text (results_intro.md, results_conclusions.md, findings.md)"
          " is not updated automatically - edit it if the conclusions change.")


if __name__ == "__main__":
    main()
