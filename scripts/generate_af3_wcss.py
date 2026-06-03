#!/usr/bin/env python3
"""
generate_af3_wcss.py
====================
Convert AF3-server JSONs to AF3-local (3.0.1) format for WCSS cluster.

Reads existing JSONs from data/af3_inputs/{labeled,discovery}/,
adds modelSeeds and adapts the schema for local AF3 3.0.1,
writes output to data/af3_wcss/{labeled,discovery}/.

Also generates a batch submission script (run_af3_wcss.sh).

Usage:
    python scripts/generate_af3_wcss.py [--seeds 50] [--subset labeled]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "af3_inputs"
OUTPUT_DIR = ROOT / "data" / "af3_wcss"
BATCH_SCRIPT = ROOT / "data" / "af3_wcss" / "run_af3_wcss.sh"

# ── AF3 WCSS settings ─────────────────────────────────────────────

WCSS_MODEL_DIR = "/mnt/db/models"   # path inside Apptainer container (/lustre/...alphafold3 → /mnt/db)
WCSS_MEMORY_GB = 96
WCSS_TIME_HOURS = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate AF3 3.0.1 JSONs for WCSS")
    p.add_argument(
        "--seeds", type=int, default=4,
        help="Number of model seeds per job (default: 50)",
    )
    p.add_argument(
        "--subset", choices=["labeled", "discovery", "all"], default="all",
        help="Which input subset to process (default: all)",
    )
    p.add_argument(
        "--wcss-base", type=str, default=None,
        help="Base path on WCSS where data/af3_wcss/ will live "
             "(e.g. /home/kackoz9157/thesis). "
             "If not set, uses absolute paths from this machine.",
    )
    p.add_argument(
        "--wcss-output", type=str, default=None,
        help="Output directory for AF3 results on WCSS "
             "(e.g. /home/kackoz9157/thesis/af3_outputs). "
             "If not set, defaults to <wcss-base>/data/af3_wcss/outputs.",
    )
    return p.parse_args()


def convert_server_to_local(server_json: dict, n_seeds: int) -> dict:
    """Convert AF3-server JSON format to AF3-local 3.0.1 format.

    Key differences:
    - Local uses 'protein' with 'id' field instead of 'proteinChain' with 'count'
    - Local requires 'modelSeeds' list
    - Remove server-specific fields: dialect, version, _pipeline_meta
    """
    # Build seed list: [1, 2, 3, ..., n_seeds]
    seeds = list(range(1, n_seeds + 1))

    # Convert sequences
    sequences = []
    chain_id = "A"
    for seq_entry in server_json["sequences"]:
        if "proteinChain" in seq_entry:
            sequences.append({
                "protein": {
                    "id": chain_id,
                    "sequence": seq_entry["proteinChain"]["sequence"],
                }
            })
            chain_id = chr(ord(chain_id) + 1)
        else:
            # Pass through unknown sequence types
            sequences.append(seq_entry)

    local_json = {
        "name": server_json["name"],
        "modelSeeds": seeds,
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 2,
    }

    return local_json


def process_subset(subset_name: str, n_seeds: int) -> list[str]:
    """Process one subset (labeled or discovery). Returns list of output paths."""
    input_dir = INPUT_DIR / subset_name
    output_dir = OUTPUT_DIR / subset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"  WARNING: {input_dir} does not exist, skipping")
        return []

    json_files = sorted(input_dir.glob("*.json"))
    output_paths = []

    for jf in json_files:
        with open(jf) as f:
            server_data = json.load(f)

        local_data = convert_server_to_local(server_data, n_seeds)
        out_path = output_dir / jf.name
        with open(out_path, "w") as f:
            json.dump(local_data, f, indent=2)

        output_paths.append(str(out_path))

    print(f"  {subset_name}: {len(json_files)} JSONs → {output_dir}")
    return output_paths


def generate_batch_script(
    output_paths: list[str], n_seeds: int, subset_name: str,
    wcss_base: str | None = None,
    wcss_output: str | None = None,
) -> None:
    """Generate a SLURM batch submission script for WCSS."""
    script_dir = OUTPUT_DIR
    script_dir.mkdir(parents=True, exist_ok=True)

    # Rewrite input JSON paths for WCSS: local data/af3_wcss/{subset}/ → af3_inputs/{subset}/
    def wcss_path(local_path: str) -> str:
        if wcss_base is None:
            return local_path
        filename = Path(local_path).name
        parent = Path(local_path).parent.name   # 'labeled' or 'discovery'
        return f"{wcss_base}/thesis/af3_inputs/{parent}/{filename}"

    # Output dir: explicit --wcss-output or fallback
    if wcss_output:
        af3_output_dir = wcss_output
    elif wcss_base:
        af3_output_dir = f"{wcss_base}/thesis/af3_outputs"
    else:
        af3_output_dir = str(OUTPUT_DIR / "outputs")

    lines = [
        "#!/bin/bash",
        f"# AF3 3.0.1 batch submission for WCSS — {len(output_paths)} jobs, {n_seeds} seeds each",
        f"# Generated by generate_af3_wcss.py",
        "",
        f'OUTPUT_BASE="{af3_output_dir}"',
        f'MODEL_DIR="{WCSS_MODEL_DIR}"',
        "",
        "mkdir -p ${OUTPUT_BASE}",
        "",
        "# Submit each job (sleep 0.3s between submits to avoid SLURM socket timeouts)",
    ]

    for p in output_paths:
        lines.append(
            f'sub-alphafold-3.0.1 "{wcss_path(p)}" "${{OUTPUT_BASE}}" '
            f"--model ${{MODEL_DIR}} -m {WCSS_MEMORY_GB} -t {WCSS_TIME_HOURS}"
        )
        lines.append("sleep 0.3")

    lines.append("")
    lines.append(f"echo 'Submitted {len(output_paths)} AF3 jobs'")

    script_path = script_dir / f"run_af3_{subset_name}.sh"
    with open(script_path, "w") as f:
        f.write("\n".join(lines))
    script_path.chmod(0o755)

    print(f"\n  Batch script: {script_path}")
    print(f"  Total jobs: {len(output_paths)}")
    print(f"  Seeds per job: {n_seeds}")
    print(f"  Estimated GPU-hours: ~{len(output_paths) * n_seeds / 50 * 1:.0f}h")
    print(f"  (assuming ~1h per job with 50 seeds on H100)")


def main() -> None:
    args = parse_args()
    print(f"Generating AF3 3.0.1 JSONs for WCSS ({args.seeds} seeds)")

    subsets = ["labeled"] if args.subset == "all" else [args.subset]
    all_paths = []

    for subset in subsets:
        paths = process_subset(subset, args.seeds)
        all_paths.extend(paths)

        if paths:
            generate_batch_script(paths, args.seeds, subset, args.wcss_base, args.wcss_output)

    # Also generate a combined script
    if len(subsets) > 1 and all_paths:
        generate_batch_script(all_paths, args.seeds, "all", args.wcss_base, args.wcss_output)

    print(f"\nDone. Total: {len(all_paths)} jobs ready.")
    print(f"\nUsage on WCSS:")
    print(f"  bash data/af3_wcss/run_af3_labeled.sh")
    print(f"  # or for all:")
    print(f"  bash data/af3_wcss/run_af3_all.sh")


if __name__ == "__main__":
    main()
