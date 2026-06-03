#!/usr/bin/env python3
"""
collect_selected.py — Bundle CIF + JSON files for manually-accepted AF3 jobs.

Standalone script (no local imports) — safe to copy to WCSS as a single file.

Workflow:
  1. You provide a list of accepted job names in accepted_jobs.txt.
  2. For each job the script locates the AF3 output folder on WCSS
     (searching across multiple output directories from successive runs),
     picks the best CIF (by hetc_centrality), and copies:
       - the CIF
       - *_confidences.json         (contact_probs, PAE, token_chain_ids)
       - *_summary_confidences.json  (iptm, ptm, ranking_score)
       - *_job_request.json          (input sequences)
     All files go into a flat subfolder: bundle_dir/<job_name>/

Two ways to specify which seed to use per job
─────────────────────────────────────────────
A) Auto-pick — filename is plain job name (from --copy-best workflow):
     wcss/selected/<batch>/d1_chehdap_conf_vs_c2_l1.cif
   collect_selected re-runs hetc_centrality and picks the best seed.

B) Explicit seed — filename encodes the seed (from --copy-n workflow):
     wcss/selected/<batch>/d1_chehdap_conf_vs_c2_l1__seed-3_sample-0.cif
   collect_selected splits on '__' and uses that exact seed folder.
   This is the intended flow when the auto-picked CIF showed wrapping
   and you manually chose a different seed.

In both cases, generate accepted_jobs.txt the same way:
  ls wcss/selected/original/ wcss/selected/rerun_1/ wcss/selected/rerun_2/ \
     wcss/selected/rerun_3/ | sed 's/.cif$//' > accepted_jobs.txt
  scp accepted_jobs.txt wcss:~/thesis/

collect_selected detects the format automatically from the name.

Multiple output directories
───────────────────────────
AF3 outputs may be spread across multiple directories from successive
batch runs (af3_outputs, af3_outputs_rerun_1, _rerun_2, _rerun_3).
Use --af3_dir multiple times to search all of them:

  python collect_selected.py \\
      --accepted accepted_jobs.txt \\
      --af3_dir /lustre/.../af3_outputs \\
      --af3_dir /lustre/.../af3_outputs_rerun_1 \\
      --af3_dir /lustre/.../af3_outputs_rerun_2 \\
      --af3_dir /lustre/.../af3_outputs_rerun_3 \\
      --bundle_dir thesis/ready

The script searches directories in order and uses the first match.

Then tar + scp back:
  tar czf af3_selected.tar.gz thesis/ready/
  scp af3_selected.tar.gz local:~/praca_magisterska/thesis/

Author: Kacper Koźmin
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import shutil
from pathlib import Path


# ── Defaults ────────────────────────────────────────────────────────────────

# Default search locations for AF3 output bundles on the HPC cluster.
# These are placeholders: replace <scratch> with your own cluster scratch
# path, or override them at run time with one or more --af3_dir arguments.
DEFAULT_AF3_DIRS = [
    Path("/lustre/<scratch>/af3_outputs"),
    Path("/lustre/<scratch>/af3_outputs_rerun_1"),
    Path("/lustre/<scratch>/af3_outputs_rerun_2"),
    Path("/lustre/<scratch>/af3_outputs_rerun_3"),
]
DEFAULT_BUNDLE_DIR = Path("thesis/ready")

PROPELLER_LENGTH = 294  # residues 1..294 = propeller 1, 295.. = propeller 2


# ── CIF parsing ─────────────────────────────────────────────────────────────

def parse_ca_coords(cif_path: Path) -> dict[str, list[tuple[int, float, float, float]]]:
    """Extract Cα coordinates per chain. Returns {chain: [(resnum,x,y,z), ...]}."""
    with open(cif_path) as f:
        lines = f.readlines()

    col_map: dict[str, int] = {}
    col_idx = 0
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("_atom_site."):
                col_map[lines[j].strip()] = col_idx
                col_idx += 1
                j += 1
            if col_map:
                start_line = j
                break
        i += 1
    else:
        return {}

    if not col_map:
        return {}

    grp = col_map.get("_atom_site.group_PDB")
    atm = col_map.get("_atom_site.label_atom_id")
    chn = (col_map.get("_atom_site.label_asym_id")
           or col_map.get("_atom_site.auth_asym_id"))
    seq = (col_map.get("_atom_site.label_seq_id")
           or col_map.get("_atom_site.auth_seq_id"))
    xi  = col_map.get("_atom_site.Cartn_x")
    yi  = col_map.get("_atom_site.Cartn_y")
    zi  = col_map.get("_atom_site.Cartn_z")

    if None in (grp, atm, chn, seq, xi, yi, zi):
        return {}

    max_col = max(grp, atm, chn, seq, xi, yi, zi)
    result: dict[str, list[tuple[int, float, float, float]]] = {}

    for i in range(start_line, len(lines)):
        s = lines[i].strip()
        if not s or s.startswith("#") or s.startswith("_"):
            break
        parts = s.split()
        if len(parts) <= max_col:
            continue
        try:
            if parts[grp] == "ATOM" and parts[atm] == "CA" and parts[seq] != ".":
                chain = parts[chn]
                result.setdefault(chain, []).append((
                    int(parts[seq]),
                    float(parts[xi]),
                    float(parts[yi]),
                    float(parts[zi]),
                ))
        except (ValueError, IndexError):
            pass

    return {ch: sorted(v, key=lambda c: c[0]) for ch, v in result.items()}


# ── Geometry ─────────────────────────────────────────────────────────────────

def _com(coords: list[tuple[int, float, float, float]]) -> tuple[float, float, float]:
    n = len(coords)
    if n == 0:
        return (0.0, 0.0, 0.0)
    return (sum(c[1] for c in coords) / n,
            sum(c[2] for c in coords) / n,
            sum(c[3] for c in coords) / n)


def _dist(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def hetc_centrality(cif_path: Path,
                    split_at: int = PROPELLER_LENGTH) -> float | None:
    """Return abs(d(HET-C, P1_COM) - d(HET-C, P2_COM)). Lower = less wrapping."""
    by_chain = parse_ca_coords(cif_path)
    chain_a = by_chain.get("A", [])
    chain_b = by_chain.get("B", [])
    if not chain_a or not chain_b:
        return None

    p1 = [c for c in chain_a if c[0] <= split_at]
    p2 = [c for c in chain_a if c[0] > split_at]
    if not p1 or not p2:
        return None

    com_b = _com(chain_b)
    d1 = _dist(com_b, _com(p1))
    d2 = _dist(com_b, _com(p2))
    return abs(d1 - d2)


# ── Seed enumeration and selection ────────────────────────────────────────────

def find_all_cifs(job_dir: Path) -> list[tuple[int, int, Path]]:
    """Return all (seed, sample, cif_path) tuples in a job directory."""
    results = []
    for subdir in job_dir.iterdir():
        if not subdir.is_dir():
            continue
        m = re.match(r"seed-(\d+)_sample-(\d+)$", subdir.name)
        if not m:
            continue
        cifs = list(subdir.glob("*.cif"))
        if cifs:
            results.append((int(m.group(1)), int(m.group(2)), cifs[0]))
    return results


def pick_best_cif(job_dir: Path) -> Path | None:
    """Auto-pick the seed with the lowest hetc_centrality (least wrapping)."""
    entries = find_all_cifs(job_dir)
    if not entries:
        return None
    best_path = None
    best_score = float("inf")
    for _seed, _sample, cif_path in entries:
        c = hetc_centrality(cif_path)
        if c is not None and c < best_score:
            best_score = c
            best_path = cif_path
    return best_path


def resolve_cif(job_dir: Path, explicit_seed: str | None) -> Path | None:
    """Return the CIF to use for a job.

    If explicit_seed is given (e.g. 'seed-3_sample-0'), look for that
    subfolder directly. Falls back to auto-pick if not found.
    If explicit_seed is None, uses auto-pick.
    """
    if explicit_seed:
        seed_dir = job_dir / explicit_seed
        if seed_dir.is_dir():
            cifs = list(seed_dir.glob("*.cif"))
            if cifs:
                return cifs[0]
        print(f"  WARN: explicit seed '{explicit_seed}' not found in "
              f"{job_dir.name}, falling back to auto-pick")
    return pick_best_cif(job_dir)


# ── Job directory search across multiple AF3 output folders ──────────────────

def find_job_dir(job_name: str, af3_dirs: list[Path]) -> Path | None:
    """Find the AF3 job folder across multiple output directories.

    Searches directories in order, returns the first match (case-insensitive).
    """
    for af3_dir in af3_dirs:
        if not af3_dir.exists():
            continue
        for d in af3_dir.iterdir():
            if d.is_dir() and d.name.lower() == job_name.lower():
                return d
    return None


# ── Bundling ─────────────────────────────────────────────────────────────────

def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def bundle_job(job_name: str, explicit_seed: str | None,
               af3_dirs: list[Path], bundle_dir: Path) -> dict:
    """Copy CIF + JSONs for one job into bundle_dir/<job_name>/."""
    job_dir = find_job_dir(job_name, af3_dirs)

    if job_dir is None:
        searched = ", ".join(str(d) for d in af3_dirs if d.exists())
        return {"job": job_name, "status": "NOT_FOUND", "seed_sample": "",
                "cif_md5": "", "source_dir": "",
                "message": f"Job folder not found in: {searched}"}

    best_cif = resolve_cif(job_dir, explicit_seed)
    if best_cif is None:
        return {"job": job_name, "status": "NO_CIF", "seed_sample": "",
                "cif_md5": "", "source_dir": str(job_dir.parent.name),
                "message": "No CIF files found"}

    seed_sample_dir = best_cif.parent
    seed_sample = seed_sample_dir.name
    cif_md5 = md5_file(best_cif)
    source_dir = job_dir.parent.name  # e.g. 'af3_outputs_rerun_2'

    dest = bundle_dir / job_name.lower()
    dest.mkdir(parents=True, exist_ok=True)

    # Copy CIF
    shutil.copy2(best_cif, dest / best_cif.name)

    # Copy confidences JSON (the big one, not the summary)
    for f in seed_sample_dir.glob("*confidences*.json"):
        if "summary" not in f.name:
            shutil.copy2(f, dest / f.name)

    # Copy summary confidences JSON
    for f in seed_sample_dir.glob("*summary_confidences*.json"):
        shutil.copy2(f, dest / f.name)

    # Copy job_request JSON (usually in job_dir or one level up)
    job_request = None
    for candidate in (list(job_dir.glob("*job_request.json"))
                      + list(job_dir.parent.glob(
                             f"{job_dir.name}*job_request.json"))):
        job_request = candidate
        break
    if job_request:
        shutil.copy2(job_request, dest / job_request.name)
    else:
        for f in seed_sample_dir.glob("*job_request.json"):
            shutil.copy2(f, dest / f.name)
            break

    n_files = len(list(dest.iterdir()))
    return {"job": job_name, "status": "OK", "seed_sample": seed_sample,
            "cif_md5": cif_md5, "source_dir": source_dir,
            "message": f"{n_files} files bundled"}


# ── Accepted-jobs file parsing ────────────────────────────────────────────────

def read_accepted(path: Path) -> list[tuple[str, str | None]]:
    """Parse accepted_jobs.txt into [(job_name, seed_or_None), ...].

    Each line is a stem from wcss/selected/<batch>/ (filename without .cif).
    Two formats are detected automatically:

      d1_chehdap_conf_vs_c2_l1
          → job = full line, seed = None (auto-pick by metric)

      d1_chehdap_conf_vs_c2_l1__seed-3_sample-0
          → job = part before '__', seed = part after '__'
          → produced by check_propellers --copy-n when you choose
            a non-default seed after seeing wrapping in the best pick

    Lines starting with '#' and blank lines are ignored.
    """
    entries: list[tuple[str, str | None]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "__seed-" in line:
            job_name, explicit_seed = line.split("__", 1)
        elif "__" in line:
            job_name, explicit_seed = line.split("__", 1)
        else:
            job_name, explicit_seed = line, None
        if job_name:
            entries.append((job_name, explicit_seed))
    return entries


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bundle CIF + JSONs for accepted AF3 jobs into flat folders")
    p.add_argument("--accepted", type=Path, required=True,
                   help="Text file: one job per line. Optionally includes "
                        "__seed-N_sample-M to override auto-pick for that job.")
    p.add_argument("--af3_dir", type=Path, action="append", dest="af3_dirs",
                   help="AF3 output directory to search. Can be specified "
                        "multiple times. Searched in order, first match wins. "
                        f"Default: {' '.join(str(d) for d in DEFAULT_AF3_DIRS)}")
    p.add_argument("--bundle_dir", type=Path, default=DEFAULT_BUNDLE_DIR,
                   help=f"Output bundle dir (default: {DEFAULT_BUNDLE_DIR})")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    af3_dirs = args.af3_dirs if args.af3_dirs else DEFAULT_AF3_DIRS
    args.bundle_dir.mkdir(parents=True, exist_ok=True)

    # Report which dirs exist
    print(f"AF3 output directories ({len(af3_dirs)}):")
    for d in af3_dirs:
        exists = "OK" if d.exists() else "NOT FOUND"
        print(f"  {d}  [{exists}]")

    entries = read_accepted(args.accepted)
    n_explicit = sum(1 for _, s in entries if s is not None)
    print(f"\nAccepted jobs to bundle: {len(entries)} "
          f"({n_explicit} with explicit seed, "
          f"{len(entries) - n_explicit} auto-pick)")

    ok = 0
    problems: list[dict] = []
    manifest_rows: list[dict] = []

    for job_name, explicit_seed in sorted(entries, key=lambda x: x[0]):
        result = bundle_job(job_name, explicit_seed, af3_dirs, args.bundle_dir)
        manifest_rows.append(result)

        seed_tag = f"[explicit: {explicit_seed}]" if explicit_seed else "[auto-pick]"
        if result["status"] == "OK":
            ok += 1
            print(f"  OK  {job_name:<45} {result['seed_sample']:<22} "
                  f"{seed_tag}  {result['source_dir']:<25} {result['message']}")
        else:
            problems.append(result)
            print(f"  !!  {job_name:<45} {result['status']}: {result['message']}")

    # Write manifest (records which seed was actually used)
    manifest_path = args.bundle_dir / "manifest.tsv"
    with open(manifest_path, "w") as f:
        f.write("job\tstatus\tseed_sample\tsource_dir\tcif_md5\tmessage\n")
        for r in manifest_rows:
            f.write(f"{r['job']}\t{r['status']}\t{r['seed_sample']}\t"
                    f"{r['source_dir']}\t{r['cif_md5']}\t{r['message']}\n")

    print(f"\nBundled: {ok}/{len(entries)}")
    if problems:
        print(f"Problems: {len(problems)}")
        for p in problems:
            print(f"  - {p['job']}: {p['status']} — {p['message']}")
    print(f"Manifest: {manifest_path}")
    print(f"\nNext steps:")
    print(f"  tar czf af3_selected.tar.gz {args.bundle_dir}/")
    print(f"  scp <wcss>:~/thesis/af3_selected.tar.gz .")


if __name__ == "__main__":
    main()
