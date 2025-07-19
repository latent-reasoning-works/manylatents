import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections.abc import MutableMapping
import wandb
from concurrent.futures import ThreadPoolExecutor, as_completed


# -------------------------
# Utils
# -------------------------
def flatten_dict(d, parent_key='', sep='.', skip_keys=None):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if skip_keys and new_key in skip_keys:
            continue
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def parse_admixture_Ks(config_raw):
    """Extract list of Ks from nested config dict; return [] if missing."""
    try:
        k_str = config_raw["data"]["files"]["admixture_K"]
    except Exception:
        return []
    return [k.strip() for k in str(k_str).split(",") if k.strip()]


# -------------------------
# Artifact handling
# -------------------------
def download_logged_artifacts(run, artifact_root):
    """
    Download artifacts logged by this run.
    Returns: list of (artifact_obj, safe_alias, local_dir) and a dict of alias->semicolon file list.
    """
    os.makedirs(artifact_root, exist_ok=True)
    artifacts_info = []
    alias_filemap = {}

    for art in run.logged_artifacts():
        alias = art.aliases[0] if art.aliases else art.name
        safe_alias = alias.replace("/", "_")
        local_dir = os.path.join(artifact_root, run.id, safe_alias)
        os.makedirs(local_dir, exist_ok=True)

        try:
            dl_path = art.download(root=local_dir)
        except Exception as e:
            alias_filemap[f"artifact.{safe_alias}"] = f"ERROR:{e}"
            continue

        # collect relative file paths
        file_list = []
        for dirpath, _, filenames in os.walk(dl_path):
            for f in filenames:
                rel = os.path.relpath(os.path.join(dirpath, f), artifact_root)
                file_list.append(rel)
        alias_filemap[f"artifact.{safe_alias}"] = ";".join(file_list)

        artifacts_info.append((art, safe_alias, dl_path))

    return artifacts_info, alias_filemap


def extract_admix_from_artifact_dir(local_dir, admix_Ks, verbose=False):
    """
    Look through JSON-ish files for columns starting with 'dataset.admixture_preservation'.
    Returns flat dict of scalar metrics.
    """
    out = {}
    if not admix_Ks:
        return out
    if not os.path.isdir(local_dir):
        return out

    for root, _, files in os.walk(local_dir):
        for fname in files:
            if not fname.endswith(".json") and ".table.json" not in fname:
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
            except Exception as e:
                if verbose:
                    print(f"[skip bad json] {fpath}: {e}")
                continue

            # Expect a W&B Table style structure
            cols = data.get("columns")
            rows = data.get("data")
            if not isinstance(cols, list) or rows is None:
                continue

            try:
                df = pd.DataFrame(rows, columns=cols)
            except Exception as e:
                if verbose:
                    print(f"[bad table] {fpath}: {e}")
                continue

            admix_cols = [c for c in df.columns if c.startswith("dataset.admixture_preservation")]
            if not admix_cols:
                continue
            if verbose:
                print(f"[admix found] {fpath} cols={admix_cols}")

            # choose first row; adjust if needed
            row0 = df.iloc[0]

            for col in admix_cols:
                val = row0[col]

                # If JSON stored nested structure (list inside dict, etc.)
                if isinstance(val, dict):
                    # try keys sorted by K; fallback to values
                    cand_vals = list(val.values())
                    if len(cand_vals) == len(admix_Ks):
                        for k, v in zip(admix_Ks, cand_vals):
                            out[f"admixture_preservation.{col}.{k}"] = v
                        continue
                    # otherwise, try broadcast scalar fallback
                    val = cand_vals

                if isinstance(val, (list, tuple, np.ndarray)) and len(val) == len(admix_Ks):
                    for k, v in zip(admix_Ks, val):
                        out[f"admixture_preservation.{col}.{k}"] = v
                else:
                    for k in admix_Ks:
                        out[f"admixture_preservation.{col}.{k}"] = val

    return out

def process_run(run, artifact_dir):
    try:
        summary = run.summary._json_dict
        config_raw = {k: v for k, v in run.config.items() if not k.startswith("_")}
        name = run.name

        config_flat = {}
        for k, v in config_raw.items():
            if isinstance(v, dict):
                config_flat.update(flatten_dict(v, 
                                                parent_key=f"config.{k}"))
            else:
                config_flat[f"config.{k}"] = v
        summary_flat = flatten_dict(summary, 
                                    parent_key="summary")
        admix_Ks = parse_admixture_Ks(config_raw)

        artifacts_info, artifact_paths = download_logged_artifacts(run, artifact_dir)

        admix_metrics = {}
        for _, safe_alias, dl_path in artifacts_info:
            admix_metrics.update(extract_admix_from_artifact_dir(dl_path, 
                                                                 admix_Ks))

        record = {}
        record.update(config_flat)
        record.update(summary_flat)
        record.update(artifact_paths)
        record.update(admix_metrics)
        record["name"] = name
        record["run_id"] = run.id
        record["run_path"] = "/".join(run.path)

        return record
    except Exception as e:
        print(f"[!] Failed to process run {run.id}: {e}")
        return None

# -------------------------
# Main
# -------------------------
def main(project_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    artifact_dir = os.path.join(output_dir, "artifacts")

    api = wandb.Api()
    runs = api.runs(project_name)

    records = []
    
    # Parallel processing of runs
    max_workers = min(8, os.cpu_count() or 4)
    records = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_run, 
                                   run, 
                                   artifact_dir) for run in runs]
        for future in tqdm(as_completed(futures), 
                           total=len(futures), 
                           desc="Processing W&B runs"):
            result = future.result()
            if result:
                records.append(result)

    df = pd.DataFrame(records)
    out_csv = os.path.join(output_dir, "wandb_runs.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Exported {len(df)} runs to {out_csv}")
    print(f"📦 Artifacts saved in: {artifact_dir}")


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export W&B runs, flatten config/summary, download artifacts, extract admixture metrics."
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="W&B project name in the form 'entity/project-name'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/wandb_export",
        help="Directory to store output CSV and artifacts",
    )
    args = parser.parse_args()
    main(args.project, args.output_dir)