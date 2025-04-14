import re
import pandas as pd
from typing import List

from omegaconf import OmegaConf
from pathlib import Path

# add the metric keys to extract
metric_keys = ['geographic_preservation', 'admixture_preservation', 'trustworthiness']

config_sections = [
                    "algorithm.dimensionality_reduction",
                    "metrics.embedding.trustworthiness",
                    "metrics.dataset.geographic_preservation", # can be removed if not needed
                    "metrics.dataset.admixture_preservation", # can be removed if not needed
                    ]

# === extracts the metric value from the log ===

def extract_metric_value(metric_str: str):
    # Convert array([...]) to list of floats
    array_match = re.match(r'array\(\[(.*?)\]\)', metric_str, re.DOTALL)
    if array_match:
        return [float(x.strip()) for x in array_match.group(1).split(',')]
    else:
        try:
            return float(metric_str.strip())
        except ValueError:
            return None

def extract_selected_metrics_from_text(log_text: str, metric_names: List[str]):
    result = {}
    for metric in metric_names:
        pattern = rf"'{metric}':\s*(array\(\[.*?\]\)|[0-9.eE+-]+)"
        match = re.search(pattern, log_text, re.DOTALL)
        if match:
            value = extract_metric_value(match.group(1))
            if isinstance(value, list):
                for i, v in enumerate(value):
                    result[f"{metric}_{i+2}"] = v
            else:
                result[metric] = value
        else:
            print(f"Metric {metric} not found in log")
    return result

def load_metrics_from_logs(log_paths: List[str], metric_names: List[str]) -> pd.DataFrame:
    records = []
    for log_path in log_paths:
        with open(log_path, 'r') as f:
            text = f.read()
        metrics = extract_selected_metrics_from_text(text, metric_names)
        if metrics:
            records.append(metrics)
    return pd.DataFrame(records)

def flatten_omegaconf(cfg_section, parent_key=''):
    """Recursively flatten OmegaConf config."""
    items = {}
    for k, v in cfg_section.items():
        full_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, (dict, OmegaConf)):
            items.update(flatten_omegaconf(v, full_key))
        else:
            items[full_key] = v
    return items

# === add configuration parameters to the metrics DataFrame ===

def add_config_to_df(df: pd.DataFrame, config_path: Path, sections: list[str]) -> pd.DataFrame:
    cfg = OmegaConf.load(config_path)
    for section in sections:
        section_cfg = OmegaConf.select(cfg, section)
        if section_cfg is not None:
            flat_cfg = flatten_omegaconf(section_cfg, section)
            for key, value in flat_cfg.items():
                if key == "algorithm.dimensionality_reduction._target_":
                    key = "algorithm.dimensionality_reduction.method"

                if key.endswith("._partial_") or key.endswith("._target_") or key.endswith(".verbose") or key.endswith(".n_jobs"):
                    continue
                      
                # Strip the first two levels of the key
                parts = key.split(".")
                if len(parts) > 2:
                    new_key = ".".join(parts[2:])
                else:
                    new_key = key

                df[new_key] = value

    return df

def load_all_runs_from_sweeps(
    sweep_dirs: list[Path], 
    metric_keys: list[str], 
    config_sections: list[str]
) -> pd.DataFrame:
    all_rows = []

    for sweep_dir in sweep_dirs:
        for run_dir in sorted(sweep_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            log_file = run_dir / "main.log"
            config_file = run_dir / ".hydra" / "config.yaml"

            if not log_file.exists() or not config_file.exists():
                print(f"Skipping {run_dir} (missing log or config)")
                continue

            # Step 1: Extract metrics
            with open(log_file, "r") as f:
                log_text = f.read()
            metrics = extract_selected_metrics_from_text(log_text, metric_keys)

            # Step 2: Create DataFrame row
            df_row = pd.DataFrame([metrics])

            # Step 3: Add config info
            df_row = add_config_to_df(df_row, config_file, config_sections)

            all_rows.append(df_row)

    return pd.concat(all_rows, ignore_index=True)

if __name__ == "__main__":
    # Example usage
    # sweep path can be one or more directories
    sweep_paths = [Path("manifold_genetics/outputs/2025-04-14/15-31-42"),
                   Path("manifold_genetics/outputs/2025-04-14/14-48-50")]
    
    # add the metric keys to extract
    metric_keys = ['geographic_preservation', 'admixture_preservation', 'trustworthiness']
    
    config_sections = [
        "algorithm.dimensionality_reduction",
        "metrics.embedding.trustworthiness",
        "metrics.dataset.geographic_preservation", # can be removed if not needed
        "metrics.dataset.admixture_preservation", # can be removed if not needed
    ]


    df_all = load_all_runs_from_sweeps(sweep_paths, metric_keys, config_sections)
    df_all.to_csv("./outputs/DR_merged_results.csv", index=False)

    
    # Preview
    print(df_all.head())
