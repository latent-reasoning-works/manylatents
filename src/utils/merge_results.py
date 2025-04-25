import argparse
import re
import pandas as pd
from typing import List
from omegaconf import OmegaConf
from pathlib import Path

def extract_metric_value(metric_str: str):
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

def flatten_omegaconf(cfg_section, parent_key=''):
    items = {}
    for k, v in cfg_section.items():
        full_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, (dict, OmegaConf)):
            items.update(flatten_omegaconf(v, full_key))
        else:
            items[full_key] = v
    return items

def add_config_to_df(df: pd.DataFrame, config_path: Path, sections: List[str]) -> pd.DataFrame:
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
                parts = key.split(".")
                if len(parts) > 2:
                    new_key = ".".join(parts[2:])
                else:
                    new_key = key
                df[new_key] = value
    return df

def load_all_runs_from_sweeps(
    sweep_dirs: List[Path], 
    metric_keys: List[str], 
    config_sections: List[str]
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
            with open(log_file, "r") as f:
                log_text = f.read()
            metrics = extract_selected_metrics_from_text(log_text, metric_keys)
            if not metrics:
                continue
            df_row = pd.DataFrame([metrics])
            df_row = add_config_to_df(df_row, config_file, config_sections)
            all_rows.append(df_row)
    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    else:
        return pd.DataFrame()  # Return empty DataFrame if no rows

def main():
    parser = argparse.ArgumentParser(description="Merge results from multiple sweep directories.")
    parser.add_argument(
        "--sweep_paths", 
        type=str, 
        nargs='+',
        required=True,
        help="One or more sweep directories to merge (space separated)"
    )
    parser.add_argument(
        "--metric_keys", 
        type=str, 
        nargs='+',
        required=True,
        help="Metric keys to extract from logs (space separated)"
    )
    parser.add_argument(
        "--config_sections",
        type=str,
        nargs='*',
        default=[
            "algorithm.dimensionality_reduction",
            "metrics.embedding.trustworthiness",
            "metrics.dataset.geographic_preservation",
            "metrics.dataset.admixture_preservation",
        ],
        help="Config sections to extract (default: the four main ones)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/DR_merged_results.csv",
        help="Output CSV path"
    )

    args = parser.parse_args()
    sweep_dirs = [Path(p) for p in args.sweep_paths]
    df_all = load_all_runs_from_sweeps(sweep_dirs, args.metric_keys, args.config_sections)
    if df_all.empty:
        print("No data found. No CSV written.")
        return
    df_all.to_csv(args.output, index=False)
    print(f"Written {len(df_all)} rows to {args.output}")
    print(df_all.head())

if __name__ == "__main__":
    main()