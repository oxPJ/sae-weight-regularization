"""
Multi-Trainer Class Pareto Radar Comparison

Creates radar plots comparing Pareto frontiers across different SAE trainer classes.
Pareto optimal trainers shown in RED with hyperparameter labels.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        if (cand / "scripts" / "download_artifact.py").is_file():
            return cand
    raise FileNotFoundError("Could not find repo root.")


REPO_ROOT = _repo_root()
DATA_RESULTS_DIR = REPO_ROOT / "data_results"
DATA_MODEL_WEIGHTS_DIR = REPO_ROOT / "data_model_weights"

# Configuration for each experiment
EXPERIMENTS = {
    "BatchTopK_L1": {
        "eval_dir": "eval_results_SAEBenchingBatchTopK",
        "models_dir": "models_SAEBenchingBatchTopK",
        "color": "#E24A33",
    },
    "BatchTopK_L2": {
        "eval_dir": "eva_batch_topK_L2",
        "models_dir": "sae_models_batch_topK_L2",
        "color": "#FF6B35",
    },
    "Matryoshka_L1": {
        "eval_dir": "eval_results_matryoshka_l1",
        "models_dir": "models_matryoshka_l1",
        "color": "#348ABD",
    },
    "Matryoshka_L2": {
        "eval_dir": "eval_matryoshka_L2",
        "models_dir": "sae_models_matryoshka_L2",
        "color": "#1E90FF",
    },
    "TopK_L1": {
        "eval_dir": "eval_results_SAEBenchingTopK",
        "models_dir": "models_SAEBenchingTopK",
        "color": "#7A68A6",
    },
    "TopK_L2": {
        "eval_dir": "eval_results_top_k_l2",
        "models_dir": "models_topk_l2",
        "color": "#467821",
    },
}

METRICS = ["ce_score", "tpp_10", "scr_10", "sp_top1"]
METRIC_LABELS = {
    "ce_score": "CE Score",
    "tpp_10": "TPP@10",
    "scr_10": "SCR@10",
    "sp_top1": "SP Top-1",
}


def load_experiment_data(eval_dir: Path, models_dir: Path) -> pd.DataFrame:
    """Load all trainer data for an experiment."""
    trainer_dirs = []
    for subdir in models_dir.iterdir():
        if subdir.is_dir():
            for trainer_dir in subdir.iterdir():
                if trainer_dir.is_dir() and trainer_dir.name.startswith("trainer_"):
                    trainer_dirs.append(trainer_dir)
    
    if not trainer_dirs:
        trainer_dirs = [d for d in models_dir.iterdir() 
                       if d.is_dir() and d.name.startswith("trainer_")]
    
    data = []
    for trainer_dir in sorted(trainer_dirs, key=lambda x: int(x.name.split('_')[1])):
        config_path = trainer_dir / "config.json"
        if not config_path.exists():
            continue
            
        with open(config_path) as f:
            cfg = json.load(f)["trainer"]
        
        trainer_name = trainer_dir.name + "_custom_sae"
        # Alternative naming pattern (with layer prefix)
        trainer_name_alt = f"resid_post_layer_3_{trainer_dir.name}_custom_sae"
        
        l1_w = cfg.get("l1_w", 0) or 0
        l2_w = cfg.get("l2_w", 0) or 0
        
        if l2_w != 0:
            reg_type, reg_value = "l2_w", l2_w
        else:
            reg_type, reg_value = "l1_w", l1_w
        
        entry = {
            "trainer": trainer_dir.name,
            "k": cfg.get("k"),
            "reg_type": reg_type,
            "reg_value": reg_value,
            "trainer_class": cfg.get("trainer_class", "unknown"),
        }
        
        # Helper to find eval file with either naming pattern
        def find_eval_path(subdir: str, suffix: str) -> Path | None:
            path1 = eval_dir / subdir / f"{trainer_name}_{suffix}"
            if path1.exists():
                return path1
            path2 = eval_dir / subdir / f"{trainer_name_alt}_{suffix}"
            if path2.exists():
                return path2
            return None
        
        # Load metrics
        core_path = find_eval_path("core", "eval_results.json")
        if core_path:
            with open(core_path) as f:
                core = json.load(f)
                entry["ce_score"] = core["eval_result_metrics"]["model_performance_preservation"]["ce_loss_score"]
        
        sp_path = find_eval_path("sparse_probing", "eval_results.json")
        if sp_path:
            with open(sp_path) as f:
                sp = json.load(f)["eval_result_metrics"]["sae"]
                entry["sp_top1"] = sp.get("sae_top_1_test_accuracy")
        
        tpp_path = find_eval_path("tpp", "eval_results.json")
        if tpp_path:
            with open(tpp_path) as f:
                tpp = json.load(f)["eval_result_metrics"]["tpp_metrics"]
                entry["tpp_10"] = tpp.get("tpp_threshold_10_total_metric")
        
        scr_path = find_eval_path("scr", "eval_results.json")
        if scr_path:
            with open(scr_path) as f:
                scr = json.load(f)["eval_result_metrics"]["scr_metrics"]
                entry["scr_10"] = scr.get("scr_metric_threshold_10")
        
        data.append(entry)
    
    return pd.DataFrame(data)


def compute_pareto_front(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Compute Pareto frontier (all metrics maximized)."""
    df = df.copy()
    df['pareto_optimal'] = False
    
    values = df[metrics].values
    
    for idx in range(len(df)):
        point = values[idx]
        is_dominated = False
        
        for other_idx in range(len(df)):
            if idx == other_idx:
                continue
            other = values[other_idx]
            
            at_least_as_good = all(other[i] >= point[i] for i in range(len(metrics)))
            strictly_better = any(other[i] > point[i] for i in range(len(metrics)))
            
            if at_least_as_good and strictly_better:
                is_dominated = True
                break
        
        if not is_dominated:
            df.iloc[idx, df.columns.get_loc('pareto_optimal')] = True
    
    return df


def normalize_metrics(df: pd.DataFrame, metrics: list[str], global_ranges: dict = None) -> pd.DataFrame:
    """Normalize metrics to [0, 1] range."""
    df_norm = df.copy()
    
    for metric in metrics:
        if global_ranges and metric in global_ranges:
            min_val, max_val = global_ranges[metric]
        else:
            min_val = df[metric].min()
            max_val = df[metric].max()
        
        if max_val > min_val:
            df_norm[f"{metric}_norm"] = (df[metric] - min_val) / (max_val - min_val)
        else:
            df_norm[f"{metric}_norm"] = 0.5
    
    return df_norm


def format_reg(reg_value):
    """Format regularization value for display."""
    if reg_value == 0:
        return "0"
    elif reg_value < 1e-3:
        return f"{reg_value:.0e}"
    else:
        return f"{reg_value:g}"


def plot_single_experiment_radar_labeled(df: pd.DataFrame, exp_name: str, exp_color: str, 
                                          output_path: Path, global_ranges: dict):
    """Create radar plot highlighting only the best Pareto config in red."""
    
    df_norm = normalize_metrics(df, METRICS, global_ranges)
    
    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # First, plot all dominated trainers in gray (background)
    dominated_df = df_norm[df_norm['pareto_optimal'] == False]
    for idx, row in dominated_df.iterrows():
        values = [row[f"{m}_norm"] for m in METRICS]
        values += values[:1]
        ax.plot(angles, values, '-', color='lightgray', linewidth=1, alpha=0.5, zorder=10)
    
    # Then plot Pareto optimal in RED with labels
    pareto_df = df_norm[df_norm['pareto_optimal'] == True]
    
    # Color mapping for remaining Pareto (non-best)
    k_values = sorted(df['k'].dropna().unique())
    k_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(k_values))) if k_values else []
    k_to_color = {k: k_colors[i] for i, k in enumerate(k_values)}
    
    # Determine best Pareto config (highest avg normalized score)
    pareto_df = pareto_df.copy()
    pareto_df['avg_score'] = pareto_df[[f"{m}_norm" for m in METRICS]].mean(axis=1)
    best_idx = pareto_df['avg_score'].idxmax()
    
    # Plot non-best Pareto configs
    for idx, row in pareto_df.iterrows():
        values = [row[f"{m}_norm"] for m in METRICS]
        values += values[:1]
        
        if idx == best_idx:
            continue
        
        k = row['k']
        color = k_to_color.get(k, exp_color)
        ax.plot(angles, values, '-', color=color, linewidth=2, alpha=0.9, zorder=40)
        ax.fill(angles, values, color=color, alpha=0.02)
    
    # Plot best config in red with label
    best_row = pareto_df.loc[best_idx]
    best_values = [best_row[f"{m}_norm"] for m in METRICS]
    best_values += best_values[:1]
    best_k = int(best_row['k'])
    best_reg = format_reg(best_row['reg_value'])
    
    best_line, = ax.plot(angles, best_values, 'o-', color='red', linewidth=3, 
                         markersize=9, alpha=0.95, zorder=60)
    ax.fill(angles, best_values, color='red', alpha=0.08)
    
    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=12)
    ax.set_ylim(0, 1)
    
    # Legend: include best config + color-by-k reference
    legend_handles = [best_line]
    legend_labels = [f"HIGHLIGHT: k={best_k}, reg={best_reg}"]
    
    k_handles = [Line2D([0], [0], color=k_to_color[k], linewidth=2, label=f"k={int(k)}") 
                 for k in k_values]
    legend_handles.extend(k_handles)
    legend_labels.extend([f"k={int(k)}" for k in k_values])
    
    ax.legend(legend_handles, legend_labels, loc='upper right', 
              bbox_to_anchor=(1.45, 1.0), fontsize=9, title="Lines", title_fontsize=10)
    
    n_pareto = len(pareto_df)
    n_total = len(df)
    ax.set_title(f"{exp_name}\nPareto Optimal: {n_pareto}/{n_total} (RED) | Dominated: {n_total - n_pareto} (gray)", 
                fontsize=14, fontweight='bold', pad=25, color=exp_color)
    
    plt.tight_layout()
    filename = output_path / f"pareto_radar_{exp_name.lower().replace(' ', '_')}_labeled.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_combined_radar_labeled(all_data: dict, output_path: Path, global_ranges: dict):
    """Create 2x3 comparison plot highlighting only the best Pareto config in red."""
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 16), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]
    
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<', '>', 'p', '8', 'H']
    
    for ax_idx, (exp_name, exp_info) in enumerate(EXPERIMENTS.items()):
        ax = axes[ax_idx]
        
        if exp_name not in all_data:
            ax.set_visible(False)
            continue
            
        df = all_data[exp_name]
        df_norm = normalize_metrics(df, METRICS, global_ranges)
        
        k_values = sorted(df['k'].dropna().unique())
        k_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(k_values))) if k_values else []
        k_to_color = {k: k_colors[i] for i, k in enumerate(k_values)}
        
        # Plot dominated in gray
        dominated_df = df_norm[df_norm['pareto_optimal'] == False]
        for idx, row in dominated_df.iterrows():
            values = [row[f"{m}_norm"] for m in METRICS]
            values += values[:1]
            ax.plot(angles, values, '-', color='lightgray', linewidth=0.8, alpha=0.4, zorder=10)
        
        best_line = None
        best_label = ""
        # Plot Pareto optimal configurations
        pareto_df = df_norm[df_norm['pareto_optimal'] == True]
        
        # Determine best Pareto config
        pareto_df = pareto_df.copy()
        pareto_df['avg_score'] = pareto_df[[f"{m}_norm" for m in METRICS]].mean(axis=1)
        best_idx = pareto_df['avg_score'].idxmax()
        
        for i, (idx, row) in enumerate(pareto_df.iterrows()):
            values = [row[f"{m}_norm"] for m in METRICS]
            values += values[:1]
            
            k = row['k']
            color = k_to_color.get(k, exp_info['color'])
            
            if idx == best_idx:
                label = f"k={int(k)}, reg={format_reg(row['reg_value'])}"
                marker = markers[i % len(markers)]
                best_line, = ax.plot(angles, values, 'o-', color='red', linewidth=2.5,
                                     markersize=7, alpha=0.95, zorder=60, marker=marker)
                ax.fill(angles, values, color='red', alpha=0.07)
                best_label = label
            else:
                ax.plot(angles, values, '-', color=color, linewidth=1.8, alpha=0.85, zorder=40)
                ax.fill(angles, values, color=color, alpha=0.02)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=10)
        ax.set_ylim(0, 1)
        
        # Legends: highlight best + k colors
        handles = []
        labels = []
        if best_line is not None:
            handles.append(best_line)
            labels.append(f"Best: {best_label}")
        k_handles = [Line2D([0], [0], color=k_to_color[k], linewidth=2, label=f"k={int(k)}") 
                     for k in k_values]
        handles.extend(k_handles)
        labels.extend([f"k={int(k)}" for k in k_values])
        ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=7,
                  title="Legend", title_fontsize=8)
        
        n_pareto = len(pareto_df)
        n_total = len(df)
        ax.set_title(f"{exp_name}\n{n_pareto}/{n_total} Pareto optimal", 
                    fontsize=12, fontweight='bold', pad=15, color=exp_info['color'])
    
    plt.suptitle("Pareto Frontier Comparison: RED = Pareto Optimal, Gray = Dominated\n(Normalized metrics, higher = better)", 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    
    filename = output_path / "pareto_radar_comparison_labeled.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_best_comparison_labeled(all_data: dict, output_path: Path, global_ranges: dict, top_n: int = 5):
    """Create single radar comparing best configs from each experiment."""
    
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]
    
    all_handles = []
    all_labels = []
    
    for exp_name, exp_info in EXPERIMENTS.items():
        if exp_name not in all_data:
            continue
            
        df = all_data[exp_name]
        df_norm = normalize_metrics(df, METRICS, global_ranges)
        
        # Get Pareto optimal, sorted by avg score
        pareto_df = df_norm[df_norm['pareto_optimal'] == True].copy()
        if pareto_df.empty:
            continue
            
        pareto_df['avg_score'] = pareto_df[[f"{m}_norm" for m in METRICS]].mean(axis=1)
        top_configs = pareto_df.nlargest(top_n, 'avg_score')
        
        # Plot top configs for this experiment
        for i, (idx, row) in enumerate(top_configs.iterrows()):
            values = [row[f"{m}_norm"] for m in METRICS]
            values += values[:1]
            
            k = int(row['k'])
            reg = format_reg(row['reg_value'])
            label = f"{exp_name}: k={k}, reg={reg}"
            
            # Vary alpha for rank
            alpha = 0.9 - i * 0.15
            linewidth = 2.5 - i * 0.3
            
            line, = ax.plot(angles, values, 'o-', color=exp_info['color'], 
                           linewidth=linewidth, markersize=6, alpha=alpha)
            
            if i == 0:  # Only fill best one
                ax.fill(angles, values, color=exp_info['color'], alpha=0.05)
            
            all_handles.append(line)
            all_labels.append(label)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=12)
    ax.set_ylim(0, 1)
    
    # Legend
    ax.legend(all_handles, all_labels, loc='upper right', 
             bbox_to_anchor=(1.5, 1.0), fontsize=9,
             title=f"Top {top_n} Pareto Configs per Architecture")
    
    ax.set_title(f"Best Pareto Configurations Comparison\n(Top {top_n} per architecture by avg normalized score)", 
                fontsize=16, fontweight='bold', pad=25)
    
    plt.tight_layout()
    filename = output_path / "pareto_radar_best_comparison_labeled.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def main():
    output_path = Path("./my_plots/pareto_comparison")
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("MULTI-EXPERIMENT PARETO RADAR (LABELED)")
    print("=" * 70)
    
    # Load all experiments
    all_data = {}
    all_metrics_values = {m: [] for m in METRICS}
    
    for exp_name, exp_info in EXPERIMENTS.items():
        print(f"\nLoading {exp_name}...")
        eval_dir = Path(exp_info['eval_dir'])
        models_dir = Path(exp_info['models_dir'])
        
        if not eval_dir.exists() or not models_dir.exists():
            print(f"  Skipping - directories not found")
            continue
        
        df = load_experiment_data(eval_dir, models_dir)
        df = df.dropna(subset=METRICS)
        
        if df.empty:
            print(f"  Skipping - no complete data")
            continue
        
        df = compute_pareto_front(df, METRICS)
        all_data[exp_name] = df
        
        for m in METRICS:
            all_metrics_values[m].extend(df[m].tolist())
        
        n_pareto = df['pareto_optimal'].sum()
        print(f"  Loaded {len(df)} trainers, {n_pareto} Pareto optimal")
    
    if not all_data:
        print("\nNo data loaded!")
        return
    
    # Compute global ranges
    global_ranges = {}
    for m in METRICS:
        if all_metrics_values[m]:
            global_ranges[m] = (min(all_metrics_values[m]), max(all_metrics_values[m]))
    
    print(f"\nGlobal metric ranges:")
    for m, (lo, hi) in global_ranges.items():
        print(f"  {m}: [{lo:.3f}, {hi:.3f}]")
    
    # Generate plots
    print("\n" + "-" * 70)
    print("GENERATING LABELED PLOTS")
    print("-" * 70)
    
    # 1. Individual labeled radar plots
    print("\n1. Creating individual labeled radar plots...")
    for exp_name, exp_info in EXPERIMENTS.items():
        if exp_name in all_data:
            plot_single_experiment_radar_labeled(
                all_data[exp_name], exp_name, exp_info['color'], 
                output_path, global_ranges
            )
    
    # 2. Combined 2x2 labeled plot
    print("\n2. Creating combined labeled comparison plot...")
    plot_combined_radar_labeled(all_data, output_path, global_ranges)
    
    # 3. Best configs comparison
    print("\n3. Creating best configs comparison plot...")
    plot_best_comparison_labeled(all_data, output_path, global_ranges, top_n=3)
    
    # Print Pareto optimal configs
    print("\n" + "=" * 70)
    print("PARETO OPTIMAL CONFIGURATIONS")
    print("=" * 70)
    
    for exp_name in all_data:
        df = all_data[exp_name]
        pareto = df[df['pareto_optimal']]
        print(f"\n{exp_name} ({len(pareto)} Pareto optimal):")
        for _, row in pareto.iterrows():
            k = int(row['k'])
            reg = format_reg(row['reg_value'])
            print(f"  k={k:>3}, reg={reg:>8}: CE={row['ce_score']:.3f}, TPP={row['tpp_10']:.3f}, SCR={row['scr_10']:.3f}, SP={row['sp_top1']:.3f}")
    
    print(f"\nDone! Plots saved to: {output_path}")


if __name__ == "__main__":
    main()
