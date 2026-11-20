"""Create performance charts from ML/results/model_comparison.csv."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLE = RESULTS / "model_comparison.csv"


def _write_svg(frame: pd.DataFrame, metric: str, title: str, filename: str, color: str,
               limits: tuple[float, float] | None = None) -> Path:
    """Write a dependency-free SVG bar chart when matplotlib is unavailable."""
    width, height = max(900, 100 * len(frame)), 470
    left, right, top, bottom = 78, 28, 58, 132
    plot_w, plot_h = width - left - right, height - top - bottom
    values = pd.to_numeric(frame[metric], errors="coerce").fillna(0).to_numpy(dtype=float)
    low, high = limits or (0.0, max(float(values.max()) * 1.15, 1.0))
    labels = frame["feature_set"].str.upper() + " / " + frame["model"].str.replace("_", " ")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2:.0f}" y="32" text-anchor="middle" font-family="Arial,sans-serif" font-size="20" font-weight="600" fill="#172b3a">{escape(title)}</text>',
    ]
    for tick in range(6):
        value = low + (high - low) * tick / 5
        y = top + plot_h * (1 - tick / 5)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e4e9ed"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="12" fill="#526370">{value:.2f}</text>')
    slot = plot_w / max(len(values), 1)
    bar_w = min(58, slot * 0.62)
    for i, value in enumerate(values):
        x = left + i * slot + (slot - bar_w) / 2
        fraction = min(1.0, max(0.0, (value - low) / max(high - low, 1e-12)))
        bar_h = fraction * plot_h
        y = top + plot_h - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{max(top+14, y-6):.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#34495a">{value:.3f}</text>')
        label_x = left + (i + 0.5) * slot
        label = escape(str(labels.iloc[i]))
        parts.append(f'<text x="{label_x:.1f}" y="{height-bottom+24}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#34495a">{label}</text>')
    parts.append(f'<text x="20" y="{top+plot_h/2:.1f}" transform="rotate(-90 20 {top+plot_h/2:.1f})" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#34495a">{escape(metric.upper())}</text>')
    parts.append('</svg>')
    path = RESULTS / filename
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def main():
    if not TABLE.exists():
        raise FileNotFoundError(f"Run python -m ML.training.train first: {TABLE}")
    frame = pd.read_csv(TABLE)
    if frame.empty:
        raise ValueError("The comparison table has no rows.")
    report_path = RESULTS / "run_report.json"
    cv_title = "Pair-grouped stratified cross-validation model comparison"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            cv_title = (
                f"{report.get('folds', '?')}-fold pair-grouped CV x "
                f"{report.get('repeats', '?')} repeat(s) model comparison"
            )
        except (OSError, ValueError, TypeError):
            pass

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        files = [_write_svg(frame, "rmse", "Cross-validation RMSE by feature group and model", "rmse_comparison.svg", "#48a995")]
        if "auc_s_ge_10" in frame and frame["auc_s_ge_10"].notna().any():
            files.append(_write_svg(frame, "auc_s_ge_10", "Synergy classification performance (S >= 10)", "auc_comparison.svg", "#7398e6", (0, 1)))
        for metric, title, color in (
            ("balanced_accuracy_s_ge_10", "Balanced accuracy at S >= 10", "#e0a64e"),
            ("f1_s_ge_10", "F1 score at S >= 10", "#8c79d8"),
        ):
            if metric in frame and frame[metric].notna().any():
                files.append(_write_svg(frame, metric, title, f"{metric}_comparison.svg", color, (0, 1)))
        graph_dir = RESULTS / "graphs"
        graph_dir.mkdir(parents=True, exist_ok=True)
        for model, group in frame.groupby("model", sort=True):
            safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(model).lower())
            files.append(_write_svg(
                group, "rmse", f"{str(model).replace('_', ' ').title()} RMSE by feature view",
                f"graphs/{safe_name}_rmse.svg", "#48a995",
            ))
            if "auc_s_ge_10" in group and group["auc_s_ge_10"].notna().any():
                files.append(_write_svg(
                    group, "auc_s_ge_10", f"{str(model).replace('_', ' ').title()} AUC by feature view",
                    f"graphs/{safe_name}_auc.svg", "#7398e6", (0, 1),
                ))
            for metric, title, color in (
                ("balanced_accuracy_s_ge_10", "balanced accuracy", "#e0a64e"),
                ("f1_s_ge_10", "F1 score", "#8c79d8"),
            ):
                if metric in group and group[metric].notna().any():
                    files.append(_write_svg(
                        group, metric,
                        f"{str(model).replace('_', ' ').title()} {title} by feature view",
                        f"graphs/{safe_name}_{metric}.svg", color, (0, 1),
                    ))
        print("Matplotlib is unavailable; saved SVG charts: " + ", ".join(map(str, files)))
        return

    labels = frame["feature_set"].str.upper() + "\n" + frame["model"].str.replace("_", " ")
    errors_low = (frame["rmse"] - frame["rmse_ci95_low"]).clip(lower=0).fillna(0)
    errors_high = (frame["rmse_ci95_high"] - frame["rmse"]).clip(lower=0).fillna(0)
    fig, ax = plt.subplots(figsize=(max(9, len(frame) * 0.8), 5.5))
    ax.bar(range(len(frame)), frame["rmse"], color="#48a995", yerr=[errors_low, errors_high], capsize=3)
    ax.set_xticks(range(len(frame)), labels, rotation=35, ha="right")
    ax.set_ylabel("RMSE (DrugComb S score)")
    ax.set_title(cv_title)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULTS / "rmse_comparison.png", dpi=180)
    fig.savefig(RESULTS / "rmse_comparison.svg")
    plt.close(fig)

    if "auc_s_ge_10" in frame and frame["auc_s_ge_10"].notna().any():
        fig, ax = plt.subplots(figsize=(max(9, len(frame) * 0.8), 5.5))
        ax.bar(range(len(frame)), frame["auc_s_ge_10"], color="#7398e6")
        ax.axhline(0.5, color="#9b6070", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(len(frame)), labels, rotation=35, ha="right")
        ax.set_ylabel("ROC AUC (S >= 10)")
        ax.set_title("Synergy classification performance")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(RESULTS / "auc_comparison.png", dpi=180)
        fig.savefig(RESULTS / "auc_comparison.svg")
        plt.close(fig)

    for metric, title, color in (
        ("r2", "R² comparison by feature view and model", "#e0a64e"),
        ("spearman", "Spearman correlation by feature view and model", "#8c79d8"),
        ("balanced_accuracy_s_ge_10", "Balanced accuracy at S >= 10", "#e0a64e"),
        ("f1_s_ge_10", "F1 score at S >= 10", "#8c79d8"),
    ):
        if metric not in frame or not frame[metric].notna().any():
            continue
        fig, ax = plt.subplots(figsize=(max(9, len(frame) * 0.8), 5.5))
        ax.bar(range(len(frame)), frame[metric], color=color)
        ax.axhline(0, color="#526370", linewidth=0.8)
        ax.set_xticks(range(len(frame)), labels, rotation=35, ha="right")
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(RESULTS / f"{metric}_comparison.png", dpi=180)
        fig.savefig(RESULTS / f"{metric}_comparison.svg")
        plt.close(fig)

    # Combined view: every model is grouped on the x-axis; feature views use color.
    feature_order = ["structure", "ds", "drs", "multilevel"]
    model_candidates = ("lasso", "random_forest", "adaboost", "xgboost", "synergyx", "stacked_random_forest")
    available_models = set(frame["model"].astype(str))
    model_order = [name for name in model_candidates if name in available_models]
    available_views = set(frame["feature_set"].astype(str))
    view_order = [name for name in feature_order if name in available_views]
    view_colors = {
        "structure": "#48a995",
        "ds": "#7398e6",
        "drs": "#e0a64e",
        "multilevel": "#8c79d8",
    }
    metric_panels = (
        ("rmse", "RMSE (lower is better)"),
        ("r2", "R-squared"),
        ("spearman", "Spearman rho"),
        ("auc_s_ge_10", "ROC AUC (S >= 10)"),
        ("balanced_accuracy_s_ge_10", "Balanced accuracy (S >= 10)"),
        ("f1_s_ge_10", "F1 score (S >= 10)"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    x = np.arange(len(model_order))
    bar_width = 0.8 / max(len(view_order), 1)
    for ax, (metric, title) in zip(axes.flat, metric_panels):
        if metric not in frame or not pd.to_numeric(frame[metric], errors="coerce").notna().any():
            ax.set_visible(False)
            continue
        pivot = frame.pivot(index="model", columns="feature_set", values=metric)
        pivot = pivot.reindex(index=model_order, columns=view_order)
        for view_index, view in enumerate(view_order):
            values = pd.to_numeric(pivot[view], errors="coerce").to_numpy(dtype=float)
            offset = (view_index - (len(view_order) - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                values,
                width=bar_width * 0.9,
                color=view_colors[view],
                label=view.upper(),
            )
        ax.axhline(0, color="#526370", linewidth=0.8)
        if metric in {
            "auc_s_ge_10",
            "balanced_accuracy_s_ge_10",
            "f1_s_ge_10",
        }:
            ax.axhline(0.5, color="#9b6070", linestyle="--", linewidth=1)
            ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace("_", " ").title() for name in model_order], rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(f"All model families by feature view | {cv_title}", fontsize=15)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="lower center", ncol=len(view_order), title="Feature view")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    combined_png = RESULTS / "all_models_comparison.png"
    combined_svg = RESULTS / "all_models_comparison.svg"
    fig.savefig(combined_png, dpi=180)
    fig.savefig(combined_svg)
    plt.close(fig)

    graph_dir = RESULTS / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    manifest["all_models_comparison"] = [str(combined_png), str(combined_svg)]
    for model, group in frame.groupby("model", sort=True):
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(model).lower())
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, metric, title, color in zip(
            axes.flat,
            (
                "rmse", "r2", "spearman", "auc_s_ge_10",
                "balanced_accuracy_s_ge_10", "f1_s_ge_10",
            ),
            (
                "RMSE", "R²", "Spearman", "ROC AUC (S ≥ 10)",
                "Balanced accuracy", "F1 (S ≥ 10)",
            ),
            (
                "#48a995", "#e0a64e", "#8c79d8", "#7398e6",
                "#e0a64e", "#8c79d8",
            ),
        ):
            if metric not in group or not group[metric].notna().any():
                ax.set_visible(False)
                continue
            ax.bar(group["feature_set"].str.upper(), group[metric], color=color)
            if metric in {
                "auc_s_ge_10",
                "balanced_accuracy_s_ge_10",
                "f1_s_ge_10",
            }:
                ax.axhline(0.5, color="#9b6070", linestyle="--", linewidth=1)
                ax.set_ylim(0, 1)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle(f"{str(model).replace('_', ' ').title()} by feature view", fontsize=15)
        fig.tight_layout()
        png_path = graph_dir / f"{safe_name}_performance.png"
        svg_path = graph_dir / f"{safe_name}_performance.svg"
        fig.savefig(png_path, dpi=180)
        fig.savefig(svg_path)
        plt.close(fig)
        manifest[str(model)] = [str(png_path), str(svg_path)]
    (graph_dir / "index.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved model-comparison plots under {RESULTS}")


if __name__ == "__main__":
    main()

