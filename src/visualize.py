"""
Visualizations for HR analytics pipeline: charts and figures for GitHub/portfolio.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# Style for portfolio-ready plots
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass
sns.set_palette("husl")


def plot_attrition_by_department(df, save_path: Path | None = None) -> None:
    """Bar chart: attrition % by department."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = df["Department"]
    y = df["attrition_pct"]
    bars = ax.bar(x, y, color=sns.color_palette("husl", len(x)), edgecolor="gray", linewidth=0.5)
    ax.set_xlabel("Department")
    ax.set_ylabel("Attrition %")
    ax.set_title("Employee Attrition Rate by Department")
    ax.set_ylim(0, None)
    for bar, pct in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{pct}%", ha="center", fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_confusion_matrix(cm, save_path: Path | None = None) -> None:
    """Heatmap of confusion matrix (0 = Stay, 1 = Leave)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Stay", "Leave"], yticklabels=["Stay", "Leave"],
        cbar_kws={"label": "Count"}
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Attrition Prediction)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_feature_importance(pipe, feature_names: list, top_n: int = 15, save_path: Path | None = None) -> None:
    """Bar chart of top N feature importances from the Random Forest."""
    clf = pipe.named_steps["clf"]
    importances = clf.feature_importances_
    names = feature_names[: len(importances)]
    idx = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)[:top_n]
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = range(len(idx))
    ax.barh(y_pos, [importances[i] for i in idx], color=sns.color_palette("husl", len(idx)))
    ax.set_yticks(y_pos)
    ax.set_yticklabels([names[i] for i in idx], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances (Random Forest)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_target_distribution(attrition_counts: dict, save_path: Path | None = None) -> None:
    """Pie or bar showing class balance (Stay vs Leave)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = list(attrition_counts.keys())
    sizes = list(attrition_counts.values())
    colors = sns.color_palette("husl", 2)
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    ax.set_title("Attrition (Target) Distribution")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def generate_all_plots(
    dept_df=None,
    confusion_matrix=None,
    pipe=None,
    feature_names=None,
    target_counts=None,
    output_dir: Path | None = None,
) -> None:
    """Generate all figures and save to output_dir."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    if dept_df is not None and not dept_df.empty:
        plot_attrition_by_department(dept_df, save_path=figures_dir / "attrition_by_department.png")
    if confusion_matrix is not None:
        plot_confusion_matrix(confusion_matrix, save_path=figures_dir / "confusion_matrix.png")
    if pipe is not None and feature_names is not None:
        plot_feature_importance(pipe, feature_names, save_path=figures_dir / "feature_importance.png")
    if target_counts is not None:
        plot_target_distribution(target_counts, save_path=figures_dir / "target_distribution.png")

    print(f"  Figures saved to: {figures_dir}")
