import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

ENCODER_COLORS = {
    "cnn": "#9E9E9E",
    "hybrid": "#3B82F6",
}

ABLATION_COLORS = {
    "cnn_baseline": "#9E9E9E",
    "hybrid_full": "#3B82F6",
    "hybrid_1layer": "#10B981",
    "hybrid_mean_pool": "#F59E0B",
}

ENCODER_LABELS = {
    "cnn": "Rainbow (CNN)",
    "hybrid": "Rainbow + Hybrid",
}

GAME_ORDER = [
    "pong",
    "breakout",
    "space_invaders",
    "seaquest",
    "qbert",
    "beam_rider",
    "enduro",
]


def experiment_dirs(root: Path):
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if path.is_dir()])


def parse_standard_name(name: str):
    parts = name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected experiment name: {name}")
    seed = parts[-1].replace("seed", "")
    encoder = parts[-2]
    game = "_".join(parts[:-2])
    return game, encoder, seed


def load_metrics_csv(exp_dir: Path) -> pd.DataFrame:
    metrics_path = exp_dir / "metrics.csv"
    if not metrics_path.exists():
        return pd.DataFrame()
    return pd.read_csv(metrics_path)


def style():
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#9A9A9A",
            "axes.linewidth": 0.7,
            "grid.color": "#ECECEC",
            "grid.linewidth": 0.4,
            "grid.linestyle": "-",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def draw_box(ax, x, y, w, h, text, facecolor, edgecolor="#8F8F8F", fontsize=8):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=0.8,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def draw_arrow(ax, start, end, color="#7A7A7A"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=0.9,
        color=color,
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def draw_line(ax, start, end, color="#7A7A7A", linewidth=0.9, linestyle="-"):
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        solid_capstyle="round",
    )


def gather_main_benchmark(base_dir: Path) -> pd.DataFrame:
    rows = []
    for exp_dir in experiment_dirs(base_dir / "code" / "results"):
        game, encoder, seed = parse_standard_name(exp_dir.name)
        metrics = load_metrics_csv(exp_dir)
        if metrics.empty:
            continue
        for _, row in metrics.iterrows():
            rows.append(
                {
                    "game": game,
                    "encoder": encoder,
                    "seed": seed,
                    "step_m": row["step"] / 1e6,
                    "raw_score": row["raw_score"],
                    "hns": row["human_normalized_score"],
                }
            )
    return pd.DataFrame(rows)


def gather_multiseed_timeseries(base_dir: Path, game: str) -> pd.DataFrame:
    rows = []
    for root in [base_dir / "code" / "results", base_dir / "code" / "results_extra_seeds"]:
        for exp_dir in experiment_dirs(root):
            exp_game, encoder, seed = parse_standard_name(exp_dir.name)
            if exp_game != game:
                continue
            metrics = load_metrics_csv(exp_dir)
            if metrics.empty:
                continue
            for _, row in metrics.iterrows():
                rows.append(
                    {
                        "game": exp_game,
                        "encoder": encoder,
                        "seed": seed,
                        "step_m": row["step"] / 1e6,
                        "raw_score": row["raw_score"],
                    }
                )
    return pd.DataFrame(rows)


def gather_long_horizon(base_dir: Path, game: str) -> pd.DataFrame:
    rows = []
    for exp_dir in experiment_dirs(base_dir / "code" / "results_long"):
        exp_game, encoder, seed = parse_standard_name(exp_dir.name)
        if exp_game != game:
            continue
        metrics = load_metrics_csv(exp_dir)
        if metrics.empty:
            continue
        for _, row in metrics.iterrows():
            rows.append(
                {
                    "game": exp_game,
                    "encoder": encoder,
                    "seed": seed,
                    "step_m": row["step"] / 1e6,
                    "raw_score": row["raw_score"],
                }
            )
    return pd.DataFrame(rows)


def gather_main_game_timeseries(base_dir: Path, game: str) -> pd.DataFrame:
    rows = []
    for exp_dir in experiment_dirs(base_dir / "code" / "results"):
        exp_game, encoder, seed = parse_standard_name(exp_dir.name)
        if exp_game != game:
            continue
        metrics = load_metrics_csv(exp_dir)
        if metrics.empty:
            continue
        for _, row in metrics.iterrows():
            rows.append(
                {
                    "game": exp_game,
                    "encoder": encoder,
                    "seed": seed,
                    "step_m": row["step"] / 1e6,
                    "raw_score": row["raw_score"],
                }
            )
    return pd.DataFrame(rows)


def gather_qbert_ablation(base_dir: Path) -> pd.DataFrame:
    rows = []
    variants = {
        "cnn_baseline": base_dir / "code" / "results" / "qbert_cnn_seed42",
        "hybrid_full": base_dir / "code" / "results" / "qbert_hybrid_seed42",
        "hybrid_1layer": base_dir
        / "code"
        / "results_ablations"
        / "results_ablation_hybrid_1layer"
        / "ablation_qbert_hybrid_1layer_seed42",
        "hybrid_mean_pool": base_dir
        / "code"
        / "results_ablations"
        / "results_ablation_hybrid_mean_pool"
        / "ablation_qbert_hybrid_mean_pool_seed42",
    }
    for variant, exp_dir in variants.items():
        metrics = load_metrics_csv(exp_dir)
        if metrics.empty:
            continue
        for _, row in metrics.iterrows():
            rows.append(
                {
                    "variant": variant,
                    "step_m": row["step"] / 1e6,
                    "raw_score": row["raw_score"],
                }
            )
    return pd.DataFrame(rows)


def setup_axis(ax: plt.Axes):
    ax.grid(axis="y", alpha=0.8)
    ax.grid(axis="x", alpha=0)
    ax.tick_params(length=2.5, width=0.7, color="#666666")
    ax.spines["left"].set_color("#9A9A9A")
    ax.spines["bottom"].set_color("#9A9A9A")


def plot_series(ax: plt.Axes, x, y, color: str, label: str):
    ax.plot(
        x,
        y,
        color=color,
        linewidth=1.1,
        label=label,
        solid_capstyle="round",
    )


def place_legend_above_axes(
    ax: plt.Axes, ncol: int = 1, fontsize: float = 7.0
) -> None:
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=ncol,
        frameon=False,
        handlelength=1.0,
        borderaxespad=0.0,
        labelspacing=0.25,
        columnspacing=0.9,
        fontsize=fontsize,
    )


def place_legend_above_figure(
    fig: plt.Figure, handles, labels, ncol: int = 2, fontsize: float = 7.0
) -> None:
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
        frameon=False,
        handlelength=1.0,
        columnspacing=1.0,
        fontsize=fontsize,
    )


def plot_architecture_diagram(out_path: Path):
    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Shared pipeline blocks
    shared_y = 0.42
    branch_top_y = 0.73
    branch_bottom_y = 0.18
    h = 0.16
    w = 0.13

    # Subtle lane backgrounds
    baseline_bg = FancyBboxPatch(
        (0.48, 0.60),
        0.46,
        0.22,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=0.0,
        facecolor="#F6F6F6",
        alpha=1.0,
    )
    hybrid_bg = FancyBboxPatch(
        (0.48, 0.08),
        0.46,
        0.28,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=0.0,
        facecolor="#F2F7FF",
        alpha=1.0,
    )
    ax.add_patch(baseline_bg)
    ax.add_patch(hybrid_bg)

    draw_box(ax, 0.05, shared_y, w, h, "Input\n4 × 84 × 84\nstacked frames", "#FAFAFA")
    draw_box(ax, 0.24, shared_y, w, h, "Shared CNN\nfront-end\n3 conv layers", "#F3F3F3")
    draw_box(ax, 0.43, shared_y, w, h, "Feature map\n64 × 7 × 7", "#F8F8F8")

    draw_arrow(ax, (0.05 + w, shared_y + h / 2), (0.24, shared_y + h / 2))
    draw_arrow(ax, (0.24 + w, shared_y + h / 2), (0.43, shared_y + h / 2))

    # Split point
    split_x = 0.43 + w
    split_y = shared_y + h / 2
    draw_line(ax, (split_x, split_y), (0.62, split_y))
    draw_line(ax, (0.62, split_y), (0.62, branch_top_y + h / 2))
    draw_line(ax, (0.62, split_y), (0.62, branch_bottom_y + h / 2))

    # Baseline branch
    draw_box(ax, 0.66, branch_top_y, w, h, "Flatten", "#FAFAFA")
    draw_box(ax, 0.84, branch_top_y, w, h, "Rainbow heads\n(value +\nadvantage)", "#FAFAFA")
    draw_arrow(ax, (0.62, branch_top_y + h / 2), (0.66, branch_top_y + h / 2))
    draw_arrow(ax, (0.66 + w, branch_top_y + h / 2), (0.84, branch_top_y + h / 2))

    # Hybrid branch
    draw_box(ax, 0.57, branch_bottom_y, w, h, "Spatial\ntokenization\n49 tokens", "#EAF2FF")
    draw_box(ax, 0.73, branch_bottom_y, w, h, "Transformer\n2 layers, 4 heads\nembed dim 64", "#DCE9FF")
    draw_box(ax, 0.89, branch_bottom_y, w, h, "CLS-token\npooling", "#EAF2FF")
    draw_arrow(ax, (0.62, branch_bottom_y + h / 2), (0.57, branch_bottom_y + h / 2))
    draw_arrow(ax, (0.57 + w, branch_bottom_y + h / 2), (0.73, branch_bottom_y + h / 2))
    draw_arrow(ax, (0.73 + w, branch_bottom_y + h / 2), (0.89, branch_bottom_y + h / 2))

    # Route back to the same heads family notion
    draw_line(ax, (0.89 + w, branch_bottom_y + h / 2), (0.98, branch_bottom_y + h / 2))
    draw_line(ax, (0.98, branch_bottom_y + h / 2), (0.98, 0.50))

    ax.text(0.71, 0.86, "Baseline Rainbow encoder path", ha="center", va="center", fontsize=8.5, color="#4F4F4F")
    ax.text(0.71, 0.38, "Proposed Hybrid CNN-Transformer encoder path", ha="center", va="center", fontsize=8.5, color="#2F5EA8")
    ax.text(0.305, 0.66, "Shared visual front-end", ha="center", va="center", fontsize=8, color="#555555")
    ax.text(0.80, 0.07, "Only the lower branch replaces the standard flattening stage.", ha="center", va="center", fontsize=7.5, color="#666666")

    save_figure(fig, out_path)


def apply_sparse_frame_ticks(ax: plt.Axes, step_values) -> None:
    unique_steps = sorted({int(v) for v in step_values})
    if not unique_steps:
        return
    max_step = unique_steps[-1]
    if max_step <= 10:
        ticks = list(range(1, max_step + 1, 2))
        if max_step not in ticks:
            ticks.append(max_step)
    elif max_step <= 30:
        ticks = list(range(5, max_step + 1, 5))
        if unique_steps[0] < ticks[0]:
            ticks = [unique_steps[0]] + ticks
        if max_step not in ticks:
            ticks.append(max_step)
    else:
        ticks = unique_steps[:: max(1, len(unique_steps) // 5)]
        if max_step not in ticks:
            ticks.append(max_step)
    ticks = sorted(set(ticks))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])


def plot_median_hns(main_df: pd.DataFrame, out_path: Path):
    if main_df.empty:
        return
    median_df = (
        main_df.groupby(["encoder", "step_m"], as_index=False)["hns"]
        .median()
        .sort_values(["encoder", "step_m"])
    )

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    for encoder in ["cnn", "hybrid"]:
        subset = median_df[median_df["encoder"] == encoder]
        plot_series(
            ax,
            subset["step_m"],
            subset["hns"],
            ENCODER_COLORS[encoder],
            ENCODER_LABELS[encoder],
        )

    setup_axis(ax)
    ax.set_xlabel("Millions of frames")
    ax.set_ylabel("Median human-normalized score")
    apply_sparse_frame_ticks(ax, median_df["step_m"].unique())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}%"))
    place_legend_above_axes(ax, ncol=1)
    save_figure(fig, out_path)


def plot_games_above_thresholds(main_df: pd.DataFrame, out_path: Path):
    if main_df.empty:
        return
    thresholds = [20, 50, 100]
    rows = []
    for threshold in thresholds:
        grouped = (
            main_df.assign(hit=main_df["hns"] >= threshold)
            .groupby(["encoder", "step_m"], as_index=False)["hit"]
            .sum()
        )
        grouped["threshold"] = threshold
        rows.append(grouped)
    plot_df = pd.concat(rows, ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8), sharey=True)
    legend_handles = None
    legend_labels = None
    for ax, threshold in zip(axes, thresholds):
        subset = plot_df[plot_df["threshold"] == threshold]
        for encoder in ["cnn", "hybrid"]:
            enc_df = subset[subset["encoder"] == encoder]
            plot_series(
                ax,
                enc_df["step_m"],
                enc_df["hit"],
                ENCODER_COLORS[encoder],
                ENCODER_LABELS[encoder],
            )
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        setup_axis(ax)
        ax.set_title(f"{threshold}% Human")
        ax.set_xlabel("Millions of frames")
        ax.set_ylabel("Games")
        apply_sparse_frame_ticks(ax, subset["step_m"].unique())
        ax.set_ylim(0, len(GAME_ORDER))
        ax.set_yticks(range(0, len(GAME_ORDER) + 1))
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    place_legend_above_figure(fig, legend_handles, legend_labels, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, out_path)


def plot_two_line_raw(df: pd.DataFrame, out_path: Path, title: str):
    if df.empty:
        return
    summary = (
        df.groupby(["encoder", "step_m"], as_index=False)["raw_score"]
        .mean()
        .sort_values(["encoder", "step_m"])
    )

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    for encoder in ["cnn", "hybrid"]:
        subset = summary[summary["encoder"] == encoder]
        plot_series(
            ax,
            subset["step_m"],
            subset["raw_score"],
            ENCODER_COLORS[encoder],
            ENCODER_LABELS[encoder],
        )

    setup_axis(ax)
    ax.set_xlabel("Millions of frames")
    ax.set_ylabel("Raw score")
    apply_sparse_frame_ticks(ax, summary["step_m"].unique())
    place_legend_above_axes(ax, ncol=1)
    save_figure(fig, out_path)


def plot_ablation_raw(df: pd.DataFrame, out_path: Path):
    if df.empty:
        return
    order = [
        "cnn_baseline",
        "hybrid_full",
        "hybrid_1layer",
        "hybrid_mean_pool",
    ]
    labels = {
        "cnn_baseline": "CNN baseline",
        "hybrid_full": "Hybrid (full)",
        "hybrid_1layer": "Hybrid (1-layer)",
        "hybrid_mean_pool": "Hybrid (mean-pool)",
    }

    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    for variant in order:
        subset = df[df["variant"] == variant]
        plot_series(
            ax,
            subset["step_m"],
            subset["raw_score"],
            ABLATION_COLORS[variant],
            labels[variant],
        )

    setup_axis(ax)
    ax.set_xlabel("Millions of frames")
    ax.set_ylabel("Raw score")
    apply_sparse_frame_ticks(ax, df["step_m"].unique())
    place_legend_above_axes(ax, ncol=2)
    save_figure(fig, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "paper" / "Latex" / "figures"
    )
    args = parser.parse_args()

    style()

    plot_architecture_diagram(args.output_dir / "figure_1_architecture.png")

    main_df = gather_main_benchmark(args.base_dir)
    qbert_df = gather_multiseed_timeseries(args.base_dir, "qbert")
    breakout_df = gather_multiseed_timeseries(args.base_dir, "breakout")
    beam_df = gather_long_horizon(args.base_dir, "beam_rider")
    seaquest_df = gather_long_horizon(args.base_dir, "seaquest")
    pong_df = gather_main_game_timeseries(args.base_dir, "pong")
    ablation_df = gather_qbert_ablation(args.base_dir)

    plot_median_hns(main_df, args.output_dir / "figure_2_median_hns_10m.png")
    plot_games_above_thresholds(
        main_df, args.output_dir / "figure_3_games_above_human_10m.png"
    )
    plot_two_line_raw(
        qbert_df,
        args.output_dir / "figure_4_qbert_raw_multiseed.png",
        "Qbert Raw Score vs Frames",
    )
    plot_two_line_raw(
        beam_df,
        args.output_dir / "figure_5_beam_rider_30m_raw.png",
        "Beam Rider Raw Score vs Frames (30M)",
    )
    plot_two_line_raw(
        breakout_df,
        args.output_dir / "appendix_a1_breakout_raw_multiseed.png",
        "Breakout Raw Score vs Frames",
    )
    plot_two_line_raw(
        seaquest_df,
        args.output_dir / "appendix_a2_seaquest_30m_raw.png",
        "Seaquest Raw Score vs Frames (30M)",
    )
    plot_two_line_raw(
        pong_df,
        args.output_dir / "figure_6_pong_raw_10m.png",
        "Pong Raw Score vs Frames",
    )
    plot_ablation_raw(
        ablation_df, args.output_dir / "figure_7_qbert_ablation_raw.png"
    )

    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
