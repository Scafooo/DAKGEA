#!/usr/bin/env python3
"""Advanced visualization utilities - heatmaps, boxplots, violin plots, scatter plots."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import numpy as np
import shutil
import subprocess
import tempfile

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# Red → White → Green diverging colormap (no yellow at center)
_RdWGn = mcolors.LinearSegmentedColormap.from_list(
    "RdWGn",
    [(0.0, "#d73027"), (0.5, "#fffde0"), (1.0, "#1a9850")],
)


def save_figure(fig, path: Path, dpi: int = 200, formats: list[str] = None) -> list[Path]:
    """Save figure in multiple formats.

    Args:
        fig: Matplotlib figure object
        path: Base output path (without extension)
        dpi: Resolution for raster formats
        formats: List of formats to save ('png', 'pdf', 'svg'). Default: ['png']

    Returns:
        List of paths where figure was saved
    """
    if formats is None:
        formats = ['png']

    saved_paths = []
    for fmt in formats:
        output_path = path.with_suffix(f".{fmt}")
        fig.savefig(output_path, dpi=dpi if fmt == 'png' else None,
                   bbox_inches='tight', format=fmt)
        saved_paths.append(output_path)

    return saved_paths


def plot_heatmap(
    dataset: str,
    data: Dict[float, Dict[float, float]],
    metric: str,
    plots_dir: Path,
    dpi: int = 200,
) -> None:
    """Create heatmap showing metric values across reduction/augmentation ratios."""
    if not data:
        return

    reduction_ratios = sorted(data.keys())
    augmentation_ratios_set = set()
    for aug_dict in data.values():
        augmentation_ratios_set.update(aug_dict.keys())
    augmentation_ratios = sorted(augmentation_ratios_set)

    if not reduction_ratios or not augmentation_ratios:
        return

    # Create matrix
    matrix = np.zeros((len(reduction_ratios), len(augmentation_ratios)))
    for i, red_ratio in enumerate(reduction_ratios):
        for j, aug_ratio in enumerate(augmentation_ratios):
            value = data.get(red_ratio, {}).get(aug_ratio)
            matrix[i, j] = value if value is not None else np.nan

    ensure_dir(plots_dir / "heatmaps")

    fig, ax = plt.subplots(figsize=(max(8, len(augmentation_ratios)), max(6, len(reduction_ratios))))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", interpolation="nearest")

    # Set ticks
    ax.set_xticks(np.arange(len(augmentation_ratios)))
    ax.set_yticks(np.arange(len(reduction_ratios)))
    ax.set_xticklabels([f"{r:.2f}" for r in augmentation_ratios])
    ax.set_yticklabels([f"{r:.2f}" for r in reduction_ratios])

    # Labels
    ax.set_xlabel("Augmentation Ratio", fontsize=11)
    ax.set_ylabel("Reduction Ratio", fontsize=11)
    ax.set_title(f"{dataset} - {metric} Heatmap", fontsize=13, fontweight="bold")

    # Annotate cells
    for i in range(len(reduction_ratios)):
        for j in range(len(augmentation_ratios)):
            value = matrix[i, j]
            if not np.isnan(value):
                text_color = "white" if value < 0.5 else "black"
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=8)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric, rotation=270, labelpad=15)

    plt.tight_layout()
    outfile = plots_dir / "heatmaps" / f"{dataset}_{metric.replace('@', 'at')}_heatmap.png"
    plt.savefig(outfile, dpi=dpi)
    plt.close(fig)


def _build_heatmap_matrices(
    aug_data: Dict[float, Dict[float, float]],
    red_data: Dict[float, Dict[float, float]],
    global_rr=None,
    global_ar=None,
):
    """Return (reduction_ratios, augmentation_ratios, red_matrix, delta_matrix) in %."""
    reduction_ratios = global_rr if global_rr is not None else sorted(aug_data.keys())
    augmentation_ratios = global_ar if global_ar is not None else sorted({a for d in aug_data.values() for a in d})
    n_rows, n_cols = len(reduction_ratios), len(augmentation_ratios)
    red_matrix   = np.full((n_rows, n_cols), np.nan)
    delta_matrix = np.full((n_rows, n_cols), np.nan)
    for i, r in enumerate(reduction_ratios):
        for j, a in enumerate(augmentation_ratios):
            av = aug_data.get(r, {}).get(a)
            rv = red_data.get(r, {}).get(a)
            if rv is not None:
                red_matrix[i, j] = rv * 100.0
            if av is not None and rv is not None:
                delta_matrix[i, j] = (av - rv) * 100.0
    return reduction_ratios, augmentation_ratios, red_matrix, delta_matrix


def _annotate_delta_heatmap_ax(
    ax,
    red_matrix: np.ndarray,
    delta_matrix: np.ndarray,
    reduction_ratios,
    augmentation_ratios,
    norm,
    cmap,
    dataset: str,
    metric: str,
    font_family: str = "serif",
    font_name: str = "Times New Roman",
) -> object:
    """Draw one delta-heatmap on *ax* and return the AxesImage for colorbar use."""
    n_rows, n_cols = len(reduction_ratios), len(augmentation_ratios)
    display_matrix = delta_matrix
    display_red    = red_matrix

    im = ax.imshow(display_matrix, cmap=cmap, norm=norm,
                   aspect="equal", interpolation="nearest")

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels([f"{a:.1f}" for a in augmentation_ratios],
                       fontsize=7, fontfamily=font_family)
    ax.set_yticklabels([f"{r:.1f}" for r in reduction_ratios],
                       fontsize=7, fontfamily=font_family)
    ax.set_xlabel("Augmentation Factor", fontsize=8, labelpad=4, fontfamily=font_family)
    ax.set_ylabel("Reduction Factor", fontsize=8, labelpad=4, fontfamily=font_family)
    ax.set_title(dataset, fontsize=9, fontweight="bold", pad=5, fontfamily=font_family)

    ax.set_xticks(np.arange(n_cols) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows) - 0.5, minor=True)
    ax.grid(which="minor", color="#cccccc", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate: baseline on top (always bold), delta below (colored by sign)
    for i in range(n_rows):
        for j in range(n_cols):
            rv = display_red[i, j]
            dv = display_matrix[i, j]
            if np.isnan(rv):
                continue
            base_str  = f"{rv:.2f}"
            delta_str = f"{dv:+.2f}" if not np.isnan(dv) else ""
            dv_pos    = (not np.isnan(dv)) and dv > 0
            dv_neg    = (not np.isnan(dv)) and dv < 0

            # Determine text color for legibility against background
            # Use white text on very dark cells, dark otherwise
            bg_val = dv if not np.isnan(dv) else 0.0
            norm_val = norm(bg_val) if not np.isnan(bg_val) else 0.5
            text_color = "white" if (norm_val < 0.15 or norm_val > 0.85) else "#1a1a1a"

            ax.text(j, i - 0.27, base_str,
                    ha="center", va="center", color=text_color,
                    fontsize=14, fontweight="normal", fontfamily=font_family)
            if delta_str:
                ax.text(j, i + 0.27, delta_str,
                        ha="center", va="center", color=text_color,
                        fontsize=14, fontweight="normal", fontfamily=font_family,
                        style="italic")
    return im


def plot_delta_heatmap(
    dataset: str,
    aug_data: Dict[float, Dict[float, float]],
    red_data: Dict[float, Dict[float, float]],
    metric: str,
    plots_dir: Path,
    global_vmax: Optional[float] = None,
    global_vmin: Optional[float] = None,
    dpi: int = 300,
) -> None:
    """Single-dataset delta heatmap (baseline on top, delta below per cell).

    Coloured with RdYlGn diverging scale shared across datasets via
    global_vmin/global_vmax. Values displayed in percentage points.
    """
    if not aug_data or not red_data:
        return
    reduction_ratios, augmentation_ratios, red_matrix, delta_matrix = \
        _build_heatmap_matrices(aug_data, red_data)
    if not reduction_ratios or not augmentation_ratios:
        return

    ensure_dir(plots_dir / "delta_heatmaps")

    cmap = _RdWGn
    abs_max = global_vmax if global_vmax is not None else float(
        np.nanmax(np.abs(delta_matrix)) if not np.all(np.isnan(delta_matrix)) else 1.0
    )
    vmin = global_vmin if global_vmin is not None else -abs_max
    vmax = abs_max
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=max(vmax, 1e-9))

    cell_size = 1.1
    n_rows, n_cols = len(reduction_ratios), len(augmentation_ratios)
    fig, ax = plt.subplots(figsize=(n_cols * cell_size + 2.0, n_rows * cell_size + 1.5))

    im = _annotate_delta_heatmap_ax(
        ax, red_matrix, delta_matrix,
        reduction_ratios, augmentation_ratios,
        norm, cmap, dataset, metric,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(rf"$\Delta$ {metric} (pp)", rotation=270, labelpad=14,
                   fontsize=8, fontfamily="serif")
    cbar.ax.tick_params(labelsize=7)

    stem = f"{dataset}_{metric.replace('@', 'at')}_delta_heatmap"
    fig.savefig(plots_dir / "delta_heatmaps" / f"{stem}.png",
                dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(plots_dir / "delta_heatmaps" / f"{stem}.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_delta_heatmap_combined(
    all_data: Dict[str, tuple],
    metric: str,
    plots_dir: Path,
    dpi: int = 300,
) -> None:
    """Combined figure with one subplot per dataset and a single shared colorbar.

    Args:
        all_data:  {dataset -> (aug_data, red_data)} for all datasets.
        metric:    Metric name (e.g. 'hits@1').
        plots_dir: Root output dir.
        dpi:       Output resolution.
    """
    if not all_data:
        return

    # Build all matrices and compute global symmetric scale
    built = {}
    abs_max = 0.0
    for ds, (aug_data, red_data) in all_data.items():
        rr, ar, rm, dm = _build_heatmap_matrices(aug_data, red_data)
        if rr and ar:
            built[ds] = (rr, ar, rm, dm)
            if not np.all(np.isnan(dm)):
                abs_max = max(abs_max, float(np.nanmax(np.abs(dm))))
    if not built:
        return
    abs_max = max(abs_max, 1e-9)

    cmap = _RdWGn
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

    # Use original key order (includes datasets with no data → placeholder)
    datasets = list(all_data.keys())
    n_ds = len(datasets)

    # Grid size from first available dataset
    first_built = next(iter(built.values()))
    rr0, ar0 = first_built[0], first_built[1]
    n_rows, n_cols = len(rr0), len(ar0)

    cell_size = 1.1
    fig_w = n_ds * (n_cols * cell_size + 0.5) + 1.8
    fig_h = n_rows * cell_size + 1.5

    fig, axes = plt.subplots(1, n_ds, figsize=(fig_w, fig_h),
                             sharey=True, constrained_layout=False)
    if n_ds == 1:
        axes = [axes]

    im_ref = None
    for ax, ds in zip(axes, datasets):
        if ds in built:
            rr, ar, rm, dm = built[ds]
            im_ref = _annotate_delta_heatmap_ax(
                ax, rm, dm, rr, ar, norm, cmap, ds, metric,
            )
        else:
            # Placeholder for missing dataset
            ax.set_facecolor("#f0f0f0")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(ds, fontsize=9, fontweight="bold", pad=5)
            ax.text(0.5, 0.5, "Data not available",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#888888", style="italic")
            for spine in ax.spines.values():
                spine.set_edgecolor("#cccccc")
        if ax != axes[0]:
            ax.set_ylabel("")

    # Single colorbar on the right (only if at least one real subplot was drawn)
    if im_ref is None:
        plt.close(fig)
        return
    cbar = fig.colorbar(im_ref, ax=axes, fraction=0.015, pad=0.02, shrink=0.85)
    cbar.set_label(rf"$\Delta$ {metric} (pp)", rotation=270, labelpad=14,
                   fontsize=9, fontfamily="serif")
    cbar.ax.tick_params(labelsize=7)

    ensure_dir(plots_dir / "delta_heatmaps")
    stem = f"combined_{metric.replace('@', 'at')}_delta_heatmap"
    fig.savefig(plots_dir / "delta_heatmaps" / f"{stem}.png",
                dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(plots_dir / "delta_heatmaps" / f"{stem}.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── LaTeX / TikZ delta heatmap ───────────────────────────────────────────────

def _esc(s: str) -> str:
    """Escape LaTeX special characters."""
    for old, new in [("_", r"\_"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#")]:
        s = s.replace(old, new)
    return s


def _rgb_tikz(r: float, g: float, b: float) -> str:
    return f"{{rgb,1:red,{r:.4f};green,{g:.4f};blue,{b:.4f}}}"


def _text_color_bw(r: float, g: float, b: float) -> str:
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.45 else "black"


def _tikz_heatmap_body(
    red_matrix: np.ndarray,
    delta_matrix: np.ndarray,
    reduction_ratios,
    augmentation_ratios,
    norm,
    cmap,
    dataset: str,
    cw: float,
    ch: float,
    x_off: float,
    show_ylabel: bool,
    fs_cell: str,
    fs_tick: str,
    fs_label: str,
    fs_title: str,
    y_off: float = 0.0,
    show_xlabel: bool = True,
    show_xlabel_top: bool = False,
    show_yticks: bool = True,
    show_corner_label: bool = False,
    show_title: bool = True,
    fs_cell_top: str = None,
) -> tuple:
    """Return (tikz_lines, grid_width, grid_height)."""
    n_rows, n_cols = len(reduction_ratios), len(augmentation_ratios)
    gw, gh = n_cols * cw, n_rows * ch
    lines = []

    for i in range(n_rows):
        for j in range(n_cols):
            rv = red_matrix[i, j]
            dv = delta_matrix[i, j]
            x0 = x_off + j * cw
            y0 = y_off - i * ch

            if not np.isnan(dv):
                rgba = cmap(norm(float(dv)))
                # Blend with white to soften colors (0 = white, 1 = full color)
                sat = 0.75
                rc = rgba[0] * sat + (1 - sat)
                gc = rgba[1] * sat + (1 - sat)
                bc = rgba[2] * sat + (1 - sat)
                fill = _rgb_tikz(rc, gc, bc)
                tc = _text_color_bw(rc, gc, bc)
            else:
                fill = "{gray!15}"
                tc = "black"

            lines.append(
                f"  \\fill[fill={fill}] ({x0:.3f}cm,{y0:.3f}cm)"
                f" rectangle ++({cw:.3f}cm,{-ch:.3f}cm);"
            )
            cx = x0 + cw / 2
            if not np.isnan(rv):
                ft = fs_cell_top if fs_cell_top is not None else fs_cell
                lines.append(
                    f"  \\node[{tc},font={ft},anchor=center]"
                    f" at ({cx:.3f}cm,{y0 - ch*0.29:.3f}cm) {{{rv:.1f}}};"
                )
            if not np.isnan(dv):
                val = float(dv)
                sign = r"\scalebox{0.65}{+}" if val >= 0 else ""
                num  = f"{abs(val):.1f}" if val >= 0 else f"{val:.1f}"
                content = sign + num
                # 3-digit integer part (≥10): shrink whole string slightly to prevent overflow
                dstr = (r"\scalebox{0.86}{" + content + r"}") if abs(val) >= 10 else content
                lines.append(
                    f"  \\node[{tc},font={fs_cell}\\bfseries\\itshape,anchor=center]"
                    f" at ({cx:.3f}cm,{y0 - ch*0.71:.3f}cm) {{{dstr}}};"
                )

    # Minor grid lines
    for ii in range(1, n_rows):
        y = y_off - ii * ch
        lines.append(
            f"  \\draw[black!20,line width=0.2pt]"
            f" ({x_off:.3f}cm,{y:.3f}cm) -- ({x_off+gw:.3f}cm,{y:.3f}cm);"
        )
    for jj in range(1, n_cols):
        x = x_off + jj * cw
        lines.append(
            f"  \\draw[black!20,line width=0.2pt]"
            f" ({x:.3f}cm,{y_off:.3f}cm) -- ({x:.3f}cm,{y_off-gh:.3f}cm);"
        )
    # Border
    lines.append(
        f"  \\draw[black!60,line width=0.5pt]"
        f" ({x_off:.3f}cm,{y_off:.3f}cm) rectangle ({x_off+gw:.3f}cm,{y_off-gh:.3f}cm);"
    )

    # X-axis ticks at bottom
    if show_xlabel:
        for j, a in enumerate(augmentation_ratios):
            x = x_off + j * cw + cw / 2
            lines.append(
                f"  \\node[anchor=north,font={fs_tick}]"
                f" at ({x:.3f}cm,{y_off-gh:.3f}cm) {{{int(round(a * 100))}}};"
            )

    # X-axis ticks at top (ticks only, no text label)
    if show_xlabel_top:
        for j, a in enumerate(augmentation_ratios):
            x = x_off + j * cw + cw / 2
            lines.append(
                f"  \\node[anchor=south,font={fs_tick}]"
                f" at ({x:.3f}cm,{y_off:.3f}cm) {{{int(round(a * 100))}}};"
            )

    # Y-axis numeric ticks
    if show_yticks:
        for i, r in enumerate(reduction_ratios):
            y = y_off - i * ch - ch / 2
            lines.append(
                f"  \\node[anchor=east,font={fs_tick}]"
                f" at ({x_off - 0.1:.3f}cm,{y:.3f}cm) {{{int(round(r * 100))}}};"
            )

    # Y-axis text label (once, centered on this block)
    if show_ylabel:
        lines.append(
            f"  \\node[anchor=south,rotate=90,font={fs_label}]"
            f" at ({x_off - 0.65:.3f}cm,{y_off-gh/2:.3f}cm) {{Reduction (\\%)}};"
        )

    # Corner split: diagonal line, "r%" lower-left (row axis), "a%" upper-right (col axis)
    if show_corner_label:
        cx0, cy0 = x_off - 0.46, y_off + 0.18
        cx1, cy1 = x_off,         y_off
        lines += [
            f"  \\draw[black!50,line width=0.3pt]"
            f" ({cx0:.3f}cm,{cy0:.3f}cm) -- ({cx1:.3f}cm,{cy1:.3f}cm);",
            f"  \\node[anchor=north,font={fs_tick}\\itshape]"
            f" at ({x_off-0.35:.3f}cm,{y_off+0.14:.3f}cm) {{r\\%}};",
            f"  \\node[anchor=south,font={fs_tick}\\itshape]"
            f" at ({x_off-0.13:.3f}cm,{y_off+0.04:.3f}cm) {{a\\%}};",
        ]

    # Title (above each block)
    if show_title:
        lines.append(
            f"  \\node[anchor=south,font={fs_title}]"
            f" at ({x_off + gw/2:.3f}cm,{y_off+0.45:.3f}cm) {{\\textbf{{{_esc(dataset)}}}}};"
        )

    return lines, gw, gh


def _tikz_colorbar_horizontal(
    x_pos: float,
    y_pos: float,
    width: float,
    vmin: float,
    vmax: float,
    norm,
    cmap,
    metric: str,
    height: float = 0.35,
    n_steps: int = 256,
    fs_tick: str = r"\scriptsize",
    fs_label: str = r"\footnotesize",
) -> list:
    """Draw a horizontal colorbar as stacked colored rectangles."""
    lines = []
    step_w = width / n_steps
    sat = 0.75
    for k in range(n_steps):
        v = vmin + (vmax - vmin) * k / n_steps
        rgba = cmap(norm(float(v)))
        rc = rgba[0] * sat + (1 - sat)
        gc = rgba[1] * sat + (1 - sat)
        bc = rgba[2] * sat + (1 - sat)
        fill = _rgb_tikz(rc, gc, bc)
        x = x_pos + k * step_w
        lines.append(
            f"  \\fill[fill={fill}] ({x:.3f}cm,{y_pos:.3f}cm)"
            f" rectangle ++({step_w + 0.01:.3f}cm,{-height:.3f}cm);"
        )
    lines.append(
        f"  \\draw[black!60,line width=0.4pt]"
        f" ({x_pos:.3f}cm,{y_pos:.3f}cm)"
        f" rectangle ({x_pos+width:.3f}cm,{y_pos-height:.3f}cm);"
    )
    span = vmax - vmin if (vmax - vmin) > 1e-9 else 1.0
    for tv in [vmin, vmin / 2, 0.0, vmax / 2, vmax]:
        if vmin <= tv <= vmax:
            frac = (tv - vmin) / span
            x = x_pos + frac * width
            lines.append(
                f"  \\draw[black!70] ({x:.3f}cm,{y_pos-height:.3f}cm) -- ++(0,{-0.10:.2f}cm);"
            )
            lines.append(
                f"  \\node[anchor=north,font={fs_tick}]"
                f" at ({x:.3f}cm,{y_pos-height-0.13:.3f}cm) {{{tv:.1f}}};"
            )
    lines.append(
        f"  \\node[anchor=north,font={fs_label}]"
        f" at ({x_pos+width/2:.3f}cm,{y_pos-height-0.45:.3f}cm)"
        f" {{$\\Delta$ {_esc(metric)} (pp)}};"
    )
    return lines


def _tikz_colorbar(
    x_pos: float,
    gh: float,
    vmin: float,
    vmax: float,
    norm,
    cmap,
    metric: str,
    width: float = 0.4,
    n_steps: int = 256,
    fs_tick: str = r"\footnotesize",
    fs_label: str = r"\small",
) -> list:
    """Draw a vertical colorbar at x_pos as stacked colored rectangles."""
    lines = []
    step_h = gh / n_steps
    sat = 0.75
    for k in range(n_steps):
        v = vmin + (vmax - vmin) * k / n_steps
        rgba = cmap(norm(float(v)))
        rc = rgba[0] * sat + (1 - sat)
        gc = rgba[1] * sat + (1 - sat)
        bc = rgba[2] * sat + (1 - sat)
        fill = _rgb_tikz(rc, gc, bc)
        y = -gh + k * step_h
        lines.append(
            f"  \\fill[fill={fill}] ({x_pos:.3f}cm,{y:.3f}cm)"
            f" rectangle ++({width:.3f}cm,{step_h + 0.02:.3f}cm);"
        )
    lines.append(
        f"  \\draw[black!60,line width=0.4pt]"
        f" ({x_pos:.3f}cm,{-gh:.3f}cm) rectangle ({x_pos+width:.3f}cm,0cm);"
    )
    span = vmax - vmin if (vmax - vmin) > 1e-9 else 1.0
    for tv in [vmin, vmin / 2, 0.0, vmax / 2, vmax]:
        if vmin <= tv <= vmax:
            frac = (tv - vmin) / span
            y = -gh + frac * gh
            lines.append(
                f"  \\draw[black!70] ({x_pos+width:.3f}cm,{y:.3f}cm) -- ++({0.12:.2f}cm,0);"
            )
            lines.append(
                f"  \\node[anchor=west,font={fs_tick}]"
                f" at ({x_pos+width+0.17:.3f}cm,{y:.3f}cm) {{{tv:.1f}}};"
            )
    lx = x_pos + width + 1.1
    ly = -gh / 2
    lines.append(
        f"  \\node[rotate=270,anchor=center,font={fs_label}]"
        f" at ({lx:.3f}cm,{ly:.3f}cm)"
        f" {{$\\Delta$ {_esc(metric)} (pp)}};"
    )
    return lines


def _latex_doc(body_lines: list) -> str:
    return "\n".join([
        r"\documentclass[border=6pt]{standalone}",
        r"\usepackage{tikz}",
        r"\usepackage{graphicx}",
        r"\usepackage{amsmath}",
        r"\begin{document}",
        r"\begin{tikzpicture}",
    ] + body_lines + [
        r"\end{tikzpicture}",
        r"\end{document}",
    ])


def _compile_latex_to_pdf(tex_src: str, pdf_path: Path) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        tex_file = tmp_p / "fig.tex"
        tex_file.write_text(tex_src, encoding="utf-8")
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig.tex"],
            cwd=tmp_p, capture_output=True, text=True,
        )
        pdf_out = tmp_p / "fig.pdf"
        if pdf_out.exists():
            shutil.copy(pdf_out, pdf_path)
            return True
        return False


def _pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 300) -> bool:
    subprocess.run(
        ["pdftoppm", f"-r", str(dpi), "-png", "-singlefile",
         str(pdf_path), str(png_path.with_suffix(""))],
        capture_output=True,
    )
    candidate = png_path.with_name(png_path.stem + ".png")
    if not candidate.exists():
        candidate = png_path.with_name(png_path.stem + "-1.png")
    if candidate.exists() and candidate != png_path:
        candidate.rename(png_path)
    return png_path.exists()


def plot_delta_heatmap_latex(
    dataset: str,
    aug_data: Dict[float, Dict[float, float]],
    red_data: Dict[float, Dict[float, float]],
    metric: str,
    plots_dir: Path,
    global_vmax: Optional[float] = None,
    global_vmin: Optional[float] = None,
    dpi: int = 300,
    cell_size: float = 1.1,
) -> None:
    if not aug_data or not red_data:
        return
    rr, ar, rm, dm = _build_heatmap_matrices(aug_data, red_data)
    if not rr or not ar:
        return
    ensure_dir(plots_dir / "delta_heatmaps")

    cmap = _RdWGn
    abs_max = global_vmax if global_vmax is not None else float(
        np.nanmax(np.abs(dm)) if not np.all(np.isnan(dm)) else 1.0
    )
    vmin_val = -(global_vmin if global_vmin is not None else abs_max)
    vmax_val = abs_max
    norm = mcolors.TwoSlopeNorm(vmin=vmin_val, vcenter=0.0, vmax=max(vmax_val, 1e-9))

    cw = ch = cell_size
    body, gw, gh = _tikz_heatmap_body(
        rm, dm, rr, ar, norm, cmap, dataset, cw, ch, 0.0, True,
        r"\small", r"\footnotesize", r"\small", r"\normalsize",
    )
    body += _tikz_colorbar(gw + 0.5, gh, vmin_val, vmax_val, norm, cmap, metric)

    stem = f"{dataset}_{metric.replace('@', 'at')}_delta_heatmap"
    out_dir = plots_dir / "delta_heatmaps"
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    tex_src = _latex_doc(body)
    (out_dir / f"{stem}.tex").write_text(tex_src, encoding="utf-8")
    if _compile_latex_to_pdf(tex_src, pdf_path):
        _pdf_to_png(pdf_path, png_path, dpi)


def plot_delta_heatmap_latex_combined(
    all_data: Dict[str, tuple],
    metric: str,
    plots_dir: Path,
    dpi: int = 300,
    cell_size: float = 0.85,
    show_right_labels: bool = False,
) -> None:
    if not all_data:
        return

    # Global union of ratios so missing rows/cols appear as gray cells
    global_rr = sorted({r for aug, _ in all_data.values() for r in aug})
    global_ar = sorted({a for aug, _ in all_data.values() for d in aug.values() for a in d})

    built = {}
    abs_max = 0.0
    for ds, (aug_data, red_data) in all_data.items():
        rr, ar, rm, dm = _build_heatmap_matrices(aug_data, red_data, global_rr, global_ar)
        if rr and ar:
            built[ds] = (rr, ar, rm, dm)
            if not np.all(np.isnan(dm)):
                abs_max = max(abs_max, float(np.nanmax(np.abs(dm))))
    if not built:
        return
    abs_max = max(abs_max, 1e-9)

    cmap = _RdWGn
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

    cw = ch = cell_size
    gap = 0.6
    body_lines = []
    n_ds = len(all_data)
    # Pre-compute exact grid dimensions to avoid floating-point drift
    n_rows = len(global_rr)
    n_cols = len(global_ar)
    gw = n_cols * cw
    gh = n_rows * ch

    for idx, ds in enumerate(all_data.keys()):
        y_cur = -(idx * (gh + gap))   # exact, no accumulation
        if ds in built:
            rr, ar, rm, dm = built[ds]
            lines, gw, gh = _tikz_heatmap_body(
                rm, dm, rr, ar, norm, cmap, ds, cw, ch, 0.0, False,
                r"\small", r"\scriptsize", r"\footnotesize", r"\small",
                y_off=y_cur, show_xlabel=False,
                show_xlabel_top=True, show_yticks=True, show_corner_label=True,
                show_title=False, fs_cell_top=r"\normalsize",
            )
        else:
            # Placeholder block for missing dataset
            ref_rr = sorted({r for v in built.values() for r in v[0]}) or list(range(10))
            ref_ar = sorted({a for v in built.values() for a in v[1]}) or list(range(10))
            n_r, n_c = len(ref_rr), len(ref_ar)
            gw, gh = n_c * cw, n_r * ch
            lines = []
            for i in range(n_r):
                for j in range(n_c):
                    x0 = j * cw
                    y0 = y_cur - i * ch
                    lines.append(
                        f"  \\fill[gray!15] ({x0:.3f}cm,{y0:.3f}cm)"
                        f" rectangle ++({cw:.3f}cm,{-ch:.3f}cm);"
                    )
            # grid lines
            for ii in range(1, n_r):
                y = y_cur - ii * ch
                lines.append(
                    f"  \\draw[black!20,line width=0.2pt]"
                    f" (0cm,{y:.3f}cm) -- ({gw:.3f}cm,{y:.3f}cm);"
                )
            for jj in range(1, n_c):
                x = jj * cw
                lines.append(
                    f"  \\draw[black!20,line width=0.2pt]"
                    f" ({x:.3f}cm,{y_cur:.3f}cm) -- ({x:.3f}cm,{y_cur-gh:.3f}cm);"
                )
            lines += [
                f"  \\draw[black!60,line width=0.5pt]"
                f" (0cm,{y_cur:.3f}cm) rectangle ({gw:.3f}cm,{y_cur-gh:.3f}cm);",
                f"  \\node[gray!60,font=\\scriptsize\\itshape,anchor=center]"
                f" at ({gw/2:.3f}cm,{y_cur-gh/2:.3f}cm) {{n/a}};",
            ]
            # Augmentation ticks at top
            for j, a in enumerate(ref_ar):
                x = j * cw + cw / 2
                lines.append(
                    f"  \\node[anchor=south,font=\\scriptsize]"
                    f" at ({x:.3f}cm,{y_cur:.3f}cm) {{{int(round(a * 100))}}};"
                )
            # Reduction ticks on left
            for i, r in enumerate(ref_rr):
                y = y_cur - i * ch - ch / 2
                lines.append(
                    f"  \\node[anchor=east,font=\\scriptsize]"
                    f" at (-0.1cm,{y:.3f}cm) {{{int(round(r * 100))}}};"
                )
            # Corner split: diagonal line
            cx0, cy0 = -0.46, y_cur + 0.18
            cx1, cy1 = 0.0,   y_cur
            lines += [
                f"  \\draw[black!50,line width=0.3pt]"
                f" ({cx0:.3f}cm,{cy0:.3f}cm) -- ({cx1:.3f}cm,{cy1:.3f}cm);",
                f"  \\node[anchor=north,font=\\scriptsize\\itshape]"
                f" at (-0.35cm,{y_cur+0.14:.3f}cm) {{r\\%}};",
                f"  \\node[anchor=south,font=\\scriptsize\\itshape]"
                f" at (-0.13cm,{y_cur+0.04:.3f}cm) {{a\\%}};",
            ]
        # Right-side dataset label (or phantom for bounding-box consistency)
        y_mid = y_cur - gh / 2
        label_color = "black" if show_right_labels else "white"
        body_lines.extend(lines)
        body_lines.append(
            f"  \\node[{label_color},rotate=-90,anchor=center,font=\\small]"
            f" at ({gw+0.35:.3f}cm,{y_mid:.3f}cm) {{\\textbf{{{_esc(ds)}}}}};"
        )

    # Horizontal colorbar below all blocks
    total_height = n_ds * gh + (n_ds - 1) * gap
    body_lines += _tikz_colorbar_horizontal(
        0.0, -(total_height + 0.5), gw, -abs_max, abs_max, norm, cmap, metric,
        fs_tick=r"\scriptsize", fs_label=r"\footnotesize",
    )

    ensure_dir(plots_dir / "delta_heatmaps")
    stem = f"combined_{metric.replace('@', 'at')}_delta_heatmap"
    out_dir = plots_dir / "delta_heatmaps"
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    tex_src = _latex_doc(body_lines)
    tex_path = out_dir / f"{stem}.tex"
    tex_path.write_text(tex_src, encoding="utf-8")
    if _compile_latex_to_pdf(tex_src, pdf_path):
        _pdf_to_png(pdf_path, png_path, dpi)

    # LaTeX include snippet (copy-paste ready, relative paths)
    rel_tex = tex_path.name
    rel_pdf = pdf_path.name
    snippet = "\n".join([
        r"% -------------------------------------------------------",
        r"% Delta heatmap — copy-paste into your paper",
        r"% Place this file in the same directory as main.tex, or",
        r"% adjust the path accordingly.",
        r"% -------------------------------------------------------",
        r"\begin{figure*}[t]",
        r"  \centering",
        f"  \\resizebox{{\\linewidth}}{{!}}{{%",
        f"    \\input{{{rel_tex}}}%",
        r"  }",
        f"  \\caption{{$\\Delta$~\\textsc{{hits@1}} across all datasets.",
        r"    Each cell reports the baseline (top) and the augmentation",
        r"    gain $\Delta$ (bottom, italic). Colour encodes the gain:",
        r"    \textcolor{green!50!black}{green}~$>0$,",
        r"    \textcolor{red!70!black}{red}~$<0$.}",
        f"  \\label{{fig:delta_heatmap_{metric.replace('@', 'at')}}}",
        r"\end{figure*}",
        r"",
        r"% Alternative (simpler, no font reuse):",
        f"% \\includegraphics[width=\\linewidth]{{{rel_pdf}}}",
    ]) + "\n"
    (out_dir / f"{stem}.snippet.tex").write_text(snippet, encoding="utf-8")


def plot_boxplot(
    dataset: str,
    reduction_values: List[float],
    augmentation_values: List[float],
    metric: str,
    plots_dir: Path,
    stage_colors: Dict[str, Dict[str, str]] = None,
    dpi: int = 200,
) -> None:
    """Create boxplot comparing reduction vs augmentation."""
    if not reduction_values and not augmentation_values:
        return

    # Default colors if not provided
    if stage_colors is None:
        stage_colors = {
            "reduction": {"primary": "#264653"},
            "augmentation": {"primary": "#e76f51"},
        }

    ensure_dir(plots_dir / "boxplots")

    fig, ax = plt.subplots(figsize=(6, 5))

    data = []
    labels = []
    colors = []

    if reduction_values:
        data.append(reduction_values)
        labels.append("Reduction")
        colors.append(stage_colors.get("reduction", {}).get("primary", "#264653"))

    if augmentation_values:
        data.append(augmentation_values)
        labels.append("Augmentation")
        colors.append(stage_colors.get("augmentation", {}).get("primary", "#e76f51"))

    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=True,
                     showmeans=True, meanline=True)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f"{dataset} - {metric} Distribution", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    outfile = plots_dir / "boxplots" / f"{dataset}_{metric.replace('@', 'at')}_boxplot.png"
    plt.savefig(outfile, dpi=dpi)
    plt.close(fig)


def plot_violin(
    dataset: str,
    reduction_values: List[float],
    augmentation_values: List[float],
    metric: str,
    plots_dir: Path,
    stage_colors: Dict[str, Dict[str, str]] = None,
    dpi: int = 200,
) -> None:
    """Create violin plot comparing reduction vs augmentation."""
    if (not reduction_values or len(reduction_values) < 2) and (not augmentation_values or len(augmentation_values) < 2):
        return

    # Default colors if not provided
    if stage_colors is None:
        stage_colors = {
            "reduction": {"primary": "#264653"},
            "augmentation": {"primary": "#e76f51"},
        }

    ensure_dir(plots_dir / "violins")

    fig, ax = plt.subplots(figsize=(6, 5))

    data = []
    positions = []
    labels = []
    colors = []

    pos = 1
    if reduction_values and len(reduction_values) >= 2:
        data.append(reduction_values)
        positions.append(pos)
        labels.append("Reduction")
        colors.append(stage_colors.get("reduction", {}).get("primary", "#264653"))
        pos += 1

    if augmentation_values and len(augmentation_values) >= 2:
        data.append(augmentation_values)
        positions.append(pos)
        labels.append("Augmentation")
        colors.append(stage_colors.get("augmentation", {}).get("primary", "#e76f51"))

    if not data:
        return

    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)

    # Color the violins
    for i, pc in enumerate(parts["bodies"]):
        if i < len(colors):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f"{dataset} - {metric} Violin Plot", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    outfile = plots_dir / "violins" / f"{dataset}_{metric.replace('@', 'at')}_violin.png"
    plt.savefig(outfile, dpi=dpi)
    plt.close(fig)


def plot_scatter_correlation(
    dataset: str,
    reduction_values: List[float],
    augmentation_values: List[float],
    metric: str,
    plots_dir: Path,
    dpi: int = 200,
) -> None:
    """Create scatter plot showing correlation between reduction and augmentation."""
    if len(reduction_values) != len(augmentation_values) or len(reduction_values) < 2:
        return

    ensure_dir(plots_dir / "scatter")

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(reduction_values, augmentation_values, alpha=0.6, s=80, c="#457b9d", edgecolors="black", linewidths=0.5)

    # Add diagonal reference line (y=x)
    min_val = min(min(reduction_values), min(augmentation_values))
    max_val = max(max(reduction_values), max(augmentation_values))
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.3, label="y=x (no change)")

    # Calculate and display correlation
    if len(reduction_values) > 1:
        corr = np.corrcoef(reduction_values, augmentation_values)[0, 1]
        ax.text(0.05, 0.95, f"Correlation: {corr:.3f}", transform=ax.transAxes,
                fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel(f"Reduction {metric}", fontsize=11)
    ax.set_ylabel(f"Augmentation {metric}", fontsize=11)
    ax.set_title(f"{dataset} - {metric} Correlation", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    outfile = plots_dir / "scatter" / f"{dataset}_{metric.replace('@', 'at')}_scatter.png"
    plt.savefig(outfile, dpi=dpi)
    plt.close(fig)


def plot_delta_chart(
    dataset: str,
    reduction_values: List[float],
    augmentation_values: List[float],
    metric: str,
    plots_dir: Path,
    delta_colors: Dict[str, str] = None,
    dpi: int = 200,
) -> None:
    """Create chart showing delta (augmentation - reduction) for each experiment."""
    if len(reduction_values) != len(augmentation_values) or not reduction_values:
        return

    # Default delta colors if not provided
    if delta_colors is None:
        delta_colors = {
            "positive": "#2a9d8f",
            "negative": "#e63946",
            "neutral": "#6c757d",
        }

    ensure_dir(plots_dir / "deltas")

    deltas = [aug - red for red, aug in zip(reduction_values, augmentation_values)]
    indices = list(range(len(deltas)))

    fig, ax = plt.subplots(figsize=(max(8, len(deltas) * 0.5), 5))

    colors = [delta_colors.get("positive", "#2a9d8f") if d > 0 else delta_colors.get("negative", "#e63946") for d in deltas]
    ax.bar(indices, deltas, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)

    # Add zero line
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

    # Add mean line
    mean_delta = np.mean(deltas)
    ax.axhline(y=mean_delta, color="blue", linestyle="--", linewidth=1, label=f"Mean Δ: {mean_delta:.4f}")

    ax.set_xlabel("Experiment Index", fontsize=11)
    ax.set_ylabel(f"Δ {metric} (Augmentation - Reduction)", fontsize=11)
    ax.set_title(f"{dataset} - {metric} Performance Delta", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    outfile = plots_dir / "deltas" / f"{dataset}_{metric.replace('@', 'at')}_delta.png"
    plt.savefig(outfile, dpi=dpi)
    plt.close(fig)


def plot_radar_chart(
    dataset: str,
    reduction_metrics: Dict[str, float],
    augmentation_metrics: Dict[str, float],
    metrics: List[str],
    plots_dir: Path,
    stage_colors: Dict[str, Dict[str, str]],
    dpi: int = 200,
) -> None:
    """Create radar/spider chart comparing reduction vs augmentation metrics.

    Args:
        dataset: Dataset name
        reduction_metrics: Dictionary of metric -> value for reduction
        augmentation_metrics: Dictionary of metric -> value for augmentation
        metrics: List of metrics to plot
        plots_dir: Output directory
        stage_colors: Color configuration for reduction/augmentation
        dpi: Resolution
    """
    # Filter to metrics that have values
    valid_metrics = [m for m in metrics if m in reduction_metrics or m in augmentation_metrics]
    if not valid_metrics:
        return

    ensure_dir(plots_dir / "radar_charts")

    num_vars = len(valid_metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Prepare data
    red_values = [reduction_metrics.get(m, 0) for m in valid_metrics]
    aug_values = [augmentation_metrics.get(m, 0) for m in valid_metrics]
    red_values += red_values[:1]  # Complete the circle
    aug_values += aug_values[:1]

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='polar'))

    # Plot reduction
    red_color = stage_colors.get("reduction", {}).get("primary", "#264653")
    aug_color = stage_colors.get("augmentation", {}).get("primary", "#e76f51")

    ax.plot(angles, red_values, 'o-', linewidth=2, label='Reduction',
            color=red_color, alpha=0.7)
    ax.fill(angles, red_values, alpha=0.15, color=red_color)

    # Plot augmentation
    ax.plot(angles, aug_values, 'o-', linewidth=2, label='Augmentation',
            color=aug_color, alpha=0.7)
    ax.fill(angles, aug_values, alpha=0.15, color=aug_color)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(valid_metrics)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{dataset} - Multi-Metric Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()
    outfile = plots_dir / "radar_charts" / f"{dataset}_radar.png"
    plt.savefig(outfile, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_ridge(
    dataset: str,
    reduction_values: List[float],
    augmentation_values: List[float],
    metric: str,
    plots_dir: Path,
    stage_colors: Dict[str, Dict[str, str]],
    dpi: int = 200,
) -> None:
    """Create ridge plot showing distribution comparison.

    Args:
        dataset: Dataset name
        reduction_values: Values for reduction
        augmentation_values: Values for augmentation
        metric: Metric name
        plots_dir: Output directory
        stage_colors: Color configuration
        dpi: Resolution
    """
    if (not reduction_values or len(reduction_values) < 2) and \
       (not augmentation_values or len(augmentation_values) < 2):
        return

    ensure_dir(plots_dir / "ridge_plots")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    red_color = stage_colors.get("reduction", {}).get("primary", "#264653")
    aug_color = stage_colors.get("augmentation", {}).get("primary", "#e76f51")

    # Reduction distribution
    if reduction_values and len(reduction_values) >= 2:
        axes[0].fill_between(
            np.linspace(min(reduction_values), max(reduction_values), 100),
            0,
            np.histogram(reduction_values, bins=30, density=True)[0].max(),
            alpha=0.6,
            color=red_color,
            label='Reduction'
        )
        axes[0].hist(reduction_values, bins=30, density=True, alpha=0.7,
                     color=red_color, edgecolor='black', linewidth=0.5)
        axes[0].set_ylabel("Density", fontsize=10)
        axes[0].legend(loc='upper right')
        axes[0].grid(axis='y', linestyle='--', alpha=0.3)

    # Augmentation distribution
    if augmentation_values and len(augmentation_values) >= 2:
        axes[1].fill_between(
            np.linspace(min(augmentation_values), max(augmentation_values), 100),
            0,
            np.histogram(augmentation_values, bins=30, density=True)[0].max(),
            alpha=0.6,
            color=aug_color,
            label='Augmentation'
        )
        axes[1].hist(augmentation_values, bins=30, density=True, alpha=0.7,
                     color=aug_color, edgecolor='black', linewidth=0.5)
        axes[1].set_xlabel(metric, fontsize=11)
        axes[1].set_ylabel("Density", fontsize=10)
        axes[1].legend(loc='upper right')
        axes[1].grid(axis='y', linestyle='--', alpha=0.3)

    fig.suptitle(f"{dataset} - {metric} Distribution Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    outfile = plots_dir / "ridge_plots" / f"{dataset}_{metric.replace('@', 'at')}_ridge.png"
    plt.savefig(outfile, dpi=dpi)
    plt.close(fig)


def plot_performance_matrix(
    stage_data: Dict[str, Dict[str, float]],
    metric: str,
    stage: str,
    plots_dir: Path,
    stage_colors: Dict[str, Dict[str, str]],
    dpi: int = 200,
) -> None:
    """Create performance matrix heatmap (dataset × model).

    Args:
        stage_data: Dict of dataset -> dict of model -> metric value
        metric: Metric name
        stage: Stage name ("reduction" or "augmentation")
        plots_dir: Output directory
        stage_colors: Color configuration
        dpi: Resolution
    """
    if not stage_data:
        return

    ensure_dir(plots_dir / "performance_matrices")

    # Prepare data
    datasets = sorted(stage_data.keys())
    all_models = set()
    for models_dict in stage_data.values():
        all_models.update(models_dict.keys())
    models = sorted(all_models)

    if not datasets or not models:
        return

    # Create matrix
    matrix = np.zeros((len(datasets), len(models)))
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            value = stage_data.get(dataset, {}).get(model)
            matrix[i, j] = value if value is not None else np.nan

    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 0.8), max(8, len(datasets) * 0.5)))

    # Use appropriate colormap based on stage
    stage_color = stage_colors.get(stage, {}).get("primary", "#264653")
    cmap = "YlOrRd" if stage == "augmentation" else "YlGnBu"

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", interpolation="nearest")

    # Set ticks
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticklabels(datasets)

    # Labels
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Dataset", fontsize=12)
    ax.set_title(f"{stage.capitalize()} - {metric} Performance Matrix",
                 fontsize=14, fontweight="bold")

    # Annotate cells
    for i in range(len(datasets)):
        for j in range(len(models)):
            value = matrix[i, j]
            if not np.isnan(value):
                text_color = "white" if value < 0.5 else "black"
                ax.text(j, i, f"{value:.3f}",
                       ha="center", va="center", color=text_color, fontsize=8)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric, rotation=270, labelpad=20)

    plt.tight_layout()
    outfile = plots_dir / "performance_matrices" / f"{stage}_{metric.replace('@', 'at')}_matrix.png"
    plt.savefig(outfile, dpi=dpi)
    plt.close(fig)


def plot_surface_3d(
    dataset: str,
    ratio_plot_data: Dict[float, Dict[float, Dict[str, Dict[str, float]]]],
    metric: str,
    stage: str,
    plots_dir: Path,
    stage_colors: Dict[str, Dict[str, str]],
    dpi: int = 200,
) -> None:
    """Create 3D surface plot showing metric variation across reduction and augmentation ratios.

    Args:
        dataset: Dataset name
        ratio_plot_data: Dict[red_ratio -> aug_ratio -> stage -> metric -> value]
        metric: Metric to plot
        stage: Stage name ("reduction" or "augmentation")
        plots_dir: Output directory
        stage_colors: Color configuration
        dpi: Resolution
    """
    if not ratio_plot_data:
        return

    ensure_dir(plots_dir / "surface_3d")

    # Extract data for 3D surface
    reduction_ratios = sorted(ratio_plot_data.keys())
    augmentation_ratios = set()
    for aug_dict in ratio_plot_data.values():
        augmentation_ratios.update(aug_dict.keys())
    augmentation_ratios = sorted(augmentation_ratios)

    if not reduction_ratios or not augmentation_ratios:
        return

    # Need at least 2x2 grid for meaningful 3D surface plot
    if len(reduction_ratios) < 2 or len(augmentation_ratios) < 2:
        return

    # Create meshgrid
    X, Y = np.meshgrid(reduction_ratios, augmentation_ratios)
    Z = np.zeros_like(X)

    # Fill Z values
    for i, aug_ratio in enumerate(augmentation_ratios):
        for j, red_ratio in enumerate(reduction_ratios):
            value = ratio_plot_data.get(red_ratio, {}).get(aug_ratio, {}).get(stage, {}).get(metric)
            Z[i, j] = value if value is not None else np.nan

    # Create 3D plot
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Color based on stage
    color = stage_colors.get(stage, {}).get("primary", "#264653")

    # Create surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8,
                           linewidth=0, antialiased=True,
                           edgecolor='none')

    # Add contour lines on the bottom (only if we have enough data points)
    if Z.shape[0] >= 2 and Z.shape[1] >= 2:
        ax.contour(X, Y, Z, zdir='z', offset=np.nanmin(Z) - 0.05,
                   cmap='viridis', alpha=0.4, linewidths=1)

    # Labels and title
    ax.set_xlabel('Reduction Ratio', fontsize=11, labelpad=10)
    ax.set_ylabel('Augmentation Ratio', fontsize=11, labelpad=10)
    ax.set_zlabel(metric, fontsize=11, labelpad=10)
    ax.set_title(f'{dataset} - {metric} ({stage.capitalize()})',
                 fontsize=14, fontweight='bold', pad=20)

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label(metric, rotation=270, labelpad=20)

    # Adjust viewing angle for better visualization
    ax.view_init(elev=25, azim=45)

    # Grid
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    outfile = plots_dir / "surface_3d" / f"{dataset}_{metric.replace('@', 'at')}_{stage}_surface3d.png"
    plt.savefig(outfile, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_interactive_surface_3d(
    dataset: str,
    ratio_plot_data: Dict[float, Dict[float, Dict[str, Dict[str, float]]]],
    metric: str,
    stage: str,
    plots_dir: Path,
    dpi: int = 200,
) -> None:
    """Create interactive 3D surface plot using plotly (if available).

    Args:
        dataset: Dataset name
        ratio_plot_data: Dict[red_ratio -> aug_ratio -> stage -> metric -> value]
        metric: Metric to plot
        stage: Stage name
        plots_dir: Output directory
        dpi: Resolution (not used for plotly, kept for compatibility)
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        # Fall back to matplotlib if plotly not available
        return

    if not ratio_plot_data:
        return

    ensure_dir(plots_dir / "surface_3d_interactive")

    # Extract data
    reduction_ratios = sorted(ratio_plot_data.keys())
    augmentation_ratios = set()
    for aug_dict in ratio_plot_data.values():
        augmentation_ratios.update(aug_dict.keys())
    augmentation_ratios = sorted(augmentation_ratios)

    if not reduction_ratios or not augmentation_ratios:
        return

    # Need at least 2x2 grid for meaningful 3D surface plot
    if len(reduction_ratios) < 2 or len(augmentation_ratios) < 2:
        return

    # Create meshgrid
    X, Y = np.meshgrid(reduction_ratios, augmentation_ratios)
    Z = np.zeros_like(X)

    # Fill Z values
    for i, aug_ratio in enumerate(augmentation_ratios):
        for j, red_ratio in enumerate(reduction_ratios):
            value = ratio_plot_data.get(red_ratio, {}).get(aug_ratio, {}).get(stage, {}).get(metric)
            Z[i, j] = value if value is not None else np.nan

    # Create interactive surface
    fig = go.Figure(data=[go.Surface(
        x=reduction_ratios,
        y=augmentation_ratios,
        z=Z,
        colorscale='Viridis',
        colorbar=dict(title=metric),
    )])

    fig.update_layout(
        title=f'{dataset} - {metric} ({stage.capitalize()})',
        scene=dict(
            xaxis_title='Reduction Ratio',
            yaxis_title='Augmentation Ratio',
            zaxis_title=metric,
        ),
        width=1000,
        height=800,
    )

    outfile = plots_dir / "surface_3d_interactive" / f"{dataset}_{metric.replace('@', 'at')}_{stage}_surface3d.html"
    fig.write_html(str(outfile))


__all__ = [
    "plot_heatmap",
    "plot_boxplot",
    "plot_violin",
    "plot_scatter_correlation",
    "plot_delta_chart",
    "plot_radar_chart",
    "plot_ridge",
    "plot_performance_matrix",
    "plot_surface_3d",
    "plot_interactive_surface_3d",
    "save_figure",
]
