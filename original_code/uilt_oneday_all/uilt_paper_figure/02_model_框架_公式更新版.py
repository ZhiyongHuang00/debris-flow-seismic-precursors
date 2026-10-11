# -*- coding: utf-8 -*-
"""Draw the PCS conceptual diagram using the current manuscript formulas."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


INK = "#25313B"
MUTED = "#687685"
EXPECTED = "#98A8B7"
TEAL = "#138F89"
ORANGE = "#D98C2B"
RED = "#C95743"
MAGENTA = "#B83CAD"

REFERENCE_FILL = "#E9EFF4"
PREVIOUS_5H_FILL = "#F3EEE4"
LAST_1H_FILL = "#F5DDD8"
MAGENTA_FILL = "#F8EAF5"
TEAL_FILL = "#E5F3F1"
RED_FILL = "#FAECE8"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": True,
    }
)


def cumulative_curves(time_h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the reference expectation and an illustrative nonlinear cumulative curve."""
    initial_count = 0.62
    early_rate = 0.18
    nonlinear_strength = 0.035
    expected = initial_count + early_rate * (time_h + 24.0)
    observed = expected + nonlinear_strength * np.clip(time_h + 12.0, 0.0, None) ** 2
    return expected, observed


def formula_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    edge: str,
    fill: str,
    fontsize: float,
) -> None:
    """Add a restrained formula callout in data coordinates."""
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.15,rounding_size=0.28",
        linewidth=1.15,
        edgecolor=edge,
        facecolor=fill,
        zorder=7,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        zorder=8,
    )


def draw_arrow_axes(ax: plt.Axes) -> None:
    arrow_style = dict(
        arrowstyle="-|>",
        mutation_scale=23,
        linewidth=1.55,
        color=INK,
        shrinkA=0,
        shrinkB=0,
        clip_on=False,
        zorder=10,
    )
    ax.add_patch(FancyArrowPatch((-24.0, 0.0), (0.50, 0.0), **arrow_style))
    ax.add_patch(FancyArrowPatch((-24.0, 0.0), (-24.0, 10.35), **arrow_style))

    for x, label in zip([-24, -12, -6, -1, 0], ["−24", "−12", "−6", "−1", "0"]):
        ax.plot([x, x], [0, -0.20], color=INK, lw=1.05, clip_on=False, zorder=10)
        ax.text(
            x,
            -0.53,
            label,
            ha="center",
            va="top",
            fontsize=11,
            color=MUTED,
            clip_on=False,
        )


def draw_current_formula_figure(
    output_file: str | Path | None = None,
    *,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Draw and save the formula-updated PCS conceptual figure as PNG."""
    if output_file is None:
        output_file = (
            Path(__file__).resolve().parent
            / "figure2"
            / "PCS_theory_formula_updated.png"
        )
    output_file = Path(output_file).expanduser().resolve()
    if output_file.suffix.lower() != ".png":
        output_file = output_file.with_suffix(".png")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.7, 7.15), facecolor="white")
    fig.subplots_adjust(left=0.10, right=0.94, bottom=0.16, top=0.94)

    ax.set_xlim(-26.2, 1.75)
    ax.set_ylim(-0.85, 10.85)
    ax.set_facecolor("white")

    # Windows used by the current definitions.
    ax.axvspan(-24, -12, ymin=0.072, ymax=0.952, color=REFERENCE_FILL, zorder=0)
    ax.axvspan(-6, -1, ymin=0.072, ymax=0.952, color=PREVIOUS_5H_FILL, alpha=0.62, zorder=0)
    ax.axvspan(-1, 0, ymin=0.072, ymax=0.952, color=LAST_1H_FILL, alpha=0.82, zorder=0)

    time_h = np.linspace(-24.0, 0.0, 800)
    expected, observed = cumulative_curves(time_h)

    ax.plot(
        time_h,
        expected,
        color=EXPECTED,
        lw=1.9,
        linestyle=(0, (5, 4)),
        zorder=2,
    )
    ax.plot(
        time_h,
        observed,
        color=TEAL,
        lw=3.1,
        solid_capstyle="round",
        zorder=4,
    )

    early_rate = 0.18
    nonlinear_strength = 0.035

    early_x = np.linspace(-21.7, -13.0, 60)
    early_y = 0.62 + early_rate * (early_x + 24.0) + 0.22
    ax.plot(early_x, early_y, color=ORANGE, lw=1.55, linestyle=(0, (5, 4)), zorder=3)

    tangent_x0 = -5.2
    tangent_y0 = np.interp(tangent_x0, time_h, observed)
    late_rate = early_rate + 2.0 * nonlinear_strength * (tangent_x0 + 12.0)
    late_x = np.linspace(-6.9, -2.5, 60)
    late_y = tangent_y0 + late_rate * (late_x - tangent_x0) + 0.25
    ax.plot(late_x, late_y, color=ORANGE, lw=1.55, linestyle=(0, (5, 4)), zorder=3)

    # Known initiation boundary.
    ax.plot([0, 0], [0, observed[-1]], color=RED, lw=1.05, zorder=3)

    # S is the standardized endpoint anomaly, not simply an unscaled vertical gap.
    ax.add_patch(
        FancyArrowPatch(
            (0.63, expected[-1]),
            (0.63, observed[-1]),
            arrowstyle="<|-|>",
            mutation_scale=18,
            linewidth=1.55,
            color=RED,
            clip_on=False,
            zorder=8,
        )
    )
    ax.text(
        1.15,
        (expected[-1] + observed[-1]) / 2,
        r"$S_{e,c}$",
        ha="center",
        va="center",
        fontsize=20,
        color=INK,
        clip_on=False,
    )

    ax.text(
        -22.55,
        np.interp(-22.55, time_h, expected) + 0.58,
        r"Reference expectation  $\lambda^{\mathrm{ref}}_{e,c}k\Delta t$",
        fontsize=12.3,
        fontfamily="serif",
        fontstyle="italic",
        color=INK,
    )
    ax.text(
        -9.8,
        np.interp(-9.8, time_h, observed) + 2.35,
        r"Observed  $C_{e,c}(k)$",
        fontsize=13.7,
        fontfamily="serif",
        fontstyle="italic",
        fontweight="semibold",
        color=INK,
    )
    ax.text(
        -9.8,
        2.74,
        r"$A_{e,c}=\beta^{\mathrm{late}}_{e,c}-\beta^{\mathrm{early}}_{e,c}$",
        fontsize=13.4,
        color=INK,
    )

    formula_box(
        ax,
        (-22.65, 7.45),
        9.15,
        1.30,
        r"$R_{e,c}=\mathrm{RMSE}_{\mathrm{PL}}/\mathrm{RMSE}_{\mathrm{lin}}<1$",
        MAGENTA,
        MAGENTA_FILL,
        14.5,
    )
    formula_box(
        ax,
        (-11.25, 8.45),
        9.60,
        1.18,
        (
            r"$S_{e,c}=\dfrac{C_{e,c}(N)-\lambda^{\mathrm{ref}}_{e,c}N\Delta t}"
            r"{\sqrt{\lambda^{\mathrm{ref}}_{e,c}N\Delta t+\varepsilon_S}}"
            r"\,,\quad S_{e,c}>2$"
        ),
        RED,
        RED_FILL,
        13.1,
    )
    formula_box(
        ax,
        (-5.72, 0.54),
        5.18,
        1.94,
        (
            r"$G_{e,c}=\dfrac{r^{1\mathrm{h}}_{e,c}}{r^{5\mathrm{h}}_{e,c}+\varepsilon_G}$"
            "\n"
            r"$F_{e,c}=1:\ [r^{5\mathrm{h}}_{e,c}>0\ \wedge\ G_{e,c}\geq0.5]$"
            "\n"
            r"$\mathrm{or}\ [r^{5\mathrm{h}}_{e,c}=0\ \wedge\ n^{1\mathrm{h}}_{e,c}\geq1]$"
        ),
        TEAL,
        TEAL_FILL,
        10.9,
    )

    ax.text(-18.0, 9.92, "early reference (12 h)", ha="center", va="center", fontsize=12.5, color=INK, fontstyle="italic")
    ax.text(-3.50, 9.92, "previous 5 h", ha="center", va="center", fontsize=11.5, color=MUTED, fontstyle="italic")
    ax.text(-0.50, 9.56, "last 1 h", ha="center", va="center", fontsize=11.5, color=INK, fontstyle="italic")
    ax.text(0.0, 10.53, "initiation", ha="right", va="bottom", fontsize=13.0, color=INK, fontstyle="italic", fontweight="bold")

    draw_arrow_axes(ax)

    ax.text(-12.0, -0.79, "Time to initiation (h)", ha="center", va="top", fontsize=14.5, color=INK, clip_on=False)
    ax.text(-26.0, 5.2, r"Cumulative occurrences  $C_{e,c}(k)$", ha="center", va="center", rotation=90, fontsize=14.5, color=INK, clip_on=False)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(output_file, dpi=450, bbox_inches="tight", pad_inches=0.08, facecolor="white")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def draw_simplified_figure(
    output_file: str | Path | None = None,
    *,
    no_text: bool = False,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Draw a concise PCS concept figure, optionally removing every text artist."""
    if output_file is None:
        output_file = (
            Path(__file__).resolve().parent
            / "figure2"
            / ("PCS_theory_no_text.png" if no_text else "PCS_theory_simplified.png")
        )
    output_file = Path(output_file).expanduser().resolve()
    if output_file.suffix.lower() != ".png":
        output_file = output_file.with_suffix(".png")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.7, 7.15), facecolor="white")
    fig.subplots_adjust(left=0.10, right=0.94, bottom=0.16, top=0.94)

    ax.set_xlim(-26.2, 1.75)
    ax.set_ylim(-0.85, 10.85)
    ax.set_facecolor("white")

    # Time windows used by PCS.
    ax.axvspan(-24, -12, ymin=0.072, ymax=0.952, color=REFERENCE_FILL, zorder=0)
    ax.axvspan(-6, -1, ymin=0.072, ymax=0.952, color=PREVIOUS_5H_FILL, alpha=0.62, zorder=0)
    ax.axvspan(-1, 0, ymin=0.072, ymax=0.952, color=LAST_1H_FILL, alpha=0.82, zorder=0)

    time_h = np.linspace(-24.0, 0.0, 800)
    expected, observed = cumulative_curves(time_h)

    ax.plot(
        time_h,
        expected,
        color=EXPECTED,
        lw=1.9,
        linestyle=(0, (5, 4)),
        zorder=2,
    )
    ax.plot(
        time_h,
        observed,
        color=TEAL,
        lw=3.1,
        solid_capstyle="round",
        zorder=4,
    )

    # Early and late local trends: a visual definition of the rate change A.
    early_rate = 0.18
    nonlinear_strength = 0.035
    early_x = np.linspace(-21.7, -13.0, 60)
    early_y = 0.62 + early_rate * (early_x + 24.0) + 0.22
    ax.plot(early_x, early_y, color=ORANGE, lw=1.55, linestyle=(0, (5, 4)), zorder=3)

    tangent_x0 = -5.2
    tangent_y0 = np.interp(tangent_x0, time_h, observed)
    late_rate = early_rate + 2.0 * nonlinear_strength * (tangent_x0 + 12.0)
    late_x = np.linspace(-6.9, -2.5, 60)
    late_y = tangent_y0 + late_rate * (late_x - tangent_x0) + 0.25
    ax.plot(late_x, late_y, color=ORANGE, lw=1.55, linestyle=(0, (5, 4)), zorder=3)

    ax.plot([0, 0], [0, observed[-1]], color=RED, lw=1.05, zorder=3)
    ax.add_patch(
        FancyArrowPatch(
            (0.63, expected[-1]),
            (0.63, observed[-1]),
            arrowstyle="<|-|>",
            mutation_scale=18,
            linewidth=1.55,
            color=RED,
            clip_on=False,
            zorder=8,
        )
    )

    # Direct labels replace the expanded equations.
    ax.text(
        -22.55,
        np.interp(-22.55, time_h, expected) + 0.58,
        "Reference expectation",
        fontsize=12.3,
        fontfamily="serif",
        fontstyle="italic",
        color=INK,
    )
    ax.text(
        -9.8,
        np.interp(-9.8, time_h, observed) + 2.35,
        "Observed cumulative curve",
        fontsize=13.2,
        fontfamily="serif",
        fontstyle="italic",
        fontweight="semibold",
        color=INK,
    )

    ax.text(
        -19.0,
        7.90,
        "Nonlinear temporal organization",
        ha="center",
        fontsize=13.0,
        color=MAGENTA,
        fontweight="semibold",
    )
    ax.text(-19.0, 7.43, r"$R<1$", ha="center", fontsize=14.0, color=INK)

    ax.text(
        -9.75,
        2.72,
        r"Late-stage rate enhancement  $(A)$",
        fontsize=12.7,
        color=INK,
        fontweight="semibold",
    )

    ax.text(
        -3.50,
        1.32,
        "Terminal relative activity",
        ha="center",
        fontsize=12.2,
        color=TEAL,
        fontweight="semibold",
    )
    ax.text(
        -3.50,
        0.86,
        r"$G$  ·  observable at initiation  $(F=1)$",
        ha="center",
        fontsize=11.7,
        color=INK,
    )

    ax.text(
        1.14,
        (expected[-1] + observed[-1]) / 2 + 0.25,
        "Cumulative\nanomaly",
        ha="center",
        va="center",
        fontsize=11.5,
        color=INK,
        clip_on=False,
        linespacing=1.10,
    )
    ax.text(
        1.14,
        (expected[-1] + observed[-1]) / 2 - 0.53,
        r"$S>2$",
        ha="center",
        va="center",
        fontsize=14.0,
        color=RED,
        clip_on=False,
    )

    ax.text(-18.0, 9.92, "early reference (12 h)", ha="center", va="center", fontsize=12.5, color=INK, fontstyle="italic")
    ax.text(-3.50, 9.92, "previous 5 h", ha="center", va="center", fontsize=11.5, color=MUTED, fontstyle="italic")
    ax.text(-0.50, 9.56, "last 1 h", ha="center", va="center", fontsize=11.5, color=INK, fontstyle="italic")
    ax.text(0.0, 10.53, "initiation", ha="right", va="bottom", fontsize=13.0, color=INK, fontstyle="italic", fontweight="bold")

    draw_arrow_axes(ax)
    ax.text(-12.0, -0.79, "Time to initiation (h)", ha="center", va="top", fontsize=14.5, color=INK, clip_on=False)
    ax.text(-26.0, 5.2, "Cumulative occurrences", ha="center", va="center", rotation=90, fontsize=14.5, color=INK, clip_on=False)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Keep the visual structure while removing labels, equations and tick numbers.
    if no_text:
        for text_artist in list(ax.texts):
            text_artist.remove()

    fig.savefig(output_file, dpi=450, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw the PCS conceptual figure.")
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--full-formulas", action="store_true", help="Render the expanded-formula version.")
    parser.add_argument("--with-text", action="store_true", help="Render the concise labels; the default is a text-free figure.")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.full_formulas:
        draw_current_formula_figure(args.output_file, show=not args.no_show)
    else:
        draw_simplified_figure(args.output_file, no_text=not args.with_text, show=not args.no_show)
