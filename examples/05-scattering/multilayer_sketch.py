"""Sketch of the multilayer stack in the frame of the sample for the scattering
overview example, which is also its gallery thumbnail.

The sketch is saved to the given paths or, by default, to
``multilayer_sketch.svg`` and ``multilayer_sketch.png`` next to this script,
e.g. by ``python multilayer_sketch.py``. The labels use the font Fira Sans,
which must be installed, and are converted to paths in the svg. The svg must
also be copied to ``docs/source/examples/scattering/`` for the documentation,
while the png is used for the gallery thumbnail.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update(
    {
        "font.family": "Fira Sans",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Fira Sans",
        "mathtext.it": "Fira Sans:italic",
        "mathtext.bf": "Fira Sans:bold",
        "mathtext.bfit": "Fira Sans:bold:italic",
        "svg.fonttype": "path",
        "svg.hashsalt": "udkm1Dsim",  # reproducible ids of the svg elements
        "font.size": 13,
    }
)

BLACK = "#222222"
RED, BLUE, GREEN, ORANGE, PURPLE = "#c0392b", "#2471a3", "#1e8449", "#d68910", "#7d3c98"

fig, ax = plt.subplots(figsize=[8.4, 5.8])


def arrow(start, end, color=BLACK, lw=1.8, head=0.32, both=False, zorder=2):
    """line with filled, swept-back heads as in the getting started sketch"""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    unit = (end - start) / np.hypot(*(end - start))

    def draw_head(tip, direction):
        back = tip - head * direction
        notch = tip - 0.62 * head * direction
        side = 0.36 * head * np.array([-direction[1], direction[0]])
        ax.add_patch(
            plt.Polygon(
                [tip, back + side, notch, back - side],
                closed=True,
                color=color,
                lw=0,
                zorder=zorder,
            )
        )
        return notch

    line_end = draw_head(end, unit)
    line_start = draw_head(start, -unit) if both else start
    ax.plot(
        *np.transpose([line_start, line_end]),
        color=color,
        lw=lw,
        zorder=zorder,
        solid_capstyle="butt",
    )


ax.set_xlim(-5.0, 5.6)
ax.set_ylim(-5.2, 3.2)
ax.set_aspect("equal")
ax.axis("off")

# stack from the superstrate to the substrate
left, right = -5.0, 3.4
layers = [  # name, top, bottom, color
    ("layer 1", 0.0, -0.7, "#e3eef8"),
    (None, -0.7, -1.2, "white"),
    ("layer $j$", -1.2, -2.6, "#f8ead6"),
    (None, -2.6, -3.1, "white"),
    ("layer $N$", -3.1, -3.8, "#e3eef8"),
]
for name, top, bottom, color in layers:
    ax.fill_between([left, right], bottom, top, color=color, lw=0)
    if name is None:
        ax.text(-3.9, (top + bottom) / 2, r"$\vdots$", ha="center", va="center", fontsize=16)
    else:
        ax.text(left + 0.15, (top + bottom) / 2, name, va="center", color=BLACK)
ax.fill_between([left, right], -5.2, -3.8, color="#d9d9d9", lw=0)
for z in [0.0, -0.7, -1.2, -2.6, -3.1, -3.8]:
    ax.plot([left, right], [z, z], color=BLACK, lw=0.8)
ax.text(left + 0.15, 2.9, "superstrate", va="center", color=BLACK)
ax.text(left + 0.15, -4.5, "substrate", va="center", color=BLACK)

# matrices of layer j: projection at its top interface, propagation through it
ax.annotate(
    r"$A_j^{-1} A_{j-1}$",
    (right, -1.2),
    (right + 0.35, -0.75),
    color=PURPLE,
    va="center",
    arrowprops=dict(arrowstyle="-", color=PURPLE, lw=1.0),
)
arrow((right + 0.2, -1.2), (right + 0.2, -2.6), color=PURPLE, lw=1.2, head=0.22, both=True)
ax.text(right + 0.35, -1.9, "$P_j$", color=PURPLE, va="center")

# beams in the yz plane of incidence at the grazing angle theta
theta = np.radians(30)
down = np.array([np.cos(theta), -np.sin(theta)])
up = np.array([np.cos(theta), np.sin(theta)])
start = np.array([-3.4, 3.4 * np.tan(theta)])
arrow(start, (0, 0))
arrow((0, 0), 3.3 * up)
ax.text(*(start + [0.05, 0.25]), r"$\mathbf{E}^\mathrm{in}$", color=BLACK)
ax.text(
    *(3.3 * up + [-0.9, 0.25]), r"$\mathbf{E}^\mathrm{r} = R\,\mathbf{E}^\mathrm{in}$", color=BLACK
)

# the two down- and two up-going eigenmodes coexist throughout layer j
for dx in [0.0, 0.3]:
    arrow(
        (-1.6 + dx, -1.67), np.array([-1.6 + dx, -1.67]) + 0.9 * down, color=RED, lw=1.4, head=0.24
    )
    arrow((1.0 + dx, -2.13), np.array([1.0 + dx, -2.13]) + 0.9 * up, color=BLUE, lw=1.4, head=0.24)
ax.text(-1.0, -2.35, "2 down-going modes", color=RED, ha="center", va="center", fontsize=11)
ax.text(1.55, -1.45, "2 up-going modes", color=BLUE, ha="center", va="center", fontsize=11)

# transmitted beam into the substrate
t0 = np.array([0.8, -3.8])
arrow(t0, t0 + 1.4 * down)
ax.text(-1.6, -4.65, r"$\mathbf{E}^\mathrm{t} = T\,\mathbf{E}^\mathrm{in}$", va="center")

# polarization of the incident beam: sigma along x out of the page, pi in the plane
p = start + 0.45 * (np.array([0, 0]) - start)
normal = np.array([np.sin(theta), np.cos(theta)])
ax.plot(*p, "o", ms=13, mfc="white", mec=GREEN, mew=1.6, zorder=3)
ax.plot(*p, "o", ms=4, color=GREEN, zorder=4)
ax.text(*(p + [-0.75, -0.05]), r"$E_\sigma$", color=GREEN, ha="center", va="center")
arrow(p - 0.75 * normal, p + 0.75 * normal, color=ORANGE, lw=1.6, head=0.24, both=True, zorder=2)
ax.text(*(p + 0.85 * normal + [0.05, 0.0]), r"$E_\pi$", color=ORANGE)

# grazing angle
arc = np.linspace(np.pi - theta, np.pi, 40)
ax.plot(1.1 * np.cos(arc), 1.1 * np.sin(arc), color=BLACK, lw=1.0)
ax.text(-1.55, 0.25, r"$\vartheta$", color=BLACK)

# sample frame: z along the outward surface normal, yz the plane of incidence,
# and x along sigma, i.e. out of the page
o = np.array([4.2, 1.9])
arrow(o, o + [0.9, 0], lw=1.4, head=0.24)
arrow(o, o + [0, 0.9], lw=1.4, head=0.24)
ax.plot(*o, "o", ms=11, mfc="white", mec=BLACK, mew=1.2, zorder=3)
ax.plot(*o, "o", ms=3, color=BLACK, zorder=4)
ax.text(*(o + [1.0, -0.1]), "$y$", va="center")
ax.text(*(o + [-0.08, 1.05]), "$z$", ha="center")
ax.text(*(o + [-0.45, -0.4]), "$x$", ha="center")

paths = sys.argv[1:] or [
    Path(__file__).with_name("multilayer_sketch.svg"),
    Path(__file__).with_name("multilayer_sketch.png"),
]
for path in paths:
    metadata = {"Date": None} if str(path).endswith(".svg") else {}
    fig.savefig(
        path, dpi=200, bbox_inches="tight", pad_inches=0.05, facecolor="white", metadata=metadata
    )
