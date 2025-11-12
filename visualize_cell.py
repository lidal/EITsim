#!/usr/bin/env python3
"""3D visualization of the vapor cell and intersecting laser beams."""
from __future__ import annotations

import argparse
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import proj3d


class _OverlayLabel:
    """2D overlay text that tracks a 3D point and stays on top."""

    def __init__(self, ax, position, text, color):
        self.ax = ax
        self.position = np.array(position, dtype=float)
        fig = ax.figure
        self.text = fig.text(
            0,
            0,
            text,
            transform=fig.transFigure,
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(facecolor=color, alpha=0.85, edgecolor="none", pad=1),
            zorder=1e6,
        )
        self.cid = fig.canvas.mpl_connect("draw_event", self._update)
        self._update()

    def _update(self, event=None):
        x2, y2, _ = proj3d.proj_transform(*self.position, self.ax.get_proj())
        disp_x, disp_y = self.ax.transData.transform((x2, y2))
        fig_x, fig_y = self.ax.figure.transFigure.inverted().transform((disp_x, disp_y))
        self.text.set_position((fig_x, fig_y))

    def remove(self):
        self.ax.figure.canvas.mpl_disconnect(self.cid)
        self.text.remove()


def _clear_overlay_labels(ax):
    labels = getattr(ax, "_rq_overlay_labels", [])
    for label in labels:
        label.remove()
    ax._rq_overlay_labels = []


def _add_overlay_label(ax, position, text, color):
    if not hasattr(ax, "_rq_overlay_labels"):
        ax._rq_overlay_labels = []
    label = _OverlayLabel(ax, position, text, color)
    ax._rq_overlay_labels.append(label)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a 3D sketch of the vapor cell and probe/control beams.")
    parser.add_argument("--cell-length", type=float, default=0.05,
                        help="Cell length in meters (defaults to main.py value).")
    parser.add_argument("--cell-cross", type=float, default=0.02,
                        help="Cell cross-section side (meters).")
    parser.add_argument("--probe-waist", type=float, default=100e-6,
                        help="Probe beam 1/e^2 radius in meters.")
    parser.add_argument("--control-waist", type=float, default=100e-6,
                        help="Control beam 1/e^2 radius in meters.")
    parser.add_argument("--resolution", type=int, default=100,
                        help="Number of samples along the beam axis.")
    parser.add_argument("--probe-label", default="", help="Text label near the probe laser housing.")
    parser.add_argument("--control-label", default="", help="Text label near the control laser housing.")
    return parser.parse_args()


def gaussian_radius(z, waist, zr):
    return waist * np.sqrt(1.0 + (z / zr) ** 2)


def draw_cell(ax, length, cross):
    x = [-cross/2, cross/2, cross/2, -cross/2, -cross/2]
    y = [-cross/2, -cross/2, cross/2, cross/2, -cross/2]
    z0, z1 = 0, length
    ax.plot(x, y, [z0]*5, color="gray")
    ax.plot(x, y, [z1]*5, color="gray")
    for i in range(4):
        ax.plot([x[i], x[i]], [y[i], y[i]], [z0, z1], color="gray")
    verts = [
        [(-cross/2, -cross/2, z0), (cross/2, -cross/2, z0), (cross/2, cross/2, z0), (-cross/2, cross/2, z0)],
        [(-cross/2, -cross/2, z1), (cross/2, -cross/2, z1), (cross/2, cross/2, z1), (-cross/2, cross/2, z1)],
        [(-cross/2, -cross/2, z0), (-cross/2, -cross/2, z1), (-cross/2, cross/2, z1), (-cross/2, cross/2, z0)],
        [(cross/2, -cross/2, z0), (cross/2, -cross/2, z1), (cross/2, cross/2, z1), (cross/2, cross/2, z0)],
        [(-cross/2, -cross/2, z0), (cross/2, -cross/2, z0), (cross/2, -cross/2, z1), (-cross/2, -cross/2, z1)],
        [(-cross/2, cross/2, z0), (cross/2, cross/2, z0), (cross/2, cross/2, z1), (-cross/2, cross/2, z1)],
    ]
    poly = Poly3DCollection(verts, facecolors="lightgray", linewidths=0, alpha=0.2)
    ax.add_collection3d(poly)


def draw_beam(ax, length, waist, color, resolution, extend_negative=0.0, extend_positive=0.0,
              turn_length=0.02, turn_dir_negative=1.0, turn_dir_positive=-1.0):
    z = np.linspace(-extend_negative, length + extend_positive, resolution)
    lambda_placeholder = 500e-9
    zr = np.pi * waist**2 / lambda_placeholder
    radii = gaussian_radius(z - length/2, waist, zr)
    for angle in np.linspace(0, 2*np.pi, 20):
        x = radii * np.cos(angle)
        y = radii * np.sin(angle)
        ax.plot(x, y, z, color=color, alpha=0.1)
        if extend_negative > 0:
            z_val = -extend_negative
            r_end = gaussian_radius(z_val - length/2, waist, zr)
            x_end = r_end * np.cos(angle)
            y_end = r_end * np.sin(angle)
            x_turn = np.linspace(x_end, x_end + turn_dir_negative * turn_length, 20)
            y_turn = np.full_like(x_turn, y_end)
            z_turn = np.full_like(x_turn, z_val)
            ax.plot(x_turn, y_turn, z_turn, color=color, alpha=0.1)
        if extend_positive > 0:
            z_val = length + extend_positive
            r_end = gaussian_radius(z_val - length/2, waist, zr)
            x_end = r_end * np.cos(angle)
            y_end = r_end * np.sin(angle)
            x_turn = np.linspace(x_end, x_end + turn_dir_positive * turn_length, 20)
            y_turn = np.full_like(x_turn, y_end)
            z_turn = np.full_like(x_turn, z_val)
            ax.plot(x_turn, y_turn, z_turn, color=color, alpha=0.1)


def render_scene(ax, cell_length, cell_cross, probe_waist, control_waist, resolution,
                 zoom=1.0, probe_label="", control_label=""):
    _clear_overlay_labels(ax)
    ax.clear()
    draw_cell(ax, cell_length, cell_cross)
    red_extend_neg = cell_length * 0.25
    red_extend_pos = cell_length * 0.50
    blue_extend_neg = cell_length * 0.50
    blue_extend_pos = cell_length * 0.25
    turn_len = 0.02

    draw_beam(ax, cell_length, probe_waist, "red", resolution,
              extend_negative=red_extend_neg, extend_positive=red_extend_pos,
              turn_length=turn_len, turn_dir_negative=1.0, turn_dir_positive=0.0)
    draw_beam(ax, cell_length, control_waist, "blue", resolution,
              extend_negative=blue_extend_neg, extend_positive=blue_extend_pos,
              turn_length=turn_len, turn_dir_negative=0.0, turn_dir_positive=-1.0)

    red_turn_point = [0, 0, -red_extend_neg]
    blue_turn_point = [0, 0, cell_length + blue_extend_pos]
    draw_mirror(ax, red_turn_point, [1, 0, 1])
    draw_mirror(ax, blue_turn_point, [-1, 0, -1])

    box_size = (0.03, 0.01, 0.01)
    red_prism_center = [turn_len + box_size[0]/2, 0, red_turn_point[2]]
    blue_prism_center = [-turn_len - box_size[0]/2, 0, blue_turn_point[2]]
    draw_prism(ax, red_prism_center, box_size, color="#5a1f1f", alpha=1.0, label=probe_label)
    draw_prism(ax, blue_prism_center, box_size, color="#2c3f7a", alpha=1.0, label=control_label)
    max_range = max(cell_cross, cell_length) * zoom
    ax.set_xlim(-max_range/2, max_range/2)
    ax.set_ylim(-max_range/2, max_range/2)
    ax.set_zlim(0, max_range)
    ax.set_facecolor("#2b2b2b")
    ax.figure.set_facecolor("#2b2b2b")
    ax.set_axis_off()
    ax.grid(False)
    ax.figure.canvas.draw_idle()


def main():
    args = parse_args()
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111, projection="3d")
    render_scene(ax, args.cell_length, args.cell_cross, args.probe_waist,
                 args.control_waist, args.resolution, zoom=1.0,
                 probe_label=args.probe_label, control_label=args.control_label)
    plt.tight_layout()
    plt.show()


def draw_mirror(ax, center, normal, size=0.01, color="dimgrey"):
    center = np.array(center)
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)
    if abs(normal[2]) > 0.9:
        u = np.array([1, 0, 0])
    else:
        u = np.array([0, 0, 1])
    u = u - np.dot(u, normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    half = size / 2
    corners = [
        center + half * ( u + v),
        center + half * ( u - v),
        center + half * (-u - v),
        center + half * (-u + v),
    ]
    xs, ys, zs = zip(*(corners + [corners[0]]))
    ax.plot(xs, ys, zs, color=color, linewidth=1.2)
    face = Poly3DCollection([corners], facecolors=color, alpha=0.9, linewidths=0)
    ax.add_collection3d(face)
def draw_prism(ax, center, size, axis="x", color="gray", alpha=0.4, label=""):
    cx, cy, cz = center
    lx, ly, lz = size
    dx = lx / 2
    dy = ly / 2
    dz = lz / 2
    vertices = [
        [cx - dx, cy - dy, cz - dz],
        [cx + dx, cy - dy, cz - dz],
        [cx + dx, cy + dy, cz - dz],
        [cx - dx, cy + dy, cz - dz],
        [cx - dx, cy - dy, cz + dz],
        [cx + dx, cy - dy, cz + dz],
        [cx + dx, cy + dy, cz + dz],
        [cx - dx, cy + dy, cz + dz],
    ]
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [1, 2, 6, 5],
        [0, 3, 7, 4],
    ]
    polys = [[vertices[idx] for idx in face] for face in faces]
    for i, face in enumerate(polys):
        shade = 0.9 - 0.1 * i
        base_rgb = colors.to_rgb(color)
        face_color = tuple(min(1.0, c * shade) for c in base_rgb)
        poly = Poly3DCollection([face], facecolors=face_color, alpha=alpha,
                                linewidths=0.5, edgecolors=color)
        ax.add_collection3d(poly)
    if label:
        _add_overlay_label(ax, center, label, color)



if __name__ == "__main__":
    main()
