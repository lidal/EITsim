#!/usr/bin/env python3
"""
Back-out RF field amplitude and detuning from measured Autler–Townes splitting.

Inputs: selected n/n_p (or derived RF transition), measured peak-to-peak splitting,
and optional RF detuning. Uses ARC/Rydiqule to compute transition wavelengths
and dipole moments, then estimates the RF electric field.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.constants import hbar

from rydiqule import Cell
from rydiqule.atom_utils import A_QState


PA_TO_TORR = 7.50061683 / 1000.0  # unused here; kept for completeness


@dataclass(frozen=True)
class SimulationStates:
    ground: A_QState
    intermediate: A_QState
    rydberg_d: A_QState
    rydberg_p: A_QState

    def as_dict(self) -> Dict[str, A_QState]:
        return {
            "g": self.ground,
            "e": self.intermediate,
            "r": self.rydberg_d,
            "r_prime": self.rydberg_p,
        }


def build_states(isotope: str, n_d: int, n_p: int | None) -> SimulationStates:
    """Choose representative hyperfine states for the ladder system."""
    if isotope == "Rb87":
        g = A_QState(n=5, l=0, j=0.5, f=2, m_f=0)
        e = A_QState(n=5, l=1, j=1.5, f=3, m_f=0)
    else:
        g = A_QState(n=5, l=0, j=0.5, f=3, m_f=0)
        e = A_QState(n=5, l=1, j=1.5, f=4, m_f=0)

    r_state = A_QState(n=n_d, l=2, j=2.5, m_j=0.5)
    p_state = A_QState(n=n_p if n_p is not None else n_d + 1, l=1, j=1.5, m_j=0.5)
    return SimulationStates(g, e, r_state, p_state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode RF amplitude and detuning from measured AT splitting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--isotope", choices=("Rb85", "Rb87"), default="Rb87",
                        help="Rubidium isotope used.")
    parser.add_argument("--n", type=int, required=True,
                        help="Selected principal quantum number for nD5/2.")
    parser.add_argument("--np", type=int, default=None,
                        help="Selected principal quantum number for nP (RF-coupled). Defaults to n+1.")
    parser.add_argument("--measured-splitting", type=float, required=True,
                        help="Measured peak-to-peak splitting (MHz) from the experiment.")
    parser.add_argument("--peak-ratio", type=float, default=1.0,
                        help="Measured peak height ratio (left/right). Used to infer RF detuning. Defaults to 1.0.")
    parser.add_argument("--df-correction", type=float, default=1.6,
                        help="Multiplicative correction factor applied to the Df slope (default 1.6).")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON output file with the decoded results.")
    return parser.parse_args()


def compute_wavelengths(atom, states: SimulationStates) -> Dict[str, float]:
    probe_lambda_nm = atom.get_transition_wavelength(states.ground, states.intermediate) * 1e9
    control_lambda_nm = atom.get_transition_wavelength(states.intermediate, states.rydberg_d) * 1e9
    rf_res_hz = atom.get_transition_frequency(states.rydberg_p, states.rydberg_d)
    return {
        "probe_lambda_nm": probe_lambda_nm,
        "control_lambda_nm": control_lambda_nm,
        "rf_res_hz": rf_res_hz,
    }


def decode():
    args = parse_args()
    states = build_states(args.isotope, args.n, args.np)

    # Minimal cell for atomic data lookup
    cell = Cell(args.isotope, list(states.as_dict().values()), temp=300.0, cell_length=0.01)
    atom = cell.atom
    wl = compute_wavelengths(atom, states)

    # Dipole for RF transition
    dipole_au = atom.get_dipole_matrix_element(states.rydberg_p, states.rydberg_d, q=0)
    dipole_si = dipole_au * 5.29177210903e-11 * 1.602176634e-19  # Bohr radius * e (C·m)

    lambda_ratio = (wl["control_lambda_nm"] / wl["probe_lambda_nm"]) * args.df_correction
    meas = args.measured_splitting

    # Infer Ω_RF and Δ_RF from splitting and peak ratio.
    # Model: splitting_obs = lambda_ratio * sqrt(Ω_RF^2 + Δ_RF^2)
    # Peak height ratio R ≈ ((Ω_RF + Δ_RF)/(Ω_RF - Δ_RF))^2 -> sqrt(R) = (Ω+Δ)/(Ω-Δ)
    a = np.sqrt(args.peak_ratio)
    if a <= 0 or np.isclose(a, 1.0):
        # Symmetric peaks imply Δ_RF ~ 0
        omega_rf_mhz = meas * (wl["probe_lambda_nm"] / wl["control_lambda_nm"]) * args.df_correction
        delta_rf_mhz = 0.0
    else:
        # Solve for Ω_RF using both equations
        ratio_term = (a - 1) / (a + 1)
        denom = np.sqrt(1 + ratio_term ** 2)
        omega_rf_mhz = (meas / lambda_ratio) / denom
        delta_rf_mhz = omega_rf_mhz * ratio_term

    omega_rf_rad_s = omega_rf_mhz * 1e6 * 2 * np.pi
    if dipole_si == 0:
        e_field_v_m = np.nan
    else:
        e_field_v_m = omega_rf_rad_s * hbar / dipole_si
    e_field_v_cm = e_field_v_m / 100.0 if not np.isnan(e_field_v_m) else np.nan

    result = {
        "isotope": args.isotope,
        "n": args.n,
        "np": args.np if args.np is not None else args.n + 1,
        "probe_lambda_nm": wl["probe_lambda_nm"],
        "control_lambda_nm": wl["control_lambda_nm"],
        "rf_res_hz": wl["rf_res_hz"],
        "measured_splitting_mhz": meas,
        "inferred_rf_detuning_mhz": delta_rf_mhz,
        "omega_rf_mhz_est": omega_rf_mhz,
        "e_field_v_cm_est": e_field_v_cm,
        "peak_ratio": args.peak_ratio,
        "df_correction": args.df_correction,
    }

    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)


if __name__ == "__main__":
    decode()
