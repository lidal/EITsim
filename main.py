#!/usr/bin/env python3
"""
Rydiqule-based Rydberg EIT simulation with RF-induced Autler-Townes splitting.

Features
--------
* Rubidium ladder system driven by ~780 nm probe and ~480 nm coupling lasers.
* RF field couples two Rydberg states; user supplies the RF frequency.
* Reports the required coupling-laser frequency for the chosen Rydberg level.
* Simulates Doppler-averaged probe transparency vs detuning, including a temperature option.
* Sweeps over multiple RF amplitudes to highlight the Autler-Townes splitting.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Qt5Agg")  # or "Qt5Agg" if PyQt5 is installed

import matplotlib.pyplot as plt
import numpy as np
import rydiqule as rq
from rydiqule import Cell
from rydiqule.atom_utils import A_QState
from scipy.constants import c, epsilon_0, hbar, k, physical_constants
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from threading import Lock
import time

MASSU = physical_constants["atomic mass constant"][0]
A0 = physical_constants["Bohr radius"][0]
E_CHARGE = physical_constants["elementary charge"][0]
PA_TO_MTORR = 7.50061683
PA_TO_TORR = PA_TO_MTORR / 1000.0
TORR_TO_PA = 133.32236842105263
RB_MASSES = {
    "Rb85": 84.9117893 * MASSU,
    "Rb87": 86.909180531 * MASSU,
}
ARC_DATA_LOCK = Lock()
BACKEND_MODE = False


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


def cprint(message: str) -> None:
    if not BACKEND_MODE:
        print(message)


def vprint(enabled: bool, message: str) -> None:
    if enabled and not BACKEND_MODE:
        print(message)


def f_or_none(value):
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    return float(value)


def write_backend_file(path: str, payload: Dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as backend_file:
            json.dump(payload, backend_file, indent=2)
    except Exception as exc:
        cprint(f"Failed to write backend JSON: {exc}")


def apply_pressure_override(cell: Cell, args: argparse.Namespace) -> float:
    """Optionally override vapor density using user-specified pressure."""

    if args.pressure_torr is None:
        return cell.density

    temp_k = max(args.temperature, 1e-6)
    density = args.pressure_torr * TORR_TO_PA / (k * temp_k)
    cell.density = density
    vprint(args.verbose,
           f"Overriding density: P={args.pressure_torr:.3g} Torr => n={density:.3e} m^-3 at T={temp_k:.1f} K.")
    return density


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate RF-dressed Rydberg EIT in Rubidium with Rydiqule.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--isotope", choices=("Rb85", "Rb87"), default="Rb87",
                        help="Rubidium isotope to model.")
    parser.add_argument("--n", type=int, default=50,
                        help="Principal quantum number for the nD5/2 Rydberg level.")
    parser.add_argument("--np", type=int, default=None,
                        help="Principal quantum number for the P-state coupled by the RF field. "
                             "Defaults to n+1 if not provided.")
    parser.add_argument("--auto-n", action="store_true",
                        help="Automatically choose the Rydberg n that makes the RF transition match the requested RF frequency.")
    parser.add_argument("--n-min", type=int, default=5,
                        help="Minimum n to consider when --auto-n is enabled.")
    parser.add_argument("--n-max", type=int, default=2000,
                        help="Maximum n to consider when --auto-n is enabled.")
    parser.add_argument("--np-offsets", type=int, nargs="+", default=[1],
                        help="Offsets (n_P - n_D) to explore when --auto-n is enabled. "
                             "Values can be negative, zero, or positive.")
    parser.add_argument("--rf-frequency", type=float, required=True,
                        help="Applied RF frequency in MHz that couples the Rydberg states.")
    parser.add_argument("--rf-amplitudes", type=float, nargs="+",
                        default=[0.0, 0.01, 0.02, 0.05, 0.085, 0.1],
                        help="List of RF electric-field amplitudes in V/cm to sweep.")
    parser.add_argument("--probe-power", type=float, default=5e-5,
                        help="Probe laser power in watts.")
    parser.add_argument("--control-power", type=float, default=10e-3,
                        help="Control laser power in watts.")
    parser.add_argument("--probe-waist", type=float, default=1e-4,
                        help="Probe 1/e^2 beam waist in meters.")
    parser.add_argument("--control-waist", type=float, default=1e-4,
                        help="Control 1/e^2 beam waist in meters.")
    parser.add_argument("--control-detuning", type=float, default=0.0,
                        help="Detuning of the control laser in MHz.")
    parser.add_argument("--probe-span", type=float, default=300.0,
                        help="Half-width of the probe detuning scan in MHz.")
    parser.add_argument("--probe-points", type=int, default=401,
                        help="Number of probe detuning points.")
    parser.add_argument("--temperature", type=float, default=300.0,
                        help="Cell temperature in Kelvin. Set <=0 to disable Doppler averaging.")
    parser.add_argument("--pressure-torr", type=float, default=None,
                        help="Override vapor pressure in Torr (ideal gas, sets density via P/(kT)).")
    parser.add_argument("--cell-length", type=float, default=10e-2,
                        help="Length of the vapor cell in meters.")
    parser.add_argument("--transit-rate", type=float, default=0.15,
                        help="Transit broadening rate in MHz.")
    parser.add_argument("--probe-direction", type=int, choices=(-1, 1), default=1,
                        help="Propagation direction of the probe beam (+z or -z) for Doppler shifts.")
    parser.add_argument("--control-direction", type=int, choices=(-1, 1), default=-1,
                        help="Propagation direction of the control beam for Doppler shifts.")
    parser.add_argument("--probe-linewidth", type=float, default=0.0,
                        help="Probe laser linewidth (FWHM) in MHz added to decoherence.")
    parser.add_argument("--control-linewidth", type=float, default=0.0,
                        help="Control laser linewidth (FWHM) in MHz added to decoherence.")
    parser.add_argument("--doppler-method", choices=("split", "uniform", "isopop"), default="split",
                        help="Sampling strategy for Doppler averaging.")
    parser.add_argument("--doppler-width", type=float, default=2.0,
                        help="Velocity-space half-width (in units of most probable speed) for Doppler sampling.")
    parser.add_argument("--coherent-width", type=float, default=0.4,
                        help="Fine-mesh half-width for the split Doppler mesh.")
    parser.add_argument("--doppler-points", type=int, default=201,
                        help="Number of coarse Doppler points (split/uniform methods).")
    parser.add_argument("--coherent-points", type=int, default=401,
                        help="Number of fine Doppler points for the split method.")
    parser.add_argument("--output", default="eit_rf_scan.png",
                        help="Filename for the Autler-Townes plot.")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip displaying the Matplotlib window (still saves the figure).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed status information during the simulation.")
    parser.add_argument("--normalize-baseline", action="store_true",
                        help="Subtract a baseline trace (default: largest RF amplitude) from all curves.")
    parser.add_argument("--baseline-rf-amplitude", type=float, default=None,
                        help="Optional RF amplitude (V/cm) to use solely for baseline normalization.")
    parser.add_argument("--timing", action="store_true",
                        help="Print timing information for the overall simulation.")
    parser.add_argument("--fit-peaks", action="store_true",
                        help="Fit the transmission curve to extract Δ_peak even when peaks overlap.")
    parser.add_argument("--fit-profile", choices=("gaussian", "lorentzian"), default="gaussian",
                        help="Line shape to use when fitting peaks.")
    parser.add_argument("--auto-n-only", action="store_true",
                        help="Only determine the (auto-)selected n/n_p and transition info without running the EIT simulation.")
    parser.add_argument("--sweep-plot", action="store_true",
                        help="Generate an additional Δ_fit vs RF amplitude plot by sweeping many amplitudes.")
    parser.add_argument("--sweep-points", type=int, default=20,
                        help="Number of RF amplitudes to sample for the sweep plot.")
    parser.add_argument("--sweep-output", default="eit_rf_sweep.png",
                        help="Filename for the sweep plot.")
    parser.add_argument("--backend-json", type=str, default=None,
                        help="Write structured output to this JSON file and suppress console chatter.")
    return parser.parse_args()


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


def rubidium_most_probable_speed(isotope: str, temperature: float) -> float:
    mass = RB_MASSES[isotope]
    return np.sqrt(2 * k * temperature / mass)


def propagation_axis(direction: int) -> Tuple[float, float, float]:
    """Return a unit vector along the beam propagation direction."""
    return (0.0, 0.0, float(direction))


def doppler_mesh_args(args: argparse.Namespace) -> Dict:
    if args.doppler_method == "uniform":
        return {"method": "uniform", "width_doppler": args.doppler_width, "n_uniform": args.doppler_points}
    if args.doppler_method == "isopop":
        return {"method": "isopop", "n_isopop": args.doppler_points}
    return {
        "method": "split",
        "width_doppler": args.doppler_width,
        "n_doppler": args.doppler_points,
        "width_coherent": args.coherent_width,
        "n_coherent": args.coherent_points,
    }


def compute_rf_rabi(dipole_coulomb_meter: float, rf_field_v_per_m: float) -> float:
    """Return the RF Rabi rate in Mrad/s."""
    omega_rad_s = dipole_coulomb_meter * rf_field_v_per_m / hbar
    return omega_rad_s / 1e6  # convert to Mrad/s


def rabi_from_power(power_w: float, waist_m: float, dipole_au: float) -> float:
    """Return the optical Rabi rate in rad/s from beam power and waist."""
    if power_w <= 0 or waist_m <= 0 or dipole_au == 0:
        return 0.0
    intensity = 2 * power_w / (np.pi * waist_m ** 2)
    electric_field = np.sqrt(2 * intensity / (c * epsilon_0))
    dipole_si = dipole_au * A0 * E_CHARGE
    return dipole_si * electric_field / hbar


def _simulate_task(task_args):
    return simulate_transparency(*task_args)


def auto_select_n(atom,
                  isotope: str,
                  rf_frequency_mhz: float,
                  n_min: int,
                  n_max: int,
                  base_n: int,
                  base_np: int | None,
                  np_offsets: Tuple[int, ...],
                  verbose: bool = False) -> Tuple[int, int, SimulationStates, float]:
    """Choose the principal quantum number that best matches the requested RF frequency."""
    if n_min > n_max:
        raise ValueError(f"n_min ({n_min}) must be <= n_max ({n_max}) for auto selection.")

    target_hz = rf_frequency_mhz * 1e6
    offsets = np_offsets if len(np_offsets) > 0 else (1,)
    if base_np is not None:
        default_offset = base_np - base_n
        if default_offset not in offsets:
            offsets = tuple(offsets) + (default_offset,)

    best: Tuple[float, int, int, SimulationStates, float] | None = None

    for n_candidate in range(n_min, n_max + 1):
        for offset in offsets:
            n_p = int(n_candidate + offset)
            if n_p <= 0:
                continue
            candidate_states = build_states(isotope, n_candidate, n_p)
            rf_res_hz = atom.get_transition_frequency(candidate_states.rydberg_p, candidate_states.rydberg_d)
            detuning_hz = abs(target_hz - rf_res_hz)
            vprint(verbose, f"Checked n={n_candidate}, n_p={n_p}: RF={rf_res_hz / 1e9:.3f} GHz "
                   f"(detuning {(target_hz - rf_res_hz)/1e6:+.3f} MHz)")
            if best is None or detuning_hz < best[0]:
                best = (detuning_hz, n_candidate, n_p, candidate_states, rf_res_hz)

    if best is None:
        raise ValueError("Failed to identify a suitable (n, n_p) pair in the requested range.")

    _, n_selected, np_selected, states, rf_res_hz = best
    detuning_mhz = (target_hz - rf_res_hz) / 1e6
    cprint(f"Auto-selected n={n_selected}, n_p={np_selected} (offset {np_selected - n_selected}) "
           f"with RF resonance {rf_res_hz / 1e9:.3f} GHz (detuning {detuning_mhz:+.3f} MHz).")
    if n_selected in (n_min, n_max):
        cprint("Warning: best match hits the n search boundary; consider expanding --n-min/--n-max.")
    return n_selected, np_selected, states, rf_res_hz


def add_couplings(cell: Cell,
                  states: SimulationStates,
                  probe_detuning_scan_mrad: np.ndarray,
                  probe_rabi_mrad: float,
                  control_rabi_mrad: float,
                  control_detuning_mrad: float,
                  rf_rabi_mrad: float,
                  rf_detuning_mrad: float,
                  probe_kunit,
                  control_kunit) -> None:
    probe = dict(
        states=(states.ground, states.intermediate),
        detuning=probe_detuning_scan_mrad,
        rabi_frequency=probe_rabi_mrad,
        label="probe",
        kunit=probe_kunit,
    )
    control = dict(
        states=(states.intermediate, states.rydberg_d),
        detuning=control_detuning_mrad,
        rabi_frequency=control_rabi_mrad,
        label="control",
        kunit=control_kunit,
    )
    rf_coupling = dict(
        states=(states.rydberg_p, states.rydberg_d),  # ensure descending energy order
        detuning=rf_detuning_mrad,
        rabi_frequency=rf_rabi_mrad,
        label="rf",
    )
    cell.add_couplings(probe, control, rf_coupling)
    cell.probe_tuple = (states.ground, states.intermediate)


def report_transition_frequencies(atom, states: SimulationStates, rf_frequency_input_mhz: float,
                                  quiet: bool = False) -> Dict[str, float]:
    """Compute and display the relevant transition frequencies."""
    probe_freq_hz = atom.get_transition_frequency(states.ground, states.intermediate)
    control_freq_hz = atom.get_transition_frequency(states.intermediate, states.rydberg_d)
    rf_res_hz = atom.get_transition_frequency(states.rydberg_p, states.rydberg_d)

    probe_lambda_nm = atom.get_transition_wavelength(states.ground, states.intermediate) * 1e9
    control_lambda_nm = atom.get_transition_wavelength(states.intermediate, states.rydberg_d) * 1e9

    rf_detuning_hz = rf_frequency_input_mhz * 1e6 - rf_res_hz

    if not quiet:
        cprint(f"Probe transition:   {probe_freq_hz / 1e12:.6f} THz  (~{probe_lambda_nm:.2f} nm)")
        cprint(f"Coupling transition:{control_freq_hz / 1e12:.6f} THz  (~{control_lambda_nm:.2f} nm)")
        cprint(f"RF transition:      {rf_res_hz / 1e9:.3f} GHz")
        cprint(f"RF detuning:        {rf_detuning_hz / 1e6:+.3f} MHz from resonance")
        cprint(f"Recommended coupling laser frequency for n={states.rydberg_d.n}: "
               f"{control_freq_hz / 1e12:.6f} THz")
    return {
        "probe_freq_hz": probe_freq_hz,
        "control_freq_hz": control_freq_hz,
        "rf_res_hz": rf_res_hz,
        "rf_detuning_hz": rf_detuning_hz,
        "control_lambda_nm": control_lambda_nm,
        "probe_lambda_nm": probe_lambda_nm,
        "probe_lambda_nm": probe_lambda_nm,
    }


def simulate_transparency(args: argparse.Namespace,
                          states: SimulationStates,
                          probe_rabi_mrad: float,
                          control_rabi_mrad: float,
                          rf_field_v_cm: float,
                          probe_detuning_mhz: np.ndarray,
                          probe_detuning_mrad: np.ndarray,
                          rf_detuning_mrad: float,
                          probe_kunit,
                          control_kunit,
                          doppler_enabled: bool,
                          doppler_kwargs: Dict) -> Tuple[np.ndarray, float, float]:
    """Run a single steady-state solve for a given RF amplitude."""
    with ARC_DATA_LOCK:
        cell = Cell(
            args.isotope,
            list(states.as_dict().values()),
            cell_length=args.cell_length,
            gamma_transit=2 * np.pi * args.transit_rate,
            temp=args.temperature,
        )
        apply_pressure_override(cell, args)
    if doppler_enabled:
        cell.vP = rubidium_most_probable_speed(args.isotope, args.temperature)

    dipole_au = cell.atom.get_dipole_matrix_element(states.rydberg_p, states.rydberg_d, q=0)
    dipole_coulomb_meter = dipole_au * A0 * E_CHARGE
    rf_field_v_per_m = rf_field_v_cm * 100.0
    rf_rabi_mrad = compute_rf_rabi(dipole_coulomb_meter, rf_field_v_per_m)
    vprint(args.verbose,
           f"  RF dipole moment={dipole_coulomb_meter:.3e} C·m (raw {dipole_au:.3e} a0·e), "
           f"Ω_RF={rf_rabi_mrad/(2*np.pi):.3f} MHz.")
    add_couplings(
        cell,
        states,
        probe_detuning_mrad,
        probe_rabi_mrad=probe_rabi_mrad,
        control_rabi_mrad=control_rabi_mrad,
        control_detuning_mrad=2 * np.pi * args.control_detuning,
        rf_rabi_mrad=rf_rabi_mrad,
        rf_detuning_mrad=rf_detuning_mrad,
        probe_kunit=probe_kunit,
        control_kunit=control_kunit,
    )
    if args.probe_linewidth > 0:
        cell.add_self_broadening(states.intermediate, 2 * np.pi * args.probe_linewidth, label="probe_lw")
    if args.control_linewidth > 0:
        cell.add_self_broadening(states.intermediate, 2 * np.pi * (0.5 * args.control_linewidth), label="ctrl_lw_e")
        cell.add_self_broadening(states.rydberg_d, 2 * np.pi * args.control_linewidth, label="ctrl_lw_r")
        cell.add_self_broadening(states.rydberg_p, 2 * np.pi * args.control_linewidth, label="ctrl_lw_rp")

    with ARC_DATA_LOCK:
        solution = rq.solve_steady_state(
            cell,
            doppler=doppler_enabled,
            doppler_mesh_method=doppler_kwargs if doppler_enabled else None,
        )
    transmission = np.real(solution.get_transmission_coef())
    rf_rabi_mhz = rf_rabi_mrad / (2 * np.pi)
    vprint(args.verbose, "  Steady-state solve complete.")
    return transmission, rf_rabi_mhz, rf_rabi_mrad


def peak_spacing(detunings_mhz: np.ndarray, transmission: np.ndarray) -> float:
    """Return spacing between the dominant positive and negative detuning peaks."""
    peak_indices, _ = find_peaks(transmission)
    if peak_indices.size == 0:
        return np.nan

    negatives = [idx for idx in peak_indices if detunings_mhz[idx] < 0]
    positives = [idx for idx in peak_indices if detunings_mhz[idx] > 0]

    if not negatives or not positives:
        return np.nan

    neg_idx = max(negatives, key=lambda i: transmission[i])
    pos_idx = max(positives, key=lambda i: transmission[i])

    return abs(detunings_mhz[pos_idx] - detunings_mhz[neg_idx])


def _double_gaussian(det, center, splitting, width, amp1, amp2, offset):
    half = splitting / 2.0
    term1 = amp1 * np.exp(-0.5 * ((det - (center - half)) / width) ** 2)
    term2 = amp2 * np.exp(-0.5 * ((det - (center + half)) / width) ** 2)
    return term1 + term2 + offset


def _double_lorentzian(det, center, splitting, width, amp1, amp2, offset):
    half = splitting / 2.0
    term1 = amp1 / (1.0 + ((det - (center - half)) / width) ** 2)
    term2 = amp2 / (1.0 + ((det - (center + half)) / width) ** 2)
    return term1 + term2 + offset


def _gaussian(det, center, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((det - center) / sigma) ** 2)


def _lorentzian(det, center, gamma, amplitude):
    return amplitude / (1.0 + ((det - center) / gamma) ** 2)


def fit_peak_spacing(detunings_mhz: np.ndarray,
                     transmission: np.ndarray,
                     guess_split: float | None,
                     baseline: np.ndarray | None = None,
                     profile: str = "gaussian"):
    if detunings_mhz.size < 5:
        return np.nan, None

    if guess_split is None or np.isnan(guess_split) or guess_split <= 0:
        guess_split = max(1.0, 0.1 * (detunings_mhz.max() - detunings_mhz.min()))

    fit_data = transmission if baseline is None else transmission - baseline

    sigma_guess = max(0.5, guess_split / 4.0)
    amp_guess = max(1e-3, fit_data.max() - fit_data.min())
    offset_guess = fit_data.min()

    p0 = [0.0, guess_split, sigma_guess, amp_guess, amp_guess * 0.8, offset_guess]
    bounds = (
        (-np.inf, 0.0, 1e-3, -np.inf, -np.inf, -np.inf),
        (np.inf, (detunings_mhz.max() - detunings_mhz.min()), np.inf, np.inf, np.inf, np.inf),
    )

    try:
        profile_func = _double_gaussian if profile == "gaussian" else _double_lorentzian
        popt, _ = curve_fit(
            profile_func,
            detunings_mhz,
            fit_data,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )
        return abs(popt[1]), popt
    except Exception:
        return np.nan, None


def state_lifetime_linewidth(atom, state: A_QState, temperature: float) -> float:
    """Return the natural linewidth (FWHM) in MHz for a given state."""
    try:
        lifetime = atom.getStateLifetime(state.n, state.l, state.j, temperature=temperature)
    except Exception:
        lifetime = None
    if not lifetime or lifetime <= 0:
        return 0.0
    return 1.0 / (2 * np.pi * lifetime) / 1e6


def estimate_coherence_rates(atom,
                             states: SimulationStates,
                             transit_rate_mhz: float,
                             temperature: float,
                             probe_linewidth_mhz: float,
                             control_linewidth_mhz: float) -> Tuple[float, float, float]:
    """Estimate coherence decay rates (MHz) for |g>-|e|, |g>-|r|, and |r|-|r'|."""
    gamma_e = state_lifetime_linewidth(atom, states.intermediate, temperature)
    gamma_r = state_lifetime_linewidth(atom, states.rydberg_d, temperature)
    gamma_rp = state_lifetime_linewidth(atom, states.rydberg_p, temperature)

    gamma_ge = 0.5 * gamma_e + transit_rate_mhz + probe_linewidth_mhz
    gamma_gr = (0.5 * (gamma_r + gamma_e)
                + transit_rate_mhz
                + 0.5 * (probe_linewidth_mhz + control_linewidth_mhz))
    gamma_rp_coh = 0.5 * (gamma_r + gamma_rp) + transit_rate_mhz + control_linewidth_mhz
    return gamma_ge, gamma_gr, gamma_rp_coh


def analytic_autler_townes(control_rabi_mhz: float,
                           rf_rabi_mhz: float,
                           control_detuning_mhz: float,
                           rf_detuning_mhz: float,
                           gamma_ge: float,
                           gamma_gr: float,
                           gamma_rp: float) -> float:
    """Analytic peak-to-peak spacing from cubic susceptibility roots (MHz)."""
    if control_rabi_mhz == 0:
        return np.nan

    r = 1j * gamma_ge
    p = control_detuning_mhz + 1j * gamma_gr
    q = control_detuning_mhz + rf_detuning_mhz + 1j * gamma_rp

    s = p + q
    t = p * q - (rf_rabi_mhz ** 2) / 4.0
    coeffs = np.array([
        1.0,
        s + r,
        t + r * s - (control_rabi_mhz ** 2) / 4.0,
        r * t - (control_rabi_mhz ** 2) / 4.0 * q
    ], dtype=complex)

    roots = np.roots(coeffs)
    neg = [root for root in roots if root.real <= 0.0]
    pos = [root for root in roots if root.real >= 0.0]

    if not neg or not pos:
        return np.nan

    neg_root = max(neg, key=lambda z: z.real)
    pos_root = min(pos, key=lambda z: z.real)
    return pos_root.real - neg_root.real



def main() -> None:
    args = parse_args()
    global BACKEND_MODE
    BACKEND_MODE = bool(args.backend_json)
    t_start = time.perf_counter() if args.timing else None
    states = build_states(args.isotope, args.n, args.np)
    selected_n = args.n

    vprint(args.verbose,
           f"Initialized ladder with isotope={args.isotope}, n={args.n}, n_p={args.np or args.n + 1}.")

    selected_np = args.np if args.np is not None else args.n + 1

    with ARC_DATA_LOCK:
        analysis_cell = Cell(
            args.isotope,
            list(states.as_dict().values()),
            cell_length=args.cell_length,
            gamma_transit=2 * np.pi * args.transit_rate,
            temp=args.temperature,
        )
        apply_pressure_override(analysis_cell, args)
    atom = analysis_cell.atom

    if args.auto_n:
        selected_n, selected_np, states, _ = auto_select_n(
            atom,
            args.isotope,
            args.rf_frequency,
            args.n_min,
            args.n_max,
            args.n,
            args.np,
            tuple(args.np_offsets),
            verbose=args.verbose,
        )
        args.n = selected_n
        args.np = selected_np
        with ARC_DATA_LOCK:
            analysis_cell = Cell(
                args.isotope,
                list(states.as_dict().values()),
                cell_length=args.cell_length,
                gamma_transit=2 * np.pi * args.transit_rate,
                temp=args.temperature,
            )
            apply_pressure_override(analysis_cell, args)
        atom = analysis_cell.atom
    else:
        selected_np = args.np if args.np is not None else args.n + 1

    freq_info = report_transition_frequencies(atom, states, args.rf_frequency, quiet=BACKEND_MODE)
    gas_density = analysis_cell.density
    temp_for_pressure = max(args.temperature, 1e-6)
    gas_pressure_pa = gas_density * k * temp_for_pressure
    gas_pressure_torr = gas_pressure_pa * PA_TO_TORR
    gas_pressure_1e8_torr = gas_pressure_torr * 1e8
    rf_detuning_mhz = args.rf_frequency - freq_info["rf_res_hz"] / 1e6
    rf_detuning_mrad = 2 * np.pi * rf_detuning_mhz

    if args.auto_n_only:
        cprint("Auto-n only mode selected; skipping full EIT simulation.")
        backend_payload = {
            "selected_n": selected_n,
            "selected_np": selected_np,
            "probe_freq_hz": freq_info["probe_freq_hz"],
            "control_freq_hz": freq_info["control_freq_hz"],
            "probe_lambda_nm": freq_info["probe_lambda_nm"],
            "control_lambda_nm": freq_info["control_lambda_nm"],
            "rf_res_hz": freq_info["rf_res_hz"],
            "rf_detuning_mhz": float(rf_detuning_mhz),
            "temperature_K": args.temperature,
            "pressure_override_torr": args.pressure_torr,
            "gas_density_m3": gas_density,
            "gas_pressure_torr": gas_pressure_torr,
            "baseline_amplitude_v_cm": None,
            "amplitudes": [],
            "plots": {},
            "sweep": None,
            "auto_n_only": True,
        }
        if args.backend_json:
            write_backend_file(args.backend_json, backend_payload)
        return

    probe_dipole_au = atom.get_dipole_matrix_element(states.intermediate, states.ground, q=0)
    control_dipole_au = atom.get_dipole_matrix_element(states.rydberg_d, states.intermediate, q=0)
    probe_rabi_rad = rabi_from_power(args.probe_power, args.probe_waist, probe_dipole_au)
    control_rabi_rad = rabi_from_power(args.control_power, args.control_waist, control_dipole_au)
    probe_rabi_mrad = probe_rabi_rad / 1e6
    control_rabi_mrad = control_rabi_rad / 1e6
    control_rabi_mhz = control_rabi_rad / (2 * np.pi * 1e6)
    vprint(args.verbose,
           f"Derived Ω_probe={probe_rabi_rad/(2*np.pi*1e6):.3f} MHz, Ω_control={control_rabi_mhz:.3f} MHz "
           f"from optical powers.")

    gamma_ge, gamma_gr, gamma_rp = estimate_coherence_rates(
        atom, states, args.transit_rate, args.temperature,
        args.probe_linewidth, args.control_linewidth)

    probe_detuning_mhz = np.linspace(-args.probe_span, args.probe_span, args.probe_points)
    probe_detuning_mrad = 2 * np.pi * probe_detuning_mhz

    probe_kunit = propagation_axis(args.probe_direction)
    control_kunit = propagation_axis(args.control_direction)

    doppler_enabled = args.temperature > 0
    doppler_kwargs = doppler_mesh_args(args)
    vprint(args.verbose,
           f"Doppler averaging {'enabled' if doppler_enabled else 'disabled'} "
           f"(T={args.temperature} K).")

    results = [
        _simulate_task((
            args,
            states,
            probe_rabi_mrad,
            control_rabi_mrad,
            rf_field,
            probe_detuning_mhz,
            probe_detuning_mrad,
            rf_detuning_mrad,
            probe_kunit,
            control_kunit,
            doppler_enabled,
            doppler_kwargs,
        ))
        for rf_field in args.rf_amplitudes
    ]

    raw_transmissions = np.array([res[0] for res in results])
    rf_rabi_rates = np.array([res[1] for res in results])
    rf_rabi_mrads = np.array([res[2] for res in results])

    baseline_transmission = None
    baseline_amp_used = None
    baseline_needed = args.normalize_baseline or args.baseline_rf_amplitude is not None or args.fit_peaks
    if baseline_needed and len(raw_transmissions) > 0:
        if args.baseline_rf_amplitude is not None:
            baseline_amp = args.baseline_rf_amplitude
        else:
            max_amp = max(args.rf_amplitudes) if args.rf_amplitudes else 0.0
            baseline_amp = 100.0 * max_amp
        vprint(args.verbose,
               f"Computing baseline using RF amplitude {baseline_amp:.4f} V/cm.")
        baseline_amp_used = baseline_amp
        baseline_transmission, _, _ = simulate_transparency(
            args,
            states,
            probe_rabi_mrad,
            control_rabi_mrad,
            baseline_amp,
            probe_detuning_mhz,
            probe_detuning_mrad,
            rf_detuning_mrad,
            probe_kunit,
            control_kunit,
            doppler_enabled,
            doppler_kwargs,
        )
        baseline_transmission = baseline_transmission.copy()

    fit_curves = raw_transmissions.copy()
    if args.normalize_baseline and baseline_transmission is not None:
        fit_curves = fit_curves - baseline_transmission

    fit_baseline_curve = None
    if baseline_transmission is not None and not args.normalize_baseline:
        fit_baseline_curve = baseline_transmission

    if args.normalize_baseline:
        transmissions = fit_curves.copy()
        if baseline_transmission is not None:
            norm_msg = "custom RF amplitude" if args.baseline_rf_amplitude else "largest RF amplitude"
            vprint(args.verbose, f"Normalized transmission curves to baseline ({norm_msg}).")
    else:
        transmissions = raw_transmissions.copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    amplitude_results = []
    for idx, rf_field in enumerate(args.rf_amplitudes):
        transmission = transmissions[idx]
        fitting_curve = fit_curves[idx]
        rf_rabi_mhz = rf_rabi_rates[idx]
        expected_split_mhz = analytic_autler_townes(
            control_rabi_mhz=control_rabi_mhz,
            rf_rabi_mhz=rf_rabi_mhz,
            control_detuning_mhz=args.control_detuning,
            rf_detuning_mhz=rf_detuning_mhz,
            gamma_ge=gamma_ge,
            gamma_gr=gamma_gr,
            gamma_rp=gamma_rp,
        )
        df_value = (freq_info["control_lambda_nm"] / freq_info["probe_lambda_nm"]) * rf_rabi_mhz
        measured_spacing = peak_spacing(probe_detuning_mhz, fitting_curve)
        spacing_text = "Δ_peak=NA" if np.isnan(measured_spacing) else f"Δ_peak={measured_spacing:.2f} MHz"
        if np.isnan(expected_split_mhz):
            expected_text = f"Δ_AT=NA (Df={df_value:.2f} MHz)"
        else:
            expected_text = f"Δ_AT={expected_split_mhz:.2f} MHz (Df={df_value:.2f} MHz)"
        fit_text = ""
        fit_params = None
        fit_spacing_value = np.nan
        if args.fit_peaks:
            guess_for_fit = measured_spacing if not np.isnan(measured_spacing) else expected_split_mhz
            if guess_for_fit is None or np.isnan(guess_for_fit) or guess_for_fit <= 0:
                guess_for_fit = df_value
            baseline_for_fit = None if args.normalize_baseline else fit_baseline_curve
            fit_spacing, fit_params = fit_peak_spacing(
                probe_detuning_mhz,
                fitting_curve,
                guess_for_fit,
                baseline=baseline_for_fit,
                profile=args.fit_profile,
            )
            fit_spacing_value = fit_spacing
            fit_text = ", Δ_fit=NA" if np.isnan(fit_spacing) else f", Δ_fit={fit_spacing:.2f} MHz"

        label = (f"E_RF={rf_field*1e3:.1f} mV/cm, "
                 f"{expected_text}, {spacing_text}{fit_text}")
        ax.plot(probe_detuning_mhz, transmission, label=label)

        if args.fit_peaks and fit_params is not None:
            center, splitting, width_param, amp1, amp2, offset = fit_params
            half = splitting / 2.0
            profile_component = _gaussian if args.fit_profile == "gaussian" else _lorentzian
            comp1 = profile_component(probe_detuning_mhz, center - half, width_param, amp1)
            comp2 = profile_component(probe_detuning_mhz, center + half, width_param, amp2)
            baseline_plot = fit_baseline_curve if fit_baseline_curve is not None else 0.0
            total_fit = comp1 + comp2 + offset + baseline_plot
            ax.plot(probe_detuning_mhz, total_fit, "--", color="black", linewidth=1, alpha=0.7)
        amplitude_results.append({
            "rf_field_v_cm": float(rf_field),
            "rf_rabi_mhz": float(rf_rabi_mhz),
            "df_mhz": float(df_value),
            "expected_split_mhz": f_or_none(expected_split_mhz),
            "delta_peak_mhz": f_or_none(measured_spacing),
            "delta_fit_mhz": f_or_none(fit_spacing_value),
        })

    ax.set_xlabel("Probe detuning (MHz)")
    ax.set_ylabel("Transmission")
    ax.set_title(
        f"{args.isotope} Rydberg EIT @ n={selected_n} ({freq_info['control_lambda_nm']:.1f} nm control) "
        f"T={args.temperature:.0f} K, P≈{gas_pressure_1e8_torr:.2f} x10^-8 Torr, n≈{gas_density:.2e} m⁻³"
    )
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    cprint(f"Saved Autler-Townes plot to {args.output}")

    sweep_payload = None
    if args.sweep_plot:
        rf_min = min(args.rf_amplitudes)
        rf_max = max(args.rf_amplitudes)
        if np.isclose(rf_min, rf_max):
            cprint("Sweep plot skipped: RF amplitudes span zero range.")
        else:
            sweep_fields = np.linspace(rf_min, rf_max, max(2, args.sweep_points))
            sweep_splittings = []
            df_splittings = []
            cprint(f"Running sweep plot over {len(sweep_fields)} RF amplitudes...")
            for rf_field in sweep_fields:
                transmission, rf_rabi_mhz, _ = simulate_transparency(
                    args,
                    states,
                    probe_rabi_mrad,
                    control_rabi_mrad,
                    rf_field,
                    probe_detuning_mhz,
                    probe_detuning_mrad,
                    rf_detuning_mrad,
                    probe_kunit,
                    control_kunit,
                    doppler_enabled,
                    doppler_kwargs,
                )
                sweep_curve = transmission.copy()
                if args.normalize_baseline and baseline_transmission is not None:
                    sweep_curve = sweep_curve - baseline_transmission
                measured_spacing = peak_spacing(probe_detuning_mhz, sweep_curve)
                df_value = (freq_info["control_lambda_nm"] / freq_info["probe_lambda_nm"]) * rf_rabi_mhz
                df_splittings.append(df_value)
                guess_for_fit = measured_spacing if not np.isnan(measured_spacing) else df_value
                if guess_for_fit is None or np.isnan(guess_for_fit) or guess_for_fit <= 0:
                    guess_for_fit = (freq_info["control_lambda_nm"] / freq_info["probe_lambda_nm"]) * rf_rabi_mhz
                baseline_for_fit = None if args.normalize_baseline else fit_baseline_curve
                fit_spacing, _ = fit_peak_spacing(
                    probe_detuning_mhz,
                    sweep_curve,
                    guess_for_fit,
                    baseline=baseline_for_fit,
                    profile=args.fit_profile,
                )
                if np.isnan(fit_spacing):
                    fit_spacing = measured_spacing
                if np.isnan(fit_spacing):
                    fit_spacing = np.nan
                sweep_splittings.append(fit_spacing)
            fig_sweep, ax_sweep = plt.subplots(figsize=(7, 4))
            ax_sweep.plot(sweep_fields, sweep_splittings, "o-", label="Δ_fit")
            ax_sweep.plot(sweep_fields, df_splittings, "--", label="Ω_RF * λ_c/λ_p expectation")
            ax_sweep.set_xlabel("RF amplitude (V/cm)")
            ax_sweep.set_ylabel("Splitting (MHz)")
            ax_sweep.set_title(f"Δ_fit vs RF amplitude (n={selected_n})")
            ax_sweep.grid(True)
            ax_sweep.legend()
            fig_sweep.tight_layout()
            fig_sweep.savefig(args.sweep_output, dpi=300)
            cprint(f"Saved sweep plot to {args.sweep_output}")
            sweep_payload = {
                "fields_v_cm": [float(val) for val in sweep_fields],
                "delta_fit_mhz": [f_or_none(val) for val in sweep_splittings],
                "df_mhz": [f_or_none(val) for val in df_splittings],
                "plot": str(Path(args.sweep_output).resolve()),
            }

    if args.timing and t_start is not None:
        elapsed = time.perf_counter() - t_start
        cprint(f"Simulation completed in {elapsed:.2f} s.")

    if not args.no_show:
        backend = plt.get_backend().lower()
        non_interactive_backends = {"agg", "pdf", "ps", "svg", "cairo", "template"}
        if backend in non_interactive_backends:
            cprint(f"Matplotlib backend '{backend}' is non-interactive; skipping GUI display.")
            cprint("Re-run with an interactive backend or pass --no-show (already implied here).")
        else:
            plt.show()

    if args.backend_json:
        plots_dict = {"transmission": str(Path(args.output).resolve())}
        if sweep_payload:
            plots_dict["sweep"] = sweep_payload["plot"]
        backend_payload = {
            "selected_n": selected_n,
            "selected_np": selected_np,
            "probe_freq_hz": freq_info["probe_freq_hz"],
            "control_freq_hz": freq_info["control_freq_hz"],
            "probe_lambda_nm": freq_info["probe_lambda_nm"],
            "control_lambda_nm": freq_info["control_lambda_nm"],
            "rf_res_hz": freq_info["rf_res_hz"],
            "rf_detuning_mhz": float(rf_detuning_mhz),
            "temperature_K": args.temperature,
            "pressure_override_torr": args.pressure_torr,
            "gas_density_m3": gas_density,
            "gas_pressure_torr": gas_pressure_torr,
            "baseline_amplitude_v_cm": baseline_amp_used,
            "amplitudes": amplitude_results,
            "plots": plots_dict,
            "sweep": sweep_payload,
            "auto_n_only": False,
        }
        write_backend_file(args.backend_json, backend_payload)

if __name__ == "__main__":
    main()
