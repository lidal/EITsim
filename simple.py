import numpy as np
import matplotlib.pyplot as plt
import rydiqule as rq
from rydiqule import Cell
from rydiqule.atom_utils import A_QState
from scipy.constants import physical_constants, pi, hbar

# Define the atomic system
atom = "Rb85"
r_level_n = 40

# Define the states using A_QState for rydiqule.Cell
# This allows automatic calculation of spontaneous decay rates
states = {
    "g": A_QState(n=5, l=0, j=0.5, f=3, m_f=0),
    "e": A_QState(n=5, l=1, j=1.5, f=4, m_f=0),  # 5P3/2
    "r": A_QState(n=r_level_n, l=2, j=2.5, m_j=0.5),  # 40D5/2
    "r_prime": A_QState(n=r_level_n + 1, l=1, j=1.5, m_j=0.5),  # 41P3/2
}

# Define the laser detunings in MHz
probe_detunings_mhz = np.linspace(-20, 20, 201)

# Define the lasers
probe_laser = {
    "states": (states["g"], states["e"]),
    "rabi_frequency": 2 * np.pi * 5,  # Mrad/s
    "detuning": 2 * np.pi * probe_detunings_mhz,  # Mrad/s
    "q": 0,  # Assume pi-polarization
}
coupling_laser = {
    "states": (states["e"], states["r"]),
    "rabi_frequency": 2 * np.pi * 10,  # Mrad/s
    "detuning": 0,
    "q": 0,  # Assume pi-polarization
}

# Define the RF electric field strengths to loop over in V/cm
rf_E_strengths_V_cm = [0, 0.005, 0.01, 0.02]

plt.figure()

# Calculate dipole moment for the RF transition to convert E-field to Rabi frequency.
# This requires an `arc` atom object, which we can get from a `rydiqule.Cell`.
temp_cell = Cell(atom, list(states.values()))

# get_dipole_matrix_element returns value in units of a_0*e
dme = temp_cell.atom.get_dipole_matrix_element(states["r_prime"], states["r"], q=0)

# Convert to SI units (C*m)
a0 = physical_constants["Bohr radius"][0]
e_charge = physical_constants["elementary charge"][0]
d_si = dme * a0 * e_charge

for rf_E_V_cm in rf_E_strengths_V_cm:
    # Create the Cell. This automatically includes spontaneous decay
    # from all excited states, making the model more physical.
    # We also pass the transit broadening and cell length to the constructor.
    s = Cell(atom, list(states.values()), cell_length=10e-3,
             gamma_transit=2 * np.pi * 0.1)

    # Convert E-field in V/cm to Rabi frequency in Mrad/s for the solver
    # Omega (rad/s) = d_SI * E_SI / hbar
    E_V_m = rf_E_V_cm * 100
    rabi_frequency_rad_s = d_si * E_V_m / hbar
    rabi_frequency_Mrad_s = rabi_frequency_rad_s / 1e6

    # Define the RF field coupling
    rf_field = {
        # Note: 41P is lower energy than 40D, so r_prime must be the first state
        "states": (states["r_prime"], states["r"]),
        "rabi_frequency": rabi_frequency_Mrad_s,
        "detuning": 0,
        "q": 0,  # Assume pi-polarization
    }

    s.add_couplings(probe_laser, coupling_laser, rf_field)

    # Solve the system
    solution = rq.solve_steady_state(s)

    # Plot the results
    # Get the transmission of the probe laser through the cell
    transmission = solution.get_transmission_coef()
    plt.plot(probe_detunings_mhz, transmission, label=f"RF E-field = {rf_E_V_cm:.3f} V/cm")

plt.xlabel("Probe Detuning (MHz)")
plt.ylabel("Transmission")
plt.title("EIT Transmission with RF E-field")
plt.legend()
plt.grid(True)
plt.savefig("eit_rf_scan.png")
print("Plot saved to eit_rf_scan.png")