import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
num_ions = 100
implant_E = 1000 * 1.60218e-19  # eV → J
energy_threshold = 10 * 1.60218e-19  # eV → J
Z1 = 30  # Br atomic number
Z2 = 6   # C atomic number
projectile_mass = 2 * Z1 * 1.66054e-27  # Br in kg
target_mass = 2* Z2 * 1.66054e-27      # C in kg
step_size = 1e-9  # 1 nm in meters
theta_std_deg = 1.2
N = 1e22 * 1e6  # atoms/cm³ → atoms/m³
verbose = False

def nuclear_stopping_power(E_ion, M1, M2, theta_rad):
    return ((4 * M1 * M2) / ((M1 + M2)**2)) * E_ion * (np.sin(theta_rad / 2)**2)

def electronic_stopping_power(Z1, Z2, v, N, a0=5.29177e-11):
    num = Z1**(7/6) * Z2
    denom = (Z1**(2/36) + Z2**(2/3))**(3/2)
    return (num / denom) * 4 * a0 * N * v

def velocity_from_energy(E, m):
    return np.sqrt(2 * E / m)

def simulate_ion():
    pos = np.array([0.0, 0.0])  # Start at surface
    energy = implant_E
    traj = [pos.copy()]
    direction = np.array([0.0, 1.0])  # Moving into the material (positive z)
    recoils = []

    while energy > energy_threshold:
        pos += direction * step_size
        traj.append(pos.copy())

        v = velocity_from_energy(energy, projectile_mass)
        dE_e = electronic_stopping_power(Z1, Z2, v, N) * step_size
        dE_e = 1e-3 * energy
        
        # stochastic nuclear scattering
        if np.random.rand() < 0.75:  # 75% chance per step
            angle = np.radians(np.random.normal(0, theta_std_deg))
            dE_n = nuclear_stopping_power(energy, projectile_mass, target_mass, angle)

            # scatter direction
            R = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle),  np.cos(angle)]])
            direction = R @ direction
            direction /= np.linalg.norm(direction)

            recoils.append(pos.copy())
        else:
            dE_n = 0

        if verbose:
            print(f"Position: {pos}, Energy: {energy:.2e}, dE_e: {dE_e:.2e}, dE_n: {dE_n:.2e}")
        energy -= (dE_e + dE_n)
        energy = max(0, energy)

    return np.array(traj), pos[1], np.array(recoils)

# Run simulation
all_trajectories = []
depths = []
all_recoils = []

for _ in range(num_ions):
    traj, final_depth, recoils = simulate_ion()
    all_trajectories.append(traj)
    depths.append(final_depth)
    all_recoils.extend(recoils)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Ion tracks (depth in nm)
axs[0].set_title("Ion Tracks")
for traj in all_trajectories:
    axs[0].plot(traj[:, 0]*1e9, traj[:, 1]*1e9, color='black', linewidth=0.6)
axs[0].set_xlabel("Lateral Position (nm)")
axs[0].set_ylabel("Depth (nm)")
axs[0].invert_yaxis()
axs[0].grid(True, alpha=0.2)

# Depth histogram
axs[1].hist(np.array(depths)*1e9, bins=20, color='gray', edgecolor='black')
axs[1].set_title("Implantation Depths")
axs[1].set_xlabel("Final Depth (nm)")
axs[1].set_ylabel("Number of Ions")
axs[1].grid(True, alpha=0.2)

plt.tight_layout()
plt.show()
