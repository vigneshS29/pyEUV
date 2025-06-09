import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
num_ions = 200
initial_energy_keV = 50.0
energy_threshold = 0.5
projectile_mass = 1.0
target_mass = 28.0
step_size = 1.0
theta_std_deg = 10

def get_stopping_powers(E_keV):
    S_e = 0.02 * np.sqrt(E_keV)
    S_n = 0.1 / (np.sqrt(E_keV) + 1e-6)
    return S_e, S_n

def velocity_from_energy(E, m):
    return np.sqrt(2 * E / m)

def simulate_ion():
    pos = np.array([0.0, 0.0])
    energy = initial_energy_keV
    traj = [pos.copy()]
    direction = np.array([0.0, 1.0])
    recoils = []

    while energy > energy_threshold:
        pos += direction * step_size
        traj.append(pos.copy())
        S_e, S_n = get_stopping_powers(energy)
        v = velocity_from_energy(energy, projectile_mass)
        dE_e = S_e * step_size
        dE_n = S_n * step_size

        if np.random.rand() < 0.2:
            angle = np.radians(np.random.normal(0, theta_std_deg))
            R = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle),  np.cos(angle)]])
            direction = R @ direction
            direction /= np.linalg.norm(direction)
            recoils.append(pos.copy())

        energy -= (dE_e + dE_n)

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

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Ion tracks
axs[0].set_title("Ion Tracks")
for traj in all_trajectories:
    axs[0].plot(traj[:, 0], traj[:, 1], color='black', linewidth=0.6)
axs[0].set_xlabel("Lateral Position")
axs[0].set_ylabel("Depth")
axs[0].invert_yaxis()
axs[0].grid(True,alpha=0.2)

# Histogram
axs[1].hist(depths, bins=20, color='gray', edgecolor='black')
axs[1].set_title("Implantation Depths")
axs[1].set_xlabel("Final Depth")
axs[1].set_ylabel("Number of Ions")
axs[1].grid(True,alpha=0.2)

plt.tight_layout()
plt.show()
