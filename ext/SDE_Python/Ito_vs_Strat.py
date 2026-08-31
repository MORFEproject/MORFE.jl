import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
np.random.seed(42)
T = 4.0                     # total time
dt_fine = 0.01             # fine step for smooth plotting
N_fine = int(T / dt_fine)

# Generate a fine Brownian path
dW_fine = np.sqrt(dt_fine) * np.random.randn(N_fine)
W_fine = np.zeros(N_fine + 1)
W_fine[1:] = np.cumsum(dW_fine)
t_fine = np.arange(N_fine + 1) * dt_fine

# --- Coarse integration grid (downsample) ---
scale = 1                  # every 20th fine step becomes one coarse step
dt_coarse = scale * dt_fine
N_coarse = int(T / dt_coarse)

# Pre-allocate coarse arrays
Ito_int = np.zeros(N_coarse + 1)
Strat_int = np.zeros(N_coarse + 1)
Ito_iterates = np.zeros(N_coarse)
Strat_iterates = np.zeros(N_coarse)
times_coarse = np.arange(N_coarse + 1) * dt_coarse

# Compute integrals on the coarse grid
for k in range(N_coarse):
    # index in the fine array corresponding to the left boundary of this coarse step
    i = k * scale
    dWi = dW_fine[i]               # increment over the coarse step
    Wi = W_fine[i]                 # W at left
    W_next = W_fine[i + scale]     # W at right (since we skip `scale` fine steps)
    
    # Iterates
    Ito_iterates[k] = Wi
    Strat_iterates[k] = (Wi + W_next) / 2.0
    
    # Increments
    Ito_inc = Wi * dWi
    Strat_inc = Strat_iterates[k] * dWi
    
    # Cumulative integrals
    Ito_int[k+1] = Ito_int[k] + Ito_inc
    Strat_int[k+1] = Strat_int[k] + Strat_inc

# Difference and theory
diff = Strat_int - Ito_int
theory_diff = 0.5 * times_coarse

# --- Plotting (2×2) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Brownian motion (fine)
ax1 = axes[0, 0]
ax1.plot(t_fine, W_fine, 'k-', lw=1.5)
ax1.set_title('Brownian path W(t) (fine resolution)')
ax1.set_xlabel('Time t')
ax1.set_ylabel('W(t)')
ax1.grid(True)

# 2. Iterates on the coarse grid
ax2 = axes[0, 1]
step_subset = 3   # show every 3rd coarse point to avoid clutter
idx = np.arange(0, N_coarse, step_subset)
ax2.scatter(times_coarse[idx], Ito_iterates[idx], s=50, color='blue',
            label='Itô iterate: W(tᵢ)', zorder=3)
ax2.scatter(times_coarse[idx] + dt_coarse/2, Strat_iterates[idx], s=50, color='red',
            label='Strat iterate: W(tᵢ+Δt/2)', zorder=3, marker='s')
# Connect the pairs with vertical dashed lines
for j in idx:
    ax2.plot([times_coarse[j], times_coarse[j] + dt_coarse/2],
             [Ito_iterates[j], Strat_iterates[j]],
             'k--', linewidth=0.8, alpha=0.3)
ax2.set_title('Evaluation points per coarse step (iterates)')
ax2.set_xlabel('Time')
ax2.set_ylabel('W at evaluation point')
ax2.legend()
ax2.grid(True)

# 3. Cumulative integrals
ax3 = axes[1, 0]
ax3.plot(times_coarse, Ito_int, 'b-', lw=2, label='Itô integral')
ax3.plot(times_coarse, Strat_int, 'r-', lw=2, label='Stratonovich integral')
# Theoretical curves (using the coarse path values for a fair comparison)
W_coarse = W_fine[::scale]   # take every `scale`-th point (coarse grid)
ax3.plot(times_coarse, 0.5 * (W_coarse**2 - times_coarse), 'b--', lw=1.5, alpha=0.5, label='Ito theory')
ax3.plot(times_coarse, 0.5 * W_coarse**2, 'r--', lw=1.5, alpha=0.5, label='Strat theory')
ax3.set_title('Cumulative integrals')
ax3.set_xlabel('Time t')
ax3.set_ylabel('Integral value')
ax3.legend()
ax3.grid(True)

# 4. Cumulative difference (Strat - Ito)
ax4 = axes[1, 1]
ax4.plot(times_coarse, diff, 'g-', lw=2, label='Simulated S(t) - I(t)')
ax4.plot(times_coarse, theory_diff, 'k--', lw=2, label='Theory: t/2')
ax4.fill_between(times_coarse, 0, diff, alpha=0.2, color='green')
ax4.set_title('Accumulated effect of the iterate choice')
ax4.set_xlabel('Time t')
ax4.set_ylabel('S(t) - I(t)')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show()

# --- Print final values ---
print(f"Final Itô integral       = {Ito_int[-1]:.4f}")
print(f"Final Stratonovich integral = {Strat_int[-1]:.4f}")
print(f"Difference S - I        = {diff[-1]:.4f}")
print(f"Theory t/2              = {T/2:.4f}")