#%%
# 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial

# ============================================
# 1. Load monomial-level CSV (new format)
# ============================================
mono_file = "benchmark_results/benchmark_per_monomial.csv"
df_mono = pd.read_csv(mono_file)

# Global monomial index = row number (1‑based)
global_index = df_mono['monomial_idx']

# Total cumulative time is already provided (seconds)
total_cumulative_s = df_mono['cumul_time_s'].values

# ============================================
# 2. Find order boundaries
# ============================================
# Last row index (global_index) for each order
order_end_indices = df_mono.groupby('order').apply(
    lambda g: g.index[-1] + 1   # convert 0‑based index to 1‑based global index
).values

# Cumulative time at the end of each order (for polynomial fitting)
y_orange = global_index   # -1 because index is 0‑based

# ============================================
# 3. Polynomial fit at order endpoints
# ============================================
poly_orange = Polynomial.fit(global_index, total_cumulative_s, deg=3)
print(f"Polynomial coefficients (total cumulative): {poly_orange.coef}")

# Smooth x for plotting the fit
x_smooth = np.linspace(1, max(global_index), 500)

# ============================================
# 4. Plot
# ============================================
fig, ax1 = plt.subplots(figsize=(12, 6))

# Total cumulative time (orange area)
ax1.fill_between(global_index + 4, total_cumulative_s / 60, 0, color='bisque', alpha=1.0)
ax1.plot(global_index + 4, total_cumulative_s / 60,
         color='darkorange', linewidth=2, label='Total cumulative time (minutes)')

# Polynomial fit (shifted +4)
ax1.plot(x_smooth + 4, poly_orange(x_smooth) / 60,
         color='darkorange', linestyle='--', linewidth=2,
         label=f'Polynomial fit (deg={poly_orange.degree()}) – total')

# Vertical dashed lines at order boundaries
for end_idx in order_end_indices:
    # start of each order = previous end + 1; here we draw at the start (end_idx - 0.5)
    ax1.axvline(x=end_idx + 4 - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

ax1.set_xlabel('Global monomial index')
ax1.set_ylabel('Cumulative time (minutes)', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_title('Total cumulative time (monomials only, new benchmark)')
ax1.set_xlim(1, max(global_index) + 4)
ax1.set_ylim(0, total_cumulative_s.max() / 60 * 1.05)
ax1.legend(loc='upper left')
ax1.grid(False)

plt.tight_layout()
plt.show()
# %%
