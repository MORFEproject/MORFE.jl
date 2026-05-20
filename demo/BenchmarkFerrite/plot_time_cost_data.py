#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial

# ============================================
# 1. Load monomial-level CSV
# ============================================
mono_file = "benchmark_results/benchmark_per_monomial.csv"
df_mono = pd.read_csv(mono_file)

global_index = df_mono['monomial_idx'].values
total_cumulative_s = df_mono['cumul_time_s'].values

# ============================================
# 2. End-of-order points (for polynomial fit and boundary lines)
# ============================================
order_ends = df_mono.groupby('order').agg(
    x_end=('monomial_idx', 'last'),
    y_end=('cumul_time_s', 'last'),
).reset_index()
x_ends = order_ends['x_end'].values
y_ends = order_ends['y_end'].values

# ============================================
# 3. Polynomial fit at order endpoints only
# ============================================
poly_orange = Polynomial.fit(x_ends, y_ends, deg=3)
print(f"Polynomial coefficients (total cumulative): {poly_orange.coef}")

x_smooth = np.linspace(global_index.min(), global_index.max(), 500)

# ============================================
# 4. Plot
# ============================================
fig, ax1 = plt.subplots(figsize=(12, 6))

# Total cumulative time (orange area)
ax1.fill_between(global_index, total_cumulative_s / 60, 0, color='bisque', alpha=1.0)
ax1.plot(global_index, total_cumulative_s / 60,
         color='darkorange', linewidth=2, label='Total cumulative time (minutes)')

# Polynomial fit
ax1.plot(x_smooth, poly_orange(x_smooth) / 60,
         color='darkorange', linestyle='--', linewidth=2,
         label=f'Polynomial fit (deg={poly_orange.degree()}) – total')

# Vertical dashed lines at order boundaries (after each order except the last)
for x_end in x_ends[:-1]:
    ax1.axvline(x=x_end + 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

ax1.set_xlabel('Global monomial index')
ax1.set_ylabel('Cumulative time (minutes)', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_title('Total cumulative time (monomials only, new benchmark)')
ax1.set_xlim(global_index.min() - 1, global_index.max() + 1)
ax1.set_ylim(0, total_cumulative_s.max() / 60 * 1.05)
ax1.legend(loc='upper left')
ax1.grid(False)

plt.tight_layout()
plt.show()
# %%
