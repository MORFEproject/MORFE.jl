import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial

# ============================================
# 1. Load monomial-level CSV
# ============================================
mono_file = "benchmark_results/benchmark_per_monomial.csv"
df_mono = pd.read_csv(mono_file)

# Compute total allocated bytes per monomial (RHS + solve)
df_mono['monomial_total_alloc_bytes'] = (
    df_mono['rhs_alloc_bytes'].fillna(0) +
    df_mono['solve_alloc_bytes'].fillna(0)
)

# Global monomial index (1‑based row number)
global_index = df_mono['monomial_idx']

# Cumulative memory allocation (no extra order overhead)
cumulative_mono = df_mono['monomial_total_alloc_bytes'].cumsum().values
total_cumulative = cumulative_mono   # all memory is from monomials

# ============================================
# 2. Find order boundaries (last monomial of each order)
# ============================================
# Last row index (1‑based) per order
order_end_indices = df_mono.groupby('order').apply(
    lambda g: g.index[-1] + 1   # convert 0‑based to 1‑based
).values

# Cumulative memory at those endpoints (for polynomial fitting)
y_memory = total_cumulative[order_end_indices - 1]

# ============================================
# 3. Polynomial interpolation at order endpoints
# ============================================
x_endpoints = order_end_indices
poly_memory = Polynomial.fit(global_index, cumulative_mono, deg=3)
print(f"Polynomial coefficients (cumulative memory, bytes): {poly_memory.coef}")

# Smooth x for plotting the polynomial fit
x_smooth = np.linspace(4, max(global_index), 500)

# ============================================
# 4. Plot (memory in GB)
# ============================================
fig, ax1 = plt.subplots(figsize=(12, 6))

# Fill area under total cumulative memory (convert bytes to GB)
ax1.fill_between(global_index, total_cumulative / (1024**3), 0,
                 color='bisque', alpha=1.0, label='Total cumulative memory (GB)')
ax1.plot(global_index, total_cumulative / (1024**3),
         color='darkorange', linewidth=2, label='Total cumulative memory (GB)')

# Polynomial fit (shifted by +4 to align with the data)
ax1.plot(x_smooth, poly_memory(x_smooth) / (1024**3),
         color='darkorange', linestyle='--', linewidth=2,
         label=f'Polynomial fit (deg={poly_memory.degree()}) – total')

# Vertical dashed lines at order boundaries (start of each order)
# start index = previous end index + 1; we draw a line at start - 0.5
for end_idx in order_end_indices:
    start_idx = end_idx - df_mono[df_mono['order'] == df_mono.iloc[end_idx-1]['order']].shape[0] + 1
    ax1.axvline(x=start_idx + 4 - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

ax1.set_xlabel('Global monomial index')
ax1.set_ylabel('Cumulative memory (GB)', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_title('Total cumulative memory (monomials only, new benchmark)')
ax1.set_xlim(1, max(global_index))
ax1.set_ylim(0, total_cumulative.max() / (1024**3) * 1.05)
ax1.legend(loc='upper left')
ax1.grid(False)

plt.tight_layout()
plt.show()