#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial

# ============================================
# 1. Load NEW benchmark
# ============================================
mono_new = pd.read_csv("benchmark_results/benchmark_per_monomial.csv")
mono_new['exp_tuple'] = mono_new['exponents'].apply(
    lambda s: tuple(int(x) for x in s.split('_'))
)

# ============================================
# 2. Load LEGACY benchmark and compute cumulative time
# ============================================
legacy_root = "../../legacy_morfe/MORFE2.0/output/beam_o_9_eps_9_Fmode_1_2026-05-20T01_11_38"

df_mono_leg = pd.read_csv(f"{legacy_root}/benchmark_per_monomial.csv", na_values='NaN')
df_mono_leg['monomial_total_time_s'] = df_mono_leg['monomial_total_time_s'].fillna(0)
df_mono_leg['exp_tuple'] = df_mono_leg['alpha_vector'].apply(
    lambda s: tuple(int(x) for x in s.strip('[]').replace(' ', '').split(','))
)

cumul_mono_leg = df_mono_leg['monomial_total_time_s'].cumsum().values

df_order_leg = pd.read_csv(f"{legacy_root}/benchmark_per_order.csv")
df_order_leg['total_time_order'] = (
    df_order_leg['fillrhsG_time_s'] +
    df_order_leg['fillrhsH_time_s'] +
    df_order_leg['fillWf_time_s']
)
cumul_order_leg = df_order_leg['total_time_order'].cumsum().values
df_order_leg['cumulative_monomials'] = df_order_leg['n_monomials'].cumsum().values
start_indices_leg = [1] + (df_order_leg['cumulative_monomials'][:-1] + 1).tolist()

cum_order_step = np.zeros(len(df_mono_leg))
for i, start in enumerate(start_indices_leg):
    end = df_order_leg['cumulative_monomials'].iloc[i]
    cum_order_step[start - 1:end] = cumul_order_leg[i]

df_mono_leg['cumul_leg_s'] = cumul_mono_leg + cum_order_step

# ============================================
# 3. Align by multiindex — use new monomial_idx as shared x-axis
# ============================================
df_merged = mono_new[['monomial_idx', 'order', 'exp_tuple', 'cumul_time_s']].merge(
    df_mono_leg[['exp_tuple', 'cumul_leg_s']],
    on='exp_tuple', how='inner'
).sort_values('monomial_idx')

x_shared        = df_merged['monomial_idx'].values
cumul_new_s     = df_merged['cumul_time_s'].values
cumul_leg_s     = df_merged['cumul_leg_s'].values

# End-of-order points (for polynomial fits and ratio)
order_ends = df_merged.groupby('order').agg(
    x_end   = ('monomial_idx', 'last'),
    y_new   = ('cumul_time_s', 'last'),
    y_leg   = ('cumul_leg_s',  'last'),
).reset_index()
x_ends     = order_ends['x_end'].values
y_ends_new = order_ends['y_new'].values
y_ends_leg = order_ends['y_leg'].values

# ============================================
# 4. Polynomial fits (end-of-order points only, same x-axis for both)
# ============================================
poly_new = Polynomial.fit(x_ends, y_ends_new, deg=3)
poly_leg = Polynomial.fit(x_ends, y_ends_leg, deg=3)
x_smooth = np.linspace(x_shared.min(), x_shared.max(), 500)

print(f"Poly fit new (deg={poly_new.degree()}): {poly_new.coef}")
print(f"Poly fit leg (deg={poly_leg.degree()}): {poly_leg.coef}")

# ============================================
# 5. Plot
# ============================================
fig, ax = plt.subplots(figsize=(13, 6))

# --- Legacy (blue) ---
ax.fill_between(x_shared, cumul_leg_s / 60, 0, color='lightsteelblue', alpha=0.6)
ax.plot(x_shared, cumul_leg_s / 60,
        color='steelblue', linewidth=2, label='Legacy MORFE2.0 – cumulative time')
ax.plot(x_smooth, poly_leg(x_smooth) / 60,
        color='steelblue', linestyle='--', linewidth=1.5,
        label=f'Poly fit (deg={poly_leg.degree()}) – legacy')

# --- New (orange) ---
ax.fill_between(x_shared, cumul_new_s / 60, 0, color='bisque', alpha=0.8)
ax.plot(x_shared, cumul_new_s / 60,
        color='darkorange', linewidth=2, label='New benchmark – cumulative time')
ax.plot(x_smooth, poly_new(x_smooth) / 60,
        color='darkorange', linestyle='--', linewidth=1.5,
        label=f'Poly fit (deg={poly_new.degree()}) – new')

# Order boundary lines (after each order except the last)
for x_end in x_ends[:-1]:
    ax.axvline(x=x_end + 0.5, color='grey', linestyle='--', alpha=0.35, linewidth=0.8)

# ============================================
# Right axis: ratio legacy / new at order endpoints
# ============================================
ax2 = ax.twinx()
ratio_at_ends = y_ends_leg / np.where(y_ends_new > 0, y_ends_new, np.nan)

ax2.scatter(x_ends, ratio_at_ends,
            color='mediumseagreen', s=60, zorder=5, label='Ratio legacy / new (at order end)')
ax2.plot(x_ends, ratio_at_ends,
         color='mediumseagreen', linewidth=1.2, linestyle='-', alpha=0.6)
#ax2.axhline(y=1.0, color='mediumseagreen', linestyle=':', linewidth=1.0, alpha=0.7)
ax2.set_ylabel('Ratio legacy / new (cumulative time)', color='mediumseagreen')
ax2.tick_params(axis='y', labelcolor='mediumseagreen')
ax2.set_ylim(0, np.nanmax(ratio_at_ends) * 1.2)

ax.set_xlabel('Global monomial index')
ax.set_ylabel('Cumulative time (minutes)')
ax.set_title('Benchmark comparison: Legacy MORFE2.0 vs New BenchmarkFerrite')
ax.set_xlim(x_shared.min() - 1, x_shared.max() + 1)
ax.set_ylim(0, max(cumul_new_s.max(), cumul_leg_s.max()) / 60 * 1.1)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax.grid(False)

plt.tight_layout()
plt.show()

# ============================================
# 6. Separate plot: ratio vs expansion order
# ============================================
orders = order_ends['order'].values

fig2, ax3 = plt.subplots(figsize=(7, 4))
ax3.bar(orders, ratio_at_ends, color='mediumseagreen', alpha=0.8, width=0.6)
ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=1.0)
ax3.set_xlabel('Expansion order')
ax3.set_ylabel('Cumulative time ratio\nlegacy / new')
ax3.set_title('Speedup: Legacy MORFE2.0 vs New BenchmarkFerrite')
ax3.set_xticks(orders)
ax3.set_ylim(0, np.nanmax(ratio_at_ends) * 1.2)
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
# %%
