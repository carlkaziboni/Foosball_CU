"""
Comprehensive analysis: models_new_formation vs models_new_longtrain
Produces 11 individual images, one graph each.
Run with:  conda run -n foosballrl python3 analyze_formation_vs_longtrain.py
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Rollout data from 30-episode evaluation ───────────────────────────────────
FORMATION_REWARDS = [101.48, 191.20, 110.61, 68.54, 92.83, 88.18, 181.38,
                     69.12, 222.79, 92.31, 77.59, 60.06, 231.90, 91.69,
                     90.84, 86.63, 184.28, 92.97, 75.12, 90.95, 91.46,
                     222.59, 212.82, 113.80, 216.53, 206.40, 84.58, 181.92,
                     83.41, 197.11]
FORMATION_LENGTHS = [300] * 30

LONGTRAIN_REWARDS  = [1036.04, 106.23, 250.87, 1010.24, 140.51, 96.30,
                      101.42, 151.98, 125.15, 164.38, 183.38, 112.38,
                      226.73, 118.67, 132.37, 122.79, 92.93, 102.50,
                      96.19, 527.38, 160.36, 574.83, 92.53, 209.02,
                      110.28, 509.46, 115.60, 275.20, 447.57, 549.81]
LONGTRAIN_LENGTHS  = [1333, 300, 300, 1369, 300, 300, 300, 300, 300, 300,
                      300, 300, 300, 300, 300, 300, 300, 300, 300, 956,
                      300, 688, 300, 300, 300, 1392, 300, 300, 852, 1016]

CF = '#4C9BE8'   # formation blue
CL = '#E8844C'   # longtrain orange

WEIGHT_LAYERS = ['L1 (38→512)', 'L2 (512→512)', 'L3 (512→256)', 'Out (256→8)']
ACTOR_ABS_MEAN = {'formation': [0.2563, 0.0837, 0.0472, 0.0310],
                  'longtrain':  [0.5720, 0.1499, 0.0678, 0.0436]}
ACTOR_STD      = {'formation': [0.3998, 0.1284, 0.0743, 0.0411],
                  'longtrain':  [0.9541, 0.2460, 0.0960, 0.0637]}
CRITIC_LAYERS  = ['L1 (46→512)', 'L2 (512→512)', 'L3 (512→256)', 'Out (256→1)']
CRITIC_ABS_MEAN = {'formation': [0.3848, 0.1285, 0.0969, 0.4440],
                   'longtrain':  [3.2179, 2.7345, 4.9514, 18.3947]}

HYPERPARAMS = {
    'Timesteps':         {'formation': '500 000',       'longtrain': '1 500 000'},
    'Gradient steps':    {'formation': '2',             'longtrain': '4'},
    'Replay buffer':     {'formation': '100 000',       'longtrain': '300 000'},
    'Learning rate':     {'formation': '3e-4',          'longtrain': '3e-4'},
    'γ (gamma)':         {'formation': '0.99',          'longtrain': '0.99'},
    'τ (tau)':           {'formation': '0.005',         'longtrain': '0.005'},
    'Entropy coef':      {'formation': '0.16',          'longtrain': '~9800 (!)'},
    'Policy updates':    {'formation': '249 750',       'longtrain': '1 499 000'},
    'Network':           {'formation': '[512,512,256]', 'longtrain': '[512,512,256]'},
    'SDE':               {'formation': 'No',            'longtrain': 'No'},
    'Training platform': {'formation': 'Kaggle GPU',    'longtrain': 'Kaggle GPU'},
}

def save(fig, name):
    fig.savefig(name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {name}')

def rolling(data, w=5):
    return [np.mean(data[max(0, i-w+1):i+1]) for i in range(len(data))]

eps = np.arange(1, 31)

# ─────────────────────────────────────────────────────────────────────────────
# 01 — Per-episode reward bar chart
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(eps - 0.2, FORMATION_REWARDS, width=0.38, color=CF, alpha=0.85, label='formation')
ax.bar(eps + 0.2, LONGTRAIN_REWARDS, width=0.38, color=CL, alpha=0.85, label='longtrain')
ax.axhline(np.mean(FORMATION_REWARDS), color=CF, ls='--', lw=1.5,
           label=f'formation mean ({np.mean(FORMATION_REWARDS):.0f})')
ax.axhline(np.mean(LONGTRAIN_REWARDS), color=CL, ls='--', lw=1.5,
           label=f'longtrain mean ({np.mean(LONGTRAIN_REWARDS):.0f})')
ax.set_xlabel('Episode')
ax.set_ylabel('Cumulative Reward')
ax.set_title('Per-Episode Reward — formation vs longtrain (30 deterministic episodes)')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
save(fig, 'chart_01_per_episode_reward.png')

# ─────────────────────────────────────────────────────────────────────────────
# 02 — Reward distribution box + strip
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
data = [FORMATION_REWARDS, LONGTRAIN_REWARDS]
bp = ax.boxplot(data, positions=[1, 2], widths=0.4, patch_artist=True,
                medianprops=dict(color='white', linewidth=2))
bp['boxes'][0].set_facecolor(CF)
bp['boxes'][1].set_facecolor(CL)
for r, c, x in zip(data, [CF, CL], [1, 2]):
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(r))
    ax.scatter(x + jitter, r, alpha=0.55, s=20, color=c, zorder=3)
ax.set_xticks([1, 2])
ax.set_xticklabels(['formation\n(500k steps)', 'longtrain\n(1.5M steps)'])
ax.set_ylabel('Cumulative Reward')
ax.set_title('Reward Distribution')
ax.grid(axis='y', alpha=0.3)
for i, (model, d, c) in enumerate(zip(['formation', 'longtrain'], data, [CF, CL])):
    ax.text(i+1, max(d) + 30, f'μ={np.mean(d):.0f}\nσ={np.std(d):.0f}',
            ha='center', va='bottom', fontsize=9, color=c)
save(fig, 'chart_02_reward_distribution.png')

# ─────────────────────────────────────────────────────────────────────────────
# 03 — Episode length distribution
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(FORMATION_LENGTHS, bins=20, alpha=0.7, color=CF,
        label='formation (always 300)', edgecolor='white', linewidth=0.5)
ax.hist(LONGTRAIN_LENGTHS, bins=20, alpha=0.7, color=CL,
        label=f'longtrain (μ={np.mean(LONGTRAIN_LENGTHS):.0f})', edgecolor='white', linewidth=0.5)
ax.set_xlabel('Episode Length (steps)')
ax.set_ylabel('Count')
ax.set_title('Episode Length Distribution')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.text(0.97, 0.95, 'Ball stalls at step 300\nin formation model',
        transform=ax.transAxes, ha='right', va='top', fontsize=9, color=CF,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
save(fig, 'chart_03_episode_lengths.png')

# ─────────────────────────────────────────────────────────────────────────────
# 04 — Reward vs episode length scatter
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(FORMATION_LENGTHS, FORMATION_REWARDS, c=CF, s=65, alpha=0.75,
           label='formation', zorder=3)
ax.scatter(LONGTRAIN_LENGTHS, LONGTRAIN_REWARDS, c=CL, s=65, alpha=0.75,
           label='longtrain', zorder=3)
for r, l in zip(LONGTRAIN_REWARDS, LONGTRAIN_LENGTHS):
    if r > 500:
        ax.annotate(f'{r:.0f}', (l, r), xytext=(6, 4), textcoords='offset points',
                    fontsize=8, color=CL)
ax.set_xlabel('Episode Length (steps)')
ax.set_ylabel('Cumulative Reward')
ax.set_title('Reward vs Episode Length\n(longer episodes correlate with higher reward in longtrain)')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
save(fig, 'chart_04_reward_vs_length.png')

# ─────────────────────────────────────────────────────────────────────────────
# 05 — Reward progression with rolling mean
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(eps, FORMATION_REWARDS, 'o', ms=4, alpha=0.35, color=CF)
ax.plot(eps, rolling(FORMATION_REWARDS), '-', color=CF, lw=2.5, label='formation (rolling-5)')
ax.plot(eps, LONGTRAIN_REWARDS, 'o', ms=4, alpha=0.35, color=CL)
ax.plot(eps, rolling(LONGTRAIN_REWARDS), '-', color=CL, lw=2.5, label='longtrain (rolling-5)')
ax.fill_between(eps,
                np.array(rolling(FORMATION_REWARDS)) - np.std(FORMATION_REWARDS)*0.3,
                np.array(rolling(FORMATION_REWARDS)) + np.std(FORMATION_REWARDS)*0.3,
                alpha=0.15, color=CF)
ax.fill_between(eps,
                np.array(rolling(LONGTRAIN_REWARDS)) - np.std(LONGTRAIN_REWARDS)*0.3,
                np.array(rolling(LONGTRAIN_REWARDS)) + np.std(LONGTRAIN_REWARDS)*0.3,
                alpha=0.15, color=CL)
ax.set_xlabel('Episode')
ax.set_ylabel('Cumulative Reward')
ax.set_title('Reward Progression Over Evaluation Episodes (Rolling Mean, w=5)')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
save(fig, 'chart_05_reward_progression.png')

# ─────────────────────────────────────────────────────────────────────────────
# 06 — Reward percentiles
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
pcts = [10, 25, 50, 75, 90]
f_pcts = [np.percentile(FORMATION_REWARDS, p) for p in pcts]
l_pcts = [np.percentile(LONGTRAIN_REWARDS, p) for p in pcts]
x5 = np.arange(len(pcts))
ax.bar(x5 - 0.2, f_pcts, width=0.38, color=CF, alpha=0.85, label='formation')
ax.bar(x5 + 0.2, l_pcts, width=0.38, color=CL, alpha=0.85, label='longtrain')
ax.set_xticks(x5)
ax.set_xticklabels([f'P{p}' for p in pcts])
ax.set_xlabel('Percentile')
ax.set_ylabel('Reward')
ax.set_title('Reward Percentiles (P10 – P90)')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
for i, (f, l) in enumerate(zip(f_pcts, l_pcts)):
    ax.text(i - 0.2, f + 8, f'{f:.0f}', ha='center', fontsize=8, color=CF)
    ax.text(i + 0.2, l + 8, f'{l:.0f}', ha='center', fontsize=8, color=CL)
save(fig, 'chart_06_reward_percentiles.png')

# ─────────────────────────────────────────────────────────────────────────────
# 07 — Actor weight magnitudes
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(WEIGHT_LAYERS))
w = 0.35
ax.bar(x - w/2, ACTOR_ABS_MEAN['formation'], width=w,
       yerr=ACTOR_STD['formation'], capsize=4, color=CF, alpha=0.85, label='formation')
ax.bar(x + w/2, ACTOR_ABS_MEAN['longtrain'], width=w,
       yerr=ACTOR_STD['longtrain'], capsize=4, color=CL, alpha=0.85, label='longtrain')
ax.set_xticks(x)
ax.set_xticklabels(WEIGHT_LAYERS)
ax.set_ylabel('Mean |weight| (± std)')
ax.set_title('Actor Network — Weight Magnitude per Layer\n(longtrain weights are stronger, especially at input)')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
save(fig, 'chart_07_actor_weights.png')

# ─────────────────────────────────────────────────────────────────────────────
# 08 — Critic weight magnitudes (log scale)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
cx = np.arange(len(CRITIC_LAYERS))
ax.bar(cx - w/2, CRITIC_ABS_MEAN['formation'], width=w, color=CF, alpha=0.85, label='formation')
ax.bar(cx + w/2, CRITIC_ABS_MEAN['longtrain'], width=w, color=CL, alpha=0.85, label='longtrain')
ax.set_xticks(cx)
ax.set_xticklabels(CRITIC_LAYERS)
ax.set_ylabel('Mean |weight|  (log scale)')
ax.set_title('Critic Q₀ — Weight Magnitude per Layer\n(longtrain critic saturated by ±5000 goal rewards)')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_yscale('log')
ax.annotate('34× larger output\n(log scale)', xy=(3, CRITIC_ABS_MEAN['longtrain'][3]),
            xytext=(2.35, 12), fontsize=9, color='#cc3300',
            arrowprops=dict(arrowstyle='->', color='#cc3300', lw=1.5))
save(fig, 'chart_08_critic_weights.png')

# ─────────────────────────────────────────────────────────────────────────────
# 09 — Training state indicators (entropy + layer-1 magnitudes)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
categories = ['log_ent_coef', 'actor L1\nabs_mean', 'critic L1\nabs_mean']
f_vals = [np.log(0.16), 0.2563, 0.3848]
l_vals = [9.19,         0.5720, 3.2179]
x3 = np.arange(len(categories))
ax.bar(x3 - w/2, f_vals, width=w, color=CF, alpha=0.85, label='formation')
ax.bar(x3 + w/2, l_vals, width=w, color=CL, alpha=0.85, label='longtrain')
ax.set_xticks(x3)
ax.set_xticklabels(categories)
ax.set_ylabel('Value')
ax.set_title('Training State Indicators\n(log_ent_coef=9.19 in longtrain → entropy instability)')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.text(0 + 0.2, 9.19 + 0.15, 'log(ent)=9.19\n⚠ instability', color='red', fontsize=8.5,
        ha='center')
save(fig, 'chart_09_training_state.png')

# ─────────────────────────────────────────────────────────────────────────────
# 10 — Hyperparameter comparison table
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.axis('off')
row_labels = list(HYPERPARAMS.keys())
table_data = [[k, HYPERPARAMS[k]['formation'], HYPERPARAMS[k]['longtrain']] for k in row_labels]
tbl = ax.table(cellText=table_data, colLabels=['Parameter', 'formation', 'longtrain'],
               loc='center', cellLoc='left')
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.0, 2.0)
for i, key in enumerate(row_labels):
    tbl[(i+1, 1)].set_facecolor('#DAE8FC')
    tbl[(i+1, 2)].set_facecolor('#FFD0D0' if key == 'Entropy coef' else '#FFE6CC')
    tbl[(i+1, 0)].set_facecolor('#F5F5F5')
for j in range(3):
    tbl[(0, j)].set_facecolor('#333333')
    tbl[(0, j)].set_text_props(color='white', fontweight='bold')
ax.set_title('Hyperparameter Comparison', fontsize=13, pad=16)
save(fig, 'chart_10_hyperparameters.png')

# ─────────────────────────────────────────────────────────────────────────────
# 11 — Training scale comparison (horizontal bars)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
metrics = {
    'Total Timesteps\n(×10³)':      [500,    1500],
    'Policy Updates\n(×10³)':       [249.75, 1499.0],
    'Replay Buffer\n(×10³ steps)':  [100,    300],
    'Gradient Steps\nper Env Step': [2,      4],
}
ys = np.arange(len(metrics))
height = 0.35
for i, (label, (fval, lval)) in enumerate(metrics.items()):
    max_val = max(fval, lval)
    ax.barh(i - height/2, fval / max_val, height=height, color=CF, alpha=0.85,
            label='formation' if i == 0 else '')
    ax.barh(i + height/2, lval / max_val, height=height, color=CL, alpha=0.85,
            label='longtrain' if i == 0 else '')
    ax.text(fval/max_val + 0.01, i - height/2, f'{fval:g}', va='center', fontsize=10, color=CF)
    ax.text(lval/max_val + 0.01, i + height/2, f'{lval:g}', va='center', fontsize=10, color=CL)
ax.set_yticks(ys)
ax.set_yticklabels(list(metrics.keys()), fontsize=10)
ax.set_xlim(0, 1.3)
ax.set_xlabel('Fraction of maximum')
ax.set_title('Training Scale Comparison (normalized)')
ax.legend(loc='lower right', fontsize=11)
ax.grid(axis='x', alpha=0.3)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
save(fig, 'chart_11_training_scale.png')

# ── Text summary ──────────────────────────────────────────────────────────────
print('\n' + '='*64)
print('ANALYSIS SUMMARY')
print('='*64)
print(f'{"Metric":<28} {"formation":>14} {"longtrain":>14}')
print('-'*64)
print(f'{"Timesteps trained":<28} {"500,000":>14} {"1,500,000":>14}')
print(f'{"Mean reward (30 eps)":<28} {np.mean(FORMATION_REWARDS):>14.1f} {np.mean(LONGTRAIN_REWARDS):>14.1f}')
print(f'{"Std reward":<28} {np.std(FORMATION_REWARDS):>14.1f} {np.std(LONGTRAIN_REWARDS):>14.1f}')
print(f'{"Median reward":<28} {np.median(FORMATION_REWARDS):>14.1f} {np.median(LONGTRAIN_REWARDS):>14.1f}')
print(f'{"P90 reward":<28} {np.percentile(FORMATION_REWARDS,90):>14.1f} {np.percentile(LONGTRAIN_REWARDS,90):>14.1f}')
print(f'{"Max reward":<28} {np.max(FORMATION_REWARDS):>14.1f} {np.max(LONGTRAIN_REWARDS):>14.1f}')
print(f'{"Mean episode length":<28} {np.mean(FORMATION_LENGTHS):>14.0f} {np.mean(LONGTRAIN_LENGTHS):>14.0f}')
print(f'{"Episodes at 300 steps":<28} {sum(l==300 for l in FORMATION_LENGTHS):>14} {sum(l==300 for l in LONGTRAIN_LENGTHS):>14}')
print(f'{"log_ent_coef":<28} {"-1.84":>14} {"9.19 (⚠)":>14}')
print(f'{"Critic L1 abs_mean":<28} {0.3848:>14.4f} {3.2179:>14.4f}')
print(f'{"Total model params":<28} {"2,091,284":>14} {"2,091,284":>14}')
print('='*64)

# ─────────────────────────────────────────────────────────────────────────────
# Report-optimised figures (200 DPI, single-column LaTeX width, larger fonts)
# Output: report_fig_*.png  — drop these directly into the LaTeX document.
# ─────────────────────────────────────────────────────────────────────────────
def save_report(fig, name):
    """Save at 200 DPI with tight layout; no suptitle (caption goes in LaTeX)."""
    fig.savefig(name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {name}')

matplotlib.rcParams.update({'font.size': 11})

# ── report_fig_reward_distribution.png ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
data = [FORMATION_REWARDS, LONGTRAIN_REWARDS]
bp = ax.boxplot(data, positions=[1, 2], widths=0.42, patch_artist=True,
                medianprops=dict(color='white', linewidth=2.5))
bp['boxes'][0].set_facecolor(CF)
bp['boxes'][1].set_facecolor(CL)
for r, c, x in zip(data, [CF, CL], [1, 2]):
    jitter = np.random.default_rng(42).uniform(-0.13, 0.13, len(r))
    ax.scatter(x + jitter, r, alpha=0.55, s=22, color=c, zorder=3)
ax.set_xticks([1, 2])
ax.set_xticklabels(['500k-step model\n(formation)', '1.5M-step model\n(longtrain)'])
ax.set_ylabel('Cumulative Episode Reward')
ax.grid(axis='y', alpha=0.3)
for i, (d, c) in enumerate(zip(data, [CF, CL])):
    ax.text(i+1, max(d) + 25, f'μ={np.mean(d):.0f}, σ={np.std(d):.0f}',
            ha='center', va='bottom', fontsize=10, color=c, fontweight='bold')
save_report(fig, 'report_fig_reward_distribution.png')

# ── report_fig_reward_percentiles.png ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
pcts = [10, 25, 50, 75, 90]
f_pcts = [np.percentile(FORMATION_REWARDS, p) for p in pcts]
l_pcts = [np.percentile(LONGTRAIN_REWARDS, p) for p in pcts]
x5 = np.arange(len(pcts))
ax.bar(x5 - 0.2, f_pcts, width=0.38, color=CF, alpha=0.85, label='formation (500k)')
ax.bar(x5 + 0.2, l_pcts, width=0.38, color=CL, alpha=0.85, label='longtrain (1.5M)')
ax.set_xticks(x5)
ax.set_xticklabels([f'P{p}' for p in pcts])
ax.set_xlabel('Percentile')
ax.set_ylabel('Cumulative Episode Reward')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
for i, (f, l) in enumerate(zip(f_pcts, l_pcts)):
    ax.text(i - 0.2, f + 6, f'{f:.0f}', ha='center', fontsize=9, color=CF)
    ax.text(i + 0.2, l + 6, f'{l:.0f}', ha='center', fontsize=9, color=CL)
save_report(fig, 'report_fig_reward_percentiles.png')

# ── report_fig_reward_vs_length.png ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(FORMATION_LENGTHS, FORMATION_REWARDS, c=CF, s=55, alpha=0.75,
           label='formation (500k)', zorder=3)
ax.scatter(LONGTRAIN_LENGTHS, LONGTRAIN_REWARDS, c=CL, s=55, alpha=0.75,
           label='longtrain (1.5M)', zorder=3)
for r, l in zip(LONGTRAIN_REWARDS, LONGTRAIN_LENGTHS):
    if r > 500:
        ax.annotate(f'{r:.0f}', (l, r), xytext=(6, 3), textcoords='offset points',
                    fontsize=9, color=CL)
ax.set_xlabel('Episode Length (steps)')
ax.set_ylabel('Cumulative Episode Reward')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.text(0.97, 0.05,
        'Formation: all episodes\nterminate at 300 steps\n(ball-stall timeout)',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=9, color=CF,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
save_report(fig, 'report_fig_reward_vs_length.png')

matplotlib.rcParams.update({'font.size': 10})  # reset to default
