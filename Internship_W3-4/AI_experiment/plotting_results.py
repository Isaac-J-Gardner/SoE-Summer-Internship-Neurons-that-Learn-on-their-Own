import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results = json.load(open('results.json'))
conditions = list(results.keys())
colors = {'NK-NaN': '#1b9e77', 'MNIST-NaN raw': '#d95f02',
          'MNIST-NaN mean-sub': '#7570b3', 'MNIST-NaN whitened': '#e7298a'}

panels = [
    ('r_eff',  'Effective rank  $r_{eff}=e^{H_\\lambda}$', (0, 20.5), 20),
    ('R_spec', 'Spectral redundancy  $R_{spec}=1-r_{eff}/D$', (-0.02, 1.0), None),
    ('R_KL',   'Gaussian total correlation  $R_{KL}=-\\frac{1}{2}\\log\\det C$', None, None),
    ('R_frob', 'Weak-dep. approx.  $\\frac{1}{4}\\|C-I\\|_F^2$', None, None),
    ('cos',    'Mean off-diag encoder-row cosine', (-0.1, 0.8), 0.0),
    ('recon',  'Reconstruction loss (MSE)', None, None),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
for ax, (key, title, ylim, hline) in zip(axes.flat, panels):
    for c in conditions:
        y = [h[key] for h in results[c]]
        x = np.arange(len(y))
        ax.plot(x, y, marker='o', ms=3, lw=1.8, color=colors[c], label=c)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('epoch')
    if ylim: ax.set_ylim(*ylim)
    if hline is not None:
        ax.axhline(hline, color='k', lw=0.6, ls=':', alpha=0.5)
    ax.grid(alpha=0.25)
    if key == 'R_KL':
        ax.set_yscale('symlog', linthresh=1.0)
    if key == 'recon':
        ax.set_yscale('log')

axes.flat[0].legend(fontsize=9, loc='center right', framealpha=0.9)
fig.suptitle('Neuron-autoencoder collapse in activation space  (D=20 hidden, SGD lr=10)',
             fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig('collapse_trajectories.png', dpi=130, bbox_inches='tight')
print('saved collapse_trajectories.png')

# --- companion: final-epoch bar summary of the three redundancy proxies ---
fig2, ax = plt.subplots(figsize=(9, 4.2))
metrics = ['R_spec', 'R_KL', 'R_frob']
xpos = np.arange(len(conditions))
w = 0.25
for i, m in enumerate(metrics):
    vals = [results[c][-1][m] for c in conditions]
    ax.bar(xpos + (i - 1) * w, vals, w, label=m)
ax.set_xticks(xpos); ax.set_xticklabels(conditions, rotation=12, ha='right', fontsize=9)
ax.set_title('Final-epoch redundancy proxies')
ax.legend(); ax.grid(axis='y', alpha=0.25)
ax.set_yscale('symlog', linthresh=1.0)
fig2.tight_layout()
fig2.savefig('final_redundancy.png', dpi=130, bbox_inches='tight')
print('saved final_redundancy.png')