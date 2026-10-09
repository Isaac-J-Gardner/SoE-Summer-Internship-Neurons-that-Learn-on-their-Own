"""
nan_sweep.py -- sweep the decoder-bias learning rate and flag encoder collapse.

For each lr_Bik it trains the chosen rule, then prints read-out accuracy, activation
effective rank, dead-neuron fraction and mean firing rate, tagging any run whose
encoder collapsed. A summary plot (acc / rank / dead-fraction vs lr_Bik) is saved to
the rule's image directory.

Edit `RULE`, `LR_BIK_VALUES`, and any Config fields below.
"""
from nan_core import Config, sweep_lr_Bik

RULE = 1
LR_BIK_VALUES = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5]

if __name__ == "__main__":
    base = Config(
        rule=RULE,
        epochs=20,
        make_plots=True,     # saves the summary plot; per-run grids are skipped in sweeps
        verbose=True,       # quiet per-batch; the sweep prints one line per lr_Bik
    )
    results = sweep_lr_Bik(base, LR_BIK_VALUES)
