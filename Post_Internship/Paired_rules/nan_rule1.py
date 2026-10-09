"""
nan_rule1.py -- run paired learning Rule 1.
Thin wrapper around nan_core; edit the Config below or import run() elsewhere.
Rule 1: independent decoder weights, with decoder bias
"""
from nan_core import Config, run

if __name__ == "__main__":
    cfg = Config(
        rule=1,
        use_chain_rule=True,     # False -> treat n_i as fixed (SAILnet / no S')
        lr_Q=0.1, lr_R=0.001, lr_Bi=0.001, lr_Bik=0.001,
        epochs=20, N=20,
    )
    run(cfg)
