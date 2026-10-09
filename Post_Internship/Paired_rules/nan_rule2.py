"""
nan_rule2.py -- run paired learning Rule 2.
Thin wrapper around nan_core; edit the Config below or import run() elsewhere.
Rule 2: independent decoder weights, no decoder bias
"""
from nan_core import Config, run

if __name__ == "__main__":
    cfg = Config(
        rule=2,
        use_chain_rule=True,     # False -> treat n_i as fixed (SAILnet / no S')
        lr_Q=0.0001, lr_R=0.0001, lr_Bi=0.0001, lr_Bik=0.0001,
        epochs=20, N=20,
    )
    run(cfg)
