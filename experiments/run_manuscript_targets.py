"""Run a full, independent BO campaign for each manuscript target.

Requires HFSS, its project file and a licensed local AEDT installation.
Run from the repository root. All output stays in RESULTS_DIR.
"""
import argparse
import os
from pathlib import Path
import pipeline_bo as bo
from configs.paper_targets import target_config

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frequencies', nargs='+', type=float, default=[10.0, 11.0])
    p.add_argument('--permittivities', nargs='+', type=float, default=[8.0, 9.0, 10.0])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if not os.environ.get('HFSS_PROJECT_PATH'):
        p.error('Set HFSS_PROJECT_PATH to the local .aedt project before running')
    for freq in args.frequencies:
        cfg = target_config(bo.ANTENNA_CONFIG, freq)
        cfg['base_dir'] = str(Path(bo.ANTENNA_CONFIG['base_dir']) / f'f{freq:g}THz')
        bo.run_bayesian_optimization_adaptativo(
            cfg, f_res_alvo=freq, er_alvo=args.permittivities,
            retomar=args.resume, seed=args.seed, kernel='matern52_ard',
            n_init=32, n_iter=40,
        )
    bo.fechar_sessao()

if __name__ == '__main__':
    main()
