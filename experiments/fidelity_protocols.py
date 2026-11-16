"""Independent, single-fidelity BO campaigns for the L/M/H mesh protocols.

This is NOT a multi-fidelity surrogate: no observations are shared between
levels. Each run starts from the paper's original frequency-specific bounds.
It does not reuse optima from any other level and does not modify the original
.aedt file: each run receives an independent project copy.
"""
import argparse
import os
import shutil
from pathlib import Path
import pipeline_bo as bo
from configs.paper_targets import target_config

PROTOCOLS = {
    'L': dict(MaxDeltaS=0.0100, MaximumPasses=12, MinimumPasses=2,
              PercentRefinement=30, mesh_slider=6),
    'M': dict(MaxDeltaS=0.0050, MaximumPasses=20, MinimumPasses=2,
              PercentRefinement=30, mesh_slider=7),
    'H': dict(MaxDeltaS=0.0025, MaximumPasses=26, MinimumPasses=2,
              PercentRefinement=30, mesh_slider=8),
}

def run(proj, out, levels, frequencies, permittivity, seed):
    if not proj.is_file() or proj.suffix.lower() != '.aedt':
        raise ValueError('Provide a local .aedt file via HFSS_PROJECT_PATH')
    for level in levels:
        params = PROTOCOLS[level]
        for freq in frequencies:
            bo.fechar_sessao()
            folder = out / level / f'f{freq:g}THz_er{permittivity:g}'
            folder.mkdir(parents=True, exist_ok=True)
            project_copy = folder / proj.name
            if not project_copy.exists():
                shutil.copy2(proj, project_copy)
            cfg = target_config(bo.ANTENNA_CONFIG, freq)
            cfg['project_path'] = str(project_copy)
            cfg['base_dir'] = str(folder / 'output')
            cfg['mesh_slider'] = params['mesh_slider']
            cfg['malha'] = {k: v for k, v in params.items() if k != 'mesh_slider'}
            bo.run_bayesian_optimization_adaptativo(
                cfg, f_res_alvo=freq, er_alvo=permittivity,
                retomar=True, seed=seed, n_init=32, n_iter=40,
                kernel='matern52_ard', max_remodelagens=4,
            )
    bo.fechar_sessao()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--levels', nargs='+', choices=sorted(PROTOCOLS), default=['L','M','H'])
    parser.add_argument('--frequencies', nargs='+', type=float, default=[10.0,11.0])
    parser.add_argument('--permittivity', type=float, default=9.0)
    parser.add_argument('--seed', type=int, default=42)
    a = parser.parse_args()
    project = os.environ.get('HFSS_PROJECT_PATH')
    if not project:
        parser.error('Set HFSS_PROJECT_PATH')
    run(Path(project), Path(os.environ.get('RESULTS_DIR','results')) / 'fidelity_lmh',
        a.levels, a.frequencies, a.permittivity, a.seed)

if __name__ == '__main__':
    main()
