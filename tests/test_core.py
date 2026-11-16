"""CPU-only regression checks of the manuscript configurations and search primitives."""
import numpy as np
import pytest

import pipeline_bo as bo
from configs.paper_targets import target_config
from experiments.fidelity_protocols import PROTOCOLS


def test_fabrication_grid_rounding():
    assert bo.arredondar_fabricacao({'a': 9.12349}) == {'a': 9.123}
    assert bo.arredondar_fabricacao({'a': 9.12351}) == {'a': 9.124}


def test_target_specific_bounds():
    ten = target_config(bo.ANTENNA_CONFIG, 10.0)
    eleven = target_config(bo.ANTENNA_CONFIG, 11.0)
    assert ten['geometric_vars']['a'] == (8.6, 9.3, 'um')
    assert eleven['geometric_vars']['d'] == (5.5, 6.9, 'um')
    assert ten['freq_sweep_points'] == eleven['freq_sweep_points'] == 1000
    assert ten['geometric_vars'] != eleven['geometric_vars']


def test_latin_hypercube_reproducibility():
    cfg = target_config(bo.ANTENNA_CONFIG, 10.0)
    names, bounds = bo.montar_espaco_busca(cfg)
    steps = bo.grades_do_config(cfg, names)
    lhs1 = bo.amostrar_lhs(32, bounds, seed=42, grades=steps)
    lhs2 = bo.amostrar_lhs(32, bounds, seed=42, grades=steps)
    assert np.array_equal(lhs1, lhs2)
    assert np.allclose(lhs1 / steps, np.round(lhs1 / steps))
    assert len({tuple(row) for row in lhs1}) == 32


def test_fidelity_protocols():
    assert [PROTOCOLS[x]['mesh_slider'] for x in ('L', 'M', 'H')] == [6, 7, 8]
    assert [PROTOCOLS[x]['MaxDeltaS'] for x in ('L', 'M', 'H')] == [0.01, 0.005, 0.0025]


def test_hfss_slider_call_without_hfss():
    class Mesh:
        def InitialMeshSettings(self, settings):
            self.settings = settings

    class Design:
        def __init__(self):
            self.mesh = Mesh()

        def GetModule(self, name):
            assert name == 'MeshSetup'
            return self.mesh

    class HFSS:
        def __init__(self):
            self.odesign = Design()

    obj = HFSS()
    bo.configurar_malha_inicial(obj, 8)
    assert obj.odesign.mesh.settings[1][-1] == 8
    with pytest.raises(ValueError):
        bo.configurar_malha_inicial(obj, 10)
