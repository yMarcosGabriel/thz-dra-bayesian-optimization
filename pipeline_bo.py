"""Bayesian inverse design of rectangular terahertz dielectric resonator antennas.

The computational kernel uses HFSS responses, constrained search and Gaussian processes."""
import os
import sys
import json
import time
import copy
import glob
import math
import hashlib
import inspect
import itertools
import logging
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import qmc, norm
from scipy.optimize import differential_evolution, brentq
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF

ANTENNA_CONFIG = {
    'project_path': os.environ.get('HFSS_PROJECT_PATH', ''),
    'base_dir': os.environ.get('RESULTS_DIR', 'results'),

    'freq_unit':         'THz',
    'freq_sweep':        (9, 13),
    'freq_sweep_points': 1000,
    'freq_valid_range':  (9.2, 12.8),

    's11_threshold_db': -15.0,
    's11_reject_db':    -20.0,
    'bw_threshold_db':  -10.0,

    'material_name': 'rutile',

    'geometric_vars': {
        'a':      (8.0,  8.7,  'um'),
        'b':      (8.0,  9.0,  'um'),
        'd':      (5.5,  6.9,  'um'),
        'lstrip': (13.2, 13.6, 'um'),
        'wstrip': (1.4,  1.7,  'um'),
        'lslot':  (0.8,  1.1,  'um'),
        'wslot':  (5.6,  6.7,  'um'),
    },

    'grades_por_variavel': {},

    'limites_fisicos': {},

    'hfss': {'non_graphical': False, 'version': None, 'aedt_process_id': None,
             'cores': 6, 'tasks': None, 'gpus': None},

    'malha': {},

    'modelo_analitico': {'x': 'a', 'y': 'b', 'h': 'd', 'convencao': 'A'},

    'area_vars': ('a', 'b'),
}

N_INIT_MIN      = 32
N_ITER_MIN      = 40
FATOR_ORCAMENTO = 4
FRACAO_REFINO   = 0.25
RAIO_REFINO     = 0.10
EI_MIN          = 1e-4
SEMENTE         = 42

INIT_PARADA_ANTECIPADA     = True
INIT_MIN_VALIDOS           = 8
INIT_CUSTO_BOM             = 0.30
INIT_MIN_VALIDOS_COM_CUSTO_BOM = 5

GRADE_FABRICACAO  = 0.001

OTIMIZAR_NA_GRADE = True

MAX_CANDIDATOS_REVALIDACAO = 5

PESOS_PADRAO     = {'freq': 0.60, 's11': 0.20, 're_z': 0.10, 'im_z': 0.10, 'vswr': 0.00}
RESTRICOES_PADRAO = {'s11_max_db': -20.0, 'bw_min': None}
PENALIDADE_RESTRICAO = 10.0
PENALIDADE       = 100.0
S11_IDEAL, S11_NORM = -40.0, 20.0

TOL_QUASE_OTIMO  = 0.10
CRITERIO_DESEMPATE = 'custo'

MARGEM_BORDA        = 0.08
FATOR_EXPANSAO      = 0.35
MAX_REMODELAGENS    = 4
VARS_PRIORITARIAS   = ('a', 'b', 'd')
FRACAO_NOVOS        = 0.5
MIN_NOVOS           = 4
FRACAO_ITER_REMODEL = 0.5
MIN_ITER_REMODEL    = 4

KERNELS = ('matern52_ard', 'matern52_iso', 'matern32_ard', 'rbf_ard')
KERNEL_PADRAO = 'matern52_ard'

NOME_LOG = 'pipeline_bo'
log = logging.getLogger(NOME_LOG)
if not log.handlers:
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

def configurar_log(log_path: str, nome_logger: str = NOME_LOG) -> logging.Logger:
    logger = logging.getLogger(nome_logger)
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)
    formato = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    for h in (logging.FileHandler(log_path, encoding='utf-8'), logging.StreamHandler()):
        h.setFormatter(formato)
        logger.addHandler(h)
    logger.propagate = False
    return logger

def info_ambiente(hfss=None) -> dict:
    """Return the software and machine configuration used for a simulation campaign."""
    info = {'python': sys.version.split()[0], 'so': platform.platform(),
            'cpu': platform.processor() or platform.machine(), 'nucleos_logicos': os.cpu_count()}
    try:
        import psutil
        info['ram_gb'] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except Exception:
        pass
    for pacote in ('numpy', 'scipy', 'sklearn', 'pandas', 'pyaedt'):
        try:
            info[pacote] = __import__(pacote).__version__
        except Exception:
            pass
    if hfss is not None:
        for attr in ('aedt_version_id', 'odesktop'):
            try:
                v = getattr(hfss, attr)
                info['aedt'] = str(v.GetVersion() if attr == 'odesktop' else v)
                break
            except Exception:
                continue
    return info

def _validar_config(config: dict) -> None:
    obrig = ['project_path', 'base_dir', 'freq_unit', 'freq_sweep', 'freq_sweep_points',
             'freq_valid_range', 's11_threshold_db', 's11_reject_db', 'material_name',
             'geometric_vars']
    faltando = [k for k in obrig if k not in config]
    if faltando:
        raise ValueError(f"ANTENNA_CONFIG incompleto — faltam as chaves: {faltando}")
    config.setdefault('bw_threshold_db', -10.0)
    fs, fv = config['freq_sweep'], config['freq_valid_range']
    if fs[0] >= fs[1]:
        raise ValueError("freq_sweep deve ser (inicio, fim) com inicio < fim.")
    if not (fs[0] <= fv[0] and fv[1] <= fs[1]):
        raise ValueError("freq_valid_range deve estar contido em freq_sweep.")
    if not (config['s11_reject_db'] < config['s11_threshold_db'] < config['bw_threshold_db']):
        raise ValueError("Esperado s11_reject_db < s11_threshold_db < bw_threshold_db "
                         "(ex.: -20 < -15 < -10).")
    if not config['geometric_vars']:
        raise ValueError("geometric_vars esta vazio.")
    for nome, spec in config['geometric_vars'].items():
        if len(spec) != 3 or spec[0] >= spec[1]:
            raise ValueError(f"geometric_vars['{nome}'] deve ser (min < max, unidade).")
        if grade_da_variavel(nome, config) <= 0:
            raise ValueError(f"Grade de fabricacao de '{nome}' deve ser > 0.")
    extras = set(config.get('grades_por_variavel') or {}) - set(config['geometric_vars'])
    if extras:
        raise ValueError(f"grades_por_variavel tem variaveis desconhecidas: {sorted(extras)}")

def _como_lista(x) -> list:
    return [float(x)] if np.isscalar(x) else [float(v) for v in x]

def _nome_pasta(f_alvo: float, er: float, unidade: str) -> str:
    return f"f{f_alvo:g}{unidade}_er{er:g}"

def montar_espaco_busca(config: dict):
    nomes = list(config['geometric_vars'])
    bounds = np.array([[mn, mx] for mn, mx, _ in config['geometric_vars'].values()], float)
    return nomes, bounds

def orcamento(d: int, n_init=None, n_iter=None):
    """Determine the Latin-hypercube and sequential optimization budgets."""
    n_init = n_init if n_init is not None else max(N_INIT_MIN, FATOR_ORCAMENTO * (d + 1))
    n_iter = n_iter if n_iter is not None else max(N_ITER_MIN, FATOR_ORCAMENTO * (d + 1))
    return int(n_init), int(n_iter)

def grade_da_variavel(nome: str, config: dict | None = None) -> float:
    """Return the fabrication-grid step for the requested geometric variable."""
    over = (config or {}).get('grades_por_variavel') or {}
    return float(over.get(nome, GRADE_FABRICACAO))

def grades_do_config(config: dict, nomes: list | None = None) -> np.ndarray:
    nomes = nomes or list(config['geometric_vars'])
    return np.array([grade_da_variavel(n, config) for n in nomes], float)

def arredondar_fabricacao(valores: dict, config: dict | None = None) -> dict:
    """Project geometric values onto their fabrication grids."""
    saida = {}
    for k, v in valores.items():
        g = grade_da_variavel(k, config)
        saida[k] = round(round(float(v) / g) * g, 6)
    return saida

def indices_grade(x, grades) -> tuple:
    """Convert a point to integer grid indices for duplicate detection."""
    return tuple(int(i) for i in np.round(np.asarray(x, float) / grades))

def na_grade(x, grades, tol: float = 1e-6) -> bool:
    r = np.asarray(x, float) / grades
    return bool(np.all(np.abs(r - np.round(r)) < tol))

def mascara_na_grade(df: pd.DataFrame, nomes: list, grades, tol: float = 1e-6) -> np.ndarray:
    r = df[nomes].values / grades
    return np.all(np.abs(r - np.round(r)) < tol, axis=1)

def alinhar_limites(mn: float, mx: float, g: float):
    """Align a bounded interval to the interior fabrication-grid points."""
    lo = math.ceil(mn / g - 1e-9) * g
    hi = math.floor(mx / g + 1e-9) * g
    return round(lo, 6), round(hi, 6)

def alinhar_config_a_grade(config: dict) -> dict:
    """Return a copy of the configuration with grid-aligned search bounds."""
    novo = copy.deepcopy(config)
    for nome, (mn, mx, unid) in config['geometric_vars'].items():
        g = grade_da_variavel(nome, config)
        lo, hi = alinhar_limites(mn, mx, g)
        if hi <= lo:
            raise ValueError(f"A faixa de '{nome}' [{mn}, {mx}] tem menos de 2 niveis "
                             f"na grade de {g} {unid}.")
        if (lo, hi) != (round(mn, 6), round(mx, 6)):
            log.info(f"  Grade: '{nome}' [{mn}, {mx}] -> [{lo}, {hi}] (passo {g:g} {unid})")
        novo['geometric_vars'][nome] = (lo, hi, unid)
    return novo

def conectar_hfss(config: dict, max_tentativas: int = 3):
    """Attach to an existing AEDT desktop if available, otherwise open a session."""
    from pyaedt import Hfss
    opc = config.get('hfss') or {}
    try:
        params = set(inspect.signature(Hfss.__init__).parameters)
    except (TypeError, ValueError):
        params = set()
    novos = ('project' in params) or ('new_desktop' in params)
    pares = [('project', 'new_desktop'), ('projectname', 'new_desktop_session')]
    if not novos:
        pares.reverse()

    ultimo_erro = None
    for tentativa in range(1, max_tentativas + 1):
        for p_proj, p_new in pares:
            kw = {p_proj: config['project_path'], p_new: False,
                  'non_graphical': opc.get('non_graphical', False), 'close_on_exit': False}
            if opc.get('aedt_process_id'):
                kw['aedt_process_id'] = opc['aedt_process_id']
            if opc.get('version'):
                kw['version' if 'version' in params else 'specified_version'] = opc['version']
            try:
                return Hfss(**kw)
            except TypeError as e:
                ultimo_erro = e
            except Exception as e:
                ultimo_erro = e
                log.warning(f"Tentativa {tentativa} de conexao falhou: {e}")
                time.sleep(5)
                break
    raise RuntimeError(f"Nao foi possivel conectar ao HFSS: {ultimo_erro}")

class SessaoHFSS:
    """Manage the reusable connection to the local HFSS installation."""

    def __init__(self):
        self.hfss, self.chave = None, None

    def _viva(self) -> bool:
        try:
            self.hfss.odesign.GetName()
            return True
        except Exception:
            return False

    def obter(self, config: dict):
        if self.hfss is not None and self.chave == config['project_path'] and self._viva():
            return self.hfss
        self.fechar()
        self.hfss = conectar_hfss(config)
        self.chave = config['project_path']
        return self.hfss

    def fechar(self) -> None:
        """Release this Python connection while keeping the AEDT desktop open."""
        if self.hfss is not None:
            try:
                self.hfss.release_desktop(close_projects=False, close_desktop=False)
            except Exception as e:
                log.warning(f"Falha ao liberar a sessao HFSS: {e}")
        self.hfss = None

_SESSAO = SessaoHFSS()

def fechar_sessao() -> None:
    _SESSAO.fechar()

AJUSTES_MALHA_PADRAO = {'MaxDeltaS': 0.01, 'MaximumPasses': 12,
                        'MinimumPasses': 2, 'PercentRefinement': 30}

def configurar_malha_inicial(hfss, slider: int) -> None:
    """Set the initial surface-mesh slider in HFSS (1 through 9)."""
    slider = int(slider)
    if not 1 <= slider <= 9:
        raise ValueError('mesh_slider must be between 1 and 9')
    mesh = hfss.odesign.GetModule('MeshSetup')
    mesh.InitialMeshSettings([
        'NAME:MeshSettings',
        ['NAME:GlobalSurfApproximation',
         'CurvedSurfaceApproxChoice:=', 'UseSlider',
         'SliderMeshSettings:=', slider],
        ['NAME:GlobalCurvilinear', 'Apply:=', False],
        ['NAME:GlobalModelRes', 'UseAutoLength:=', True],
        'MeshMethod:=', 'Auto',
        'UseLegacyFaceterForTauVolumeMesh:=', False,
        'DynamicSurfaceResolution:=', False,
        'UseFlexMeshingForTAUvolumeMesh:=', False,
        'UseAlternativeMeshMethodsAsFallBack:=', True,
        'AllowPhiForLayeredGeometry:=', True,
    ])

def configurar_hfss_simulacao(hfss, config: dict, ajustes: dict | None = None):
    props = {**AJUSTES_MALHA_PADRAO, **(config.get('malha') or {}), **(ajustes or {})}
    for s in hfss.setup_names:
        hfss.delete_setup(s)

    setup = hfss.create_setup("Setup1")
    setup.props["Frequency"] = f"{sum(config['freq_sweep']) / 2}{config['freq_unit']}"
    for k, v in props.items():
        setup.props[k] = v
    setup.update()
    if config.get('mesh_slider') is not None:
        configurar_malha_inicial(hfss, config['mesh_slider'])

    for sw in hfss.get_sweeps("Setup1"):
        try:
            hfss.delete_sweep("Setup1", sw)
        except Exception as e:
            log.debug(f"delete_sweep('{sw}') ignorado: {e}")

    hfss.create_linear_count_sweep(
        setup="Setup1", units=config['freq_unit'],
        start_frequency=config['freq_sweep'][0], stop_frequency=config['freq_sweep'][1],
        num_of_freq_points=config['freq_sweep_points'],
        name="Sweep_dataset", sweep_type="Interpolating", save_fields=False)
    return hfss.get_sweeps("Setup1")[0], props

def _deletar_reports(oModule, nomes) -> None:
    for nome in nomes:
        try:
            oModule.DeleteReports([nome])
        except Exception as e:
            log.debug(f"DeleteReports('{nome}') ignorado: {e}")

def _exportar(oModule, sweep_nome, nome_rel, expressoes, csv_path, max_tentativas=3) -> None:
    for tentativa in range(1, max_tentativas + 1):
        try:
            _deletar_reports(oModule, [nome_rel])
            time.sleep(2)
            oModule.CreateReport(
                nome_rel, "Modal Solution Data", "Rectangular Plot",
                f"Setup1 : {sweep_nome}", ["Domain:=", "Sweep"], ["Freq:=", ["All"]],
                ["X Component:=", "Freq", "Y Component:=", expressoes], [])
            oModule.ExportToFile(nome_rel, csv_path, False)
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                return
            raise FileNotFoundError("CSV nao gerado ou vazio.")
        except Exception as e:
            log.warning(f"Exportacao {nome_rel} tentativa {tentativa} falhou: {e}")
            time.sleep(3)
    raise RuntimeError(f"Falha na exportacao {nome_rel} apos {max_tentativas} tentativas.")

def exportar_s11_simulacao(oModule, sweep_nome, csv_path):
    _exportar(oModule, sweep_nome, "S11_data", ["dB(S(1,1))"], csv_path)

def exportar_z11_simulacao(oModule, sweep_nome, csv_path):
    _exportar(oModule, sweep_nome, "Z11_data", ["re(Z(1,1))", "im(Z(1,1))"], csv_path)

def _regioes(mask: np.ndarray) -> list:
    """Identify consecutive regions where the input mask is true."""
    m = np.concatenate(([False], mask, [False])).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))

def _refinar_minimo(freq, s11_db, i):
    """Estimate a resonance minimum using neighboring samples."""
    if i <= 0 or i >= len(freq) - 1:
        return float(freq[i]), float(s11_db[i])
    p0, pm, pp = (10 ** (s11_db[j] / 10) for j in (i, i - 1, i + 1))
    den = pm - 2 * p0 + pp
    if den <= 0:
        return float(freq[i]), float(s11_db[i])
    x = float(np.clip(0.5 * (pm - pp) / den, -1, 1))
    pmin = p0 + 0.5 * (pp - pm) * x + 0.5 * den * x * x
    passo = freq[i + 1] - freq[i]
    return float(freq[i] + x * passo), float(10 * np.log10(max(pmin, 1e-30)))

def _cruza(f1, s1, f2, s2, thr):
    return f1 + (thr - s1) * (f2 - f1) / (s2 - s1)

def largura_de_banda(freq, s11_db, i, thr_db):
    """Measure the connected return-loss bandwidth around a resonance."""
    n = len(freq)
    j = i
    while j > 0 and s11_db[j - 1] <= thr_db:
        j -= 1
    k = i
    while k < n - 1 and s11_db[k + 1] <= thr_db:
        k += 1
    trunc = (j == 0) or (k == n - 1)
    f_lo = freq[0] if j == 0 else _cruza(freq[j - 1], s11_db[j - 1], freq[j], s11_db[j], thr_db)
    f_hi = freq[-1] if k == n - 1 else _cruza(freq[k], s11_db[k], freq[k + 1], s11_db[k + 1], thr_db)
    return float(f_hi - f_lo), bool(trunc)

def analisar_s11(freq, s11_db, config: dict):
    """Extract the resonant frequency and apply electromagnetic feasibility checks."""
    freq, s11_db = np.asarray(freq, float), np.asarray(s11_db, float)
    if len(freq) < 10:
        return None, "CSV_INVALIDO"
    regs = _regioes(s11_db <= config['s11_threshold_db'])
    if len(regs) != 1:
        return None, f"MULTIPLAS ({len(regs)})"
    i0, i1 = regs[0]
    i = i0 + int(np.argmin(s11_db[i0:i1 + 1]))
    f_res, s11_min = _refinar_minimo(freq, s11_db, i)

    f_min, f_max = config['freq_valid_range']
    if not (f_min <= f_res <= f_max):
        return None, "LIMITE_SWEEP"
    if s11_min > config['s11_reject_db']:
        return None, "S11_FRACO"
    ig = int(np.argmin(s11_db))
    if ig > 1 and np.any(np.diff(s11_db[:ig]) > 1.0):
        return None, "NAO_MONOTONICA"
    bw, trunc = largura_de_banda(freq, s11_db, i, config['bw_threshold_db'])
    if bw <= 0:
        return None, "BW_ZERO"
    return {'f_res': round(f_res, 6), 's11_min': round(s11_min, 4), 'bw': round(bw, 6),
            'bw_pct': round(100 * bw / f_res, 4), 'bw_trunc': trunc}, "OK"

def _vswr_from_s11_db(s11_db: float) -> float:
    g = min(abs(10 ** (s11_db / 20.0)), 0.9999)
    return (1 + g) / (1 - g)

def _ler_csv(path, colunas):
    return pd.read_csv(path, skiprows=1, header=None, names=colunas)

class Avaliador:
    """Evaluate a geometry using HFSS and collect its electromagnetic response."""

    def __init__(self, hfss, config: dict, er: float, ajustes_malha: dict | None = None):
        self.hfss, self.cfg = hfss, config
        self.sweep, props = configurar_hfss_simulacao(hfss, config, ajustes_malha)
        self.oModule = hfss.odesign.GetModule("ReportSetup")
        self.tmp = os.path.join(config['base_dir'], '_tmp')
        os.makedirs(self.tmp, exist_ok=True)
        self.kw_analise = self._args_analise(hfss, config)
        self.er_atual = None
        self.set_er(er)
        chave = json.dumps([config['freq_sweep'], config['freq_sweep_points'], props,
                            config['material_name']], sort_keys=True, default=str)
        self.setup_id = hashlib.md5(chave.encode()).hexdigest()[:8]
        self._avisou = False

    @staticmethod
    def _args_analise(hfss, config: dict) -> dict:
        """Prepare solver resource arguments for the installed PyAEDT API."""
        opc = config.get('hfss') or {}
        try:
            params = set(inspect.signature(hfss.analyze_setup).parameters)
        except (TypeError, ValueError):
            params = set()
        kw = {}
        for chave, candidatos in (('cores', ('cores', 'num_cores')),
                                  ('tasks', ('tasks', 'num_tasks')),
                                  ('gpus', ('gpus', 'num_gpu'))):
            if opc.get(chave) is None:
                continue
            nome = next((c for c in candidatos if c in params), None)
            if nome is None:
                log.warning(f"Esta versao do pyaedt nao aceita '{chave}' em analyze_setup — ignorado.")
                continue
            kw[nome] = int(opc[chave])
        if kw:
            log.info(f"HFSS analyze_setup com: {kw}")
        return kw

    def set_er(self, er: float) -> None:
        if self.er_atual != float(er):
            self.hfss.materials[self.cfg['material_name']].permittivity = float(er)
            self.er_atual = float(er)

    def limpar(self) -> None:
        _deletar_reports(self.oModule, ["S11_data", "Z11_data"])
        try:
            self.hfss.odesign.DeleteFullVariation(["All"], False)
        except Exception as e1:
            try:
                self.hfss.cleanup_solution()
            except Exception as e2:
                if not self._avisou:
                    log.warning(f"Limpeza de solucoes indisponivel ({e1}; {e2}) — "
                                "o cache do HFSS nao esta sendo limpo.")
                    self._avisou = True
        try:
            self.hfss.odesktop.ClearMessages("", "", 3)
        except Exception:
            pass

    def simular(self, valores: dict, er: float | None = None,
                csv_dir: str | None = None, prefixo: str = 'tmp'):
        """Evaluate the geometry and return its response and any rejection reason."""
        if er is not None:
            self.set_er(er)
        csv_dir = csv_dir or self.tmp
        for nome, (_, _, unid) in self.cfg['geometric_vars'].items():
            self.hfss[nome] = f"{valores[nome]}{unid}"
        try:
            self.hfss.analyze_setup("Setup1", **self.kw_analise)
            csv_s11 = os.path.join(csv_dir, f'{prefixo}_s11.csv')
            csv_z11 = os.path.join(csv_dir, f'{prefixo}_z11.csv')
            exportar_s11_simulacao(self.oModule, self.sweep, csv_s11)
            exportar_z11_simulacao(self.oModule, self.sweep, csv_z11)
            ds = _ler_csv(csv_s11, ['freq', 's11_db'])
            resp, motivo = analisar_s11(ds['freq'].values, ds['s11_db'].values, self.cfg)
            if resp is None:
                return None, motivo
            dz = _ler_csv(csv_z11, ['freq', 're', 'im'])
            resp['re_z11'] = round(float(np.interp(resp['f_res'], dz['freq'], dz['re'])), 4)
            resp['im_z11'] = round(float(np.interp(resp['f_res'], dz['freq'], dz['im'])), 4)
            resp['vswr'] = round(_vswr_from_s11_db(resp['s11_min']), 4)
            if not all(np.isfinite(resp[k]) for k in ('re_z11', 'im_z11', 'vswr')):
                return None, "Z11_VSWR_NAN"
            return resp, "OK"
        finally:
            self.limpar()

def novo_custo(config: dict, modo: str = 'ponderado', pesos: dict | None = None,
               restricoes: dict | None = None, freq_norm: float | None = None) -> dict:
    fs = config['freq_sweep']
    return {'modo': modo, 'pesos': {**PESOS_PADRAO, **(pesos or {})},
            'restricoes': {**RESTRICOES_PADRAO, **(restricoes or {})},
            'freq_norm': freq_norm or (fs[1] - fs[0]) / 80,
            's11_ideal': S11_IDEAL, 's11_norm': S11_NORM}

def calcular_custo(resp: dict, f_alvo: float, cc: dict):
    """Return the objective value and its individual unweighted components."""
    t = {'t_freq': abs(resp['f_res'] - f_alvo) / cc['freq_norm'],
         't_s11':  max((resp['s11_min'] - cc['s11_ideal']) / cc['s11_norm'], 0.0),
         't_re_z': abs(resp['re_z11'] - 50.0) / 50.0,
         't_im_z': abs(resp['im_z11']) / 20.0,
         't_vswr': abs(resp['vswr'] - 1.0)}
    r = cc['restricoes']
    v = max((resp['s11_min'] - r['s11_max_db']) / cc['s11_norm'], 0.0)
    if r.get('bw_min'):
        v += max((r['bw_min'] - resp['bw']) / r['bw_min'], 0.0)
    t['v_restr'] = v
    if cc['modo'] == 'restrito':
        return t['t_freq'] + PENALIDADE_RESTRICAO * v, t
    w = cc['pesos']
    return (w['freq'] * t['t_freq'] + w['s11'] * t['t_s11'] + w['re_z'] * t['t_re_z'] +
            w['im_z'] * t['t_im_z'] + w['vswr'] * t['t_vswr']), t

def criar_kernel(nome: str, d: int):
    familia, _, tipo = nome.partition('_')
    ls = 1.0 if tipo == 'iso' else np.ones(d)
    b = (1e-2, 1e2)
    if familia == 'matern52':
        base = Matern(ls, b, nu=2.5)
    elif familia == 'matern32':
        base = Matern(ls, b, nu=1.5)
    elif familia == 'rbf':
        base = RBF(ls, b)
    else:
        raise ValueError(f"kernel '{nome}' desconhecido. Opcoes: {KERNELS}")
    return ConstantKernel(1.0) * base + WhiteKernel(1e-3, (1e-5, 1e0))

def _discretizar(P, bounds: np.ndarray, grades):
    """Project sample points onto the fabrication grid inside the search bounds."""
    if grades is None:
        return P
    Q = np.round(np.asarray(P, float) / grades) * grades
    return np.round(np.clip(Q, bounds[:, 0], bounds[:, 1]), 6)

def _unicos_na_grade(P, n: int, bounds: np.ndarray, grades, seed: int) -> np.ndarray:
    """Keep unique grid points and sample replacements when needed."""
    rng = np.random.default_rng(seed)
    vistos, saida = set(), []

    def add(p):
        k = indices_grade(p, grades)
        if k not in vistos:
            vistos.add(k)
            saida.append(p)

    for p in P:
        add(p)
    tent = 0
    while len(saida) < n and tent < 200 * n:
        add(_discretizar(rng.uniform(bounds[:, 0], bounds[:, 1]), bounds, grades))
        tent += 1
    return np.array(saida[:n])

def amostrar_lhs(n: int, bounds: np.ndarray, seed: int, excluir=None, ref_u=None, grades=None):
    """Generate Latin-hypercube candidates with optional exclusion and grid projection."""
    d = len(bounds)

    def lhs(m, s):
        try:
            sam = qmc.LatinHypercube(d=d, seed=s, optimization='random-cd')
        except TypeError:
            sam = qmc.LatinHypercube(d=d, seed=s)
        return _discretizar(qmc.scale(sam.random(m), bounds[:, 0], bounds[:, 1]), bounds, grades)

    if excluir is None:
        P = lhs(n, seed)
        return P if grades is None else _unicos_na_grade(P, n, bounds, grades, seed)

    lo0, hi0 = excluir
    def fora(p):
        return p[~np.all((p >= lo0 - 1e-12) & (p <= hi0 + 1e-12), axis=1)]
    cand = fora(lhs(10 * n, seed))
    rng = np.random.default_rng(seed)
    while len(cand) < 4 * n:
        extra = _discretizar(rng.uniform(bounds[:, 0], bounds[:, 1], size=(20 * n, d)), bounds, grades)
        cand = np.vstack([cand, fora(extra)])
    larg = bounds[:, 1] - bounds[:, 0]
    cu = (cand - bounds[:, 0]) / larg
    ref = [] if ref_u is None or len(ref_u) == 0 else [r for r in ref_u]
    escolhidos = []
    for _ in range(n):
        if ref:
            dist = np.min(np.linalg.norm(cu[:, None, :] - np.array(ref)[None, :, :], axis=2), axis=1)
        else:
            dist = np.ones(len(cu))
        dist[escolhidos] = -1
        i = int(np.argmax(dist))
        escolhidos.append(i)
        ref.append(cu[i])
    return cand[escolhidos]

def expected_improvement(X, gp, y_best, xi=0.01):
    mu, sigma = gp.predict(X, return_std=True)
    s = np.maximum(sigma, 1e-9)
    imp = y_best - mu - xi
    z = imp / s
    ei = imp * norm.cdf(z) + s * norm.pdf(z)
    return np.where(sigma < 1e-9, 0.0, ei)

def treinar_classificador(X, viavel, seed):
    if len(set(viavel)) < 2:
        return None
    try:
        return GaussianProcessClassifier(kernel=RBF(length_scale=1.0), random_state=seed,
                                         n_restarts_optimizer=2).fit(X, viavel)
    except Exception as e:
        log.warning(f"Classificador de viabilidade nao treinado nesta iteracao: {e}")
        return None

def acquisition(X, gp, y_best, clf=None, xi=0.01):
    ei = expected_improvement(X, gp, y_best, xi)
    return ei if clf is None else ei * clf.predict_proba(X)[:, 1]

def proximo_ponto(gp, y_best, lim, seed, clf=None, xi=0.01, n_restarts=4):
    """Maximize feasibility-weighted expected improvement using differential evolution."""
    def neg(x):
        x = np.asarray(x, float)
        if x.ndim == 1:
            return -float(acquisition(x.reshape(1, -1), gp, y_best, clf, xi)[0])
        return -acquisition(x.T, gp, y_best, clf, xi)

    melhor_x, melhor_v = None, np.inf
    for r in range(n_restarts):
        res = differential_evolution(neg, lim, seed=seed + r * 997, maxiter=300, tol=1e-8,
                                     polish=True, vectorized=True, updating='deferred')
        if res.fun < melhor_v:
            melhor_x, melhor_v = res.x, res.fun
    return melhor_x, -melhor_v

def variaveis_na_borda(valores: dict, config: dict, margem: float = MARGEM_BORDA) -> dict:
    na_borda = {}
    for nome, (mn, mx, _) in config['geometric_vars'].items():
        pos = (valores[nome] - mn) / (mx - mn)
        if pos <= margem:
            na_borda[nome] = 'inferior'
        elif pos >= 1 - margem:
            na_borda[nome] = 'superior'
    return na_borda

def expandir_bounds(config: dict, na_borda: dict, fator: float = FATOR_EXPANSAO):
    """Extend search bounds without crossing specified physical limits."""
    novo = copy.deepcopy(config)
    expandidas, bloqueadas = [], []
    for nome, lado in na_borda.items():
        mn, mx, unid = config['geometric_vars'][nome]
        delta = fator * (mx - mn) * (1.5 if nome in VARS_PRIORITARIAS else 1.0)
        lim_mn, lim_mx = (config.get('limites_fisicos') or {}).get(nome, (-np.inf, np.inf))
        lim_mn = max(lim_mn, 1e-3)
        n_mn, n_mx = (max(mn - delta, lim_mn), mx) if lado == 'inferior' \
            else (mn, min(mx + delta, lim_mx))
        if OTIMIZAR_NA_GRADE:
            n_mn, n_mx = alinhar_limites(n_mn, n_mx, grade_da_variavel(nome, config))
        nd = 6 if OTIMIZAR_NA_GRADE else 4
        n_mn, n_mx = round(n_mn, nd), round(n_mx, nd)
        if (n_mn, n_mx) == (round(mn, nd), round(mx, nd)):
            bloqueadas.append(nome)
            continue
        novo['geometric_vars'][nome] = (n_mn, n_mx, unid)
        expandidas.append(nome)
        log.info(f"  Expandindo '{nome}' ({lado}): [{mn}, {mx}] -> {novo['geometric_vars'][nome][:2]}")
    return novo, expandidas, bloqueadas

class Campanha:
    """Maintain the observations and surrogate model for one optimization target."""

    def __init__(self, av: Avaliador, config: dict, f_alvo: float, er: float, pasta: str,
                 cc: dict, seed: int, kernel: str):
        self.av, self.cfg, self.f_alvo, self.er, self.pasta = av, config, f_alvo, er, pasta
        self.cc, self.seed, self.kernel = cc, seed, kernel
        self.nomes = list(config['geometric_vars'])
        self.grades = grades_do_config(config, self.nomes)
        self.cache = {}
        self.reg, self.n_hfss, self.etapa, self.kernel_final = [], 0, 0, ''
        self.csv = os.path.join(pasta, 'resultados_bo.csv')

    @property
    def bounds(self) -> np.ndarray:
        return montar_espaco_busca(self.cfg)[1]

    def df(self) -> pd.DataFrame:
        d = pd.DataFrame(self.reg)
        if len(d):
            d['rejeitado'] = d['rejeitado'].astype(bool)
        return d

    def validos(self) -> pd.DataFrame:
        d = self.df()
        return d[~d['rejeitado']] if len(d) else d

    def melhor(self):
        v = self.validos()
        if v.empty:
            raise RuntimeError("Nenhum ponto valido ate agora.")
        return v.loc[v['custo'].idxmin()]

    def snap(self, x) -> np.ndarray:
        """Project a normalized candidate to the current fabrication-grid bounds."""
        x = np.asarray(x, float)
        return _discretizar(x, self.bounds, self.grades) if OTIMIZAR_NA_GRADE else x

    def chave(self, x) -> tuple:
        return indices_grade(x, self.grades)

    def mascara_na_grade(self, df: pd.DataFrame) -> np.ndarray:
        return mascara_na_grade(df, self.nomes, self.grades)

    def motivo_parada_init(self) -> str:
        """Check whether early termination of Latin-hypercube sampling is allowed."""
        if not INIT_PARADA_ANTECIPADA:
            return ''
        v = self.validos()
        n = len(v)
        if n >= INIT_MIN_VALIDOS:
            return f"{n} pontos validos (>= {INIT_MIN_VALIDOS})"
        if (INIT_CUSTO_BOM is not None and n >= INIT_MIN_VALIDOS_COM_CUSTO_BOM
                and float(v['custo'].min()) <= INIT_CUSTO_BOM):
            return (f"custo {float(v['custo'].min()):.3f} <= {INIT_CUSTO_BOM} "
                    f"com {n} pontos validos (>= {INIT_MIN_VALIDOS_COM_CUSTO_BOM})")
        return ''

    def escolher_na_grade(self, x_u, gp, y_best, clf, xi):
        """Find an untested fabrication-grid neighbor near the acquisition optimum."""
        b = self.bounds
        lo, hi = b[:, 0], b[:, 1]
        d = len(lo)
        base = np.floor((lo + np.asarray(x_u, float) * (hi - lo)) / self.grades + 1e-9)
        rng = np.random.default_rng(self.seed + self.n_hfss)
        if d <= 10:
            cantos = np.array(list(itertools.product((0, 1), repeat=d)))
        else:
            cantos = rng.integers(0, 2, size=(2048, d))
        amplos = rng.integers(-2, 4, size=(4000, d))
        for offs in (cantos, amplos):
            C = np.unique(_discretizar((base + offs) * self.grades, b, self.grades), axis=0)
            livres = np.array([c for c in C if self.chave(c) not in self.cache])
            if len(livres):
                aq = acquisition((livres - lo) / (hi - lo), gp, y_best, clf, xi)
                i = int(np.argmax(aq))
                return livres[i], float(aq[i])
        return None, 0.0

    def adicionar_previos(self, df_prev: pd.DataFrame) -> None:
        """Reuse compatible previously computed HFSS responses."""
        for _, r in df_prev.iterrows():
            reg = {n: float(r[n]) for n in self.nomes}
            reg.update(er=self.er, f_alvo=self.f_alvo, fase='previo', etapa=-1, iteracao=0, ei=None,
                       tempo_s=0.0, tempo_decisao_s=0.0, setup_id=self.av.setup_id,
                       motivo=r['motivo'], rejeitado=bool(r['rejeitado']), custo=np.nan)
            if not reg['rejeitado']:
                resp = {k: float(r[k]) for k in ('f_res', 's11_min', 'bw', 'bw_pct', 're_z11', 'im_z11', 'vswr')}
                resp['bw_trunc'] = bool(r.get('bw_trunc', False))
                custo, termos = calcular_custo(resp, self.f_alvo, self.cc)
                reg.update(resp, **termos, custo=custo)
            x = np.array([reg[n] for n in self.nomes])
            if OTIMIZAR_NA_GRADE and na_grade(x, self.grades):
                self.cache[self.chave(x)] = PENALIDADE if reg['rejeitado'] else reg['custo']
            self.reg.append(reg)
        self._salvar()

    def _salvar(self) -> None:
        pd.DataFrame(self.reg).to_csv(self.csv, index=False)

    def avaliar(self, x, fase: str, ei=None, t_dec: float = 0.0) -> float:
        x = self.snap(x)
        valores = dict(zip(self.nomes, (float(v) for v in x)))
        if OTIMIZAR_NA_GRADE and self.chave(x) in self.cache:
            log.info(f"  ponto ja simulado na grade — reaproveitado (sem nova simulacao): {valores}")
            return self.cache[self.chave(x)]
        t0 = time.time()
        try:
            resp, motivo = self.av.simular(valores)
        except Exception as e:
            log.error(f"Erro inesperado ao simular {valores}: {e}", exc_info=True)
            resp, motivo = None, "ERRO_HFSS"
        t = round(time.time() - t0, 2)
        self.n_hfss += 1
        reg = {**valores, 'er': self.er, 'f_alvo': self.f_alvo, 'fase': fase, 'etapa': self.etapa,
               'iteracao': self.n_hfss, 'ei': ei, 'tempo_s': t, 'tempo_decisao_s': round(t_dec, 3),
               'setup_id': self.av.setup_id, 'motivo': motivo, 'rejeitado': resp is None,
               'custo': np.nan}
        if resp is None:
            log.info(f"  [{self.n_hfss}] REJEITADO ({motivo}) {valores}")
            custo = PENALIDADE
        else:
            custo, termos = calcular_custo(resp, self.f_alvo, self.cc)
            reg.update(resp, **termos, custo=custo)
            log.info(f"  [{self.n_hfss}] f_res={resp['f_res']} S11={resp['s11_min']} dB "
                     f"BW={resp['bw']} custo={custo:.4f} ({fase}, {t:.0f}s)")
        if OTIMIZAR_NA_GRADE:
            self.cache[self.chave(x)] = custo
        self.reg.append(reg)
        self._salvar()
        return custo

    def iteracoes_bo(self, n: int, fase: str, xi_ini: float, xi_fim: float,
                     raio: float | None = None) -> bool:
        """Run the requested global or local acquisition iterations."""
        d = len(self.nomes)
        for it in range(n):
            t0 = time.time()
            df = self.df()
            lo, hi = self.bounds[:, 0], self.bounds[:, 1]
            X_all = (df[self.nomes].values - lo) / (hi - lo)
            viavel = (~df['rejeitado']).astype(int).values
            val = df[~df['rejeitado']]
            Xv = (val[self.nomes].values - lo) / (hi - lo)
            yv = val['custo'].values
            gp = GaussianProcessRegressor(criar_kernel(self.kernel, d), normalize_y=True,
                                          n_restarts_optimizer=5, random_state=self.seed).fit(Xv, yv)
            self.kernel_final = str(gp.kernel_)
            clf = treinar_classificador(X_all, viavel, self.seed)
            if raio is None:
                lim = np.column_stack([np.zeros(d), np.ones(d)])
            else:
                c = Xv[int(np.argmin(yv))]
                lim = np.column_stack([np.clip(c - raio, 0, 1), np.clip(c + raio, 0, 1)])
            xi = xi_ini + (xi_fim - xi_ini) * it / max(n - 1, 1)
            x_u, aq = proximo_ponto(gp, yv.min(), lim, self.seed + self.n_hfss, clf, xi)
            if OTIMIZAR_NA_GRADE:
                x_real, aq = self.escolher_na_grade(x_u, gp, yv.min(), clf, xi)
                if x_real is None:
                    log.info("Vizinhanca do otimo na grade ja foi toda simulada — parando esta fase.")
                    return False
            else:
                x_real = lo + x_u * (hi - lo)
            log.info(f"[{fase} {it + 1}/{n}] aquisicao={aq:.3e} xi={xi:.4f} | GP {len(yv)} pts | "
                     f"classificador {'ativo' if clf is not None else 'inativo'}")
            if aq < EI_MIN:
                log.info("Aquisicao desprezivel — parando esta fase.")
                return False
            self.avaliar(x_real, fase, ei=aq, t_dec=time.time() - t0)
        return True

def carregar_previos(raiz: str, er: float, pasta_atual: str, setup_id: str,
                     bounds: np.ndarray, nomes: list) -> pd.DataFrame:
    """Load compatible observations from another frequency campaign."""
    achados = []
    for p in glob.glob(os.path.join(raiz, f'f*_er{er:g}', 'resultados_bo.csv')):
        if os.path.abspath(os.path.dirname(p)) == os.path.abspath(pasta_atual):
            continue
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if {'setup_id', 'f_res', 'fase'} <= set(d.columns):
            achados.append(d[(d['fase'] != 'previo') & (d['setup_id'] == setup_id)])
    if not achados:
        return pd.DataFrame()
    d = pd.concat(achados, ignore_index=True)
    X = d[nomes].values
    d = d[np.all((X >= bounds[:, 0] - 1e-12) & (X <= bounds[:, 1] + 1e-12), axis=1)]
    return d.drop_duplicates(subset=nomes).reset_index(drop=True)

def selecionar_final(validos: pd.DataFrame, config: dict, criterio: str, tol: float):
    """Select a final candidate from the recorded near-optimal geometries."""
    cmin = validos['custo'].min()
    quase = validos[validos['custo'] <= cmin * (1 + tol) + 1e-3].copy()
    a, b = config.get('area_vars', ('a', 'b'))
    quase['area'] = quase[a] * quase[b]
    if criterio == 'bw':
        esc = quase.sort_values(['bw', 'custo'], ascending=[False, True]).iloc[0]
    elif criterio == 'area':
        esc = quase.sort_values(['area', 'custo']).iloc[0]
    else:
        esc = quase.sort_values('custo').iloc[0]
    return esc, quase

def gerar_figuras(df: pd.DataFrame, nomes: list, bounds: np.ndarray, pasta: str) -> list:
    plt.rcParams.update({'font.size': 11, 'font.family': 'serif', 'axes.grid': True,
                         'grid.alpha': 0.3, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
    d = df[df['fase'] != 'previo'].sort_values('iteracao')
    saidas = []

    fig, ax = plt.subplots(figsize=(7, 5))
    cores = {'init': '#888888', 'novo': '#2ca02c', 'bo': '#1f77b4', 'refino': '#ff7f0e',
             'aleatorio': '#888888', 'de': '#9467bd'}
    for fase, g in d[~d['rejeitado']].groupby('fase'):
        ax.scatter(g['iteracao'], g['custo'], s=30, color=cores.get(fase, 'k'), label=fase, zorder=3)
    ax.plot(d['iteracao'], d['custo'].where(~d['rejeitado']).cummin(), color='#d62728', lw=2,
            label='best so far', zorder=2)
    for it in d.loc[d['etapa'].diff().fillna(0) > 0, 'iteracao']:
        ax.axvline(it - 0.5, color='gray', ls=':', lw=1)
    ax.set_yscale('log'); ax.set_xlabel('HFSS evaluation index'); ax.set_ylabel('Cost (log)')
    ax.set_title('Convergence'); ax.legend(frameon=False, fontsize=8)
    saidas.append(os.path.join(pasta, 'fig1_convergencia.png')); fig.savefig(saidas[-1]); plt.close(fig)

    fig, ax = plt.subplots(figsize=(1.6 * len(nomes) + 2, 5))
    xs = np.arange(len(nomes))
    def norm_(g):
        return (g[nomes].values - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    rej, val = d[d['rejeitado']], d[~d['rejeitado']]
    for row in (norm_(rej) if len(rej) else []):
        ax.plot(xs, row, color='#999999', alpha=0.4, lw=1, ls='--')
    if len(val):
        c = val['custo'].values
        cn = (c - c.min()) / (c.max() - c.min() + 1e-12)
        cmap = plt.get_cmap('viridis_r')
        for row, ci in zip(norm_(val), cn):
            ax.plot(xs, row, color=cmap(ci), alpha=0.6, lw=1.2)
        ax.plot(xs, norm_(val)[int(np.argmin(c))], color='red', lw=2.5, label='best')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(c.min(), c.max())); sm.set_array([])
        fig.colorbar(sm, ax=ax, label='Cost (valid)')
    ax.set_xticks(xs); ax.set_xticklabels(nomes)
    ax.set_ylabel('Normalized value within final bounds'); ax.set_title('Parameter space exploration')
    ax.legend(frameon=False, fontsize=9)
    saidas.append(os.path.join(pasta, 'fig3_exploracao_espaco.png')); fig.savefig(saidas[-1]); plt.close(fig)

    bo = d[d['fase'].isin(['bo', 'refino']) & d['ei'].notna()]
    if len(bo):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(bo['iteracao'], bo['ei'].astype(float), marker='o', color='#9467bd', lw=1.5)
        ax.axhline(EI_MIN, color='gray', ls='--', lw=1, label=f'threshold ({EI_MIN:.0e})')
        ax.set_yscale('log'); ax.set_xlabel('HFSS evaluation index')
        ax.set_ylabel('max EI x P(valid)'); ax.set_title('Acquisition decay')
        ax.legend(frameon=False, fontsize=9)
        saidas.append(os.path.join(pasta, 'fig4_ei_decay.png')); fig.savefig(saidas[-1]); plt.close(fig)
    return saidas

def run_campanha(hfss, config: dict, f_alvo: float, er: float, pasta: str, *,
                 metodo: str = 'bo', seed: int = SEMENTE, kernel: str = KERNEL_PADRAO,
                 modo_custo: str = 'ponderado', pesos: dict | None = None,
                 restricoes: dict | None = None, n_init: int | None = None,
                 n_iter: int | None = None, max_remodelagens: int = MAX_REMODELAGENS,
                 margem_borda: float = MARGEM_BORDA, fracao_refino: float = FRACAO_REFINO,
                 criterio_desempate: str = CRITERIO_DESEMPATE, tol_quase_otimo: float = TOL_QUASE_OTIMO,
                 reusar_entre_alvos: bool = False, finalizar: bool = True) -> dict:
    """Run a single HFSS-in-the-loop search or its comparison baseline."""
    _validar_config(config)
    os.makedirs(pasta, exist_ok=True)
    configurar_log(os.path.join(pasta, 'bo_log.log'))
    t_ini = time.time()

    cfg = copy.deepcopy(config)
    if OTIMIZAR_NA_GRADE:
        cfg = alinhar_config_a_grade(cfg)
    nomes, _ = montar_espaco_busca(cfg)
    d = len(nomes)
    n_init, n_iter = orcamento(d, n_init, n_iter)
    n_refino = int(round(fracao_refino * n_iter)) if metodo == 'bo' else 0
    n_expl = n_iter - n_refino
    cc = novo_custo(cfg, modo_custo, pesos, restricoes)

    av = Avaliador(hfss, cfg, er, None)
    camp = Campanha(av, cfg, f_alvo, er, pasta, cc, seed, kernel)
    log.info("=" * 60)
    log.info(f"{metodo.upper()} | f_alvo={f_alvo} {cfg['freq_unit']} | er={er} | seed={seed} | kernel={kernel}")
    log.info(f"Orcamento: {n_init} iniciais + {n_expl} exploracao + {n_refino} refino | custo={modo_custo}")
    g_amostra = camp.grades if OTIMIZAR_NA_GRADE else None
    log.info("Grade de fabricacao: " + (", ".join(f"{n}={g:g}" for n, g in zip(nomes, camp.grades))
                                        if OTIMIZAR_NA_GRADE else "DESLIGADA (otimiza no continuo)"))
    log.info("Parada antecipada do LHS inicial: " +
             (f"LIGADA (>= {INIT_MIN_VALIDOS} validos, ou custo <= {INIT_CUSTO_BOM} com >= "
              f"{INIT_MIN_VALIDOS_COM_CUSTO_BOM} validos)" if INIT_PARADA_ANTECIPADA else "desligada"))
    log.info("=" * 60)

    historico_bounds = []
    n_prev = 0
    n_init_exec, motivo_init = 0, ''
    if metodo == 'aleatorio':
        rng = np.random.default_rng(seed)
        b = camp.bounds
        for x in rng.uniform(b[:, 0], b[:, 1], size=(n_init + n_iter, d)):
            camp.avaliar(x, 'aleatorio')
    elif metodo == 'de':
        class _Fim(Exception):
            pass
        total = n_init + n_iter
        def obj(x):
            if camp.n_hfss >= total:
                raise _Fim
            return camp.avaliar(x, 'de')
        try:
            differential_evolution(obj, camp.bounds, seed=seed, popsize=2, maxiter=10 ** 4, tol=0,
                                   polish=False, init='latinhypercube')
        except _Fim:
            pass
    elif metodo == 'bo':

        raiz = os.path.dirname(os.path.abspath(pasta))
        if reusar_entre_alvos:
            prev = carregar_previos(raiz, er, pasta, av.setup_id, camp.bounds, nomes)
            if len(prev):
                camp.adicionar_previos(prev)
                n_prev = int((~prev['rejeitado']).sum())
                log.info(f"Reaproveitando {len(prev)} avaliacoes de outras frequencias (er={er:g}).")
        n_lhs = max(d + 1, n_init - n_prev) if n_prev else n_init
        for i, x in enumerate(amostrar_lhs(n_lhs, camp.bounds, seed, grades=g_amostra)):
            motivo_init = camp.motivo_parada_init()
            if motivo_init:
                log.info(f"[INIT] parada antecipada apos {i} de {n_lhs} pontos do LHS: {motivo_init} "
                         f"— indo para a exploracao.")
                break
            log.info(f"[INIT {i + 1}/{n_lhs}]")
            camp.avaliar(x, 'init')
            n_init_exec = i + 1
        if len(camp.validos()) < 3:
            raise RuntimeError("Poucos pontos validos na fase inicial (< 3) — revise "
                               "geometric_vars, er_alvo ou os thresholds de S11.")
        camp.iteracoes_bo(n_expl, 'bo', 0.02, 0.005)

        for etapa in range(max_remodelagens + 1):
            best = camp.melhor()
            valores = {n: float(best[n]) for n in nomes}
            na_borda = variaveis_na_borda(valores, camp.cfg, margem_borda)
            historico_bounds.append((etapa, copy.deepcopy(camp.cfg), na_borda, [], []))
            if not na_borda:
                log.info(f"[REMODELAGEM] etapa {etapa}: nenhuma variavel na borda — bounds OK.")
                break
            if etapa == max_remodelagens:
                log.warning(f"[REMODELAGEM] limite de {max_remodelagens} atingido; na borda: {list(na_borda)}")
                break
            novo_cfg, expandidas, bloqueadas = expandir_bounds(camp.cfg, na_borda)
            historico_bounds[-1] = (etapa, copy.deepcopy(camp.cfg), na_borda, expandidas, bloqueadas)
            if not expandidas:
                log.warning(f"[REMODELAGEM] variaveis na borda no limite fisico: {bloqueadas}")
                break
            ant = camp.bounds
            camp.cfg = novo_cfg
            camp.etapa = etapa + 1
            novo = camp.bounds
            v_nova = 1 - float(np.prod((ant[:, 1] - ant[:, 0]) / (novo[:, 1] - novo[:, 0])))
            n_novos = max(MIN_NOVOS, math.ceil(FRACAO_NOVOS * n_init * v_nova))
            n_it = max(MIN_ITER_REMODEL, math.ceil(FRACAO_ITER_REMODEL * n_expl))
            log.info(f"[REMODELAGEM {etapa + 1}/{max_remodelagens}] reaproveita {camp.n_hfss + n_prev} "
                     f"pontos; novos: {n_novos} (regiao nova = {100 * v_nova:.0f}% do volume) + {n_it} iteracoes")
            ref_u = (camp.df()[nomes].values - novo[:, 0]) / (novo[:, 1] - novo[:, 0])
            for x in amostrar_lhs(n_novos, novo, seed + etapa + 1, excluir=(ant[:, 0], ant[:, 1]), ref_u=ref_u,
                              grades=g_amostra):
                camp.avaliar(x, 'novo')
            camp.iteracoes_bo(n_it, 'bo', 0.02, 0.005)

        if n_refino:
            camp.iteracoes_bo(n_refino, 'refino', 0.005, 0.002, raio=RAIO_REFINO)
    else:
        raise ValueError("metodo deve ser 'bo', 'aleatorio' ou 'de'.")

    df = camp.df()
    resultado = {'df_result': df, 'n_hfss': camp.n_hfss, 'tempo_s': round(time.time() - t_ini, 1)}
    if not finalizar:
        with open(os.path.join(pasta, 'concluido.txt'), 'w') as fh:
            fh.write('ok')
        return resultado

    val = camp.validos()
    if OTIMIZAR_NA_GRADE:

        val = val[camp.mascara_na_grade(val)]
        if val.empty:
            raise RuntimeError("Nenhum ponto valido na grade de fabricacao.")
    esc, quase = selecionar_final(val, camp.cfg, criterio_desempate, tol_quase_otimo)
    quase_ord = quase.sort_values('custo')
    quase_ord.to_csv(os.path.join(pasta, 'conjunto_quase_otimo.csv'), index=False)
    esc_dict = {n: float(esc[n]) for n in nomes}
    log.info(f"Melhor custo={val['custo'].min():.5f} | quase-otimos={len(quase)} | "
             f"escolhido por '{criterio_desempate}': {esc_dict}")
    if OTIMIZAR_NA_GRADE:

        arredondado = False
        ordem = [esc] + [r for i, r in quase_ord.iterrows() if i != esc.name]
        for k, cand in enumerate(ordem[:MAX_CANDIDATOS_REVALIDACAO]):
            valores_raw = {n: float(cand[n]) for n in nomes}
            valores_fab = arredondar_fabricacao(valores_raw, camp.cfg)
            resp_fab, motivo = av.simular(valores_fab, csv_dir=pasta, prefixo='design_final')
            if resp_fab is not None:
                esc = cand
                break
            log.warning(f"Revalidacao do candidato {k + 1} REJEITADA ({motivo}) — tentando o proximo.")
        else:
            raise RuntimeError("Revalidacao falhou para todos os candidatos testados.")
    else:
        valores_raw = esc_dict
        valores_fab = arredondar_fabricacao(valores_raw, camp.cfg)
        resp_fab, motivo = av.simular(valores_fab, csv_dir=pasta, prefixo='design_final')
        arredondado = True
        if resp_fab is None:
            log.warning(f"Ponto arredondado REJEITADO ({motivo}) — usando o ponto bruto.")
            arredondado, valores_fab = False, valores_raw
            resp_fab, motivo = av.simular(valores_fab, csv_dir=pasta, prefixo='design_final')
            if resp_fab is None:
                raise RuntimeError(f"Revalidacao do ponto bruto falhou ({motivo}).")
    custo_fab, termos_fab = calcular_custo(resp_fab, f_alvo, cc)

    na_borda_final = variaveis_na_borda(valores_fab, camp.cfg, margem_borda)
    linha = dict(valores_fab)
    linha.update({'er': er, 'f_alvo': f_alvo, 'f_res': round(resp_fab['f_res'], 4),
                  'S11_min_dB': round(resp_fab['s11_min'], 2), 'BW10': round(resp_fab['bw'], 4),
                  'BW10_pct': round(resp_fab['bw_pct'], 3), 'BW_truncada': resp_fab['bw_trunc'],
                  'Re_Z11_ohm': round(resp_fab['re_z11'], 2), 'Im_Z11_ohm': round(resp_fab['im_z11'], 2),
                  'VSWR': round(resp_fab['vswr'], 3), 'custo_final': round(custo_fab, 6),
                  **{k + '_final': round(v, 6) for k, v in termos_fab.items()},
                  'arredondado': arredondado, 'otimizado_na_grade': OTIMIZAR_NA_GRADE,
                  'grade_fab': GRADE_FABRICACAO,
                  'custo_bruto_escolhido': round(float(esc['custo']), 6),
                  'custo_minimo_historico': round(float(val['custo'].min()), 6),
                  'd_f_res_arred': round(resp_fab['f_res'] - float(esc['f_res']), 5),
                  'd_S11_arred_dB': round(resp_fab['s11_min'] - float(esc['s11_min']), 3),
                  'd_BW10_arred': round(resp_fab['bw'] - float(esc['bw']), 5),
                  'd_Re_Z11_arred': round(resp_fab['re_z11'] - float(esc['re_z11']), 3),
                  'd_Im_Z11_arred': round(resp_fab['im_z11'] - float(esc['im_z11']), 3),
                  'criterio_desempate': criterio_desempate, 'n_quase_otimos': len(quase),
                  'metodo': metodo, 'seed': seed, 'kernel': kernel, 'modo_custo': modo_custo})
    log.info("RESUMO: " + ", ".join(f"{k}={v}" for k, v in linha.items()))

    pd.DataFrame([{'etapa': e, 'variavel': n, 'min': mn, 'max': mx, 'na_borda': na.get(n, ''),
                   'expandida': n in ex, 'bloqueada_limite_fisico': n in bl}
                  for e, c, na, ex, bl in historico_bounds
                  for n, (mn, mx, _) in c['geometric_vars'].items()]
                 ).to_csv(os.path.join(pasta, 'historico_remodelagem_bounds.csv'), index=False)
    figs = gerar_figuras(df, nomes, camp.bounds, pasta)

    motivos = df[df['fase'] != 'previo']['motivo'].value_counts().to_dict()
    extras = {'status': 'OK', 'n_aval_hfss': camp.n_hfss, 'n_previos_reaproveitados': n_prev,
              'n_init_previsto': n_init, 'n_init_executados': n_init_exec,
              'init_motivo_parada': motivo_init or 'LHS completo',
              'n_rejeitados': int(df[df['fase'] != 'previo']['rejeitado'].sum()),
              'motivos': json.dumps(motivos), 'n_etapas': camp.etapa + 1,
              'bounds_finais': json.dumps({n: list(v[:2]) for n, v in camp.cfg['geometric_vars'].items()}),
              'bounds_na_borda_final': ','.join(na_borda_final),
              'tempo_hfss_min': round(df['tempo_s'].sum() / 60, 1),
              'tempo_decisao_min': round(df['tempo_decisao_s'].sum() / 60, 2),
              'tempo_total_min': round((time.time() - t_ini) / 60, 1),
              'kernel_final': camp.kernel_final, 'pasta': pasta}
    pd.DataFrame([{**linha, **extras}]).to_csv(os.path.join(pasta, 'resultado_final.csv'), index=False)
    with open(os.path.join(pasta, 'parametros.json'), 'w', encoding='utf-8') as fh:
        json.dump({'f_alvo': f_alvo, 'er': er, 'metodo': metodo, 'seed': seed, 'kernel': kernel,
                   'custo': cc, 'n_init': n_init, 'n_iter': n_iter, 'n_refino': n_refino,
                   'raio_refino': RAIO_REFINO, 'ei_min': EI_MIN, 'tol_quase_otimo': tol_quase_otimo,
                   'criterio_desempate': criterio_desempate, 'grade_fabricacao': GRADE_FABRICACAO,
                   'otimizar_na_grade': OTIMIZAR_NA_GRADE,
                   'init_parada_antecipada': INIT_PARADA_ANTECIPADA, 'init_min_validos': INIT_MIN_VALIDOS,
                   'init_custo_bom': INIT_CUSTO_BOM,
                   'init_min_validos_custo_bom': INIT_MIN_VALIDOS_COM_CUSTO_BOM,
                   'grades_por_variavel': {n: float(g) for n, g in zip(nomes, camp.grades)},
                   'bounds_iniciais_na_grade': {n: list(v[:2]) for n, v in cfg['geometric_vars'].items()},
                   'bounds_iniciais': {n: list(v[:2]) for n, v in config['geometric_vars'].items()},
                   'bounds_finais': {n: list(v[:2]) for n, v in camp.cfg['geometric_vars'].items()},
                   'limiares_s11_db': [config['s11_reject_db'], config['s11_threshold_db'], config['bw_threshold_db']],
                   'malha': av.setup_id, 'ambiente': info_ambiente(hfss)},
                  fh, indent=1, default=str, ensure_ascii=False)
    resultado.update(linha=linha, extras=extras, figuras=figs)
    return resultado

def run_bayesian_optimization_adaptativo(config: dict, f_res_alvo, er_alvo, *,
                                         retomar: bool = False, **kwargs) -> pd.DataFrame:
    """Run independent campaigns for the requested frequencies and permittivities."""
    _validar_config(config)
    freqs, ers = _como_lista(f_res_alvo), _como_lista(er_alvo)
    f_min, f_max = config['freq_valid_range']
    fora = [f for f in freqs if not (f_min <= f <= f_max)]
    if fora:
        raise ValueError(f"f_res_alvo {fora} fora de freq_valid_range {config['freq_valid_range']}.")

    raiz = config['base_dir']
    os.makedirs(raiz, exist_ok=True)
    lote = configurar_log(os.path.join(raiz, 'lote.log'), NOME_LOG + '_lote')
    resumo_path = os.path.join(raiz, 'resumo_geral.csv')
    unid, total = config['freq_unit'], len(freqs) * len(ers)
    lote.info(f"LOTE: {freqs} x {ers} = {total} campanhas")

    linhas, t_lote = [], time.time()
    n = 0
    for f in freqs:
        for er in ers:
            n += 1
            pasta = os.path.join(raiz, _nome_pasta(f, er, unid))
            lote.info(f"[{n}/{total}] f_alvo={f} {unid} | er={er} -> {pasta}")
            final_csv = os.path.join(pasta, 'resultado_final.csv')
            if retomar and os.path.exists(final_csv):
                lote.info("   ja concluida (retomar=True) — pulando.")
                linhas.append(pd.read_csv(final_csv).iloc[0].to_dict())
                continue
            t0 = time.time()
            try:
                hfss = _SESSAO.obter(config)
                if n == 1:
                    with open(os.path.join(raiz, 'ambiente.json'), 'w', encoding='utf-8') as fh:
                        json.dump(info_ambiente(hfss), fh, indent=1, ensure_ascii=False)
                res = run_campanha(hfss, config, f, er, pasta, **kwargs)
                linha = {**res['linha'], **res['extras']}
                lote.info(f"   concluida em {(time.time() - t0) / 60:.1f} min | {res['n_hfss']} simulacoes | "
                          f"f_res={linha['f_res']} | S11={linha['S11_min_dB']} dB | BW10={linha['BW10']} | "
                          f"custo={linha['custo_final']}")
            except Exception as e:
                lote.error(f"   FALHOU: {e}", exc_info=True)
                _SESSAO.fechar()
                linha = {'f_alvo': f, 'er': er, 'status': f'FALHOU: {e}', 'pasta': pasta}
            linhas.append(linha)
            pd.DataFrame(linhas).to_csv(resumo_path, index=False)

    df_final = pd.DataFrame(linhas)
    df_final.to_csv(resumo_path, index=False)
    lote.info(f"LOTE concluido em {(time.time() - t_lote) / 3600:.2f} h | resumo: {resumo_path}")
    return df_final
