"""Initial geometry boxes used for the two manuscript frequencies (dimensions in micrometres)."""
import copy

INITIAL_BOXES = {
    10.0: {
        'a': (8.6, 9.3, 'um'), 'b': (8.1, 9.2, 'um'), 'd': (6.8, 7.4, 'um'),
        'lstrip': (13.8, 14.4, 'um'), 'wstrip': (1.3, 1.5, 'um'),
        'lslot': (0.3, 0.6, 'um'), 'wslot': (5.7, 6.4, 'um'),
    },
    11.0: {
        'a': (8.0, 8.7, 'um'), 'b': (8.0, 9.0, 'um'), 'd': (5.5, 6.9, 'um'),
        'lstrip': (13.2, 13.6, 'um'), 'wstrip': (1.4, 1.7, 'um'),
        'lslot': (0.8, 1.1, 'um'), 'wslot': (5.6, 6.7, 'um'),
    },
}
SWEEP_BANDS = {10.0: ((8.5, 12.5), (8.7, 12.3)),
               11.0: ((9.0, 13.0), (9.2, 12.8))}

def target_config(base, frequency):
    frequency = float(frequency)
    if frequency not in INITIAL_BOXES:
        raise ValueError('Only 10 and 11 THz configurations are defined in this paper')
    cfg = copy.deepcopy(base)
    cfg['geometric_vars'] = copy.deepcopy(INITIAL_BOXES[frequency])
    cfg['freq_sweep'], cfg['freq_valid_range'] = SWEEP_BANDS[frequency]
    cfg['freq_sweep_points'] = 1000
    return cfg
