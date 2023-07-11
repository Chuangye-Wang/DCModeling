""" Constants used in this package. """

GAS_CONSTANT = 8.3145  # unit: J/mol/K
CELSIUS_KELVIN_OFFSET = 273  # unit: K
ATOMIC_PERCENT_MAX = 100

A_CONSTANT = 10

DIFFUSION_TYPES = {"DT": "Tracer", "DI": "Intrinsic", "DC": "Inter"}
ELEMENTS_ORDER = {"A": 0, "B": 1}

BROWN_ASHBY_CORRELATION = {
    "fcc": [5.4E-5, 18.4],
    "bcc_rare_earth": [1.5E-6, 9.3],
    "bcc_alkali": [2.5E-5, 14.7],
    "bcc_transition": [1.6E-4, 17.8],
    "hcp": [4.9E-5, 17.3],
    "tetragonal": [3.2E-4, 21.9],
    "alkali_halide": [2.8E-3, 22.8],
    "simple_oxides": [5.3E-4, 23.4],
    "silicate": [3, 31.3],
    "trigonal": [5.8E-3, 26.8],
    "ice": [1E-3, 26.3],
    "carbide": [0.2, 24],
    "diamond_cubic": [6.3E-2, 33.9],
    "hcp_graphite": [2.4E-4, 20],
}

P_VALUES = {
    "fcc": 0.28,
    "bcc": 0.4,
    "hcp": 0.28,
}
