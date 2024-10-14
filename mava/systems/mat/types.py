from typing_extensions import NamedTuple

class ValueNormParams(NamedTuple):
    beta: float = 0.99999
    epsilon: float = 1e-5
    running_mean: float = 0.0
    running_mean_sq: float = 0.0
    debiasing_term: float = 0.0