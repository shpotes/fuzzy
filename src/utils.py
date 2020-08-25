from typing import Any, Dict, List
import numpy as np

def flatten_counter(dic: Dict[Any, int]) -> List[Any]:
    flat = []
    for k, v in dic.items():
        for _ in range(v):
            flat.append(k)
    return flat

def t_min(a, b):
    return np.fmin(a, b)

def s_max(a, b):
    return np.fmax(a, b)

def t_prod(a, b):
    return a * b

def s_sum(a, b):
    return a + b - a * b

def t_lukasiewicz(a, b):
    return max(0, a + b - 1)

def s_lukasiewicz(a, b):
    return min(1, a + b)

def t_drastic(a, b):
    return np.where(
        a == 1, b,
        np.where(b == 1, a, 0)
    )

def s_drastic(a, b):
    return np.where(
        a == 0, b,
        np.where(b == 0, a, 1)
    )
