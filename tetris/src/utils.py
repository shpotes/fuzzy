from typing import Any, Dict, List

def flatten_counter(dic: Dict[Any, int]) -> List[Any]:
    flat = []
    for k, v in dic.items():
        for _ in range(v):
            flat.append(k)

    return flat
