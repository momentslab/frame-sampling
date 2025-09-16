import json
import logging
import os



def clean_dict(d: dict) -> dict:
    clean = {}
    for k, v in d.items():
        if isinstance(v, dict):
            nested = clean_dict(v)
            if len(nested.keys()) > 0:
                clean[k] = nested
        elif isinstance(v, list):
            clean[k] = [clean_dict(item) if isinstance(item, dict) else item for item in v]
        elif v is not None:
            clean[k] = v
    return clean

