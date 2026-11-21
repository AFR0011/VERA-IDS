"""Injected train/validation readers and exact historical sampling with sidecar IDs.

No production file reader exists in Batch 1. Tests inject synthetic frames.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .plan import require, valid_hash, safe_relative


@dataclass(frozen=True)
class Part:
    name: str
    sha256: str


@dataclass
class Sample:
    frame: pd.DataFrame
    identities: pd.DataFrame


class TrainingValidationReader:
    def __init__(self, parts: dict[str, tuple[Part, ...]], read: Callable):
        require(set(parts) == {'train', 'val'}, 'Only train and val sources are permitted')
        self._parts = {key: tuple(value) for key, value in parts.items()}
        names = []
        for entries in self._parts.values():
            require(entries, 'Empty split inventory')
            for entry in entries:
                safe_relative(entry.name)
                require(valid_hash(entry.sha256), 'Invalid part digest')
                names.append(entry.name)
        require(len(names) == len(set(names)), 'Part identity repeats across splits')
        self._read = read

    def sample(self, split: str, cap: int, seed: int) -> Sample:
        require(split in ('train', 'val'), 'Test access is forbidden before sealing')
        require(type(cap) is int and cap > 0, 'Positive integer sample cap required')
        rng = np.random.RandomState(seed)
        current = ids = None
        columns = None
        for part in self._parts[split]:
            frame = self._read(part).copy()
            frame.columns = [str(c).strip().lower().replace(' ', '_') for c in frame.columns]
            require(frame.columns.is_unique, 'Canonical feature names collide')
            if columns is None:
                columns = list(frame.columns)
            require(list(frame.columns) == columns, 'Part schema/order differs')
            row_ids = pd.DataFrame({'source_part': [part.name]*len(frame),
                                    'source_sha256': [part.sha256]*len(frame),
                                    'row_offset': np.arange(len(frame), dtype=np.int64)})
            if current is None:
                mixed, mixed_ids = frame, row_ids
            else:
                mixed = pd.concat([current, frame], ignore_index=True)
                mixed_ids = pd.concat([ids, row_ids], ignore_index=True)
            if len(mixed) > cap:
                # pandas.sample's unweighted positions use the identical RNG draw
                # to the historical frame.sample; sidecar IDs never enter features.
                positions = pd.Series(np.arange(len(mixed))).sample(
                    n=cap, random_state=rng.randint(0, 2**31-1)).to_numpy()
                current = mixed.iloc[positions].reset_index(drop=True)
                ids = mixed_ids.iloc[positions].reset_index(drop=True)
            else:
                current = mixed.copy() if current is None else mixed.reset_index(drop=True)
                ids = mixed_ids.reset_index(drop=True)
        require(current is not None and len(current) > 0, 'Empty sampled split')
        return Sample(current, ids)
