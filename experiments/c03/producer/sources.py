"""Manifest-bound train/validation files. No test-reader capability is exposed."""
from pathlib import Path

import pandas as pd

from .plan import file_hash, require, safe_relative, valid_hash
from .sampling import Part, TrainingValidationReader


class TrainingValidationFiles:
    def __init__(self, root, inventory):
        self.root = Path(root).resolve()
        require(set(inventory) == {'train', 'val'}, 'Only train/val inventory is accepted')
        self.parts = {}
        for split, entries in inventory.items():
            require(entries, 'Empty source inventory')
            parts = []
            for entry in entries:
                relative = safe_relative(entry['path'])
                require(relative.parts[0] == split and valid_hash(entry['sha256']),
                        'Source is assigned to wrong split or has invalid digest')
                parts.append(Part(relative.as_posix(), entry['sha256']))
            require(len({p.name for p in parts}) == len(parts), 'Repeated part')
            self.parts[split] = tuple(parts)

    def path(self, part):
        require(any(part in group for group in self.parts.values()), 'Unregistered source part')
        path = self.root/part.name
        require(not path.is_symlink() and path.resolve().is_relative_to(self.root), 'Redirected source')
        require(file_hash(path) == part.sha256, 'Source byte drift')
        return path

    def verify(self):
        for split, parts in self.parts.items():
            actual = sorted(p.relative_to(self.root).as_posix() for p in (self.root/split).glob('*.parquet'))
            require(actual == [p.name for p in parts], 'Part ordering/inventory differs')
            # Reject extra supported historical files so preprocessing sees exactly
            # the inventory (this first adapter intentionally supports Parquet only).
            extras = list((self.root/split).glob('*.csv')) + list((self.root/split).glob('*.csv.gz'))
            extras += list(self.root.glob(split+'_*.parquet')) + list(self.root.glob(split+'_*.csv*'))
            require(not extras, 'Unexpected historical-loader source files')
            for part in parts:
                self.path(part)

    def read(self, part):
        path = self.path(part)
        frame = pd.read_parquet(path)
        require(file_hash(path) == part.sha256, 'Source changed during read')
        return frame

    def sampler(self):
        return TrainingValidationReader(self.parts, self.read)

    def inventory(self):
        return {split: [{'path': p.name, 'sha256': p.sha256} for p in entries]
                for split, entries in self.parts.items()}
