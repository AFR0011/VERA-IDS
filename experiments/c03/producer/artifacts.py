"""Streaming local artifact publication and non-pickle array fingerprints."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.sparse import csr_matrix

from .plan import digest, require, safe_relative


def array_hash(array):
    value = np.ascontiguousarray(array)
    require(not value.dtype.hasobject, 'Object arrays cannot be persisted')
    h = hashlib.sha256()
    h.update(str(value.dtype).encode('ascii'))
    h.update(str(value.shape).encode('ascii'))
    h.update(memoryview(value).cast('B'))
    return h.hexdigest()


def tensor_hash(value):
    require(isinstance(value, csr_matrix), 'Historical CSR representation required')
    return digest({'data': array_hash(value.data), 'indices': array_hash(value.indices),
                   'indptr': array_hash(value.indptr), 'shape': list(value.shape)})


def identity_hash(frame):
    require(list(frame.columns) == ['source_part', 'source_sha256', 'row_offset'],
            'Invalid source identity schema')
    require(not frame.duplicated(['source_part', 'row_offset']).any(), 'Repeated source row identity')
    return digest({name: array_hash(frame[name].to_numpy(dtype=str if name != 'row_offset' else np.int64))
                   for name in frame})


@contextmanager
def generated_file(path):
    """Writer receives a temporary path; publication is atomic and no-clobber."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.pending-', suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temporary = Path(temporary)
    try:
        yield temporary
        with temporary.open('r+b') as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_arrays(path, **arrays):
    require(all(not np.asarray(a).dtype.hasobject for a in arrays.values()), 'Object arrays forbidden')
    with generated_file(path) as temporary:
        with temporary.open('wb') as stream:
            np.savez(stream, **arrays)


def read_arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def safe_artifact(root, name):
    path = Path(root)/safe_relative(name)
    require(not path.is_symlink() and path.resolve().is_relative_to(Path(root).resolve()),
            'Artifact path redirected outside run')
    return path
