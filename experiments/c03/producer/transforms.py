"""Bounded calls to the unchanged historical row-wise transform."""
from .plan import require

BLOCK_ROWS = 8192


def transform_selected(prep, frame, mask, *, block_rows=BLOCK_ROWS):
    """Same historical row-wise transform, bounded scratch space, one final CSR.

    Count nonzeros first, then fill exact-size arrays. Never retain a list of
    transformed chunks or a full selected feature frame. The second pass costs
    computation but avoids simultaneous chunk buffers plus a concatenated tensor.
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    require(type(block_rows) is int and block_rows > 0, 'Positive transform block size required')
    require(mask.dtype == np.bool_ and mask.shape == (len(frame),), 'Invalid transform row mask')
    positions = np.flatnonzero(mask)
    require(len(positions) > 0, 'Empty selected transform population')
    def block_at(offset):
        selected = frame.iloc[positions[offset:offset+block_rows]]
        return prep.transform(selected.drop(columns=['y_stage1_attack', 'y_stage2_family']))
    total = 0
    width = dtype = index_dtype = None
    for offset in range(0, len(positions), block_rows):
        block = block_at(offset)
        require(isinstance(block, csr_matrix), 'Historical CSR required')
        if width is None:
            width, dtype, index_dtype = block.shape[1], block.dtype, block.indices.dtype
        require(block.shape == (min(block_rows, len(positions)-offset), width) and
                block.dtype == dtype and block.indices.dtype == index_dtype, 'Transform block schema drift')
        total += block.nnz
        del block
    # This probe's full tensor fits the same index width as the historical blocks.
    # Fail rather than silently changing the sparse representation.
    require(max(total, len(positions), width) <= np.iinfo(index_dtype).max, 'CSR index width exceeded')
    data = np.empty(total, dtype=dtype)
    indices = np.empty(total, dtype=index_dtype)
    indptr = np.empty(len(positions)+1, dtype=index_dtype)
    indptr[0] = 0
    cursor = 0
    for offset in range(0, len(positions), block_rows):
        block = block_at(offset)
        stop = cursor+block.nnz
        require(stop <= total and block.shape == (min(block_rows, len(positions)-offset), width) and
                block.dtype == dtype and block.indices.dtype == index_dtype, 'Transform second-pass drift')
        data[cursor:stop], indices[cursor:stop] = block.data, block.indices
        indptr[offset+1:offset+block.shape[0]+1] = block.indptr[1:]+cursor
        cursor = stop
        del block
    require(cursor == total, 'Transform nonzero count drift')
    return csr_matrix((data, indices, indptr), shape=(len(positions), width), copy=False)

