"""Generated-data materialization, atomic-copy recovery and identity checks."""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import pytest

from experiments.cicids_partition import materialize as m


def fixture(root):
    run=root/'run'; source=root/'source'; run.mkdir(); source.mkdir()
    (run/'request.json').write_text('{}')
    frame=pd.DataFrame({'predictor':np.arange(7),'y_stage2_family':list(m.assignments.FAMILIES)})
    frame.to_csv(source/'part.csv.gz',index=False)
    positions=np.asarray([10,11,13,14,15,16,18],dtype='<i8')
    item={'unit':{'id':'generated::part1','kept_rows':7,'row_start':10,'row_end':19,'source_file':'generated.csv'},
          'unit_index':0,'split':'train','output':'train/part_00000.csv.gz','identity':'identities/unit-00.npz',
          'reference':'part.csv.gz','reference_sha256':m.file_hash(source/'part.csv.gz'),
          'raw_position_sha256':hashlib.sha256(positions.tobytes()).hexdigest(),'family_support':[1]*7}
    return run,source,item,positions


@pytest.mark.parametrize('event',['copy_block','copy_published','identity_published','unit_published'])
def test_interrupted_materialization_reuses_verified_independent_copy(tmp_path,event):
    run,source,item,positions=fixture(tmp_path)
    def interrupt(actual):
        if actual==event: raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError,match='interrupted'):
        m.unit_checkpoint(run,item,positions,[1]*7,reference=source,hook=interrupt)
    result=m.unit_checkpoint(run,item,positions,[1]*7,reference=source)
    assert result==m.unit_checkpoint(run,item,positions,[1]*7,reference=source)
    assert result['output_sha256']==item['reference_sha256']
    assert not os.path.samefile(source/item['reference'],run/item['output'])
    assert not list(run.rglob('.pending-*'))


@pytest.mark.parametrize('target',['source','output','identity','receipt'])
def test_corruption_fails_closed(tmp_path,target):
    run,source,item,positions=fixture(tmp_path)
    m.unit_checkpoint(run,item,positions,[1]*7,reference=source)
    if target=='source': path=source/item['reference']
    elif target=='output': path=run/item['output']
    elif target=='identity':
        path=run/item['identity']; changed=positions.copy(); changed[0]=9
        with path.open('wb') as stream: np.savez(stream,row_offset=changed)
        path=None
    else:
        path=run/'units/unit-00.json'; value=m.read(path); value['rows']=8
        path.write_text(json.dumps(value)); path=None
    if path: path.write_bytes(path.read_bytes()+b'drift')
    with pytest.raises(ValueError,match='drift'):
        m.unit_checkpoint(run,item,positions,[1]*7,reference=source)


def test_position_and_support_contract(tmp_path):
    run,source,item,positions=fixture(tmp_path)
    bad=positions.copy(); bad[1]=bad[0]
    with pytest.raises(ValueError,match='positions'): m.unit_checkpoint(run,item,bad,[1]*7,reference=source)
    with pytest.raises(ValueError,match='support'): m.unit_checkpoint(run,item,positions,[2]*7,reference=source)
    assert not (run/'train').exists()


def test_native_label_slice_retains_raw_offsets(tmp_path):
    labels=['Other','BENIGN','Botnet','Other','FTP-Patator','DDoS','DoS','PortScan','Web Attack','Other']
    pd.DataFrame({' Label':labels,'unused':np.arange(10)}).to_csv(tmp_path/'generated.csv',index=False)
    unit={'source_file':'generated.csv','row_start':1,'row_end':9,'kept_rows':7}
    offsets,support=m.positions_for(unit,tmp_path)
    np.testing.assert_array_equal(offsets,[1,2,4,5,6,7,8]); assert support==[1]*7


def test_expired_deadline_cannot_start_copy(tmp_path):
    run,source,item,positions=fixture(tmp_path)
    req={'started_epoch':time.time()-1000,'deadline_epoch':time.time()-100}
    with pytest.raises(ValueError,match='deadline'):
        m.unit_checkpoint(run,item,positions,[1]*7,reference=source,guard=lambda:m.guard(req))
    assert not (run/'train').exists()
