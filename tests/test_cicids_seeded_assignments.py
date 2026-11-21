"""Generated-metadata tests; never read benchmark observations or fit models."""
import csv
import json
import time

import numpy as np
import pytest

from experiments.cicids_partition import seeded_assignments as audit


@pytest.fixture
def metadata(tmp_path):
    counts=np.full((20,7),100,dtype=np.int64)
    counts[0,0]+=2099879-int(counts.sum())
    units=[];support=[]
    for i in range(20):
        segment=i%4+1;name=f'day{i//4}.csv'
        start=sum(int(counts[j].sum()) for j in range(i-i%4,i))
        end=start+int(counts[i].sum())
        unit={'unit_order':i,'partition_mode':'contiguous_subfiles','source_file_name':name,
            'source_file_path':'PRIVATE_SENTINEL','segment_index':segment,'segment_count':4,
            'unit_id':f'{name}::part{segment}of4','row_start':start,'row_end':end,
            'kept_rows_after_family_drop':end-start,'dropped_rows':0,'raw_row_count':end-start}
        units.append(unit)
        support.extend({**unit,'family':family,'count':int(counts[i,j])} for j,family in enumerate(audit.FAMILIES))
    def write(name,values):
        with (tmp_path/name).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    write(audit.FILES[0],units);write(audit.FILES[1],support)
    return tmp_path,units,support,write


@pytest.fixture
def run(tmp_path,monkeypatch,metadata):
    plan={'metadata':audit.load_metadata(metadata[0])}
    monkeypatch.setattr(audit,'definition',lambda:plan)
    monkeypatch.setattr(audit.host,'resources',lambda run:{'available':1,'disk_free':100*2**30})
    target=tmp_path/'audit-run';target.mkdir()
    now=time.time()
    audit.publish(target/'request.json',audit.canonical({'definition':plan,'started_epoch':now,
        'deadline_epoch':now+audit.LIMITS['seconds']}))
    return target


def test_metadata_is_sanitized_and_population_reconciled(metadata):
    value=audit.load_metadata(metadata[0])
    assert 'PRIVATE_SENTINEL' not in json.dumps(value)
    assert sum(map(sum,value['family_counts']))==2099879
    assert len(value['units'])==20 and len(value['input_hashes'])==2


@pytest.mark.parametrize('damage',['gap','negative','duplicate','family'])
def test_metadata_rejects_identity_and_count_drift(metadata,damage):
    root,units,support,write=metadata
    if damage=='gap':
        units[1]['row_start']+=1;write(audit.FILES[0],units)
    elif damage=='negative':
        support[0]['count']=-1;write(audit.FILES[1],support)
    elif damage=='duplicate':
        support[1]=dict(support[0]);write(audit.FILES[1],support)
    else:
        support[0]['family']='Unapproved';write(audit.FILES[1],support)
    with pytest.raises((ValueError,RuntimeError)):
        audit.load_metadata(root)


def test_first_valid_draw_and_fixed_disjoint_unit_counts():
    counts=np.full((20,7),100,dtype=np.int64)
    counts[:,6]=0;counts[[0,4,5],6]=200
    result=audit.sample(counts,123)
    draws=np.random.RandomState(123)
    for index in range(1,result['attempts']+1):
        draw=draws.permutation(20)
        groups=(draw[:12],draw[12:16],draw[16:])
        valid=all(any(unit in {0,4,5} for unit in group) for group in groups)
        assert valid==(index==result['attempts'])
    assert result['attempts']>1
    assert [len(v) for v in result['assignment'].values()]==[12,4,4]
    assert sorted(sum(result['assignment'].values(),[]))==list(range(20))
    assert result==audit.sample(counts,123)


def test_exhaustion_is_explicit_and_bounded():
    calls=[]
    result=audit.sample(np.zeros((20,7),dtype=np.int64),123,max_attempts=129,guard=lambda:calls.append(1))
    assert result['status']=='EXHAUSTED' and result['attempts']==129
    assert result['assignment'] is None and len(calls)==2
    for cap in (0,-1,10001,True):
        with pytest.raises((ValueError,RuntimeError)):
            audit.sample(np.ones((20,7),dtype=np.int64),123,max_attempts=cap)


def test_duplicate_assignments_are_reported_without_redraw():
    values=[{'seed':s,'status':'FEASIBLE','assignment_sha256':'same'} for s in audit.SEEDS]
    assert audit.summary(values)=={'all_five_feasible':True,'distinct_feasible_assignments':1,
                                  'duplicate_seeds':[list(audit.SEEDS)]}


def test_checkpoint_recovery_keeps_completed_seed_bytes(run,monkeypatch):
    original=audit.evaluate_seed
    def interrupt(plan,seed,guard=lambda:None):
        if seed==125:raise RuntimeError('simulated interruption')
        return original(plan,seed,guard)
    monkeypatch.setattr(audit,'evaluate_seed',interrupt)
    with pytest.raises(RuntimeError,match='simulated'):
        audit.execute(run)
    before={p.name:p.read_bytes() for p in (run/'seeds').iterdir()}
    monkeypatch.setattr(audit,'evaluate_seed',original)
    result=audit.execute(run,recover=True)
    assert result['reused_seeds']==[123,124]
    assert all((run/'seeds'/name).read_bytes()==data for name,data in before.items())
    assert audit.verify_completed(run)==result
    assert result['new_split_files'] is False and result['models_fitted'] is False


def test_partial_final_publication_finishes_without_replacing_result(run,monkeypatch):
    original=audit.publish
    def interrupt(path,payload):
        if path.name=='complete.json':raise RuntimeError('simulated publication interruption')
        return original(path,payload)
    monkeypatch.setattr(audit,'publish',interrupt)
    with pytest.raises(RuntimeError,match='simulated'):
        audit.execute(run)
    before=(run/'result.json').read_bytes()
    monkeypatch.setattr(audit,'publish',original)
    audit.execute(run,recover=True)
    assert (run/'result.json').read_bytes()==before
    audit.verify_completed(run)


@pytest.mark.parametrize('target',['seed','summary','source'])
def test_completed_audit_rejects_corruption(run,monkeypatch,target):
    audit.execute(run)
    if target=='source':
        monkeypatch.setattr(audit,'definition',lambda:{'changed':True})
    elif target=='seed':
        (run/'seeds/123.json').write_text('{}')
    else:
        result=json.loads((run/'result.json').read_text());result['distinct_feasible_assignments']=999
        (run/'result.json').write_bytes(audit.canonical(result))
        (run/'complete.json').write_bytes(audit.canonical({'result_sha256':audit.file_hash(run/'result.json')}))
    with pytest.raises((ValueError,RuntimeError)):
        audit.verify_completed(run)


def test_recovery_cannot_reset_original_deadline(run):
    req=json.loads((run/'request.json').read_text());req['started_epoch']-=901;req['deadline_epoch']-=901
    (run/'request.json').write_bytes(audit.canonical(req))
    with pytest.raises((ValueError,RuntimeError),match='Shared time budget'):
        audit.execute(run,recover=True)
    assert not (run/'seeds').exists()


@pytest.mark.parametrize('resource',['rss','disk','artifacts'])
def test_retained_resource_guards_prevent_work(run,monkeypatch,resource):
    if resource=='rss':
        monkeypatch.setattr(audit.psutil.Process,'memory_info',lambda self:type('Memory',(),{'rss':5*2**30})())
    elif resource=='disk':
        monkeypatch.setattr(audit.host,'resources',lambda path:{'available':1,'disk_free':1})
    else:
        monkeypatch.setattr(audit.host,'artifact_size',lambda path:6*2**30)
    with pytest.raises((ValueError,RuntimeError)):
        audit.execute(run)
    assert not (run/'seeds').exists()
