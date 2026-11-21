"""One authorized engineering epoch and known-validation pass; never DOC cutoffs/test."""
import argparse
import importlib.metadata
import time
from pathlib import Path

import numpy as np
import torch

from . import doc_core as core, doc_training as training, doc_runtime as rt
from experiments.c03.producer.plan import ROOT, file_hash, require
from experiments.c03.producer.locking import file_lock


def environment():
    lines = (ROOT/'experiments/literature/doc-requirements.txt').read_text().splitlines()
    packages = {}
    for line in lines:
        if not line or line.startswith('#'): continue
        name, version = line.split('==')
        require(importlib.metadata.version(name) == version, 'Isolated DOC environment drift: '+name)
        packages[name] = version
    return packages


def run_epoch(run, plan, *, device='cuda:0', emit=lambda *a,**kw:None, hook=lambda event:None,
              diagnostic=False):
    """Test-only small fixtures may use CPU; real batch requires exact full-row GPU inputs."""
    rt.guard(plan)
    run = Path(run); folder = run/'prepare'
    prepared = rt.read(folder/'complete.json')
    require(prepared['request_sha256'] == file_hash(run/'request.json'), 'Prepared request drift')
    rt.verify_files(folder,prepared)
    if not diagnostic:
        require(device == 'cuda:0' and prepared['rows'] == {'train':1270025,'validation':599278}, 'Full-row GPU surface required')
        environment()
    train, labels, validation, val_labels = [np.load(folder/name,mmap_mode='r',allow_pickle=False) for name in
        ('train-X.npy','train-y.npy','validation-X.npy','validation-y.npy')]
    require(train.shape == (prepared['rows']['train'],46) and validation.shape == (prepared['rows']['validation'],46)
            and train.dtype == validation.dtype == np.dtype('float32'), 'Prepared tensor layout differs')
    core.labels_checked(labels,len(train),6);core.labels_checked(val_labels,len(validation),6)
    require(np.bincount(labels,minlength=6).tolist() == prepared['class_support']['train']
            and np.bincount(val_labels,minlength=6).tolist() == prepared['class_support']['validation'], 'Prepared class support differs')
    training.configure(device)
    definition = {'scope':rt.SCOPE,'request_sha256':file_hash(run/'request.json'),
                  'prepared_sha256':file_hash(folder/'complete.json'),'device':device,'diagnostic_fixture':diagnostic,
                  'torch':str(torch.__version__),'cuda':torch.version.cuda,'config':training.CONFIG,
                  'epoch_cap':1,'cutoffs':False,'test':False}
    journal = training.Journal(run/'fit',definition)
    with journal.lock():
        state = journal.open()
        model = training.network(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
        if state is None:
            state = {'phase':'fit','cursor':0,'updates':0,'val_cursor':0,'val_sum':0.,
                     'order':torch.from_numpy(np.random.default_rng(np.random.SeedSequence([123,0])).permutation(len(train)))}
        else:
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
            training.restore_rng(state['rng'],device)
        def checkpoint():
            rt.guard(plan)
            state['model']=training.snapshot(model);state['optimizer']=optimizer.state_dict()
            state['rng']=training.rng_state(device);journal.save(state,hook)
        if not journal.previous:checkpoint()
        while state['phase'] != 'complete':
            rt.guard(plan);journal.guard()
            if state['phase'] == 'fit':
                model.train()
                positions=state['order'][state['cursor']:state['cursor']+128].numpy()
                x=np.array(train[positions],dtype=np.float32,copy=True)
                require(np.isfinite(x).all(),'Nonfinite prepared input')
                y=np.array(labels[positions],dtype=np.int64,copy=True)
                optimizer.zero_grad(set_to_none=True)
                loss=training.loss_rows(model(torch.as_tensor(x,device=device)),torch.as_tensor(y,device=device)).mean()
                require(torch.isfinite(loss).item(),'Nonfinite diagnostic loss')
                loss.backward();optimizer.step();hook('after_update')
                state['cursor']+=len(positions);state['updates']+=1
                if state['cursor']==len(train):state['phase']='validation'
                if state['updates']%250==0 or state['phase']=='validation':
                    checkpoint();emit('diagnostic_fit',rows=state['cursor'],total=len(train),updates=state['updates'])
            elif state['phase']=='validation':
                model.eval();start=state['val_cursor'];end=min(start+8192,len(validation))
                x=np.array(validation[start:end],dtype=np.float32,copy=True)
                require(np.isfinite(x).all(),'Nonfinite known validation input')
                y=np.array(val_labels[start:end],dtype=np.int64,copy=True)
                with torch.no_grad():
                    loss=training.loss_rows(model(torch.as_tensor(x,device=device)),torch.as_tensor(y,device=device))
                require(torch.isfinite(loss).all().item(),'Nonfinite known validation loss')
                state['val_sum']+=loss.double().sum().item();state['val_cursor']=end
                hook('validation_block')
                if end==len(validation):
                    state['validation_bce']=state['val_sum']/len(validation);state['phase']='complete'
                checkpoint()
                if (start//8192)%10==0 or state['phase']=='complete':emit('known_validation',rows=end,total=len(validation))
            else:raise ValueError('Unknown diagnostic phase')
        # Reopen committed model AND optimizer/RNG state, then compare fixed known inputs.
        rt.guard(plan)
        before=training.probabilities(model,np.array(validation[:min(8192,len(validation))]),device)
        reloaded=journal.open()
        second=training.network(device);second.load_state_dict(reloaded['model'])
        restored=torch.optim.Adam(second.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
        restored.load_state_dict(reloaded['optimizer'])
        training.restore_rng(reloaded['rng'],device)
        after=training.probabilities(second,np.array(validation[:min(8192,len(validation))]),device)
        require(np.array_equal(before,after),'Reload inference differs')
        require(reloaded['updates']==state['updates'] and reloaded['val_sum']==state['val_sum'], 'Reload progress differs')
        result={'status':'COMPLETE','rows':prepared['rows'],'updates':state['updates'],'epochs':1,
                'known_validation_bce':state['validation_bce'],'reload_prediction_exact':True,
                'reload_rows':len(before),'device':device,'torch':str(torch.__version__),
                'prepared_sha256':file_hash(folder/'complete.json'),
                'journal_commit':journal.previous,'cutoffs_fitted':False,'test_read':False,
                'interpretation':'Engineering one-epoch sizing only; not a completed DOC scientific fit'}
        rt.guard(plan)
        return rt.bound_json(run,'fit-result.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--attempt',required=True)
    args=parser.parse_args();plan,done=rt.supervise_owner(args.run,args.attempt)
    try:
        with file_lock(args.run/'.worker.guard'):
            run_epoch(args.run,plan,emit=lambda stage,**info:rt.emit(args.run,args.attempt,stage,**info))
    finally:done.set()


if __name__=='__main__':main()
