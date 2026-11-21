"""Explicit EFC dependency semantics and independently checkpointable native steps.

The only upstream source edit supplies the user-approved discretizer keywords.
Native energy inference, class cutoffs and prediction rules are unchanged.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from experiments.c03.producer.plan import ROOT,canonical,file_hash,require,safe_relative
from experiments.c03.producer.checkpoints import publish
from .efc_discretization import make_discretizer

UPSTREAM_COMMIT='2b935be347abf7daf4420989ef391436db418eac'
CLASSIFIER_SHA='b85f95b8cd4a0e502913ac1a271f897d78167b0d0514b2ce5bb143b7daa60e4d'
OLD=b'n_bins=self.n_bins, encode="ordinal", strategy="quantile"'
NEW=OLD+b', subsample=None, quantile_method="linear", random_state=None'


def _insert_settings(source):
    require(source.count(OLD)==1,'Upstream constructor shape differs')
    patched=source.replace(OLD,NEW)
    original_tree,patched_tree=ast.parse(source),ast.parse(patched)
    targets=[node for node in ast.walk(patched_tree) if isinstance(node,ast.Call)
             and isinstance(node.func,ast.Name) and node.func.id=='KBinsDiscretizer']
    require(len(targets)==1,'Ambiguous discretizer constructor')
    added={'subsample':None,'quantile_method':'linear','random_state':None}
    for key,value in added.items():
        entries=[kw for kw in targets[0].keywords if kw.arg==key]
        require(len(entries)==1 and ast.literal_eval(entries[0].value)==value,'Explicit setting differs')
    targets[0].keywords=[kw for kw in targets[0].keywords if kw.arg not in added]
    require(ast.dump(original_tree)==ast.dump(patched_tree),'Patch changes native algorithm beyond approved binning')
    return patched


def patched_classifier(source):
    require(hashlib.sha256(source).hexdigest()==CLASSIFIER_SHA,'Unreviewed upstream classifier bytes')
    return _insert_settings(source)


def copy_for_build(source,destination):
    source,destination=Path(source),Path(destination)
    def git(*args):return subprocess.check_output(['git','-C',str(source),*args],text=True).strip()
    require(git('rev-parse','HEAD')==UPSTREAM_COMMIT and not git('status','--porcelain'),'Upstream checkout differs')
    approved=json.loads((ROOT/'experiments/literature/EFC_SOURCE_AUDIT.json').read_text())
    for name,sha in approved['files'].items():require(file_hash(source/name)==sha,'Audited source drift')
    destination.mkdir(parents=True,exist_ok=False);before={};after={};total=0
    for name in git('ls-files').splitlines():
        relative=safe_relative(name);path=source/relative
        require(path.is_file() and not path.is_symlink(),'Unexpected source entry')
        require(path.stat().st_size<10*2**20,'Unexpected large source artifact')
        payload=path.read_bytes();total+=len(payload);require(total<50*2**20,'Source-copy budget exceeded')
        before[name]=hashlib.sha256(payload).hexdigest()
        if name=='efc/_energyclassifier.py':payload=patched_classifier(payload)
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(payload)
        after[name]=file_hash(target)
    receipt={'upstream_commit':UPSTREAM_COMMIT,'original_files':before,'derived_files':after,
             'change':'Only KBinsDiscretizer: subsample=None, quantile_method=linear, random_state=None',
             'license_preserved':before['LICENSE']==after['LICENSE'],'adapter_sha256':file_hash(Path(__file__))}
    publish(destination/'adapter-provenance.json',canonical(receipt))
    return receipt


def preprocess_training(X):
    """Same all-numeric pipeline as the approved source; fits supplied TRAIN only."""
    require(X.ndim==2 and len(X)>0 and np.isfinite(X).all(),'EFC input must be a finite numerical matrix')
    pipeline=Pipeline([('scaler',MaxAbsScaler()),('discretizer',make_discretizer())])
    preprocessor=ColumnTransformer([('categorical','passthrough',[])],remainder=pipeline)
    discretized=preprocessor.fit_transform(X).astype('int64')
    return preprocessor,discretized,int(discretized.max())+1


def fit_class(discretized,max_bin):
    from efc._base import BaseEFC
    estimator=BaseEFC(max_bin,0.5,0.95).fit(discretized)
    require(all(np.isfinite(getattr(estimator,name)).all() for name in
            ('sitefreq_','pairfreq_','coupling_matrix_','local_fields_','cutoff_')),'Nonfinite native EFC model')
    return estimator


def assemble(preprocessor,estimators,classes,max_bin,n_features):
    from efc import EnergyBasedFlowClassifier
    require(len(classes)>2 and len(classes)==len(estimators)
            and np.array_equal(classes,np.unique(classes)),'Expected sorted multiclass native estimators')
    model=EnergyBasedFlowClassifier(n_jobs=1)
    model.preprocessor_=preprocessor;model.estimators_=estimators
    model.classes_=np.asarray(classes);model.max_bin_=max_bin
    model.n_features_in_=n_features;model.target_type_='multiclass'
    return model
