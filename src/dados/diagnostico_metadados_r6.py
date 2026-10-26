"""Diagnósticos descritivos de nomenclatura e ligação lógica de metadados.

Nenhuma função recebe respostas, gravidade, constante de política ou split.
As features reutilizam as transformações históricas dos textos completos.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import numpy as np
import pandas as pd
from features_nomenclatura import _n_morphology_units, _similaridade, _soma, _minimo

FEATURES=('nomenclature_delta_morphology_units_pair','nomenclature_lexical_dissimilarity_max')
META_FIELDS={'observation_id','request_id','pair_id','model','condition','context_id','text_binding_sha256',
             'cache_current_present','cache_current_key','classification','historical_use_allowed_with_caveat',
             'submission_count','physical_execution_id_observed'}

def _table(frame,columns,key):
    if set(frame.columns)!=set(columns): raise ValueError('Only explicit metadata columns are accepted')
    if frame[key].duplicated().any(): raise ValueError('Duplicate identifier')
    if not frame[key].map(lambda x:isinstance(x,str) and bool(x.strip())).all():raise ValueError('Missing identifier')

def calcular_features_primarias(entities,pairs):
    _table(entities,['drug_id','dcb','experimental_name'],'drug_id')
    _table(pairs,['pair_id','drug_id_a','drug_id_b'],'pair_id')
    by=entities.set_index('drug_id').to_dict('index')
    if (set(pairs.drug_id_a)|set(pairs.drug_id_b))-set(by):raise ValueError('Orphan endpoint')
    if pairs.drug_id_a.eq(pairs.drug_id_b).any():raise ValueError('Self edge')
    result=[]
    for p in pairs.itertuples(index=False):
        a,b=by[p.drug_id_a],by[p.drug_id_b]
        delta=_soma(_n_morphology_units(a['dcb'])-_n_morphology_units(a['experimental_name']),
                    _n_morphology_units(b['dcb'])-_n_morphology_units(b['experimental_name']))
        similarity=_minimo(_similaridade(a['dcb'],a['experimental_name']),_similaridade(b['dcb'],b['experimental_name']))
        result.append(dict(pair_id=p.pair_id,**{FEATURES[0]:delta,FEATURES[1]:1.0-similarity}))
    return pd.DataFrame(result,columns=['pair_id',*FEATURES])

def resumir_features(features,admitted_ids):
    _table(features,['pair_id',*FEATURES],'pair_id')
    if set(admitted_ids)-set(features.pair_id):raise ValueError('Unknown admitted pair')
    report={}
    subsets={'all':features,'admitted':features[features.pair_id.isin(admitted_ids)],
             'excluded':features[~features.pair_id.isin(admitted_ids)]}
    for group,frame in subsets.items():
        summary=dict(pairs=len(frame),features={})
        converted={}
        for name in FEATURES:
            raw=pd.to_numeric(frame[name],errors='coerce');values=raw[np.isfinite(raw)];converted[name]=raw
            counts=values.value_counts();unique=len(counts)
            summary['features'][name]=dict(finite=len(values),missing_or_nonfinite=len(raw)-len(values),
                distinct=unique,constant=bool(len(values)>0 and unique==1),
                minimum=float(values.min()) if len(values) else None,maximum=float(values.max()) if len(values) else None,
                q25=float(values.quantile(.25)) if len(values) else None,median=float(values.median()) if len(values) else None,
                q75=float(values.quantile(.75)) if len(values) else None,largest_tie=int(counts.max()) if len(counts) else 0,
                tied_observations=int(counts[counts.gt(1)].sum()),tie_groups=int(counts.gt(1).sum()))
        delta=converted[FEATURES[0]];finite=np.isfinite(delta);n=int(finite.sum());zero=int((finite&delta.eq(0)).sum())
        summary['morphology']=dict(zero=zero,nonzero=int((finite&delta.ne(0)).sum()),positive=int((finite&delta.gt(0)).sum()),
            negative=int((finite&delta.lt(0)).sum()),missing_or_nonfinite=len(frame)-n,
            zero_share_all=zero/len(frame) if len(frame) else None,zero_share_finite=zero/n if n else None)
        joint=pd.DataFrame(converted);joint=joint[np.isfinite(joint).all(axis=1)]
        counts=joint.value_counts()
        summary['joint_feature_ties']=dict(distinct_vectors=len(counts),largest_tie=int(counts.max()) if len(counts) else 0,
            tied_observations=int(counts[counts.gt(1)].sum()),finite_pairs=len(joint))
        report[group]=summary
    return report

def parear_metadados(observations):
    grouped=defaultdict(dict);seen_obs=set();seen_req=set()
    for row in observations:
        if set(row)!=META_FIELDS:raise ValueError('Only projected request/cache metadata accepted')
        for key in ('observation_id','request_id','pair_id','model','context_id','text_binding_sha256'):
            if not isinstance(row[key],str) or not row[key]:raise ValueError('Missing logical identity')
        if row['observation_id'] in seen_obs or row['request_id'] in seen_req:raise ValueError('Duplicate observation/request')
        seen_obs.add(row['observation_id']);seen_req.add(row['request_id'])
        if row['condition'] not in {'dcb','inn'}:raise ValueError('Unknown arm')
        if row['classification'] not in {'compativel','incompativel','indeterminada','ausente'}:raise ValueError('Unknown compatibility')
        for key in ('cache_current_present','historical_use_allowed_with_caveat'):
            if type(row[key]) is not bool:raise ValueError('Strict metadata booleans required')
        if type(row['submission_count']) is not int or row['submission_count']<0:raise ValueError('Invalid submission count')
        if row['cache_current_present'] and not row['cache_current_key']:raise ValueError('Present cache requires a key')
        group=grouped[(row['pair_id'],row['model'])]
        if row['condition'] in group:raise ValueError('Ambiguous logical arm')
        group[row['condition']]=row
    output=[]
    for (pid,model),group in sorted(grouped.items()):
        rows=list(group.values())
        for key in ('context_id','text_binding_sha256'):
            if len({r[key] for r in rows})!=1:raise ValueError('Mismatched logical pair context/text')
        missing=[c for c in ('dcb','inn') if c not in group]
        missing_cache=[c for c in ('dcb','inn') if c not in group or not group[c]['cache_current_present']]
        incompatible=any(r['classification']=='incompativel' for r in rows)
        unique=not missing
        potential=bool(unique and not missing_cache and not incompatible and all(r['historical_use_allowed_with_caveat'] for r in rows))
        identifier=hashlib.sha256(json.dumps([pid,model,rows[0]['context_id'],rows[0]['text_binding_sha256'],
            [group[c]['observation_id'] if c in group else None for c in ('dcb','inn')]],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        output.append(dict(pair_id=pid,model=model,execution_pair_id='HPAIR_'+identifier,context_id=rows[0]['context_id'],
            unique_logical_link=unique,missing_logical_arms=missing,missing_cache_arms=missing_cache,
            dcb_observation_id=group.get('dcb',{}).get('observation_id'),inn_observation_id=group.get('inn',{}).get('observation_id'),
            dcb_classification=group.get('dcb',{}).get('classification'),inn_classification=group.get('inn',{}).get('classification'),
            material_divergence_known=incompatible,both_current_cache_keys_present=not missing_cache,
            execution_pair_provenance='reconstructed_cache_slot' if unique and not missing_cache else 'logical_requests_only_no_complete_cached_pair',
            physical_execution_pair_observed=False,submission_ambiguity_present=any(r['submission_count']>1 for r in rows),
            complete_observed_compatibility=bool(unique and all(r['classification']=='compativel' and r['physical_execution_id_observed'] for r in rows)),
            potential_historical_use_with_caveat=potential,response_validity_checked=False,
            automatic_reuse_authorized=False,collection_authorized=False,new_independent_replications=0))
    return output
