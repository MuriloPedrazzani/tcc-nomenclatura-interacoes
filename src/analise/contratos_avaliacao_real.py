"""Strict in-memory joins for future real evaluation; no I/O or scope spoofing.

This module implements no reading authorization. It is tested using invented
fixtures now. A future caller must first verify the frozen files and obtain
authorization to read actual outcome fields. Synthetic evaluators stay intact.
"""
from __future__ import annotations
import math
from numbers import Real
import pandas as pd

VERSION='real-exploratory-outcome-joins-v1'
LABELS={'leve','moderado','grave'}
BINDING=('observation_id','planned_observation_id','pair_id','model','provider','family',
    'condition','execution_slot','pair_partition','request_id','request_fingerprint',
    'payload_fingerprint','text_binding_sha256','context_id','admission_status')
MASK_COLUMNS=set(BINDING)|{'drug_id_a','drug_id_b','binding_verified','available','structural_response_valid',
    'eligible','eligible_d01_response','reference_available','reference_source_id','reference_source_version',
    'identity_admitted','benchmark_candidate','primary_features_complete',
    'morphology_stratum','d01e_eligible','d01e_documentary_state','response_execution_id_observed',
    'response_execution_attribution_verified','complete_observed_compatibility'}
PREDICTION_COLUMNS=set(BINDING)|{'category','structural_response_valid','response_execution_id_observed',
    'complete_observed_compatibility'}
REFERENCE_COLUMNS={'pair_id','ref_id_a','ref_id_b','context_id','source_id','source_version','quality_status','canonical_label'}
CONTEXT_COLUMNS={'pair_id','drug_id_a','drug_id_b','ref_id_a','ref_id_b','context_id'}
REGISTRY_COLUMNS={'model','provider','family','role'}


def table(frame,columns,key,name):
    if not isinstance(frame,pd.DataFrame) or frame.columns.duplicated().any() or set(frame)!=set(columns):
        raise ValueError(name+': explicit projection required')
    for k in key:
        if not frame[k].map(lambda x:isinstance(x,str) and bool(x) and x==x.strip()).all():
            raise ValueError(name+': invalid literal identity')
    if frame.duplicated(list(key)).any():raise ValueError(name+': duplicate identity')


def validar_entradas(masks,predictions,references,registry,contexts):
    """Validate all future joins, keeping planned unavailable positions as such.

    References must cover the exact pairs in the masks; prediction rows must
    cover exactly the available observations. No categories are manufactured
    for missing observations, and reference/response availability is verified
    against pinned preparation rather than silently recomputed from outcomes.
    """
    for args in [(masks,MASK_COLUMNS,('observation_id',),'masks'),
                 (predictions,PREDICTION_COLUMNS,('observation_id',),'predictions'),
                 (references,REFERENCE_COLUMNS,('pair_id',),'references'),
                 (registry,REGISTRY_COLUMNS,('model',),'registry'),
                 (contexts,CONTEXT_COLUMNS,('pair_id',),'contexts')]:table(*args)
    boolean=['binding_verified','available','structural_response_valid','eligible','eligible_d01_response',
             'reference_available','response_execution_attribution_verified','complete_observed_compatibility',
             'identity_admitted','benchmark_candidate','primary_features_complete']
    for frame,cols in [(masks,boolean),(predictions,['structural_response_valid','complete_observed_compatibility'])]:
        if any(not frame[k].map(lambda x:type(x) is bool).all() for k in cols):raise ValueError('Strict boolean required')
    pairs=set(masks.pair_id)
    if set(references.pair_id)!=pairs or set(contexts.pair_id)!=pairs:raise ValueError('Orphan/missing reference or context')
    if set(predictions.observation_id)!=set(masks.loc[masks.available,'observation_id']):
        raise ValueError('Predictions differ from admitted available observations')
    composite=['pair_id','model','condition','execution_slot','pair_partition']
    if masks[composite].duplicated().any() or masks.planned_observation_id.duplicated().any() or masks.request_id.duplicated().any():
        raise ValueError('Duplicate planned linkage')
    if masks.groupby('pair_id').pair_partition.nunique().gt(1).any():raise ValueError('Pair crosses partitions')
    cdrugs=set(masks.loc[masks.pair_partition.eq('C'),'drug_id_a'])|set(masks.loc[masks.pair_partition.eq('C'),'drug_id_b'])
    sdrugs=set(masks.loc[masks.pair_partition.eq('calibration'),'drug_id_a'])|set(masks.loc[masks.pair_partition.eq('calibration'),'drug_id_b'])
    if cdrugs&sdrugs:raise ValueError('C and calibration share a drug')
    rmap=references.set_index('pair_id').to_dict('index');cmap=contexts.set_index('pair_id').to_dict('index')
    pmap=predictions.set_index('observation_id').to_dict('index');reg=registry.set_index('model').to_dict('index')
    out=[]
    for row in masks.to_dict('records'):
        for k in BINDING+('drug_id_a','drug_id_b','reference_source_id','reference_source_version'):
            if not isinstance(row[k],str) or not row[k] or row[k]!=row[k].strip():raise ValueError('Invalid literal binding field')
        if row['drug_id_a']==row['drug_id_b']:raise ValueError('Self pair is not admitted')
        if row['model'] not in reg:raise ValueError('Unknown model')
        if any(row[k]!=reg[row['model']][k] for k in ['provider','family']):raise ValueError('Registry linkage mismatch')
        if row['condition'] not in {'inn','dcb'} or row['pair_partition'] not in {'C','calibration'}:raise ValueError('Unexpected condition/partition')
        if row['execution_slot']!='historico_a_identificar':raise ValueError('Unexpected historical slot')
        if row['pair_partition']=='calibration' and row['family']!='openai-gpt-4.1':raise ValueError('Calibration outside source')
        if row['response_execution_id_observed'] is not None or row['complete_observed_compatibility'] or row['response_execution_attribution_verified']:
            raise ValueError('Historical provenance promoted without evidence')
        for key in ['request_fingerprint','payload_fingerprint','text_binding_sha256']:
            if not isinstance(row[key],str) or len(row[key])!=64 or set(row[key])-set('0123456789abcdef'):
                raise ValueError('Invalid binding hash')
        context=cmap[row['pair_id']];ref=rmap[row['pair_id']]
        if any(row[k]!=context[k] for k in ['drug_id_a','drug_id_b','context_id']):raise ValueError('Context mismatch')
        if any(ref[k]!=context[k] for k in ['ref_id_a','ref_id_b','context_id']):raise ValueError('Reference identity mismatch')
        if ref['source_id']!=row['reference_source_id'] or ref['source_version']!=row['reference_source_version']:
            raise ValueError('Reference source/version mismatch')
        known=ref['quality_status']=='usable'
        if known and (any(not isinstance(ref[k],str) or not ref[k] or ref[k]!=ref[k].strip()
                          for k in ['ref_id_a','ref_id_b']) or ref['ref_id_a']==ref['ref_id_b']):
            raise ValueError('Usable reference requires distinct queryable endpoints')
        if known!=row['reference_available']:raise ValueError('Reference availability differs from preparation')
        if known and ref['canonical_label'] not in LABELS:raise ValueError('Usable reference requires valid category')
        if not known and ref['canonical_label'] not in (None,''):raise ValueError('Unusable reference cannot supply category')
        category=None
        if row['available']:
            pred=pmap[row['observation_id']]
            if any(pred[k]!=row[k] for k in BINDING if k!='observation_id'):raise ValueError('Prediction binding mismatch')
            if pred['structural_response_valid']!=row['structural_response_valid']:raise ValueError('Parsing status changed')
            if pred['complete_observed_compatibility'] or pred['response_execution_id_observed'] is not None:
                raise ValueError('Prediction provenance promoted')
            if pred['structural_response_valid']:
                if pred['category'] not in LABELS:raise ValueError('Invalid normalized prediction category')
                category=pred['category']
            elif pred['category'] is not None:raise ValueError('Invalid parsing cannot supply category')
        elif row['structural_response_valid']:raise ValueError('Valid response cannot be unavailable')
        response_gate=(row['available'] and row['structural_response_valid'] and row['binding_verified'] and known
                       and row['identity_admitted'] and row['benchmark_candidate'])
        if row['eligible_d01_response']!=response_gate:raise ValueError('Incoherent response eligibility')
        if row['eligible']!=(response_gate and row['primary_features_complete']):raise ValueError('Incoherent policy eligibility')
        d01e=row['d01e_eligible']
        if d01e is not None and type(d01e) is not bool:raise ValueError('D01-E requires true/false/null')
        if d01e is True and row['d01e_documentary_state']!='demonstrated_support':raise ValueError('D01-E support not demonstrated')
        if d01e is False and row['d01e_documentary_state']!='demonstrated_nonfulfillment':raise ValueError('D01-E false needs demonstrated nonfulfillment')
        if d01e is None and row['d01e_documentary_state'] not in {'insufficient_information','not_assessed','outside_population'}:
            raise ValueError('D01-E state/value mismatch')
        out.append({**row,'category':category,'severity_label':ref['canonical_label'] if known else None,
                    'historical_caveat_preserved':True})
    return pd.DataFrame(out,columns=[*sorted(MASK_COLUMNS),'category','severity_label','historical_caveat_preserved']).sort_values('observation_id').reset_index(drop=True)


def pontuar_c(features,masks,state):
    """Future operation only: score C from frozen constants without fitting."""
    from .metodos_freeze1_v1_3_1 import pontuar_politica_primaria
    names=['nomenclature_delta_morphology_units_pair','nomenclature_lexical_dissimilarity_max']
    table(features,{'pair_id',*names},('pair_id',),'features')
    for name in names:
        if not features[name].map(lambda x:isinstance(x,Real) and not isinstance(x,bool) and math.isfinite(x)).all():
            raise ValueError('Primary features must be finite numeric values, not booleans')
    if not masks.pair_partition.eq('C').all():raise ValueError('Only declared C target may be scored here')
    if set(features.pair_id)!=set(masks.pair_id):raise ValueError('Features do not cover exact C population')
    result=features[['pair_id']].copy();result['risk_score']=pontuar_politica_primaria(features,state)
    if not result.risk_score.map(lambda x:math.isfinite(x)).all():raise ValueError('Nonfinite score')
    return result.sort_values('pair_id').reset_index(drop=True)


def avaliar_d01_real(rows,*,snapshots,documentary_only=False):
    """Describe paired D01 (or its documentarily restricted sensitivity).

    Takes validated rows; repeats the paired binding checks before grouping.
    No selector, quota, inference, pooling or independent replicate claim.
    """
    if not set(MASK_COLUMNS|{'category','severity_label'})<=set(rows):raise ValueError('Missing validated columns')
    if not rows.pair_partition.eq('C').all():raise ValueError('D01 scope is C')
    if not snapshots or len(set(snapshots))!=len(snapshots) or not set(rows.model)<=set(snapshots):raise ValueError('Invalid planned snapshots')
    if rows.observation_id.duplicated().any():raise ValueError('Duplicate observation')
    summary=[];audit=[]
    for model in snapshots:
        decisions=[]
        for (pid,slot),g in rows.loc[rows.model.eq(model)].groupby(['pair_id','execution_slot'],sort=True):
            if len(g)!=2 or set(g.condition)!={'inn','dcb'}:raise ValueError('Missing/duplicate planned D01 arm')
            arms=g.set_index('condition').to_dict('index');a=arms['inn'];b=arms['dcb']
            for k in ['drug_id_a','drug_id_b','context_id','text_binding_sha256','reference_source_id','reference_source_version',
                      'severity_label','d01e_eligible','d01e_documentary_state']:
                if a[k]!=b[k]:raise ValueError('D01 paired binding mismatch')
            both=a['eligible_d01_response'] and b['eligible_d01_response']
            if both and (a['category'] not in LABELS or b['category'] not in LABELS or a['severity_label'] not in LABELS):
                raise ValueError('Eligible D01 without valid categories')
            doc=not documentary_only or (a['d01e_eligible'] is True and b['d01e_eligible'] is True)
            severe=both and doc and a['severity_label']=='grave'
            reason=('documentary_support_insufficient' if documentary_only and a['d01e_eligible'] is None else
                    'documentary_criterion_not_met' if not doc else 'paired_response_unavailable_or_ineligible' if not both else
                    'reference_not_severe' if not severe else '')
            row=dict(model=model,pair_id=pid,execution_slot=slot,both_eligible=both,documentary_subset_member=doc,
                denominator_member=severe,reason=reason,h_dcb=int(b['category'] in {'leve','moderado'}) if severe else None,
                h_inn=int(a['category'] in {'leve','moderado'}) if severe else None)
            decisions.append(row);audit.append(row)
        den=sum(r['denominator_member'] for r in decisions)
        nd=sum(r['h_dcb'] for r in decisions if r['denominator_member']);ni=sum(r['h_inn'] for r in decisions if r['denominator_member'])
        summary.append(dict(model=model,planned_pairs=len(decisions),paired_eligible=sum(r['both_eligible'] for r in decisions),
            documentary_subset_pairs=sum(r['documentary_subset_member'] for r in decisions),denominator_common_severe=den,
            underestimation_dcb_count=nd,underestimation_inn_count=ni,rate_dcb=nd/den if den else None,
            rate_inn=ni/den if den else None,estimate_dcb_minus_inn=(nd-ni)/den if den else None,
            status='descriptive' if den else 'no_planned_configuration' if not decisions else
                'documentary_sensitivity_not_evaluable' if documentary_only and not any(r['documentary_subset_member'] for r in decisions) else 'zero_common_severe_denominator',
            documentary_only=documentary_only,equivalence_claim=False,inferential_claim=False))
    return dict(snapshots=pd.DataFrame(summary,dtype=object),pairing_audit=pd.DataFrame(audit),
        metadata=dict(version=VERSION,selector_applied=False,historical_caveat_preserved=True,DEFINITION_OF_DONE=False))
