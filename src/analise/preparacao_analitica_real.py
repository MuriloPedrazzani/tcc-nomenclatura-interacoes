"""Category-free real preparation. No evaluator, reference-label or cache reader.

Inputs are explicit metadata/status projections. A valid historical parser
status is evidence of structural validity, never an invented prediction label.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math

import pandas as pd


VERSION='exploratory-real-analytic-preparation-v1'
FEATURES=('nomenclature_delta_morphology_units_pair','nomenclature_lexical_dissimilarity_max')
KEY_FIELDS=('observation_id','planned_observation_id','pair_id','model','provider','family',
            'condition','execution_slot','pair_partition','admission_status')
PLANNED_FIELDS=set(KEY_FIELDS)|{'drug_id_a','drug_id_b','context_id','request_id','request_fingerprint',
    'payload_fingerprint','text_binding_sha256','binding_verified','parsing_candidate',
    'response_execution_id_observed','response_execution_attribution_verified',
    'complete_observed_compatibility','submission_execution_ids'}
STATUS_FIELDS=set(KEY_FIELDS)|{'metadata_support','authorized_in_this_round','parsed_this_round',
    'cache_value_available','parsing_status','structural_response_valid','reasons'}
REGISTRY_FIELDS={'model','provider','family','role'}


def _table(frame,fields,key,name):
    if not isinstance(frame,pd.DataFrame) or frame.columns.duplicated().any() or set(frame.columns)!=set(fields):
        raise ValueError(name+': explicit category-free projection required')
    if frame[key].duplicated().any() or not frame[key].map(lambda x:isinstance(x,str) and x==x.strip() and bool(x)).all():
        raise ValueError(name+': invalid or duplicate identity')


def _boolean(row,fields):
    for field in fields:
        if type(row[field]) is not bool:
            raise ValueError('Strict boolean metadata required: '+field)


def build_analytic_masks(planned,status,documentary,features,registry):
    """Preserve every planned slot; keep absent units outside model abstention.

    Documentary availability is an already verified, category-free projection.
    The evaluator's synthetic guard is neither called nor bypassed.
    """
    _table(planned,PLANNED_FIELDS,'observation_id','planned')
    _table(status,STATUS_FIELDS,'observation_id','status')
    _table(registry,REGISTRY_FIELDS,'model','registry')
    _table(features,{'pair_id',*FEATURES},'pair_id','features')
    required_doc={'pair_id','drug_id_a','drug_id_b','context_id','reference_source_id','reference_source_version',
        'identity_admitted','benchmark_candidate','reference_record_present','reference_available',
        'reference_quality_status','literal_inn_to_dcb_supported','both_international_names_documented_inn',
        'd01e_documentary_state','d01e_reason'}
    _table(documentary,required_doc,'pair_id','documentary')
    if set(planned.observation_id)!=set(status.observation_id):raise ValueError('Planned/status identities differ')
    composite=['pair_id','model','condition','execution_slot','pair_partition']
    if (planned[composite].duplicated().any() or planned.planned_observation_id.duplicated().any()
            or planned.request_id.duplicated().any()):
        raise ValueError('Duplicated planned composite identity')
    if planned.groupby('pair_id').pair_partition.nunique().gt(1).any():
        raise ValueError('Pair assigned to more than one partition')
    c_drugs=set(planned.loc[planned.pair_partition.eq('C'),'drug_id_a'])|set(planned.loc[planned.pair_partition.eq('C'),'drug_id_b'])
    calibration_drugs=set(planned.loc[planned.pair_partition.eq('calibration'),'drug_id_a'])|set(planned.loc[planned.pair_partition.eq('calibration'),'drug_id_b'])
    if c_drugs & calibration_drugs:raise ValueError('C and calibration share a drug')
    docs=documentary.set_index('pair_id').to_dict('index')
    feature_by_id=features.set_index('pair_id').to_dict('index')
    models=registry.set_index('model').to_dict('index')
    statuses=status.set_index('observation_id').to_dict('index')
    rows=[]
    for row in planned.to_dict('records'):
        oid=row['observation_id'];st=statuses[oid]
        if any(row[k]!=st[k] for k in KEY_FIELDS if k!='observation_id'):
            raise ValueError('Status linkage mismatch')
        if row['pair_id'] not in docs or row['pair_id'] not in feature_by_id or row['model'] not in models:
            raise ValueError('Orphan planned identity')
        d=docs[row['pair_id']];f=feature_by_id[row['pair_id']];model=models[row['model']]
        if any(row[k]!=d[k] for k in ['drug_id_a','drug_id_b','context_id']):raise ValueError('Documentary binding mismatch')
        if any(row[k]!=model[k] for k in ['provider','family']):raise ValueError('Snapshot provenance mismatch')
        if row['condition'] not in {'inn','dcb'} or row['pair_partition'] not in {'C','calibration'}:
            raise ValueError('Observation outside authorized preparation scope')
        if row['pair_partition']=='calibration' and row['family']!='openai-gpt-4.1':
            raise ValueError('Only source calibration is authorized')
        if row['execution_slot']!='historico_a_identificar':raise ValueError('Unexpected execution slot')
        for k in ['request_id','request_fingerprint','payload_fingerprint','text_binding_sha256','context_id']:
            if not isinstance(row[k],str) or not row[k]:raise ValueError('Missing provenance')
        for k in ['request_fingerprint','payload_fingerprint','text_binding_sha256']:
            if len(row[k])!=64 or any(ch not in '0123456789abcdef' for ch in row[k]):
                raise ValueError('Invalid provenance SHA-256')
        _boolean(row,['binding_verified','parsing_candidate','response_execution_attribution_verified','complete_observed_compatibility'])
        _boolean(st,['metadata_support','authorized_in_this_round','parsed_this_round','cache_value_available','structural_response_valid'])
        _boolean(d,['identity_admitted','benchmark_candidate','reference_available','reference_record_present',
                    'literal_inn_to_dcb_supported','both_international_names_documented_inn'])
        if d['reference_available']!=(d['reference_record_present'] and d['reference_quality_status']=='usable'):
            raise ValueError('Reference availability contradicts presence/quality metadata')
        if d['benchmark_candidate'] and not (d['identity_admitted'] and d['reference_record_present']):
            raise ValueError('Candidate status contradicts identity/reference record')
        if row['response_execution_id_observed'] is not None or row['response_execution_attribution_verified'] or row['complete_observed_compatibility']:
            raise ValueError('This historical passage has no verified execution attribution')
        if st['metadata_support']!=row['parsing_candidate'] or st['authorized_in_this_round']!=row['parsing_candidate']:
            raise ValueError('Authorized parsing scope differs from pinned metadata')
        if st['parsed_this_round']!=st['authorized_in_this_round']:
            raise ValueError('Missing authorized parsing result')
        if st['structural_response_valid']!=(st['parsing_status']=='valid'):
            raise ValueError('Inconsistent structural parsing state')
        if st['cache_value_available'] and not st['parsed_this_round']:
            raise ValueError('Unavailable parsing cannot assert a cached value')
        if st['structural_response_valid'] and not st['cache_value_available']:
            raise ValueError('Valid response without available value')
        if not isinstance(row['submission_execution_ids'],list) or len(set(row['submission_execution_ids']))!=len(row['submission_execution_ids']):
            raise ValueError('Invalid submission provenance')
        if st['authorized_in_this_round'] and len(row['submission_execution_ids'])>1:
            raise ValueError('Ambiguous submission cannot be recovered in this passage')
        available=st['parsed_this_round'] and st['cache_value_available']
        reasons=[]
        if not available:
            reasons=['not_available_from_authorized_parsing',row['admission_status']]
        else:
            if not st['structural_response_valid']:reasons.append('invalid_structural_response')
            if not row['binding_verified']:reasons.append('unverified_individual_binding')
            if not d['identity_admitted'] or not d['benchmark_candidate']:reasons.append('outside_documentary_population')
            if not d['reference_available']:reasons.append('reference_unavailable_or_unusable')
        response_eligible=available and not reasons
        feature_complete=all(isinstance(f[k],(int,float)) and not isinstance(f[k],bool) and math.isfinite(f[k]) for k in FEATURES)
        policy_reasons=list(reasons)
        if available and not feature_complete:policy_reasons.append('primary_feature_missing_or_nonfinite')
        eligible=response_eligible and feature_complete
        delta=f[FEATURES[0]]
        delta_finite=isinstance(delta,(int,float)) and not isinstance(delta,bool) and math.isfinite(delta)
        morph=('morphology_delta_zero' if delta==0 else 'morphology_delta_nonzero') if delta_finite else 'morphology_not_assessable'
        rows.append({**row,'available':available,'structural_response_valid':st['structural_response_valid'],
            'parsing_status':st['parsing_status'],'reference_available':d['reference_available'],
            'reference_source_id':d['reference_source_id'],'reference_source_version':d['reference_source_version'],
            'identity_admitted':d['identity_admitted'],'benchmark_candidate':d['benchmark_candidate'],
            'eligible_d01_response':response_eligible,'eligible':eligible,'eligible_policy':eligible,
            'preparation_eligibility_status':('eligible_with_historical_caveat' if eligible else
                 'ineligible_available' if available else 'not_processed_metadata_exclusion'),
            'model_abstention_inferred':False,'gate_reasons':policy_reasons,
            'primary_features_complete':feature_complete,'morphology_stratum':morph,
            'literal_inn_to_dcb_supported':d['literal_inn_to_dcb_supported'],
            'both_international_names_documented_inn':d['both_international_names_documented_inn'],
            'd01e_documentary_state':d['d01e_documentary_state'],'d01e_eligible':None,
            'historical_caveat_preserved':True,'analysis_execution_authorized':False,
            'clinical_denominator_computed':False,'preparation_version':VERSION})
    return pd.DataFrame(rows)


def paired_masks(masks):
    groups=defaultdict(dict)
    for row in masks.to_dict('records'):
        key=(row['pair_id'],row['model'],row['execution_slot'],row['pair_partition'])
        if row['condition'] in groups[key]:raise ValueError('Duplicate planned arm')
        groups[key][row['condition']]=row
    rows=[]
    for (pid,model,slot,partition),arms in sorted(groups.items()):
        if set(arms)!={'inn','dcb'}:raise ValueError('Planned arm missing')
        a,b=arms['inn'],arms['dcb']
        same=all(a[k]==b[k] for k in ['drug_id_a','drug_id_b','context_id','text_binding_sha256',
                                     'reference_source_id','reference_source_version'])
        if not same:raise ValueError('Incoherent identity/context/provenance in paired join')
        verified=same and a['binding_verified'] and b['binding_verified']
        rows.append(dict(pair_id=pid,model=model,execution_slot=slot,pair_partition=partition,
            inn_observation_id=a['observation_id'],dcb_observation_id=b['observation_id'],
            same_identity_context_fingerprint=same,paired_binding_verified=verified,
            both_available=a['available'] and b['available'],
            comparable_d01=bool(verified and a['eligible_d01_response'] and b['eligible_d01_response']),
            comparable_policy=bool(verified and a['eligible'] and b['eligible']),
            morphology_stratum=a['morphology_stratum'],d01e_documentary_state=a['d01e_documentary_state'],
            d01e_comparable=None,d01e_pending_reason='direct_complete_evidence_strength_not_assessed',
            physical_pair_execution_observed=False,historical_caveat_preserved=True))
    return pd.DataFrame(rows)


def invariance_by_scope(masks,scopes):
    """Scopes are literal, caller-supplied analysis memberships, never all rows.

    Local diagnostics for source calibration do not overwrite the central base.
    N counts available responses; planned exclusions remain separately counted.
    """
    records=masks.to_dict('records');summaries=[];details=[]
    for scope in scopes:
        if set(scope)!={'scope_id','models','partition','execution_slot'}:raise ValueError('Explicit scope required')
        models=scope['models']
        if not models or len(set(models))!=len(models):raise ValueError('Duplicate/empty scope membership')
        per=[]
        for model in models:
            group=[r for r in records if r['model']==model and r['pair_partition']==scope['partition'] and r['execution_slot']==scope['execution_slot']]
            by={c:[r for r in group if r['condition']==c] for c in ['inn','dcb']}
            available={c:{r['pair_id'] for r in by[c] if r['available']} for c in by}
            eligible={c:{r['pair_id'] for r in by[c] if r['eligible']} for c in by}
            equal_available=available['inn']==available['dcb'];equal_eligible=eligible['inn']==eligible['dcb']
            has_configuration=bool(by['inn'] and by['dcb'])
            nonempty=bool(available['inn'] and available['dcb'])
            invariant=has_configuration and nonempty and equal_available and equal_eligible
            row=dict(scope_id=scope['scope_id'],model=model,pair_partition=scope['partition'],execution_slot=scope['execution_slot'],
                planned_inn=len(by['inn']),planned_dcb=len(by['dcb']),
                available_inn=len(available['inn']),available_dcb=len(available['dcb']),
                eligible_inn=len(eligible['inn']),eligible_dcb=len(eligible['dcb']),
                same_available_sets=equal_available,same_eligible_sets=equal_eligible,
                both_configurations_planned=has_configuration,both_conditions_available=nonempty,invariant=invariant,
                available_only_inn=sorted(available['inn']-available['dcb']),available_only_dcb=sorted(available['dcb']-available['inn']),
                eligible_only_inn=sorted(eligible['inn']-eligible['dcb']),eligible_only_dcb=sorted(eligible['dcb']-eligible['inn']))
            per.append(row);details.append(row)
        scope_invariant=all(r['invariant'] for r in per)
        statuses=all(r['both_configurations_planned'] and r['both_conditions_available'] for r in per)
        summaries.append(dict(**scope,invariant=scope_invariant,
            status='ASSESSED' if statuses else 'CONFIGURATION_ABSENT_OR_EMPTY',
            coverage_basis='eligible_after_gate' if scope_invariant else 'global',
            fallback_activated=bool(statuses and not scope_invariant),
            basis_is_local_diagnostic=True,grave_denominator_computed=False))
    return summaries,details


def preparation_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
