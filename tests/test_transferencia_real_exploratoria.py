"""Future API fixtures mirror COMMON wrappers; no real clinical data loaded."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
for sub in ('analise','dados'):sys.path.insert(0,str(ROOT/'src'/sub))
from calibracao_preparatoria_exploratoria import calibrate_source_grid,SCORE_COLUMNS,CONFIG_COLUMNS,HASH_NAMES,VERSION_NAMES
from transferencia_real_exploratoria import aplicar_transferencia,bind_thresholds,MODEL_MAP,SLOT,ROW_FIELDS,COMMON_COPY_FIELDS
APPROVAL='a'*64
ARTIFACT='b'*64
SOURCE_ARTIFACT='d'*64
FREEZE='c'*64


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def fixtures(n=5,*,all_tied=False,policy_state_sha256=None):
    targets,scores,source,configs,source_meta=[],[],[],[],[]
    for model,target in MODEL_MAP.items():
        for arm in ('dcb','inn'):
            configs.append((model,arm,SLOT))
            for i in range(n):
                calpair='cal'+str(i);pair='test'+str(i);value=1. if all_tied else float(i)
                source.append(dict(observation_id='|'.join((model,arm,calpair)),pair_id=calpair,model=model,
                    condition=arm,execution_slot=SLOT,pair_partition='calibration',available=True,eligible=True,risk_score=value))
                source_meta.append(dict(pair_id=calpair,drug_id_a=calpair+'a',drug_id_b=calpair+'b',context_id=calpair+'ctx'))
                targets.append(dict(observation_id='|'.join((target,arm,pair)),pair_id=pair,model=target,
                    provider='openai',family='openai-gpt-5.4',condition=arm,execution_slot=SLOT,pair_partition='C',
                    drug_id_a=pair+'a',drug_id_b=pair+'b',context_id=pair+'ctx',available=True,eligible=True,
                    eligible_d01_response=True,reference_available=True,morphology_stratum='morphology_delta_zero',
                    d01e_eligible=None,category='leve' if i%2==0 else 'grave',severity_label='grave'))
                scores.append(dict(pair_id=pair,risk_score=value))
    source=pd.DataFrame(source,columns=SCORE_COLUMNS,dtype=object)
    cfg=pd.DataFrame(configs,columns=CONFIG_COLUMNS)
    hv={k:'e'*64 for k in HASH_NAMES};versions={k:'fixture-v1' for k in VERSION_NAMES}
    if policy_state_sha256 is not None:hv['policy_state_sha256']=policy_state_sha256
    calc={basis:calibrate_source_grid(source,cfg,coverage_basis_by_configuration={tuple(r):basis for r in configs},
        hashes=hv,versions=versions)['thresholds'] for basis in ('global','eligible_after_gate')}
    commons=[]
    for e,g in zip(calc['eligible_after_gate'].to_dict('records'),calc['global'].to_dict('records')):
        assert e['threshold']==g['threshold']
        group=source.loc[source.model.eq(g['model']) & source.condition.eq(g['condition']) & source.eligible]
        selected=sorted(group.loc[group.risk_score.le(g['threshold']),'observation_id'])
        proof=dict(model=e['model'],condition=e['condition'],execution_slot=SLOT,coverage=e['target_coverage'],
            same_threshold=True,same_source_accepted_ids=True,source_accepted_ids_sha256=digest(selected),
            eligible_calculation_sha256=e['threshold_sha256'],global_calculation_sha256=g['threshold_sha256'])
        payload={k:e[k] for k in COMMON_COPY_FIELDS}
        payload.update(threshold_status='calculated_candidate_common_to_both_bases',non_estimable_reason=None,
            operational_coverage_basis=None,operational_basis_decision='pending_transfer_scope_decision',
            calculation_bases=['eligible_after_gate','global'],candidate_numeric_invariant_to_current_bases=True,
            base_equivalence_proof=proof,provenance_hashes=hv,provenance_versions=versions,tie_seed=None,
            tie_seed_required_for_threshold=False,thresholds_formally_frozen=False,target_application_performed=False,DEFINITION_OF_DONE=False)
        h=digest(payload);commons.append(dict(threshold_id='CANDTHR_COMMON_'+h[:24],threshold_sha256=h,**payload))
    return dict(targets=pd.DataFrame(targets,columns=sorted(ROW_FIELDS),dtype=object),
        scores=pd.DataFrame(scores).drop_duplicates().sort_values('pair_id').reset_index(drop=True),
        commons=pd.DataFrame(commons,dtype=object),global_calculations=calc['global'],source=source,
        source_meta=pd.DataFrame(source_meta).drop_duplicates().reset_index(drop=True))


def bind(data):
    return bind_thresholds(data['commons'],data['global_calculations'],approval_sha256=APPROVAL,
        candidate_artifact_sha256=ARTIFACT,source_calculation_artifact_sha256=SOURCE_ARTIFACT)


def evaluate(data,bound=None,*,endpoints=True):
    return aplicar_transferencia(data['targets'],data['scores'],bound if bound is not None else bind(data)['thresholds'],
        data['source'],freeze_manifest_sha256=FREEZE,approval_sha256=APPROVAL,formal_freeze_complete=True,
        source_pair_metadata=data['source_meta'] if endpoints else None)


def rehash_common(data,index=0):
    r=data['commons'].iloc[index].to_dict()
    h=digest({k:v for k,v in r.items() if k not in {'threshold_id','threshold_sha256'}})
    data['commons'].loc[index,'threshold_sha256']=h
    data['commons'].loc[index,'threshold_id']='CANDTHR_COMMON_'+h[:24]


def test_common_binding_preserves_payloads_ids_values_artifacts():
    data=fixtures();original=copy.deepcopy(data['commons']);bound=bind(data)
    assert len(bound['thresholds'])==24
    common=original.set_index('threshold_id');glob=data['global_calculations'].set_index('threshold_sha256')
    for row in bound['thresholds'].to_dict('records'):
        assert row['threshold_id'].startswith('CANDTHR_COMMON_')
        assert row['threshold']==common.at[row['threshold_id'],'threshold']
        assert row['candidate_payload']==(common.loc[row['threshold_id']].to_dict()|{'threshold_id':row['threshold_id']})
        assert row['candidate_threshold_sha256']==common.at[row['threshold_id'],'threshold_sha256']
        assert row['source_calculation_payload']['threshold']==glob.at[row['source_calculation_sha256'],'threshold']
        assert row['candidate_payload']['operational_coverage_basis'] is None and row['coverage_basis']=='global'
        assert row['candidate_artifact_sha256']==ARTIFACT and row['source_calculation_artifact_sha256']==SOURCE_ARTIFACT
    assert not bound['metadata']['thresholds_formally_frozen'] and not bound['metadata']['target_application_performed']
    assert not bound['metadata']['supplied_hashes_independently_verified']
    pd.testing.assert_frame_equal(original,data['commons'])


@pytest.mark.parametrize('fault',['missing','duplicate','value','wrong_slot','wrong_source','unknown_column',
    'bad_hash','proof_global_hash','proof_condition','proof_boolean','provenance_hash','provenance_version',
    'raw_changed','raw_wrong_base','duplicate_raw','wrapper_preapproved','wrapper_frozen'])
def test_wrapper_links_and_states_reject_incoherence(fault):
    d=fixtures()
    if fault=='missing':d['commons']=d['commons'].iloc[1:]
    if fault=='duplicate':d['commons']=pd.concat([d['commons'],d['commons'].iloc[[0]]])
    if fault=='value':d['commons'].loc[0,'threshold']=999.;rehash_common(d)
    if fault=='wrong_slot':d['commons'].loc[0,'execution_slot']='fixture';rehash_common(d)
    if fault=='wrong_source':d['commons'].loc[0,'model']=next(iter(MODEL_MAP.values()));rehash_common(d)
    if fault=='unknown_column':d['commons']['category']='forbidden'
    if fault=='bad_hash':d['commons'].loc[0,'threshold_sha256']='0'*64
    if fault.startswith('proof_'):
        p=copy.deepcopy(d['commons'].at[0,'base_equivalence_proof'])
        p[{'proof_global_hash':'global_calculation_sha256','proof_condition':'condition','proof_boolean':'same_threshold'}[fault]]={
            'proof_global_hash':'0'*64,'proof_condition':'inn','proof_boolean':False}[fault]
        d['commons'].at[0,'base_equivalence_proof']=p;rehash_common(d)
    if fault=='provenance_hash':
        h=copy.deepcopy(d['commons'].at[0,'provenance_hashes']);h['policy_state_sha256']='0'*64
        d['commons'].at[0,'provenance_hashes']=h;rehash_common(d)
    if fault=='provenance_version':
        v=copy.deepcopy(d['commons'].at[0,'provenance_versions']);v['feature_version']='other'
        d['commons'].at[0,'provenance_versions']=v;rehash_common(d)
    if fault=='raw_changed':d['global_calculations'].loc[0,'threshold']=999.
    if fault=='raw_wrong_base':d['global_calculations'].loc[0,'coverage_basis']='eligible_after_gate'
    if fault=='duplicate_raw':d['global_calculations']=pd.concat([d['global_calculations'],d['global_calculations'].iloc[[0]]])
    if fault=='wrapper_preapproved':d['commons'].loc[0,'operational_coverage_basis']='global';rehash_common(d)
    if fault=='wrapper_frozen':d['commons'].loc[0,'thresholds_formally_frozen']=True;rehash_common(d)
    with pytest.raises(ValueError):bind(d)


def test_formal_freeze_required_source_has_no_outcomes():
    d=fixtures();bound=bind(d)['thresholds']
    assert 'category' not in d['source'] and 'severity_label' not in d['source']
    with pytest.raises(ValueError,match='formal freeze'):
        aplicar_transferencia(d['targets'],d['scores'],bound,d['source'],freeze_manifest_sha256=FREEZE,approval_sha256=APPROVAL)
    result=evaluate(d,bound)
    assert len(result['configurations'])==24
    assert not result['metadata']['source_categories_required'] and not result['metadata']['source_clinical_labels_required']
    assert result['metadata']['source_target_drug_disjoint_checked']
    assert all(not p['thresholds_formally_frozen'] for p in bound.candidate_payload)


def test_endpoint_check_absence_reported_honestly():
    result=evaluate(fixtures(),endpoints=False)
    assert not result['metadata']['source_target_drug_disjoint_checked']
    assert 'delegated' in result['metadata']['source_target_drug_disjoint_note']


def test_row_order_independence_global_basis():
    d=fixtures();bound=bind(d)['thresholds'];one=evaluate(d,bound)
    for name in d:d[name]=d[name].iloc[::-1]
    two=evaluate(d,bound.iloc[::-1])
    pd.testing.assert_frame_equal(one['configurations'],two['configurations'])
    pd.testing.assert_frame_equal(one['decisions'],two['decisions'])
    assert one['configurations'].drift_pp.eq(0.).all() and one['configurations'].drift_within_tolerance.eq(True).all()
    assert one['configurations'].quota_forced_on_target.eq(False).all()


def test_inclusive_ties_without_forced_quota_or_equivalence():
    table=evaluate(fixtures(all_tied=True))['configurations']
    assert table.target_accepted.eq(5).all() and table.target_coverage_global.eq(1.).all()
    assert table.target_selector_inactive.eq(True).all() and table.equivalence_claim.eq(False).all()


def test_operational_denominator_excludes_unavailable_and_differs_from_clinical():
    d=fixtures();rows=d['targets']
    rows.loc[rows.pair_id.eq('test4'),['available','eligible','eligible_d01_response']]=False
    rows.loc[rows.pair_id.eq('test4'),'category']=None
    rows.loc[rows.pair_id.eq('test3'),['eligible','eligible_d01_response','reference_available']]=False
    rows.loc[rows.pair_id.eq('test3'),'severity_label']=None
    result=evaluate(d);t=result['configurations']
    assert t.target_planned.eq(5).all() and t.target_available.eq(4).all()
    assert t.target_eligible.eq(3).all() and t.target_unavailable.eq(1).all()
    assert t.clinical_denominator.eq(3).all() and t.target_accepted.eq(3).all()
    assert t.target_coverage_global.eq(.75).all() and t.target_coverage_eligible_after_gate.eq(1.).all()
    assert t.target_gate_saturated.eq(True).all()
    unavailable=result['decisions'].loc[lambda x:x.pair_id.eq('test4')]
    assert unavailable.accepted.isna().all() and unavailable.decision_status.eq('unavailable_not_a_model_abstention').all()


def test_zero_clinical_denominator_remains_undefined():
    d=fixtures();d['targets']['severity_label']='moderado';t=evaluate(d)['configurations']
    assert t.clinical_denominator.eq(0).all() and t.underestimation_rate_delta1.isna().all()
    assert t.underestimation_rate_delta2.isna().all() and t.equivalence_claim.eq(False).all()


def test_descriptive_target_mechanisms_distinct_denominators():
    t=evaluate(fixtures())['configurations'].loc[lambda x:x.target_coverage.eq(.6)]
    assert t.target_accepted.eq(3).all() and t.clinical_denominator.eq(5).all()
    assert t.u_delta1_count.eq(2).all() and t.u_delta2_count.eq(2).all() and t.underestimation_rate_delta1.eq(.4).all()
    assert t.risk_among_severe_accepted_delta1.eq(2/3).all() and t.residual_error_rate.eq(2/3).all()
    assert t.severe_routing_rate.eq(.4).all()


@pytest.mark.parametrize('difference,within',[(1,True),(-1,True),(2,False),(-2,False)])
def test_exact_drift_plus_minus_five_inclusive(difference,within):
    d=fixtures(n=20);d['scores']['risk_score']-=difference
    t=evaluate(d)['configurations'].loc[lambda x:x.target_coverage.eq(.6)]
    assert t.drift_pp.eq(difference*5).all() and t.drift_within_tolerance.eq(within).all()


def test_absent_target_configuration_retained_in_24():
    d=fixtures();model=next(iter(MODEL_MAP.values()))
    d['targets']=d['targets'].loc[~(d['targets'].model.eq(model)&d['targets'].condition.eq('inn'))]
    t=evaluate(d)['configurations'];absent=t.loc[t.target_model.eq(model)&t.condition.eq('inn')]
    assert len(t)==24 and len(absent)==4 and absent.status.eq('target_configuration_absent').all()
    assert absent.target_available.eq(0).all() and absent.target_coverage_global.isna().all()


@pytest.mark.parametrize('fault',['duplicate_row','duplicate_composite','duplicate_score','orphan_score','missing_score',
    'wrong_endpoint','wrong_context','wrong_family','wrong_provider','wrong_slot','wrong_partition','shared_entity',
    'reference_incoherent','category_missing','unavailable_category','eligible_unavailable','unknown_column',
    'source_score_changed','source_category','source_reference','source_duplicate','source_slot','source_pair_shared',
    'source_missing_configuration','source_meta_orphan','binding_tampered','approval_tampered','accepted_set_proof'])
def test_invalid_joins_masks_source_and_bindings_fail(fault):
    d=fixtures();bound=bind(d)['thresholds'];rows=d['targets']
    if fault=='duplicate_row':d['targets']=pd.concat([rows,rows.iloc[[0]]])
    if fault=='duplicate_composite':
        extra=rows.iloc[[0]].copy();extra['observation_id']='other';d['targets']=pd.concat([rows,extra])
    if fault=='duplicate_score':d['scores']=pd.concat([d['scores'],d['scores'].iloc[[0]]])
    if fault=='orphan_score':d['scores'].loc[len(d['scores'])]=['orphan',0.]
    if fault=='missing_score':d['scores']=d['scores'].iloc[1:]
    if fault=='wrong_endpoint':rows.loc[0,'drug_id_a']='different'
    if fault=='wrong_context':rows.loc[0,'context_id']='different'
    if fault=='wrong_family':rows.loc[0,'family']='different'
    if fault=='wrong_provider':rows.loc[0,'provider']='anthropic'
    if fault=='wrong_slot':rows.loc[0,'execution_slot']='slot'
    if fault=='wrong_partition':rows.loc[0,'pair_partition']='calibration'
    if fault=='shared_entity':rows.loc[rows.pair_id.eq('test0'),'drug_id_a']='cal0a'
    if fault=='reference_incoherent':rows.loc[0,'severity_label']='leve'
    if fault=='category_missing':rows.loc[0,'category']=None
    if fault=='unavailable_category':rows.loc[0,['available','eligible','eligible_d01_response']]=False
    if fault=='eligible_unavailable':rows.loc[0,'available']=False
    if fault=='unknown_column':rows['raw_response']='forbidden'
    if fault=='source_score_changed':d['source'].loc[0,'risk_score']=999.
    if fault=='source_category':d['source']['category']='forbidden'
    if fault=='source_reference':d['source']['severity_label']='forbidden'
    if fault=='source_duplicate':d['source']=pd.concat([d['source'],d['source'].iloc[[0]]])
    if fault=='source_slot':d['source'].loc[0,'execution_slot']='slot'
    if fault=='source_pair_shared':d['source'].loc[0,'pair_id']='test0'
    if fault=='source_missing_configuration':d['source']=d['source'].loc[d['source'].condition.eq('dcb')]
    if fault=='source_meta_orphan':d['source_meta'].loc[0,'pair_id']='orphan'
    if fault=='binding_tampered':bound.loc[0,'threshold']=999.
    if fault=='approval_tampered':bound.loc[0,'approval_sha256']='f'*64
    if fault=='accepted_set_proof':
        proof=copy.deepcopy(d['commons'].at[0,'base_equivalence_proof']);proof['source_accepted_ids_sha256']='0'*64
        d['commons'].at[0,'base_equivalence_proof']=proof;rehash_common(d);bound=bind(d)['thresholds']
    with pytest.raises(ValueError):evaluate(d,bound)


@pytest.mark.filterwarnings('error::DeprecationWarning')
@pytest.mark.parametrize('representation', ['bool', 'numpy_object'])
def test_boolean_mask_representation_preserves_outputs_and_inputs(representation):
    data = fixtures()
    rows = data['targets']
    rows.loc[rows.pair_id.eq('test4'), ['available', 'eligible', 'eligible_d01_response']] = False
    rows.loc[rows.pair_id.eq('test4'), 'category'] = None
    rows.loc[rows.pair_id.eq('test3'), ['eligible', 'eligible_d01_response', 'reference_available']] = False
    rows.loc[rows.pair_id.eq('test3'), 'severity_label'] = None
    original = copy.deepcopy(data)
    expected = evaluate(data)
    alternate = copy.deepcopy(data)
    for name, fields in [('targets', ['available', 'eligible', 'eligible_d01_response', 'reference_available']),
                         ('source', ['available', 'eligible'])]:
        for field in fields:
            values = alternate[name][field]
            alternate[name][field] = (values.astype(bool) if representation == 'bool'
                else pd.Series([np.bool_(v) for v in values], index=values.index, dtype=object))
    alternate_before = copy.deepcopy(alternate)
    actual = evaluate(alternate)
    assert actual.keys() == expected.keys()
    for key in expected:
        if isinstance(expected[key], pd.DataFrame):
            pd.testing.assert_frame_equal(actual[key], expected[key])
        else:
            assert actual[key] == expected[key]
    for key in data:
        pd.testing.assert_frame_equal(data[key], original[key])
        pd.testing.assert_frame_equal(alternate[key], alternate_before[key])


@pytest.mark.filterwarnings('error::DeprecationWarning')
@pytest.mark.parametrize('kind', ['target_eligible', 'target_d01_only', 'source_eligible'])
def test_unavailable_eligible_masks_still_rejected(kind):
    from transferencia_real_exploratoria import _validate_rows, _validate_source
    data = fixtures()
    if kind == 'source_eligible':
        data['source'].loc[0, 'available'] = False
        with pytest.raises(ValueError, match='Source eligible without availability'):
            _validate_source(data['source'], data['targets'], data['source_meta'])
    else:
        data['targets'].loc[0, 'available'] = False
        if kind == 'target_d01_only':
            data['targets'].loc[0, 'eligible'] = False
        with pytest.raises(ValueError, match='Eligible without admitted availability'):
            _validate_rows(data['targets'], data['scores'])


@pytest.mark.filterwarnings('error::DeprecationWarning')
@pytest.mark.parametrize('invalid', [0, 1, 'False', None, pd.NA])
def test_availability_comparison_does_not_coerce_non_booleans(invalid):
    from transferencia_real_exploratoria import _validate_rows, _validate_source
    data = fixtures()
    for field in ['available', 'eligible', 'eligible_d01_response', 'reference_available']:
        rows = data['targets'].copy(deep=True)
        rows.loc[0, field] = invalid
        with pytest.raises(ValueError, match='Explicit mask booleans required'):
            _validate_rows(rows, data['scores'])
    for field in ['available', 'eligible']:
        source = data['source'].copy(deep=True)
        source.loc[0, field] = invalid
        with pytest.raises(ValueError, match='Explicit source mask booleans required'):
            _validate_source(source, data['targets'], data['source_meta'])
