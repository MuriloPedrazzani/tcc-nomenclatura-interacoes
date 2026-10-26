"""Invented in-memory evaluation fixtures; never read historical categories."""
import copy
import hashlib
import itertools
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/analise'))
import politica_real_exploratoria as policy

CENTRAL=['gpt-5.4-2026-03-05','gpt-5.4-mini-2026-03-17','gpt-5.4-nano-2026-03-17']
SOURCE='gpt-4.1-2025-04-14'
CLAUDE='claude-haiku-4-5-20251001'


def fixture(models=None,count=5):
    models=models or CENTRAL
    rows=[]
    for model in models:
        for i in range(count):
            rows.append(dict(observation_id=f'fixture-{model}-{i}',pair_id=f'fixture-p{i:02}',
                model=model,provider='anthropic' if model.startswith('claude-') else 'openai',
                family=policy.KNOWN_FAMILIES[model],condition='dcb',execution_slot='historico_a_identificar',
                pair_partition='C',drug_id_a=f'fixture-a{i}',drug_id_b=f'fixture-b{i}',context_id=f'fixture-ctx{i}',
                available=True,eligible=True,eligible_d01_response=True,reference_available=True,
                morphology_stratum=policy.STRATA[i%2],d01e_eligible=None,
                category=['leve','moderado','grave','leve','grave'][i%5],severity_label='grave'))
    frame=pd.DataFrame(rows,columns=policy.ROW_COLUMNS)
    scores=pd.DataFrame([dict(pair_id=f'fixture-p{i:02}',risk_score=float(i)) for i in range(count)],
                        columns=['pair_id','risk_score'])
    return frame,scores


def run(rows=None,scores=None,models=None,basis='eligible_after_gate',**kwargs):
    if rows is None:rows,scores=fixture(models)
    return policy.evaluate_fixed_quota(rows,scores,models=models or CENTRAL,basis=basis,**kwargs)


def test_four_quotas_analytic_baseline_and_parallel_snapshot_weighting():
    result=run()
    summary=result['summary_bycoverage']
    assert len(summary)==4 and len(result['by_snapshot'])==12
    assert summary.accepted_count.tolist()==[9,12,12,15]
    assert summary.clinical_denominator.eq(15).all()
    assert summary.baseline_expected_u_delta1_count.tolist()==pytest.approx([5.4,7.2,7.2,9.])
    assert summary.policy_u_delta1_count.tolist()==[6,9,9,9]
    assert summary.contrast.tolist()==pytest.approx([.04,.12,.12,0.])
    assert len(result['decisions'])==60
    assert not summary.equivalence_claim.any() and not result['metadata']['inference_performed']
    for _,batch in result['decisions'].groupby(['target_coverage','model','execution_slot']):
        assert batch.accepted.sum()==batch.batch_quota.iloc[0]
        assert batch.baseline_acceptance_probability.sum()==pytest.approx(batch.accepted.sum())


def test_baseline_expectation_matches_complete_enumeration_small_fixture():
    rows,scores=fixture([CENTRAL[0]])
    result=run(rows,scores,models=[CENTRAL[0]])
    at60=result['decisions'].query('target_coverage == 0.6')
    potential=at60.potential_delta1.astype(int).to_numpy()
    oracle=np.mean([sum(potential[list(ids)]) for ids in itertools.combinations(range(5),3)])
    assert result['summary_bycoverage'].iloc[0].baseline_expected_u_delta1_count==pytest.approx(oracle)


def test_literal_hash_and_residual_collision_tie_are_order_independent(monkeypatch):
    value='fixture-p00';seed=20261003
    expected=int.from_bytes(hashlib.sha256(f'{seed}\x1f{value}'.encode('utf-8')).digest()[:8],'big')/2**64
    assert policy._hash_uniforme(value,seed)==expected
    rows,scores=fixture([CENTRAL[0]])
    scores['risk_score']=0.
    monkeypatch.setattr(policy,'_hash_uniforme',lambda pid,seed:.25)
    first=run(rows,scores,models=[CENTRAL[0]])
    second=run(rows.iloc[::-1],scores.iloc[::-1],models=[CENTRAL[0]])
    pd.testing.assert_frame_equal(first['decisions'],second['decisions'])
    selected=first['decisions'].query('target_coverage == 0.6 and accepted')
    assert selected.pair_id.tolist()==['fixture-p00','fixture-p01','fixture-p02']
    assert first['metadata']['tie_seed']==20261003


def test_permutation_does_not_change_outputs_or_mutate_inputs():
    rows,scores=fixture();original_rows=rows.copy(deep=True);original_scores=scores.copy(deep=True)
    a=run(rows,scores);b=run(rows.sample(frac=1,random_state=1),scores.iloc[::-1])
    for name in ['summary_bycoverage','by_snapshot','decisions','mechanisms','strata']:
        pd.testing.assert_frame_equal(a[name],b[name])
    pd.testing.assert_frame_equal(rows,original_rows);pd.testing.assert_frame_equal(scores,original_scores)


def test_planned_absence_is_not_model_abstention_and_invalid_available_stays_in_n():
    rows,scores=fixture([SOURCE])
    rows.loc[4,['available','eligible','eligible_d01_response']]=False
    rows.loc[4,['category','severity_label']]=None
    rows.loc[3,['eligible','eligible_d01_response']]=False
    rows.loc[3,'category']=None
    result=run(rows,scores,models=[SOURCE],basis='global')
    first=result['summary_bycoverage'].iloc[0]
    assert first.planned_count==5 and first.available_count==4 and first.unavailable_count==1
    assert first.eligible_count==3 and first.accepted_count==2
    assert first.gate_ineligible_available_count==1 and first.selector_abstention_count==1
    assert first.available_not_accepted_count==2
    missing=result['decisions'].query('not available')
    assert missing.decision_status.eq('not_available_metadata_exclusion').all()
    assert not missing.model_abstention_inferred_for_unavailable.any()


def test_operational_and_grave_denominators_separated_with_unknown_reference():
    rows,scores=fixture([SOURCE])
    rows.loc[0,'severity_label']='leve'
    rows.loc[1,['eligible','eligible_d01_response','reference_available']]=False
    rows.loc[1,'severity_label']=None
    result=run(rows,scores,models=[SOURCE],basis='global')
    summary=result['summary_bycoverage'].iloc[0]
    assert summary.available_count==5 and summary.eligible_count==4 and summary.clinical_denominator==3
    assert summary.operational_denominator==5 and summary.unknown_reference_available_count==1
    assert summary.residual_error_denominator==summary.accepted_count


def test_d0_is_none_never_zero_and_saturation_does_not_claim_equivalence():
    rows,scores=fixture([CENTRAL[0]])
    rows['severity_label']='moderado'
    summary=run(rows,scores,models=[CENTRAL[0]])['summary_bycoverage']
    assert summary.clinical_denominator.eq(0).all()
    for name in ['policy_rate','baseline_expected_rate','contrast','policy_rate_delta2','contrast_delta2']:
        assert all(value is None for value in summary[name])
    rows,scores=fixture([SOURCE]);rows.loc[2:,'eligible']=False
    summary=run(rows,scores,models=[SOURCE],basis='global')['summary_bycoverage']
    assert summary.any_batch_gate_saturated.all() and summary.selector_inactive.all()
    assert summary.contrast.eq(0.).all() and not summary.equivalence_claim.any()


def test_all_ineligible_and_no_available_are_explicit():
    rows,scores=fixture([CENTRAL[0]])
    rows['eligible']=False;rows['category']=None
    summary=run(rows,scores,models=[CENTRAL[0]])['summary_bycoverage']
    assert summary.eligible_count.eq(0).all() and summary.total_abstention.all()
    assert summary.coverage_eligible_after_gate.map(lambda x:x is None).all()
    rows['available']=False;rows['eligible_d01_response']=False;rows['severity_label']=None
    summary=run(rows,scores,models=[CENTRAL[0]])['summary_bycoverage']
    assert summary.status.eq('no_available_observations').all()
    assert summary.planned_count.eq(5).all() and summary.available_count.eq(0).all()


def test_absent_snapshot_and_empty_input_remain_in_inventory():
    rows,scores=fixture([CENTRAL[0]])
    result=run(rows,scores)
    absent=result['by_snapshot'].loc[result['by_snapshot'].model.ne(CENTRAL[0])]
    assert len(absent)==8 and absent.status.eq('configuration_absent').all()
    assert not result['summary_bycoverage'].family_complete.any()
    empty=run(rows.iloc[:0],scores.iloc[:0])
    assert len(empty['summary_bycoverage'])==4 and len(empty['by_snapshot'])==12
    assert empty['decisions'].empty and empty['by_snapshot'].status.eq('configuration_absent').all()


def test_strata_reuse_global_decisions_and_probabilities_without_requota():
    rows,scores=fixture([CENTRAL[0]],count=4)
    # Globally K=2 at 60%; the even-index stratum has two cases, both selected.
    scores['risk_score']=[0.,10.,1.,11.]
    result=run(rows,scores,models=[CENTRAL[0]])
    strata=result['strata'].query("scope == 'family' and target_coverage == 0.6")
    assert strata.accepted_count.tolist()==[2,0]
    assert strata.eligible_count.tolist()==[2,2]
    assert strata.global_decisions_reused.all() and not strata.quota_recomputed_in_stratum.any()
    selected=result['decisions'].query('target_coverage == 0.6')
    for stratum in policy.STRATA:
        sub=selected[selected.morphology_stratum.eq(stratum)]
        out=strata[strata.morphology_stratum.eq(stratum)].iloc[0]
        assert out.baseline_expected_u_delta1_count==pytest.approx(sub.baseline_expected_delta1.sum())


def test_mechanisms_report_literal_denominators_and_delta2():
    rows,scores=fixture([CENTRAL[0]])
    result=run(rows,scores,models=[CENTRAL[0]])
    mechanisms=result['mechanisms'].query("scope == 'family' and target_coverage == 0.6").set_index('mechanism')
    assert set(mechanisms.index)==set(policy.MECHANISMS)
    assert mechanisms.loc['severe_underestimation_delta2','numerator']==1
    assert mechanisms.loc['severe_underestimation_delta2','denominator']==5
    assert mechanisms.loc['severe_routing_rate','numerator']==2
    assert mechanisms.loc['risk_among_severe_accepted_delta1','denominator']==3
    assert mechanisms.loc['residual_error_rate','denominator']==3


def test_claude_is_explicit_conceptual_replication_without_protocol_equivalence():
    rows,scores=fixture([CLAUDE])
    result=run(rows,scores,models=[CLAUDE])
    assert result['metadata']['conceptual_replication']
    assert result['summary_bycoverage'].protocol_confounding_with_other_providers.all()
    assert not result['summary_bycoverage'].independent_replications_claimed.any()


@pytest.mark.parametrize('fault', ['duplicate_observation','duplicate_pair_batch','duplicate_score','score_orphan',
    'score_missing','unknown_snapshot','mixed_families','wrong_family','wrong_provider','wrong_condition',
    'wrong_partition','wrong_slot','missing_id','same_endpoints','endpoint_drift','reference_conflict',
    'nonboolean','eligible_unavailable','eligible_unknown_reference','eligible_bad_category',
    'unavailable_prediction','unavailable_reference_label','bad_score','boolean_score','unknown_stratum',
    'unassessed_eligible_stratum','wrong_seed','boolean_seed','wrong_central_basis'])
def test_inconsistent_inputs_fail_closed(fault):
    rows,scores=fixture();kwargs={}
    if fault=='duplicate_observation':rows=pd.concat([rows,rows.iloc[:1]],ignore_index=True)
    if fault=='duplicate_pair_batch':rows.loc[1,'pair_id']=rows.loc[0,'pair_id']
    if fault=='duplicate_score':scores=pd.concat([scores,scores.iloc[:1]],ignore_index=True)
    if fault=='score_orphan':scores.loc[0,'pair_id']='fixture-orphan'
    if fault=='score_missing':scores=scores.iloc[1:]
    if fault=='unknown_snapshot':kwargs['models']=['unknown']
    if fault=='mixed_families':kwargs['models']=[CENTRAL[0],SOURCE]
    if fault=='wrong_family':rows.loc[0,'family']='other'
    if fault=='wrong_provider':rows.loc[0,'provider']='anthropic'
    if fault=='wrong_condition':rows.loc[0,'condition']='inn'
    if fault=='wrong_partition':rows.loc[0,'pair_partition']='calibration'
    if fault=='wrong_slot':rows.loc[0,'execution_slot']='invented-extra-replicate'
    if fault=='missing_id':rows.loc[0,'context_id']=''
    if fault=='same_endpoints':rows.loc[0,'drug_id_b']=rows.loc[0,'drug_id_a']
    if fault=='endpoint_drift':rows.loc[0,'drug_id_a']='incoherent'
    if fault=='reference_conflict':rows.loc[0,'severity_label']='leve'
    if fault=='nonboolean':rows['available']='true'
    if fault=='eligible_unavailable':rows.loc[0,'available']=False
    if fault=='eligible_unknown_reference':rows.loc[0,'reference_available']=False
    if fault=='eligible_bad_category':rows.loc[0,'category']=None
    if fault=='unavailable_prediction':rows.loc[0,['available','eligible','eligible_d01_response']]=False
    if fault=='unavailable_reference_label':
        rows.loc[0,['eligible','reference_available']]=False
    if fault=='bad_score':scores.loc[0,'risk_score']=np.nan
    if fault=='boolean_score':scores['risk_score']=True
    if fault=='unknown_stratum':rows.loc[0,'morphology_stratum']='arbitrary'
    if fault=='unassessed_eligible_stratum':rows.loc[0,'morphology_stratum']='morphology_not_assessable'
    if fault=='wrong_seed':kwargs['seed']=1302
    if fault=='boolean_seed':kwargs['seed']=True
    if fault=='wrong_central_basis':kwargs['basis']='global'
    with pytest.raises(ValueError):run(rows,scores,**kwargs)
