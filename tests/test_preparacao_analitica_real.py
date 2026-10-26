"""Invented status-only fixtures; categories are prohibited, not fabricated."""
import copy
import pandas as pd
import pytest

from src.analise.preparacao_analitica_real import (
    PLANNED_FIELDS,STATUS_FIELDS,FEATURES,build_analytic_masks,paired_masks,invariance_by_scope)


def fixture():
    registry=pd.DataFrame([dict(model=m,provider='openai',family=f,role='fixture') for m,f in
        [('central','openai-gpt-5.4'),('source','openai-gpt-4.1')]])
    docs=pd.DataFrame([dict(pair_id=p,drug_id_a=p+'a',drug_id_b=p+'b',context_id=p+'ctx',
        reference_source_id='fixture-ref',reference_source_version='fixture-v1',identity_admitted=True,
        benchmark_candidate=True,reference_record_present=True,reference_available=True,
        reference_quality_status='usable',literal_inn_to_dcb_supported=True,
        both_international_names_documented_inn=True,d01e_documentary_state='not_assessed',
        d01e_reason='strength_not_assessed') for p in ['p1','p2','p3','p4']])
    features=pd.DataFrame([dict(pair_id=p,**{FEATURES[0]:float(i),FEATURES[1]:0.25}) for i,p in enumerate(['p1','p2','p3','p4'])])
    planned=[];statuses=[]
    for model in registry.to_dict('records'):
        for d in docs.to_dict('records'):
            if (model['model']=='central')!=(d['pair_id'] in {'p1','p2'}):continue
            for condition in ['inn','dcb']:
                oid=d['pair_id']+'-'+model['model']+'-'+condition
                row=dict(observation_id=oid,planned_observation_id='plan-'+oid,pair_id=d['pair_id'],
                    model=model['model'],provider=model['provider'],family=model['family'],condition=condition,
                    execution_slot='historico_a_identificar',pair_partition='C' if model['model']=='central' else 'calibration',
                    admission_status='supported_with_historical_caveat',drug_id_a=d['drug_id_a'],drug_id_b=d['drug_id_b'],
                    context_id=d['context_id'],request_id='req-'+oid,request_fingerprint='f'*64,payload_fingerprint='e'*64,
                    text_binding_sha256='d'*64,binding_verified=True,parsing_candidate=True,
                    response_execution_id_observed=None,response_execution_attribution_verified=False,
                    complete_observed_compatibility=False,submission_execution_ids=[])
                planned.append(row)
                statuses.append({**{k:row[k] for k in STATUS_FIELDS if k in row},'metadata_support':True,
                    'authorized_in_this_round':True,'parsed_this_round':True,'cache_value_available':True,
                    'parsing_status':'valid','structural_response_valid':True,'reasons':[]})
    return [pd.DataFrame(planned),pd.DataFrame(statuses),docs,features,registry]


def remove_available(args,model,condition,pair):
    planned,status,*_=args
    p=(planned.model==model)&(planned.condition==condition)&(planned.pair_id==pair)
    s=(status.model==model)&(status.condition==condition)&(status.pair_id==pair)
    planned.loc[p,'parsing_candidate']=False;planned.loc[p,'admission_status']='missing_current_cache_key'
    status.loc[s,'admission_status']='missing_current_cache_key'
    for key in ['metadata_support','authorized_in_this_round','parsed_this_round','cache_value_available','structural_response_valid']:
        status.loc[s,key]=False
    status.loc[s,'parsing_status']='not_parsed_metadata_exclusion'


def test_complete_masks_keep_d01e_unknown_despite_official_inn():
    masks=build_analytic_masks(*fixture());paired=paired_masks(masks)
    assert len(masks)==8 and masks.available.all() and masks.eligible.all()
    assert len(paired)==4 and paired.comparable_d01.all()
    assert masks.d01e_eligible.isna().all() and paired.d01e_comparable.isna().all()
    assert set(masks.morphology_stratum)=={'morphology_delta_zero','morphology_delta_nonzero'}
    assert not masks.analysis_execution_authorized.any() and not masks.model_abstention_inferred.any()


def test_asymmetric_source_does_not_trigger_central_fallback():
    args=fixture();remove_available(args,'source','inn','p4')
    masks=build_analytic_masks(*args)
    summaries,_=invariance_by_scope(masks,[dict(scope_id=m,models=[m],partition=p,execution_slot='historico_a_identificar')
        for m,p in [('central','C'),('source','calibration')]])
    assert summaries[0]['invariant'] and summaries[0]['coverage_basis']=='eligible_after_gate'
    assert not summaries[1]['invariant'] and summaries[1]['fallback_activated']
    missing=masks.loc[~masks.available].iloc[0]
    assert missing.preparation_eligibility_status=='not_processed_metadata_exclusion'
    assert not missing.model_abstention_inferred and len(masks)==8


def test_equal_counts_different_effective_sets_not_invariant():
    args=fixture();remove_available(args,'central','inn','p1');remove_available(args,'central','dcb','p2')
    summary,detail=invariance_by_scope(build_analytic_masks(*args),[
        dict(scope_id='central',models=['central'],partition='C',execution_slot='historico_a_identificar')])
    assert detail[0]['available_inn']==detail[0]['available_dcb']==1
    assert not detail[0]['same_available_sets'] and not detail[0]['same_eligible_sets']
    assert summary[0]['coverage_basis']=='global'


def test_absent_configuration_is_reported_not_declared_invariant():
    summary,detail=invariance_by_scope(build_analytic_masks(*fixture()),[
        dict(scope_id='missing',models=['absent'],partition='C',execution_slot='historico_a_identificar')])
    assert summary[0]['status']=='CONFIGURATION_ABSENT_OR_EMPTY'
    assert not summary[0]['invariant'] and not summary[0]['fallback_activated']
    assert detail[0]['planned_inn']==detail[0]['planned_dcb']==0


@pytest.mark.parametrize('table,column,value',[(1,'pair_id','wrong'),(1,'model','wrong'),(1,'condition','wrong'),
    (1,'pair_partition','wrong'),(1,'execution_slot','wrong'),(1,'provider','wrong'),(1,'planned_observation_id','wrong'),
    (0,'context_id','wrong'),(0,'drug_id_a','wrong'),(0,'request_id',''),(0,'family','wrong')])
def test_incoherent_join_rejected(table,column,value):
    args=fixture();args[table].loc[0,column]=value
    with pytest.raises(ValueError):build_analytic_masks(*args)


@pytest.mark.parametrize('table',[0,1,2,3,4])
def test_duplicate_input_or_outcome_field_rejected(table):
    args=fixture();args[table]=pd.concat([args[table],args[table].iloc[:1]],ignore_index=True)
    with pytest.raises(ValueError):build_analytic_masks(*args)
    args=fixture();args[table]['severity_label']='not_a_real_label'
    with pytest.raises(ValueError):build_analytic_masks(*args)


def test_same_count_different_eligibility_sets_and_no_fake_categories():
    masks=build_analytic_masks(*fixture())
    masks.loc[(masks.model=='central')&(masks.condition=='inn')&(masks.pair_id=='p1'),'eligible']=False
    masks.loc[(masks.model=='central')&(masks.condition=='dcb')&(masks.pair_id=='p2'),'eligible']=False
    summary,detail=invariance_by_scope(masks,[dict(scope_id='central',models=['central'],partition='C',execution_slot='historico_a_identificar')])
    assert detail[0]['same_available_sets'] and detail[0]['eligible_inn']==detail[0]['eligible_dcb']==1
    assert not detail[0]['same_eligible_sets'] and summary[0]['fallback_activated']


def test_invalid_present_response_is_available_but_not_eligible():
    args=fixture();args[1].loc[0,'parsing_status']='invalid';args[1].loc[0,'structural_response_valid']=False
    masks=build_analytic_masks(*args)
    assert masks.loc[0,'available'] and not masks.loc[0,'eligible']
    assert 'invalid_structural_response' in masks.loc[0,'gate_reasons']


def test_missing_feature_does_not_silently_filter_d01():
    args=fixture();args[3].loc[0,FEATURES[1]]=float('nan')
    masks=build_analytic_masks(*args);rows=masks[masks.pair_id=='p1']
    assert rows.eligible_d01_response.all() and not rows.eligible.any()
    assert rows.morphology_stratum.eq('morphology_delta_zero').all()
    assert 'primary_feature_missing_or_nonfinite' in rows.iloc[0].gate_reasons


def test_ambiguity_or_claimed_observed_provenance_cannot_be_recovered():
    args=fixture();args[0].at[0,'submission_execution_ids']=['a','b']
    with pytest.raises(ValueError):build_analytic_masks(*args)


def test_incoherent_paired_fingerprint_rejected():
    masks=build_analytic_masks(*fixture());masks.loc[0,'text_binding_sha256']='x'*64
    with pytest.raises(ValueError):paired_masks(masks)


@pytest.mark.parametrize('column,value',[('reference_record_present',False),('reference_quality_status','conflict'),
                                        ('identity_admitted',False)])
def test_documentary_projection_cannot_contradict_available_status(column,value):
    args=fixture();args[2].loc[0,column]=value
    with pytest.raises(ValueError):build_analytic_masks(*args)


def test_malformed_provenance_hash_rejected():
    args=fixture();args[0].loc[0,'request_fingerprint']='not a hash'
    with pytest.raises(ValueError):build_analytic_masks(*args)


def test_calibration_c_drug_overlap_rejected():
    args=fixture()
    args[0].loc[args[0].pair_id.eq('p3'),'drug_id_a']='p1a'
    args[2].loc[args[2].pair_id.eq('p3'),'drug_id_a']='p1a'
    with pytest.raises(ValueError):build_analytic_masks(*args)
    args=fixture();args[0].loc[0,'complete_observed_compatibility']=True
    with pytest.raises(ValueError):build_analytic_masks(*args)
