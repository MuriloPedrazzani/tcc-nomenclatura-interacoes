"""Synthetic checks for the explicitly authorized benchmark regime."""
import copy
import importlib
import json

import pandas as pd
import pytest


def j(values):
    return json.dumps(values)


@pytest.fixture
def example():
    module = importlib.import_module('src.dados.admissibilidade_benchmark')
    entities = pd.DataFrame([dict(drug_id=d, dcb='DCB '+d, experimental_name='Name '+d,
        ref_id='REF'+d, snapshot_name='Name '+d, active_reference_ids_json=j(['REF'+d])) for d in ('A','B')])
    assessments = pd.DataFrame([dict(drug_id=d, status='supported', material_conflict=False,
        required_qualifiers_json=j(['identity']), demonstrated_qualifiers_json=j(['identity']),
        evidence_ids_json=j(['E'+d]), reason='Documented correspondence') for d in ('A','B')])
    evidence = pd.DataFrame([dict(evidence_id='E'+d, drug_id=d, dcb='DCB '+d, experimental_name='Name '+d,
        ref_id='REF'+d, snapshot_name='Name '+d, source_id='S', source_version='V',
        kind='documentary_identity_correspondence', locator='source/document#'+d,
        claim='Both names refer to this qualified reference entity') for d in ('A','B')])
    pairs = pd.DataFrame([dict(pair_id='P', drug_id_a='A',drug_id_b='B',ref_id_a='REFA',ref_id_b='REFB',context_id='CTX')])
    mapping = pd.DataFrame([dict(pair_id='P',mapping_accepted=True)])
    records = pd.DataFrame([dict(pair_id='P',ref_id_a='REFA',ref_id_b='REFB',context_id='CTX',source_id='S',source_version='V',
        record_present=True,quality_status='usable',record_evidence_refs_json=j(['source/snapshot.csv#record=1']))])
    previous = pd.DataFrame([dict(pair_id='P',source_id='S',source_version='V',applicability_status='uncertain',
        evidence_refs_json='[]',reasons='Universal clinical scope not demonstrated')])
    return module,dict(entities=entities,assessments=assessments,evidence=evidence,pairs=pairs,
                      mapping=mapping,records=records,previous=previous)


def evaluate(example):
    module, data = example
    entities = module.avaliar_entidades_benchmark(data['entities'],data['assessments'],data['evidence'],
        source_id='S',source_version='V',contract_version=module.CONTRACT_VERSION)
    pairs = module.admitir_pares_benchmark(data['pairs'],data['mapping'],entities,data['records'],data['previous'],
        source_id='S',source_version='V',contract_version=module.CONTRACT_VERSION)
    return entities,pairs


def test_valid_documentary_correspondence_does_not_require_universal_clinical_scope(example):
    before=copy.deepcopy(example[1])
    entities,pairs=evaluate(example)
    assert entities.entity_admissible.all()
    assert pairs.benchmark_candidate.tolist()==[True]
    assert pairs.reference_usable_for_outcome.tolist()==[True]
    assert pairs.prior_clinical_applicability.tolist()==['uncertain']
    assert pairs.clinical_evaluation_authorized.tolist()==[False]
    for key in before: pd.testing.assert_frame_equal(before[key],example[1][key])


@pytest.mark.parametrize('kind',['label_presence','name_presence','ai_confidence','text_similarity','legacy_acceptance'])
def test_presence_similarity_or_legacy_acceptance_are_not_identity_proof(example,kind):
    example[1]['evidence'].loc[0,'kind']=kind
    assert not evaluate(example)[1].benchmark_candidate.any()


def test_missing_proof_blocks_even_with_record_and_known_label(example):
    example[1]['assessments'].loc[0,'evidence_ids_json']='[]'
    assert not evaluate(example)[1].benchmark_candidate.any()


@pytest.mark.parametrize('field',['drug_id','dcb','experimental_name','ref_id','snapshot_name','source_id','source_version'])
def test_wrong_evidence_binding_is_rejected(example,field):
    example[1]['evidence'].loc[0,field]='wrong'
    with pytest.raises(ValueError): evaluate(example)


@pytest.mark.parametrize('field',['locator','claim'])
def test_empty_substantive_proof_is_rejected(example,field):
    example[1]['evidence'].loc[0,field]=''
    with pytest.raises(ValueError): evaluate(example)


@pytest.mark.parametrize('condition',['salt','route','formulation','association'])
def test_unresolved_material_qualifiers_block(example,condition):
    example[1]['assessments'].loc[0,'required_qualifiers_json']=j(['identity',condition])
    assert not evaluate(example)[1].benchmark_candidate.any()


def test_material_conflict_blocks_despite_positive_nominal_mapping(example):
    example[1]['assessments'].loc[0,'material_conflict']=True
    assert not evaluate(example)[1].benchmark_candidate.any()


def test_unresolved_status_blocks_even_with_a_locator(example):
    example[1]['assessments'].loc[0,'status']='unresolved'
    assert not evaluate(example)[1].benchmark_candidate.any()


def test_inactive_reference_cannot_be_promoted(example):
    example[1]['entities'].loc[0,'active_reference_ids_json']=j(['CANDIDATE_OTHER'])
    assert not evaluate(example)[1].benchmark_candidate.any()


def test_missing_entity_assessment_remains_pending(example):
    example[1]['assessments']=example[1]['assessments'].iloc[1:].copy()
    assert not evaluate(example)[1].benchmark_candidate.any()


@pytest.mark.parametrize('table',['entities','assessments','evidence','pairs','mapping','records','previous'])
def test_duplicate_keys_rejected(example,table):
    frame=example[1][table]
    example[1][table]=pd.concat([frame,frame.iloc[:1]],ignore_index=True)
    with pytest.raises(ValueError): evaluate(example)


@pytest.mark.parametrize('table,field',[('assessments','drug_id'),('mapping','pair_id'),('records','pair_id'),('previous','pair_id')])
def test_orphan_keys_rejected(example,table,field):
    example[1][table].loc[0,field]='ORPHAN'
    with pytest.raises(ValueError): evaluate(example)


@pytest.mark.parametrize('field',['ref_id_a','ref_id_b','context_id','source_id','source_version'])
def test_wrong_interaction_binding_rejected(example,field):
    example[1]['records'].loc[0,field]='WRONG'
    with pytest.raises(ValueError): evaluate(example)


@pytest.mark.parametrize('status',['unknown','missing','invalid','conflict'])
def test_unusable_reference_keeps_candidate_but_not_outcome_denominator(example,status):
    example[1]['records'].loc[0,'quality_status']=status
    result=evaluate(example)[1]
    assert result.benchmark_candidate.tolist()==[True]
    assert result.reference_usable_for_outcome.tolist()==[False]


def test_absent_record_keeps_identity_in_operational_flow(example):
    example[1]['records'].loc[0,['record_present','quality_status','record_evidence_refs_json']]=[False,'not_located','[]']
    result=evaluate(example)[1]
    assert result.operational_candidate.tolist()==[True]
    assert result.benchmark_candidate.tolist()==[False]
    assert result.reference_usable_for_outcome.tolist()==[False]


def test_record_link_requires_provenance(example):
    example[1]['records'].loc[0,'record_evidence_refs_json']='[]'
    with pytest.raises(ValueError): evaluate(example)


def test_mapping_failure_not_overridden_by_new_contract(example):
    example[1]['mapping'].loc[0,'mapping_accepted']=False
    assert not evaluate(example)[1].benchmark_candidate.any()


@pytest.mark.parametrize('table,column',[('mapping','mapping_accepted'),('records','record_present'),('assessments','material_conflict')])
def test_string_boolean_not_accepted(example,table,column):
    example[1][table][column]=example[1][table][column].astype(object)
    example[1][table].loc[0,column]='True'
    with pytest.raises(ValueError): evaluate(example)


def test_prior_audit_not_relabelled_or_used_as_administrative_release(example):
    example[1]['previous'].loc[0,'applicability_status']='demonstrated'
    example[1]['assessments'].loc[0,'evidence_ids_json']='[]'
    result=evaluate(example)[1]
    assert result.prior_clinical_applicability.tolist()==['demonstrated']
    assert not result.benchmark_candidate.any()


def test_explicit_version_required_and_wrong_regime_rejected(example):
    module,data=example
    with pytest.raises(TypeError):
        module.avaliar_entidades_benchmark(data['entities'],data['assessments'],data['evidence'],source_id='S',source_version='V')
    with pytest.raises(ValueError):
        module.avaliar_entidades_benchmark(data['entities'],data['assessments'],data['evidence'],source_id='S',source_version='V',contract_version='hard-gate-documentary-v2')


@pytest.mark.parametrize('column',['severity','prediction','confidence'])
def test_identity_selection_rejects_outcomes(example,column):
    example[1]['entities'][column]='forbidden'
    with pytest.raises(ValueError): evaluate(example)


def test_pair_cannot_use_other_reference_id_than_entity_proof(example):
    example[1]['pairs'].loc[0,'ref_id_a']='OTHER'
    example[1]['records'].loc[0,'ref_id_a']='OTHER'
    assert not evaluate(example)[1].benchmark_candidate.any()


def test_documented_non_inn_name_does_not_exclude_identity(example):
    example[1]['entities']['name_nature']='outra_denominacao_documentada'
    assert evaluate(example)[1].benchmark_candidate.all()


@pytest.mark.parametrize('field',['dcb','experimental_name','snapshot_name','ref_id'])
def test_blank_nominal_fields_cannot_form_positive_proof(example,field):
    example[1]['entities'].loc[0,field]=' '
    example[1]['evidence'].loc[0,field]=' '
    if field=='ref_id': example[1]['entities'].loc[0,'active_reference_ids_json']=j([' '])
    with pytest.raises(ValueError): evaluate(example)


def test_explicit_context_mismatch_cannot_use_same_identity_proof(example):
    example[1]['pairs'].loc[0,'context_id']='ANOTHER_CONTEXT_WITH_OTHER_FORMULATION'
    with pytest.raises(ValueError): evaluate(example)
