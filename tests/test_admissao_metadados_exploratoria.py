"""Small invented metadata fixtures; never load actual predictions or severity."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from src.dados.admissao_metadados_exploratoria import construir_matriz_admissao, resumir_configuracoes, parear_admissao


def pairs():
    return [dict(pair_id='p1', drug_id_a='a', drug_id_b='b', benchmark_candidate=True,
                 pair_partition='C', context_id='ctx')]


def models():
    return [dict(model='snapshot', provider='openai', family='family', role='fixture')]


def obs(condition='dcb', **updates):
    value=dict(observation_id='obs-'+condition, request_id='req-'+condition, pair_id='p1',
        model='snapshot', condition=condition, execution_slot='historico_a_identificar', context_id='ctx',
        request_fingerprint='f'*64, payload_fingerprint='e'*64, text_binding_sha256='b'*64,
        legacy_key=('a' if condition=='dcb' else 'b')*64, binding_verified=True,
        cache_audited=True, cache_current_present=True, classification='indeterminada',
        historical_use_allowed_with_caveat=True, submission_execution_ids=[],
        response_execution_id_observed=None, complete_configuration_observed=False)
    value.update(updates)
    return value


def execution_evidence_fixture(tmp_path):
    """A pinned invented source, verified independently of supplied assertions."""
    source = tmp_path / 'invented_execution_metadata.json'
    record = dict(execution_id='exec', legacy_key='a'*64, request_fingerprint='f'*64)
    source.write_text(json.dumps([record]), encoding='utf-8')
    pinned_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    evidence = dict(source_path=str(source), source_sha256=pinned_digest,
                    source_locator='record:0', **record)

    def verify(candidate):
        if Path(candidate['source_path']) != source or candidate['source_locator'] != 'record:0':
            return False
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != pinned_digest or candidate['source_sha256'] != pinned_digest:
            return False
        records = json.loads(raw)
        expected = {key: candidate[key] for key in record}
        return records[0] == expected and sum(item == expected for item in records) == 1

    return evidence, verify


def test_complete_grid_keeps_both_absent_and_all_snapshots():
    snapshots=models()+[dict(model='absent',provider='openai',family='family',role='fixture')]
    matrix=construir_matriz_admissao(pairs(),snapshots,[])
    assert len(matrix)==4 and not any(x['parsing_candidate'] for x in matrix)
    assert all(x['link_status']=='missing_request_metadata' for x in matrix)
    summary=resumir_configuracoes(matrix,snapshots)
    assert len(summary)==20  # five partitions, every snapshot and condition
    assert sum(x['planned_observations'] for x in summary)==4
    assert len(parear_admissao(matrix))==2


def test_unique_unknown_parameters_remain_caveated_not_compatible():
    matrix=construir_matriz_admissao(pairs(),models(),[obs(),obs('inn')])
    assert all(x['parsing_candidate'] for x in matrix)
    assert all(x['link_status']=='supported_with_historical_caveat' for x in matrix)
    assert all(not x['complete_observed_compatibility'] and not x['response_validity_checked'] for x in matrix)
    assert all(x['dataset_scope']=='real_metadata_admission' for x in matrix)
    paired=parear_admissao(matrix)[0]
    assert paired['paired_parsing_candidate'] and not paired['physical_pair_execution_observed']
    assert paired['binding_fingerprints_equal'] and paired['contexts_equal']
    assert paired['both_individual_bindings_verified']


def test_multiple_submissions_not_replicates_or_arbitrary_selection():
    records=[obs(submission_execution_ids=['batch:second','batch:first']),obs('inn')]
    matrix=construir_matriz_admissao(pairs(),models(),records)
    row=next(x for x in matrix if x['condition']=='dcb')
    assert row['link_status']=='ambiguous_submission_attribution' and not row['parsing_candidate']
    assert row['new_independent_replications']==0 and row['response_execution_id_observed'] is None
    assert row['submission_execution_ids']==['batch:first','batch:second']
    assert not parear_admissao(matrix)[0]['paired_parsing_candidate']


@pytest.mark.parametrize('changes,status',[
    ({'cache_current_present':False},'missing_current_cache_key'),
    ({'cache_audited':False},'cache_access_unverified'),
    ({'classification':'incompativel'},'material_metadata_conflict'),
    ({'binding_verified':False},'unverified_text_context_binding'),
    ({'historical_use_allowed_with_caveat':False},'historical_use_not_supported'),
])
def test_reasons_remain_separate(changes,status):
    result=construir_matriz_admissao(pairs(),models(),[obs(**changes)])[0]
    assert result['link_status']==status and not result['parsing_candidate']


@pytest.mark.parametrize('partition,candidate,reason',[
    ('cross_partition',True,'outside_analysis_cross_partition'),
    ('documentary_excluded',False,'outside_documentary_population'),
])
def test_scope_exclusions_do_not_disappear(partition,candidate,reason):
    membership=pairs();membership[0].update(pair_partition=partition,benchmark_candidate=candidate)
    matrix=construir_matriz_admissao(membership,models(),[obs(),obs('inn')])
    assert len(matrix)==2 and all(x['admission_status']==reason for x in matrix)
    assert all(x['link_status']=='supported_with_historical_caveat' for x in matrix)
    assert not any(x['parsing_candidate'] for x in matrix)


def test_duplicate_requests_and_shared_key_are_not_silently_collapsed():
    matrix=construir_matriz_admissao(pairs(),models(),[obs(),obs(observation_id='other',request_id='other')])
    assert matrix[0]['link_status']=='ambiguous_request_metadata'
    matrix=construir_matriz_admissao(pairs(),models(),[obs(),obs('inn',legacy_key='a'*64)])
    assert all(x['link_status']=='shared_cache_key_requires_link_rule' for x in matrix)


def test_observed_compatible_requires_complete_evidence_and_execution(tmp_path):
    matrix=construir_matriz_admissao(pairs(),models(),[obs(classification='compativel')])
    assert not matrix[0]['complete_observed_compatibility']
    evidence,verify=execution_evidence_fixture(tmp_path)
    matrix=construir_matriz_admissao(pairs(),models(),[obs(classification='compativel',
        complete_configuration_observed=True,response_execution_id_observed='exec',
        submission_execution_ids=['exec','other'],response_execution_evidence=evidence)],
        response_execution_verifier=verify)
    assert matrix[0]['complete_observed_compatibility']
    assert matrix[0]['response_execution_attribution_verified']
    assert matrix[0]['response_execution_id_observed']=='exec'


@pytest.mark.parametrize('changes',[{'severity':'invented'},{'response':'invented'},{'cache_current_present':'False'}])
def test_response_columns_and_nonboolean_flags_rejected(changes):
    with pytest.raises(ValueError):construir_matriz_admissao(pairs(),models(),[obs(**changes)])


def test_orphans_context_mismatch_and_duplicate_population_rejected():
    for records in [[obs(pair_id='unknown')],[obs(model='unknown')],[obs(context_id='wrong')]]:
        with pytest.raises(ValueError):construir_matriz_admissao(pairs(),models(),records)
    with pytest.raises(ValueError):construir_matriz_admissao(pairs()*2,models(),[])


def test_no_mutation_and_permutation_invariance():
    records=[obs('inn'),obs()]; original=copy.deepcopy(records)
    one=construir_matriz_admissao(pairs(),models(),records)
    two=construir_matriz_admissao(pairs(),models(),list(reversed(records)))
    assert one==two and records==original


def test_equal_binding_digest_does_not_verify_an_unverified_arm():
    matrix=construir_matriz_admissao(pairs(),models(),[obs(binding_verified=False),obs('inn')])
    paired=parear_admissao(matrix)[0]
    assert paired['binding_fingerprints_equal'] and paired['contexts_equal']
    assert not paired['both_individual_bindings_verified']
    assert not paired['paired_binding_verified']
    assert not paired['paired_parsing_candidate']


def test_execution_string_alone_cannot_prove_observed_compatibility():
    matrix=construir_matriz_admissao(pairs(),models(),[obs(classification='compativel',
        complete_configuration_observed=True,response_execution_id_observed='exec',
        submission_execution_ids=['exec'])])
    assert not matrix[0]['complete_observed_compatibility']
    assert matrix[0]['response_execution_id_claimed']=='exec'
    assert matrix[0]['response_execution_id_observed'] is None


@pytest.mark.parametrize('flag',[False,None,'True',1,'missing'])
def test_pairing_requires_two_strict_verified_flags(flag):
    matrix=construir_matriz_admissao(pairs(),models(),[obs(),obs('inn')])
    if flag=='missing':
        matrix[1].pop('binding_verified')
    else:
        matrix[1]['binding_verified']=flag
    assert not parear_admissao(matrix)[0]['paired_binding_verified']


def test_traceable_dictionary_without_source_verifier_stays_unverified(tmp_path):
    evidence,_=execution_evidence_fixture(tmp_path)
    matrix=construir_matriz_admissao(pairs(),models(),[obs(classification='compativel',
        complete_configuration_observed=True,response_execution_id_observed='exec',
        submission_execution_ids=['exec','other'],response_execution_evidence=evidence)])
    row=matrix[0]
    assert not row['complete_observed_compatibility']
    assert row['link_status']=='ambiguous_submission_attribution'
    assert row['response_execution_attribution_reason']=='execution_attribution_source_not_verified'


@pytest.mark.parametrize('change',[
    {'submission_execution_ids':['other']},
    {'submission_execution_ids':[]},
    {'evidence_field':'execution_id'},
    {'evidence_field':'legacy_key'},
    {'evidence_field':'request_fingerprint'},
])
def test_incoherent_attribution_never_reaches_the_source_verifier(tmp_path,change):
    evidence,_=execution_evidence_fixture(tmp_path)
    updates=dict(response_execution_id_observed='exec',submission_execution_ids=['exec'],
                 response_execution_evidence=evidence)
    if 'evidence_field' in change:
        evidence[change['evidence_field']]='different'
    else:
        updates.update(change)
    calls=[]
    def should_not_verify(candidate):
        calls.append(candidate)
        return True
    matrix=construir_matriz_admissao(pairs(),models(),[obs(**updates)],
                                   response_execution_verifier=should_not_verify)
    assert not calls and not matrix[0]['response_execution_attribution_verified']
    assert matrix[0]['response_execution_id_observed'] is None


@pytest.mark.parametrize('field,value',[
    ('source_sha256','0'*64),('source_locator','record:9'),('source_path','unfixed_source.json')])
def test_source_pin_and_locator_must_really_be_checked(tmp_path,field,value):
    evidence,verify=execution_evidence_fixture(tmp_path)
    evidence[field]=value
    matrix=construir_matriz_admissao(pairs(),models(),[obs(response_execution_id_observed='exec',
        submission_execution_ids=['exec','other'],response_execution_evidence=evidence)],
        response_execution_verifier=verify)
    assert matrix[0]['link_status']=='ambiguous_submission_attribution'
    assert matrix[0]['response_execution_attribution_reason']=='execution_attribution_source_verification_failed'


def test_source_changed_after_pinning_is_not_attribution_evidence(tmp_path):
    evidence,verify=execution_evidence_fixture(tmp_path)
    Path(evidence['source_path']).write_text('[]',encoding='utf-8')
    matrix=construir_matriz_admissao(pairs(),models(),[obs(response_execution_id_observed='exec',
        submission_execution_ids=['exec'],response_execution_evidence=evidence)],
        response_execution_verifier=verify)
    assert not matrix[0]['response_execution_attribution_verified']


@pytest.mark.parametrize('result',[False,None,1,'True'])
def test_source_verifier_requires_a_strict_true_result(tmp_path,result):
    evidence,_=execution_evidence_fixture(tmp_path)
    matrix=construir_matriz_admissao(pairs(),models(),[obs(response_execution_id_observed='exec',
        submission_execution_ids=['exec'],response_execution_evidence=evidence)],
        response_execution_verifier=lambda candidate:result)
    assert not matrix[0]['response_execution_attribution_verified']


@pytest.mark.parametrize('damage',['extra','missing','invalid_digest','no_claim'])
def test_malformed_execution_evidence_is_rejected(tmp_path,damage):
    evidence,verify=execution_evidence_fixture(tmp_path)
    claim='exec'
    if damage=='extra': evidence['response']='forbidden fixture field'
    if damage=='missing': evidence.pop('source_locator')
    if damage=='invalid_digest': evidence['source_sha256']='not-a-hash'
    if damage=='no_claim': claim=None
    with pytest.raises(ValueError):
        construir_matriz_admissao(pairs(),models(),[obs(response_execution_id_observed=claim,
            submission_execution_ids=['exec'],response_execution_evidence=evidence)],
            response_execution_verifier=verify)
