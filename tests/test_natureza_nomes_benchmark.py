"""Synthetic tests: naming evidence does not grant benchmark admission."""
from copy import deepcopy
import importlib

import pytest


def evaluate(entities, evidence, sources):
    return importlib.import_module('src.dados.natureza_nomes_benchmark').avaliar_nomes(
        entities, evidence, source_registry=sources)


@pytest.fixture
def sample():
    entities = [{'drug_id': 'DRUG_A', 'experimental_name': 'Palbociclib'}]
    evidence = [{'evidence_id': 'EV_A', 'drug_id': 'DRUG_A',
                 'experimental_name': 'Palbociclib', 'documented_name': 'palbociclib',
                 'source_id': 'WHO_RL71', 'source_version': '28(1),2014;RL71',
                 'locator': 'p.99; English entry palbociclib',
                 'proof_kind': 'who_recommended_inn', 'relation': 'case_whitespace_only'}]
    sources = [{'source_id': 'WHO_RL71', 'source_version': '28(1),2014;RL71',
                'publisher': 'WHO', 'source_kind': 'who_recommended_inn',
                'url': 'https://www.who.int/example/rl71.pdf',
                'locator': 'Recommended INN List 71, notice p.71'}]
    return entities, evidence, sources


def test_official_inn_requires_bound_who_recommended_evidence(sample):
    out = evaluate(*sample)[0]
    assert out['naming_status'] == 'inn_oficial_documentada'
    assert out['literal_inn_documented'] is True
    assert out['experimental_name'] == 'Palbociclib'
    assert 'benchmark_admissible' not in out


def test_absent_evidence_remains_unresolved(sample):
    e, _, s = sample
    out = evaluate(e, [], s)[0]
    assert out['naming_status'] == 'natureza_nao_resolvida'
    assert out['literal_inn_documented'] is False


def test_ddinter_name_is_documented_not_automatically_inn(sample):
    e, v, s = sample
    v[0].update(proof_kind='reference_node_title', source_id='DDI', source_version='sha256:aaa')
    s = [{'source_id': 'DDI', 'source_version': 'sha256:aaa', 'publisher': 'DDInter',
          'source_kind': 'reference_node_title', 'url': 'local:identities.csv',
          'locator': 'DDInter1379; drug_name'}]
    out = evaluate(e, v, s)[0]
    assert out['naming_status'] == 'outra_denominacao_documentada'
    assert not out['literal_inn_documented']
    assert 'not_a_negative_inn_claim' in out['reasons']


def test_proposed_inn_does_not_prove_recommended_inn(sample):
    e, v, s = sample
    v[0]['proof_kind'] = s[0]['source_kind'] = 'who_proposed_inn'
    out = evaluate(e, v, s)[0]
    assert out['naming_status'] == 'outra_denominacao_documentada'


@pytest.mark.parametrize('new_name', ['Palbociclib hydrochloride', 'Palbociclib (oral)',
                                      'Palbociclib + drug B', 'Palbociclib liposomal',
                                      'Palbociclib hydrate', 'Palbociclib!'])
def test_no_qualifier_or_punctuation_removed_to_create_literal_inn(sample, new_name):
    e, v, s = sample
    e[0]['experimental_name'] = v[0]['experimental_name'] = new_name
    v[0]['relation'] = 'different_literal'
    out = evaluate(e, v, s)[0]
    assert out['naming_status'] == 'natureza_nao_resolvida'
    assert not out['literal_inn_documented']


def test_whitespace_case_equivalence_does_not_modify_input(sample):
    e, v, s = sample
    e[0]['experimental_name'] = v[0]['experimental_name'] = '  PALBOCICLIB  '
    before = deepcopy(sample)
    assert evaluate(e, v, s)[0]['literal_inn_documented']
    assert sample == before


@pytest.mark.parametrize('field,value', [('drug_id', 'DRUG_OTHER'),
    ('experimental_name', 'Other'), ('source_id', 'OTHER'),
    ('source_version', 'other-version'), ('locator', ''), ('documented_name', ''),
    ('proof_kind', 'ai_confidence'), ('relation', 'similarity')])
def test_wrong_or_missing_binding_is_rejected(sample, field, value):
    e, v, s = sample
    v[0][field] = value
    with pytest.raises(ValueError):
        evaluate(e, v, s)


def test_false_claim_of_literal_match_is_rejected(sample):
    e, v, s = sample
    v[0]['documented_name'] = 'other name'
    with pytest.raises(ValueError):
        evaluate(e, v, s)


@pytest.mark.parametrize('location', ['entities', 'evidence', 'sources'])
def test_duplicate_ids_are_rejected(sample, location):
    e, v, s = sample
    destination = {'entities': e, 'evidence': v, 'sources': s}[location]
    destination.append(deepcopy(destination[0]))
    with pytest.raises(ValueError):
        evaluate(e, v, s)


def test_duplicate_proof_with_new_id_is_rejected(sample):
    e, v, s = sample
    duplicate = deepcopy(v[0]); duplicate['evidence_id'] = 'EV_NEW'
    v.append(duplicate)
    with pytest.raises(ValueError):
        evaluate(e, v, s)


def test_non_who_registry_cannot_certify_who_recommended_inn(sample):
    e, v, s = sample
    s[0]['publisher'] = 'DDInter'
    with pytest.raises(ValueError):
        evaluate(e, v, s)


def test_atc_or_presence_cannot_be_used_as_inn_evidence(sample):
    e, v, s = sample
    s[0]['source_kind'] = 'atc_match'
    with pytest.raises(ValueError):
        evaluate(e, v, s)


def test_blank_experimental_name_never_becomes_documented(sample):
    e, _, s = sample
    e[0]['experimental_name'] = ''
    out = evaluate(e, [], s)[0]
    assert out['naming_status'] == 'natureza_nao_resolvida'


def test_all_entities_are_preserved_in_input_order(sample):
    e, v, s = sample
    e.append({'drug_id': 'DRUG_B', 'experimental_name': 'Unknown'})
    assert [x['drug_id'] for x in evaluate(e, v, s)] == ['DRUG_A', 'DRUG_B']


def test_dataframe_inputs_supported_without_mutation(sample):
    import pandas as pd
    e, v, s = sample
    out = evaluate(pd.DataFrame(e), pd.DataFrame(v), pd.DataFrame(s))
    assert out[0]['literal_inn_documented']


def test_disallows_non_who_url_for_recommended_inn(sample):
    e, v, s = sample
    s[0]['url'] = 'https://who.int.example.com/list.pdf'
    with pytest.raises(ValueError):
        evaluate(e, v, s)
