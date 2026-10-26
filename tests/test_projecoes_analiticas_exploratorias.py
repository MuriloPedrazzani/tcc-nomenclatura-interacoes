"""Invented documentary/status fixtures; no clinical labels or model responses."""
from copy import deepcopy
import json

import pandas as pd
import pytest

from src.dados.projecoes_analiticas_exploratorias import (
    CONTEXT_COLUMNS, REFERENCE_COLUMNS, UNIVERSE_COLUMNS,
    documentaire_projections,
)


def fixture():
    universe = dict(pair_id='p1', contract_version='benchmark-ddinter-entity-v1',
        drug_id_a='a', drug_id_b='b', context_id='ctx1', source_id='reference',
        source_version='fixed-v1', benchmark_identity_admissible=True,
        benchmark_candidate=True, record_present=True, quality_status='usable',
        reference_usable_for_outcome=True, literal_inn_to_dcb_supported=True,
        both_international_names_documented_inn=True)
    reference = dict(pair_id='p1', ref_id_a='r1', ref_id_b='r2', context_id='ctx1',
        source_id='reference', source_version='fixed-v1', record_present=True,
        quality_status='usable', record_evidence_refs_json=json.dumps(['fixture#record=1']))
    context = dict(pair_id='p1', drug_id_a='a', drug_id_b='b', ref_id_a='r1',
        ref_id_b='r2', context_id='ctx1')
    return [pd.DataFrame([row]) for row in (universe, reference, context)]


def test_existing_status_projection_is_sufficient_and_does_not_certify_d01e():
    tables = fixture()
    before = [x.copy(deep=True) for x in tables]
    out = documentaire_projections(*tables)
    assert len(out) == 1
    row = out.iloc[0]
    assert row.reference_available and row.identity_admitted
    assert row.literal_inn_to_dcb_supported
    assert row.d01e_documentary_state == 'not_assessed'
    assert 'direct_complete' in row.d01e_reason
    assert not {'severity_label', 'prediction', 'canonical_label'} & set(out.columns)
    for actual, original in zip(tables, before):
        pd.testing.assert_frame_equal(actual, original)


def test_availability_is_separate_from_identity_and_preserves_excluded_pairs():
    tables = fixture()
    tables[0].loc[0, ['benchmark_identity_admissible', 'benchmark_candidate',
                     'reference_usable_for_outcome', 'literal_inn_to_dcb_supported']] = False
    row = documentaire_projections(*tables).iloc[0]
    assert row.reference_available and not row.identity_admitted
    assert row.d01e_documentary_state == 'outside_population'


@pytest.mark.parametrize('status', ['unknown', 'missing', 'invalid', 'conflict'])
def test_unusable_status_keeps_candidate_and_preserves_reason(status):
    tables = fixture()
    tables[0].loc[0, 'quality_status'] = status
    tables[1].loc[0, 'quality_status'] = status
    tables[0].loc[0, 'reference_usable_for_outcome'] = False
    row = documentaire_projections(*tables).iloc[0]
    assert row.benchmark_candidate and not row.reference_available
    assert row.reference_quality_status == status
    assert row.d01e_documentary_state == 'not_assessed'


@pytest.mark.parametrize('status,ref_a', [('not_located', 'r1'), ('not_queryable', '')])
def test_absent_reference_remains_explicit_without_imputation(status, ref_a):
    tables = fixture()
    tables[0].loc[0, ['record_present', 'benchmark_candidate', 'reference_usable_for_outcome']] = False
    tables[0].loc[0, 'quality_status'] = status
    tables[1].loc[0, ['record_present']] = False
    tables[1].loc[0, 'quality_status'] = status
    tables[1].loc[0, 'record_evidence_refs_json'] = '[]'
    tables[1].loc[0, 'ref_id_a'] = ref_a
    tables[2].loc[0, 'ref_id_a'] = ref_a
    row = documentaire_projections(*tables).iloc[0]
    assert not row.reference_available and not row.reference_record_present
    assert row.identity_admitted


@pytest.mark.parametrize('which', [0, 1, 2])
def test_duplicate_identifier_rejected(which):
    tables = fixture()
    tables[which] = pd.concat([tables[which], tables[which]], ignore_index=True)
    with pytest.raises(ValueError, match='duplicate'):
        documentaire_projections(*tables)


@pytest.mark.parametrize('which', [1, 2])
def test_orphan_or_missing_identifier_rejected_even_at_same_count(which):
    tables = fixture()
    tables[which].loc[0, 'pair_id'] = 'other'
    with pytest.raises(ValueError, match='population'):
        documentaire_projections(*tables)


@pytest.mark.parametrize('which,field,value', [
    (1, 'context_id', 'other-context'), (2, 'context_id', 'other-context'),
    (1, 'source_version', 'other-v1'), (1, 'source_id', 'other-source'),
    (1, 'quality_status', 'unknown'), (1, 'ref_id_a', 'wrong-ref'),
    (2, 'drug_id_a', 'wrong-drug'), (0, 'contract_version', 'old-contract'),
])
def test_incoherent_binding_rejected(which, field, value):
    tables = fixture()
    tables[which].loc[0, field] = value
    with pytest.raises(ValueError):
        documentaire_projections(*tables)


@pytest.mark.parametrize('which,field', [
    (0, 'benchmark_candidate'), (0, 'record_present'),
    (0, 'reference_usable_for_outcome'), (0, 'literal_inn_to_dcb_supported'),
    (1, 'record_present'),
])
def test_string_boolean_is_never_accepted(which, field):
    tables = fixture()
    tables[which][field] = 'True'
    with pytest.raises(ValueError, match='boolean'):
        documentaire_projections(*tables)


@pytest.mark.parametrize('which,field', [
    (0, 'prediction'), (1, 'severity_label'), (2, 'canonical_label'),
    (0, 'raw_response'),
])
def test_extra_columns_including_outcomes_are_rejected(which, field):
    tables = fixture()
    tables[which][field] = 'invented_prohibited_value'
    with pytest.raises(ValueError, match='columns'):
        documentaire_projections(*tables)


@pytest.mark.parametrize('which', [0, 1, 2])
def test_missing_or_duplicate_column_rejected(which):
    tables = fixture()
    tables[which] = tables[which].drop(columns=['context_id'])
    with pytest.raises(ValueError, match='columns'):
        documentaire_projections(*tables)
    tables = fixture()
    tables[which] = pd.concat([tables[which], tables[which][['context_id']]], axis=1)
    with pytest.raises(ValueError, match='columns'):
        documentaire_projections(*tables)


@pytest.mark.parametrize('column', [
    'benchmark_candidate', 'reference_usable_for_outcome',
    'both_international_names_documented_inn',
])
def test_derived_documentary_flags_cannot_contradict_projection(column):
    tables = fixture()
    tables[0].loc[0, column] = False
    with pytest.raises(ValueError):
        documentaire_projections(*tables)


@pytest.mark.parametrize('proof', ['[]', '{}', '[1]', 'not-json', '[" "]'])
def test_present_reference_requires_documented_string_locators(proof):
    tables = fixture()
    tables[1].loc[0, 'record_evidence_refs_json'] = proof
    with pytest.raises(ValueError, match='evidence'):
        documentaire_projections(*tables)


def test_input_order_independent_joins_preserve_universe_order():
    tables = fixture()
    second = fixture()
    for frame in second:
        frame.loc[0, 'pair_id'] = 'p2'
        frame.loc[0, 'context_id'] = 'ctx2'
    joined = [pd.concat([a, b], ignore_index=True) for a, b in zip(tables, second)]
    joined[1] = joined[1].iloc[::-1].reset_index(drop=True)
    out = documentaire_projections(*joined)
    assert out.pair_id.tolist() == ['p1', 'p2']
    assert out.context_id.tolist() == ['ctx1', 'ctx2']


def test_allowlists_match_fixture_and_empty_population_rejected():
    tables = fixture()
    assert [set(x.columns) for x in tables] == [set(UNIVERSE_COLUMNS),
        set(REFERENCE_COLUMNS), set(CONTEXT_COLUMNS)]
    with pytest.raises(ValueError, match='empty'):
        documentaire_projections(*(x.iloc[0:0] for x in tables))
