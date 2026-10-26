from __future__ import annotations

import hashlib
import itertools
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'dados'))
from split_admissao_exploratoria import materializar_split_admissao


def fixture_population(n=100, isolated=0):
    ids = [f'D{i:03}' for i in range(n + isolated)]
    drugs = pd.DataFrame({'drug_id': ids})
    pairs = pd.DataFrame([
        {'pair_id': f'P{i:05}', 'drug_id_a': a, 'drug_id_b': b}
        for i, (a, b) in enumerate(itertools.combinations(ids[:n], 2))
    ])
    return drugs, pairs


def oracle_c_ids(drugs, seed):
    ids = sorted(drugs.drug_id)
    ordered = sorted(ids, key=lambda d: hashlib.sha256(
        f'{seed}\x1fdrug-side\x1f{d}'.encode('utf-8')
    ).digest())
    return set(ordered[:math.floor(len(ids) / 4 + .5)])


def run(drugs, pairs):
    return materializar_split_admissao(
        drugs, pairs, population_id='invented_fixture_for_metadata_admission',
    )


def test_selection_is_real_metadata_operation_without_freeze_or_synthetic_relabelling():
    drugs, pairs = fixture_population()
    original_drugs, original_pairs = drugs.copy(), pairs.copy()
    tables, report = run(drugs, pairs)
    assert report['selected_seed'] == 1301
    assert report['selection_status'] == 'SELECTED_FIRST_VIABLE_FOR_METADATA_ADMISSION'
    assert report['dataset_scope'] == 'real_metadata_admission'
    assert report['selection_materialized']
    assert report['definitive_seed'] is None and report['definitive_split'] is None
    assert not report['FREEZE2_STARTED'] and not report['FREEZE2_COMPLETE']
    assert not report['DEFINITION_OF_DONE']
    assert not report['outcomes_read'] and not report['features_used_for_selection']
    assert tables['candidate_summary'].seed.tolist() == [1301, 1302, 1303]
    assert tables['candidate_summary'].selected_for_admission.tolist() == [True, False, False]
    assert tables['candidate_summary'].auto_selected.tolist() == [True, False, False]
    for table in tables.values():
        assert table.dataset_scope.eq('real_metadata_admission').all()
    for name in ('selected_drug_assignments', 'selected_pair_assignments'):
        assert tables[name].seed.eq(1301).all()
        assert tables[name].selected_for_admission.all()
    pd.testing.assert_frame_equal(drugs, original_drugs)
    pd.testing.assert_frame_equal(pairs, original_pairs)


def test_first_failing_seed_is_skipped_without_changing_order_or_maximizing_retention():
    drugs, pairs = fixture_population()
    first_c = oracle_c_ids(drugs, 1301)
    pairs = pairs[~(pairs.drug_id_a.isin(first_c) & pairs.drug_id_b.isin(first_c))]
    tables, report = run(drugs, pairs)
    assert report['selected_seed'] == 1302
    summary = tables['candidate_summary'].set_index('seed')
    assert not summary.at[1301, 'operationally_usable']
    assert summary.at[1302, 'operationally_usable']
    assert summary.at[1303, 'operationally_usable']
    assert summary.selected_for_admission.to_dict() == {1301: False, 1302: True, 1303: False}
    assert report['evaluated_seed_sequence'] == [1301, 1302, 1303]
    assert 'C_pairs_below_minimum' in summary.at[1301, 'nonviability_reasons']


def test_materialization_preserves_isolated_entities_and_cross_pairs_and_separation():
    drugs, pairs = fixture_population(isolated=5)
    tables, report = run(drugs, pairs)
    da, pa = tables['selected_drug_assignments'], tables['selected_pair_assignments']
    assert set(da.drug_id) == set(drugs.drug_id)
    assert set(pa.pair_id) == set(pairs.pair_id)
    c = oracle_c_ids(drugs, report['selected_seed'])
    assert set(da.loc[da.entity_side.eq('C'), 'drug_id']) == c
    cross = pairs.drug_id_a.isin(c) ^ pairs.drug_id_b.isin(c)
    assert set(pa.loc[pa.pair_partition.eq('cross_partition'), 'pair_id']) == set(pairs.loc[cross, 'pair_id'])
    audit = report['audits'][str(report['selected_seed'])]
    assert audit['violation_count'] == 0
    assert audit['isolated_entities_preserved'] == [f'D{i:03}' for i in range(100, 105)]
    assert not audit['training_C_shared_entity_ids']
    assert not audit['development_calibration_shared_pair_ids']
    assert audit['development_calibration_shared_entity_ids']


def test_no_viable_candidate_returns_empty_selection_and_preserves_diagnostics():
    tables, report = run(*fixture_population(n=20))
    assert report['selection_status'] == 'NO_VIABLE_CANDIDATE_FAIL_CLOSED'
    assert report['selected_seed'] is None and report['selected_candidate_id'] is None
    assert not report['selection_materialized']
    assert tables['selected_drug_assignments'].empty
    assert tables['selected_pair_assignments'].empty
    assert tables['candidate_summary'].seed.tolist() == [1301, 1302, 1303]
    assert not tables['candidate_summary'].selected_for_admission.any()
    assert not report['minimums_relaxed'] and not report['automatic_search_expansion']


def test_reordering_inputs_preserves_selection_and_each_assignment():
    drugs, pairs = fixture_population()
    first, first_report = run(drugs, pairs)
    second, second_report = run(drugs.iloc[::-1], pairs.iloc[::-1])
    assert first_report == second_report
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


@pytest.mark.parametrize('table,column', [
    ('drugs', 'severity'), ('pairs', 'prediction'), ('pairs', 'morphological_delta'),
    ('drugs', 'atc_codes'), ('pairs', 'response_text'),
])
def test_projection_rejects_nonidentity_columns(table, column):
    drugs, pairs = fixture_population(n=5)
    (drugs if table == 'drugs' else pairs)[column] = 'not_consumed'
    with pytest.raises(ValueError, match='somente'):
        run(drugs, pairs)


@pytest.mark.parametrize('argument,value', [
    ('dataset_scope', 'synthetic'), ('config', None), ('seeds', (1303, 1302, 1301)),
])
def test_interface_does_not_accept_scope_relabelling_or_configuration_override(argument, value):
    drugs, pairs = fixture_population(n=5)
    with pytest.raises(TypeError):
        materializar_split_admissao(drugs, pairs, population_id='fixture', **{argument: value})
