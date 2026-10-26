"""Small, isolated preparation fixtures: never responses, labels or real cache."""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in ('analise', 'dados'):
    sys.path.insert(0, str(ROOT / 'src' / directory))

from calibracao_preparatoria_exploratoria import (
    build_primary_features, fit_candidate_constants, calibrate_source_grid,
    FEATURE_NAMES, UNIT_NAMES, SCORE_COLUMNS, CONFIG_COLUMNS, HASH_NAMES, VERSION_NAMES,
)
from diagnostico_metadados_r6 import calcular_features_primarias
from metodos_freeze1_v1_3_1 import pontuar_politica_primaria, validar_estado_politica_primaria


def data():
    return (pd.DataFrame([
        ['A', 'alfa sódico', 'alpha'], ['B', 'beta', 'beta'],
        ['C', 'gama mono-hidratada', 'gamma'], ['D', 'delta', 'delta drug'],
    ], columns=['drug_id', 'dcb', 'experimental_name']), pd.DataFrame([
        ['p1', 'A', 'B'], ['p2', 'B', 'C'], ['p3', 'C', 'D'], ['p4', 'A', 'D'],
    ], columns=['pair_id', 'drug_id_a', 'drug_id_b']))


def constants_data():
    values = pd.DataFrame({'pair_id': ['p1', 'p2', 'p3', 'p4', 'cal', 'test'],
                          FEATURE_NAMES[0]: [-1., 0., 2., 5., 100., -100.],
                          FEATURE_NAMES[1]: [.1, .2, .4, .9, 100., -100.],
                          UNIT_NAMES[0]: [2., 3., 5., 8., 103., -97.],
                          UNIT_NAMES[1]: [3.] * 6})
    split = pd.DataFrame({'pair_id': values.pair_id, 'pair_partition': ['development'] * 4 + ['calibration', 'C']})
    return values, split, ['p1', 'p2', 'p3', 'p4']


def calibration_data():
    configs = pd.DataFrame([['gpt-4.1-2025-04-14', 'inn', 'slot'],
                            ['gpt-4.1-2025-04-14', 'dcb', 'slot']], columns=CONFIG_COLUMNS)
    rows = []
    for model, condition, slot in configs.itertuples(index=False, name=None):
        for i in range(5):
            rows.append([f'{condition}-{i}', f'p{i}', model, condition, slot,
                         'calibration', True, True, float(i)])
    scores = pd.DataFrame(rows, columns=SCORE_COLUMNS)
    bases = {tuple(row): 'eligible_after_gate' for row in configs.itertuples(index=False, name=None)}
    hashes = {name: 'a' * 64 for name in HASH_NAMES}
    versions = {name: 'fixture-v1' for name in VERSION_NAMES}
    return scores, configs, bases, hashes, versions


def calibrate(scores=None, configs=None, bases=None, hashes=None, versions=None):
    defaults = calibration_data()
    supplied = (scores, configs, bases, hashes, versions)
    scores, configs, bases, hashes, versions = [x if x is not None else y for x, y in zip(supplied, defaults)]
    return calibrate_source_grid(scores, configs, coverage_basis_by_configuration=bases,
                                 hashes=hashes, versions=versions)


@pytest.mark.filterwarnings('error::DeprecationWarning')
@pytest.mark.parametrize('representation', ['python_object', 'numpy_object'])
def test_object_boolean_masks_preserve_calibration_and_inputs(representation):
    scores, configs, bases, hashes, versions = calibration_data()
    scores.loc[0, ['available', 'eligible']] = False
    scores.loc[1, 'eligible'] = False
    scores.loc[[0, 1], 'risk_score'] = None
    original = scores.copy(deep=True)
    expected = calibrate(scores, configs, bases, hashes, versions)
    object_scores = scores.copy(deep=True)
    object_scores[['available', 'eligible']] = object_scores[['available', 'eligible']].astype(object)
    if representation == 'numpy_object':
        for field in ['available', 'eligible']:
            object_scores[field] = pd.Series([np.bool_(v) for v in scores[field]], index=scores.index, dtype=object)
    object_before = object_scores.copy(deep=True)
    actual = calibrate(object_scores, configs, bases, hashes, versions)
    assert actual.keys() == expected.keys()
    for key in expected:
        if isinstance(expected[key], pd.DataFrame):
            pd.testing.assert_frame_equal(actual[key], expected[key])
        else:
            assert actual[key] == expected[key]
    pd.testing.assert_frame_equal(scores, original)
    pd.testing.assert_frame_equal(object_scores, object_before)


@pytest.mark.filterwarnings('error::DeprecationWarning')
@pytest.mark.parametrize('invalid', [0, 1, 'False', None, pd.NA])
def test_object_masks_reject_non_booleans_before_comparison(invalid):
    for field in ['available', 'eligible']:
        scores, configs, bases, hashes, versions = calibration_data()
        scores[['available', 'eligible']] = scores[['available', 'eligible']].astype(object)
        scores.loc[0, field] = invalid
        with pytest.raises(ValueError, match='booleanos explícitos'):
            calibrate(scores, configs, bases, hashes, versions)


@pytest.mark.filterwarnings('error::DeprecationWarning')
def test_object_masks_reject_eligible_without_availability():
    scores, configs, bases, hashes, versions = calibration_data()
    scores[['available', 'eligible']] = scores[['available', 'eligible']].astype(object)
    scores.loc[0, 'available'] = False
    with pytest.raises(ValueError, match='Elegível sem disponibilidade'):
        calibrate(scores, configs, bases, hashes, versions)


def test_features_reuse_full_text_implementation_and_validate_delta():
    entities, pairs = data()
    result = build_primary_features(entities, pairs)
    expected = calcular_features_primarias(entities, pairs)
    pd.testing.assert_frame_equal(result[['pair_id', *FEATURE_NAMES]], expected)
    np.testing.assert_array_equal(result[FEATURE_NAMES[0]], result[UNIT_NAMES[0]] - result[UNIT_NAMES[1]])
    pd.testing.assert_frame_equal(result, build_primary_features(entities.iloc[::-1], pairs.iloc[::-1]))


@pytest.mark.parametrize('fault', ['duplicate_entity', 'duplicate_pair', 'orphan', 'self_pair', 'extra_column', 'bad_id'])
def test_feature_input_integrity(fault):
    entities, pairs = data()
    if fault == 'duplicate_entity': entities = pd.concat([entities, entities.iloc[[0]]])
    if fault == 'duplicate_pair': pairs = pd.concat([pairs, pairs.iloc[[0]]])
    if fault == 'orphan': pairs.loc[0, 'drug_id_b'] = 'absent'
    if fault == 'self_pair': pairs.loc[0, 'drug_id_b'] = 'A'
    if fault == 'extra_column': entities['predicted_category'] = 'never accepted'
    if fault == 'bad_id': pairs.loc[0, 'pair_id'] = ' p1 '
    with pytest.raises(ValueError): build_primary_features(entities, pairs)


def test_missing_name_preserves_nan_without_imputation():
    entities, pairs = data()
    entities.loc[0, 'dcb'] = None
    result = build_primary_features(entities, pairs)
    assert result.loc[result.pair_id.eq('p1'), FEATURE_NAMES[0]].isna().all()


def test_constants_independent_oracle_and_exclusively_development():
    features, split, ids = constants_data()
    result = fit_candidate_constants(features, split, expected_development_ids=ids)
    values = features.iloc[:4][list(FEATURE_NAMES)].to_numpy()
    for i, constant in enumerate(result['state']['features']):
        assert constant['development_mean'] == pytest.approx(values[:, i].mean())
        assert constant['development_std_population'] == pytest.approx(values[:, i].std(ddof=0))
    np.testing.assert_allclose(pontuar_politica_primaria(features.iloc[:4], result['state']),
                               ((values - values.mean(axis=0)) / values.std(axis=0)).sum(axis=1))
    altered = features.copy()
    altered.loc[4:, list(FEATURE_NAMES)] = np.nan
    assert result == fit_candidate_constants(altered, split, expected_development_ids=ids)
    assert result == fit_candidate_constants(features.iloc[::-1], split.iloc[::-1], expected_development_ids=ids[::-1])
    assert result['metadata']['constants_formally_frozen'] is False
    validar_estado_politica_primaria(result['state'])


def test_materialized_cross_partition_literal_preserved_and_not_fitted():
    features, split, ids = constants_data()
    split.loc[5, 'pair_partition'] = 'cross_partition'
    result = fit_candidate_constants(features, split, expected_development_ids=ids)
    assert result['metadata']['development_count'] == 4
    assert result['state']['features'][0]['development_mean'] == 1.5


@pytest.mark.parametrize('fault', ['extra_feature', 'duplicate_feature', 'duplicate_split', 'missing_pair',
                                  'wrong_expected', 'duplicate_expected', 'empty_expected', 'wrong_partition',
                                  'null', 'infinity', 'boolean', 'zero_variance', 'delta_mismatch'])
def test_constants_reject_incoherent_join_or_unusable_development(fault):
    features, split, ids = constants_data()
    if fault == 'extra_feature': features['true_severity'] = 'forbidden'
    if fault == 'duplicate_feature': features = pd.concat([features, features.iloc[[0]]])
    if fault == 'duplicate_split': split = pd.concat([split, split.iloc[[0]]])
    if fault == 'missing_pair': features = features.iloc[1:]
    if fault == 'wrong_expected': ids = ['p1', 'p2', 'p3', 'test']
    if fault == 'duplicate_expected': ids.append('p1')
    if fault == 'empty_expected': ids = []
    if fault == 'wrong_partition': split.loc[0, 'pair_partition'] = 'C'
    if fault == 'null': features.loc[0, FEATURE_NAMES[0]] = np.nan
    if fault == 'infinity': features.loc[0, FEATURE_NAMES[0]] = np.inf
    if fault == 'boolean':
        features[FEATURE_NAMES[0]] = features[FEATURE_NAMES[0]].astype(object)
        features.loc[0, FEATURE_NAMES[0]] = True
    if fault == 'zero_variance': features.loc[:3, FEATURE_NAMES[1]] = 0.
    if fault == 'delta_mismatch': features.loc[0, UNIT_NAMES[0]] = 100.
    with pytest.raises(ValueError): fit_candidate_constants(features, split, expected_development_ids=ids)


def test_complete_grid_source_only_with_identifiable_thresholds():
    result = calibrate()
    table = result['thresholds']
    assert len(table) == 8 and table.threshold_id.is_unique
    assert table.groupby('condition').threshold.apply(list).tolist() == [[2., 3., 3., 4.]] * 2
    assert table.calibration_accepted_count.tolist() == [3, 4, 4, 5] * 2
    assert table.tie_seed.isna().all()
    assert table.threshold_formally_frozen.eq(False).all()
    assert table.target_application_performed.eq(False).all()
    assert result['metadata']['calculated_threshold_count'] == 8


def test_inclusive_ties_preserve_excess_coverage_without_seed():
    scores, _, _, _, _ = calibration_data()
    scores['risk_score'] = 1.
    table = calibrate(scores=scores)['thresholds']
    assert table.threshold.eq(1.).all()
    assert table.calibration_accepted_count.eq(5).all()
    assert table.at_threshold_count.eq(5).all()
    assert table.additional_accepted_due_to_ties.tolist() == [2, 1, 1, 0] * 2
    assert table.observed_source_coverage_eligible_after_gate.eq(1.).all()


def test_saturation_global_limits_to_eligible_and_preserves_operational_base():
    scores, configs, bases, _, _ = calibration_data()
    scores.loc[scores.risk_score.ge(2), 'eligible'] = False
    bases = {key: 'global' for key in bases}
    table = calibrate(scores=scores, configs=configs, bases=bases)['thresholds']
    assert table.available_count.eq(5).all() and table.eligible_count.eq(2).all()
    assert table.threshold.eq(1.).all() and table.calibration_accepted_count.eq(2).all()
    assert table.source_gate_saturated.eq(True).all()
    assert table.requested_global_quota_feasible.eq(False).all()
    assert table.observed_source_coverage_global.eq(.4).all()


def test_available_excludes_unprocessed_positions_not_model_abstentions():
    scores, _, _, _, _ = calibration_data()
    scores.loc[scores.pair_id.eq('p4'), ['available', 'eligible']] = False
    scores.loc[scores.pair_id.eq('p4'), 'risk_score'] = np.nan
    table = calibrate(scores=scores)['thresholds']
    assert table.planned_count.eq(5).all()
    assert table.available_count.eq(4).all()
    assert table.unavailable_count.eq(1).all()
    assert table.ineligible_available_count.eq(0).all()


@pytest.mark.parametrize('fault,reason', [('missing', 'configuration_absent'),
                                       ('unavailable', 'no_available_observations'),
                                       ('ineligible', 'no_eligible_observations')])
def test_non_estimable_configuration_retained(fault, reason):
    scores, _, _, _, _ = calibration_data()
    if fault == 'missing': scores = scores.loc[scores.condition.eq('dcb')]
    else:
        scores.loc[scores.condition.eq('inn'), 'eligible'] = False
        if fault == 'unavailable': scores.loc[scores.condition.eq('inn'), 'available'] = False
    table = calibrate(scores=scores)['thresholds']
    missing = table.loc[table.condition.eq('inn')]
    assert len(table) == 8 and len(missing) == 4
    assert missing.non_estimable_reason.eq(reason).all()
    assert missing.threshold.isna().all()


def test_entirely_absent_source_inventory_preserves_all_grid_positions():
    scores, _, _, _, _ = calibration_data()
    table = calibrate(scores=scores.iloc[:0])['thresholds']
    assert len(table) == 8 and table.threshold.isna().all()


def test_permutation_invariance_and_no_mutation():
    scores, configs, bases, hashes, versions = calibration_data()
    original = scores.copy(deep=True)
    before = calibrate(scores, configs, bases, hashes, versions)
    after = calibrate(scores.iloc[::-1, ::-1], configs.iloc[::-1, ::-1], bases, hashes, versions)
    pd.testing.assert_frame_equal(before['thresholds'], after['thresholds'])
    pd.testing.assert_frame_equal(scores, original)
    assert before['metadata'] == after['metadata']


@pytest.mark.parametrize('fault', ['duplicate_observation', 'duplicate_pair', 'duplicate_configuration',
                                  'target', 'target_snapshot', 'unknown_configuration', 'extra_column', 'missing_basis',
                                  'invalid_basis', 'missing_hash', 'bad_hash', 'missing_version',
                                  'nonboolean', 'eligible_without_available', 'eligible_nan', 'bool_score'])
def test_calibration_rejects_incoherent_configuration_or_leaking_inputs(fault):
    scores, configs, bases, hashes, versions = calibration_data()
    if fault == 'duplicate_observation': scores = pd.concat([scores, scores.iloc[[0]]])
    if fault == 'duplicate_pair': scores.loc[1, 'pair_id'] = scores.loc[0, 'pair_id']
    if fault == 'duplicate_configuration': configs = pd.concat([configs, configs.iloc[[0]]])
    if fault == 'target': scores.loc[0, 'pair_partition'] = 'C'
    if fault == 'target_snapshot':
        configs['model'] = 'gpt-5.4-2026-03-05'
        scores['model'] = 'gpt-5.4-2026-03-05'
        bases = {('gpt-5.4-2026-03-05', condition, 'slot'): 'eligible_after_gate' for condition in ('inn', 'dcb')}
    if fault == 'unknown_configuration': scores.loc[0, 'model'] = 'outside'
    if fault == 'extra_column': scores['predicted_category'] = 'never accepted'
    if fault == 'missing_basis': bases.pop(next(iter(bases)))
    if fault == 'invalid_basis': bases[next(iter(bases))] = 'winning'
    if fault == 'missing_hash': hashes.pop(next(iter(hashes)))
    if fault == 'bad_hash': hashes[next(iter(hashes))] = 'fixture'
    if fault == 'missing_version': versions.pop(next(iter(versions)))
    if fault == 'nonboolean': scores['available'] = 'true'
    if fault == 'eligible_without_available': scores.loc[0, 'available'] = False
    if fault == 'eligible_nan': scores.loc[0, 'risk_score'] = np.nan
    if fault == 'bool_score': scores['risk_score'] = True
    with pytest.raises(ValueError): calibrate(scores, configs, bases, hashes, versions)
