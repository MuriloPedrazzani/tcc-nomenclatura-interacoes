"""Preparation-only real adapter: features, development constants, source thresholds.

No file access, response categories, reference severities, target decisions or
synthetic-scope overrides. Callers must verify the documentary projections and
their hashes separately; supplied hashes are bindings, not proof of provenance.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Real
import re

import numpy as np
import pandas as pd

from diagnostico_metadados_r6 import calcular_features_primarias
from features_nomenclatura import _n_morphology_units, _soma
from metodos_freeze1_v1_3_1 import ajustar_politica_primaria, PRIMARY_FEATURES
from predicao_seletiva import PoliticaLimiar


VERSION = 'exploratory-real-preparation-calibration-v1'
COVERAGE_GRID = (.6, .7, .8, .9)
FEATURE_NAMES = tuple(x[0] for x in PRIMARY_FEATURES)
UNIT_NAMES = ('nomenclature_morphology_units_dcb_pair',
              'nomenclature_morphology_units_inn_pair')
FEATURE_COLUMNS = ('pair_id', *FEATURE_NAMES, *UNIT_NAMES)
CONFIG_COLUMNS = ('model', 'condition', 'execution_slot')
SCORE_COLUMNS = ('observation_id', 'pair_id', *CONFIG_COLUMNS, 'pair_partition',
                 'available', 'eligible', 'risk_score')
HASH_NAMES = frozenset({'policy_state_sha256', 'features_sha256', 'split_sha256',
                        'eligibility_sha256', 'source_projection_sha256'})
VERSION_NAMES = frozenset({'feature_version', 'denominator_version', 'provenance_version'})
SOURCE_SNAPSHOTS = frozenset({'gpt-4.1-2025-04-14', 'gpt-4.1-mini-2025-04-14',
                             'gpt-4.1-nano-2025-04-14'})


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _table(frame, columns, *, name, unique=()):
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any():
        raise ValueError(f'{name}: DataFrame sem colunas duplicadas requerido.')
    if set(frame.columns) != set(columns):
        raise ValueError(f'{name}: somente colunas projetadas exatas são aceitas.')
    if unique and frame[list(unique)].duplicated().any():
        raise ValueError(f'{name}: identidade duplicada.')


def _ids(frame, columns):
    for column in columns:
        if not frame[column].map(lambda x: isinstance(x, str) and bool(x.strip())
                                 and x == x.strip()).all():
            raise ValueError(f'{column}: identificadores literais completos requeridos.')


def _boolean(series, name):
    if not series.map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ValueError(f'{name}: booleanos explícitos requeridos.')


def build_primary_features(entities, pairs):
    """Keep the historical full-name transforms and add the delta components."""
    _table(entities, ('drug_id', 'dcb', 'experimental_name'), name='entities', unique=('drug_id',))
    _table(pairs, ('pair_id', 'drug_id_a', 'drug_id_b'), name='pairs', unique=('pair_id',))
    _ids(entities, ('drug_id',))
    _ids(pairs, ('pair_id', 'drug_id_a', 'drug_id_b'))
    for column in ('dcb', 'experimental_name'):
        if not entities[column].map(lambda x: x is None or isinstance(x, str)
                                     or (isinstance(x, Real) and math.isnan(x))).all():
            raise ValueError(f'{column}: texto ou ausência explícita requerido.')
    # The approved extractor checks referential integrity and rejects self-pairs.
    result = calcular_features_primarias(entities, pairs)
    by = entities.set_index('drug_id', verify_integrity=True)
    for name, source in zip(UNIT_NAMES, ('dcb', 'experimental_name')):
        result[name] = [_soma(_n_morphology_units(by.at[row.drug_id_a, source]),
                              _n_morphology_units(by.at[row.drug_id_b, source]))
                        for row in pairs.itertuples(index=False)]
    return result[list(FEATURE_COLUMNS)].sort_values('pair_id').reset_index(drop=True)


def fit_candidate_constants(features, split, *, expected_development_ids):
    """Fit the unchanged primary state on exactly the explicit development set."""
    _table(features, FEATURE_COLUMNS, name='features', unique=('pair_id',))
    _table(split, ('pair_id', 'pair_partition'), name='split', unique=('pair_id',))
    _ids(features, ('pair_id',))
    _ids(split, ('pair_id', 'pair_partition'))
    allowed = {'development', 'calibration', 'C', 'cross_partition', 'documentary_excluded'}
    if not set(split.pair_partition) <= allowed:
        raise ValueError('Partição não reconhecida.')
    if isinstance(expected_development_ids, (str, bytes)) or not isinstance(expected_development_ids, Sequence):
        raise ValueError('expected_development_ids deve ser sequência explícita.')
    expected = list(expected_development_ids)
    if not expected or any(not isinstance(x, str) or not x.strip() or x != x.strip() for x in expected):
        raise ValueError('IDs esperados do desenvolvimento ausentes ou inválidos.')
    if len(set(expected)) != len(expected):
        raise ValueError('IDs esperados do desenvolvimento duplicados.')
    if set(features.pair_id) != set(split.pair_id):
        raise ValueError('Universos features/split divergem; join parcial proibido.')
    actual = set(split.loc[split.pair_partition.eq('development'), 'pair_id'])
    if actual != set(expected):
        raise ValueError('Desenvolvimento diverge do conjunto exato esperado.')
    # Deliberately project development before inspecting numerical values.
    development = features.merge(split, on='pair_id', validate='one_to_one')
    development = development.loc[development.pair_partition.eq('development')].sort_values('pair_id')
    for column in (*FEATURE_NAMES, *UNIT_NAMES):
        if not development[column].map(lambda x: isinstance(x, Real)
                                       and not isinstance(x, (bool, np.bool_))
                                       and math.isfinite(x)).all():
            raise ValueError(f'Feature/componente ausente ou não finito no desenvolvimento: {column}')
    development = development.rename(columns={'pair_partition': 'split'})
    state = ajustar_politica_primaria(development)
    metadata = dict(version=VERSION, development_count=len(development),
                    development_pair_ids=sorted(expected),
                    development_pair_ids_sha256=_sha(sorted(expected)),
                    development_features_sha256=_sha(development.to_dict('records')),
                    adjustment_scope='development_only', population_std_ddof=0,
                    imputation_performed=False, feature_rows_dropped=0,
                    constants_calculated=True, constants_formally_frozen=False,
                    tie_seed=None, tie_seed_status='not_required_for_standardization',
                    calibration_or_C_used=False, freeze2_started=False,
                    freeze2_complete=False, DEFINITION_OF_DONE=False)
    return {'state': state, 'metadata': metadata}


def _validate_calibration(scores, configurations, bases, hashes, versions):
    _table(scores, SCORE_COLUMNS, name='scores', unique=('observation_id',))
    _table(configurations, CONFIG_COLUMNS, name='required_configurations', unique=CONFIG_COLUMNS)
    if configurations.empty:
        raise ValueError('Inventário explícito de configurações vazio.')
    _ids(configurations, CONFIG_COLUMNS)
    _ids(scores, ('observation_id', 'pair_id', *CONFIG_COLUMNS, 'pair_partition'))
    if not set(configurations.condition) <= {'inn', 'dcb'}:
        raise ValueError('Condição não autorizada na calibração.')
    if not set(configurations.model) <= SOURCE_SNAPSHOTS:
        raise ValueError('Somente os snapshots de origem GPT-4.1 aprovados são aceitos.')
    if not scores.pair_partition.eq('calibration').all():
        raise ValueError('Somente calibração de origem; C/alvo não são aceitos.')
    if scores[[*CONFIG_COLUMNS, 'pair_id']].duplicated().any():
        raise ValueError('Par/configuração duplicado.')
    expected = set(configurations[list(CONFIG_COLUMNS)].itertuples(index=False, name=None))
    observed = set(scores[list(CONFIG_COLUMNS)].itertuples(index=False, name=None))
    if observed - expected:
        raise ValueError('Observações fora das configurações explícitas.')
    if not isinstance(bases, Mapping) or set(bases) != expected:
        raise ValueError('Base operacional explícita deve cobrir exatamente as configurações.')
    if any(value not in {'global', 'eligible_after_gate'} for value in bases.values()):
        raise ValueError('Base operacional inválida.')
    if not isinstance(hashes, Mapping) or set(hashes) != HASH_NAMES:
        raise ValueError('Hashes de proveniência completos requeridos.')
    if any(not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None
           for value in hashes.values()):
        raise ValueError('Hash SHA-256 inválido.')
    if not isinstance(versions, Mapping) or set(versions) != VERSION_NAMES:
        raise ValueError('Versões de proveniência completas requeridas.')
    if any(not isinstance(value, str) or not value.strip() or value != value.strip()
           for value in versions.values()):
        raise ValueError('Versão literal inválida.')
    for field in ('available', 'eligible'):
        _boolean(scores[field], field)
    # Explicit booleans were validated above; comparison also supports object dtype.
    if (scores.eligible & scores.available.eq(False)).any():
        raise ValueError('Elegível sem disponibilidade.')
    for row in scores.itertuples(index=False):
        valid = isinstance(row.risk_score, Real) and not isinstance(row.risk_score, (bool, np.bool_)) and math.isfinite(row.risk_score)
        absent = row.risk_score is None or (isinstance(row.risk_score, Real) and math.isnan(row.risk_score))
        if not valid and not (not row.eligible and absent):
            raise ValueError('Escore elegível deve ser finito; ausência só permitida para inelegível.')
    return expected


def calibrate_source_grid(scores, required_configurations, *,
                          coverage_basis_by_configuration, hashes, versions):
    """Prepare four inclusive thresholds per source configuration, never target.

The explicit configuration inventory and denominator bases are caller decisions
linked by hashes. This numerical interface does not certify their approval.
Availability defines N; eligibility after the approved gate defines E. Missing
configurations remain in the output with null thresholds and an explicit reason.
"""
    expected = _validate_calibration(scores, required_configurations,
                                    coverage_basis_by_configuration, hashes, versions)
    # Normalize only validated masks in a local copy; never coerce missing/invalid input.
    scores = scores.astype({'available': bool, 'eligible': bool})
    rows = []
    for key in sorted(expected):
        group = scores.loc[np.logical_and.reduce([scores[column].eq(value)
                          for column, value in zip(CONFIG_COLUMNS, key)])].sort_values('pair_id')
        available = group.loc[group.available]
        eligible = available.loc[available.eligible]
        basis = coverage_basis_by_configuration[key]
        n, e = len(available), len(eligible)
        reason = ('configuration_absent' if group.empty else 'no_available_observations' if not n
                  else 'no_eligible_observations' if not e else None)
        source_records = group.copy()
        source_records['risk_score'] = source_records['risk_score'].astype(object)
        source_records.loc[source_records['risk_score'].isna(), 'risk_score'] = None
        source_sha = _sha(source_records.to_dict('records'))
        for coverage in COVERAGE_GRID:
            payload = dict(version=VERSION, **dict(zip(CONFIG_COLUMNS, key)),
                           pair_partition='calibration', target_coverage=coverage,
                           coverage_basis=basis, **dict(hashes), **dict(versions),
                           source_configuration_sha256=source_sha,
                           planned_count=len(group), available_count=n, eligible_count=e,
                           unavailable_count=len(group) - n, ineligible_available_count=n - e,
                           threshold=None, threshold_status='not_estimable' if reason else 'calculated_candidate',
                           non_estimable_reason=reason, desired_quota_raw=None,
                           nominal_eligible_quota=None, calibration_quantile_fraction=None,
                           calibration_accepted_count=None, observed_source_coverage_global=None,
                           observed_source_coverage_eligible_after_gate=None,
                           strictly_below_threshold_count=None, at_threshold_count=None,
                           additional_accepted_due_to_ties=None, source_gate_saturated=None,
                           source_selector_active=None, requested_global_quota_feasible=None,
                           comparator='score <= threshold', floating_tolerance_added=False,
                           tie_seed=None, tie_seed_status='not_required_inclusive_threshold',
                           source_only=True, target_application_performed=False,
                           threshold_formally_frozen=False, clinical_labels_used=False,
                           DEFINITION_OF_DONE=False)
            if reason is None:
                raw = int(math.floor((n if basis == 'global' else e) * coverage + .5))
                nominal = min(raw, e)
                fraction = nominal / e if basis == 'global' else coverage
                policy = PoliticaLimiar.calibrar(eligible.risk_score, fraction)
                accepted = int(eligible.risk_score.le(policy.limiar).sum())
                below = int(eligible.risk_score.lt(policy.limiar).sum())
                at = int(eligible.risk_score.eq(policy.limiar).sum())
                # The inclusive backend ensures at least one accepted when c>0.
                effective_order = max(1, nominal) if fraction > 0 else 0
                payload.update(threshold=float(policy.limiar), desired_quota_raw=raw,
                               nominal_eligible_quota=nominal,
                               calibration_quantile_fraction=fraction,
                               calibration_accepted_count=accepted,
                               observed_source_coverage_global=accepted / n,
                               observed_source_coverage_eligible_after_gate=accepted / e,
                               strictly_below_threshold_count=below, at_threshold_count=at,
                               additional_accepted_due_to_ties=max(0, accepted - effective_order),
                               source_gate_saturated=(basis == 'global' and raw >= e and accepted == e),
                               source_selector_active=accepted < e,
                               requested_global_quota_feasible=(raw <= e if basis == 'global' else None))
            digest = _sha(payload)
            rows.append(dict(threshold_id='CANDTHR_' + digest[:24], threshold_sha256=digest, **payload))
    return {'thresholds': pd.DataFrame(rows, dtype=object), 'metadata': dict(
        version=VERSION, source_configuration_count=len(expected),
        planned_threshold_count=len(expected) * len(COVERAGE_GRID),
        calculated_threshold_count=sum(row['threshold_status'] == 'calculated_candidate' for row in rows),
        coverage_grid=list(COVERAGE_GRID), source_only=True,
        provenance_hashes_supplied_not_independently_verified=True,
        operational_bases_supplied_not_methodological_approval=True,
        target_application_performed=False, thresholds_formally_frozen=False,
        categories_used=False, clinical_labels_used=False, freeze2_started=False,
        freeze2_complete=False, DEFINITION_OF_DONE=False)}
