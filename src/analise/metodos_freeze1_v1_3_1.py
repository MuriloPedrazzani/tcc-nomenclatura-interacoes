# -*- coding: utf-8 -*-
"""Métodos congeláveis da rodada técnica incremental v1.3.1."""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


METHOD_VERSION = 'technical-freeze1-methods-v1.3.1'
PRIMARY_POLICY_VERSION = 'two-feature-unit-standardized-v1.3.1'
COVERAGE_RULE_VERSION = 'coverage-star-power-then-routing-v1.3.1'
TRANSFER_RULE_VERSION = 'gpt41-to-gpt54-frozen-threshold-v1.3.1'
MULTIPLICITY_VERSION = 'single-primary-secondary-bh05-v1.3.1'
NULL_PATH_VERSION = 'null-unfavorable-inconclusive-path-v1.3.1'
STATE_VALIDATION_VERSION = 'primary-policy-state-validation-v1.3.3'
COVERAGE_GRID = (0.60, 0.70, 0.80, 0.90)
COVERAGE_DRIFT_TOLERANCE = 0.05
PRIMARY_FEATURES = (
    ('nomenclature_delta_morphology_units_pair', 1, 1.0),
    ('nomenclature_lexical_dissimilarity_max', 1, 1.0),
)
SENSITIVITY_FEATURE_GROUPS = (
    'salt_extra', 'pubmed_frequency', 'o200k', 'transductive_features'
)
MODEL_FAMILIES = {
    'gpt-4.1-2025-04-14': 'openai-gpt-4.1',
    'gpt-4.1-mini-2025-04-14': 'openai-gpt-4.1',
    'gpt-4.1-nano-2025-04-14': 'openai-gpt-4.1',
    'gpt-5.4-2026-03-05': 'openai-gpt-5.4',
    'gpt-5.4-mini-2026-03-17': 'openai-gpt-5.4',
    'gpt-5.4-nano-2026-03-17': 'openai-gpt-5.4',
    'claude-haiku-4-5-20251001': 'anthropic-claude-4.5',
    'claude-sonnet-4-5-20250929': 'anthropic-claude-4.5',
    'claude-opus-4-5-20251101': 'anthropic-claude-4.5',
}
HISTORICAL_REFERENCE_MODELS = {
    'gpt-4o-mini-2024-07-18': 'historical_secondary_reference'
}
PRIMARY_MEMBER = (
    'PRIMARY_GPT54_FAMILY_C_POLICY_VS_FIXED_QUOTA_RANDOM_'
    'DELTA1_AT_COVERAGE_STAR'
)
SECONDARY_FAMILY_MEMBERS = {
    'SEC_COVERAGE_GRID': tuple(f'coverage_{value:.2f}' for value in COVERAGE_GRID),
    'SEC_MODEL_SNAPSHOTS_OPENAI': (
        'gpt-4.1-2025-04-14', 'gpt-4.1-mini-2025-04-14',
        'gpt-4.1-nano-2025-04-14', 'gpt-5.4-2026-03-05',
        'gpt-5.4-mini-2026-03-17', 'gpt-5.4-nano-2026-03-17',
    ),
    'SEC_MECHANISMS': (
        'severe_underestimation_delta2', 'severe_routing_rate',
        'risk_among_severe_accepted_delta1', 'residual_error_rate',
        'morphology_delta_zero', 'morphology_delta_nonzero',
    ),
    'SEC_TRANSFER_SNAPSHOT_PAIRS': (
        'gpt-4.1-2025-04-14__to__gpt-5.4-2026-03-05',
        'gpt-4.1-mini-2025-04-14__to__gpt-5.4-mini-2026-03-17',
        'gpt-4.1-nano-2025-04-14__to__gpt-5.4-nano-2026-03-17',
    ),
    'SEC_ANTHROPIC_CONCEPTUAL_REPLICATION': (
        'claude-haiku-4-5-20251001', 'claude-sonnet-4-5-20250929',
        'claude-opus-4-5-20251101',
    ),
    'SEC_FEATURE_SENSITIVITIES': SENSITIVITY_FEATURE_GROUPS,
}


def _normal_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    return values.astype(str).str.strip().str.casefold().eq('true')


def _sha_payload(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


def validar_delta_dcb_menos_inn(features: pd.DataFrame) -> dict[str, object]:
    required = {
        'nomenclature_delta_morphology_units_pair',
        'nomenclature_morphology_units_dcb_pair',
        'nomenclature_morphology_units_inn_pair',
    }
    if missing := sorted(required - set(features.columns)):
        raise ValueError(f'Colunas ausentes para validar delta DCB-INN: {missing}')
    observed = pd.to_numeric(
        features['nomenclature_delta_morphology_units_pair'], errors='raise'
    ).astype(float)
    expected = (
        pd.to_numeric(features['nomenclature_morphology_units_dcb_pair'], errors='raise')
        - pd.to_numeric(features['nomenclature_morphology_units_inn_pair'], errors='raise')
    ).astype(float)
    comparable = np.isfinite(observed) & np.isfinite(expected)
    mismatch = comparable & ~np.isclose(observed, expected, atol=1e-12, rtol=0)
    if mismatch.any():
        ids = (
            features.loc[mismatch, 'pair_id'].astype(str).head(10).tolist()
            if 'pair_id' in features else features.index[mismatch].astype(str).tolist()[:10]
        )
        raise ValueError(f'Delta morfológico não é DCB-INN em: {ids}')
    return {
        'definition': 'morphology_units_DCB_pair_minus_morphology_units_INN_pair',
        'dcb_base_used_in_delta': False,
        'comparable_count': int(comparable.sum()),
        'nonfinite_count': int((~comparable).sum()),
        'mismatch_count': 0,
        'validated': True,
    }


def ajustar_politica_primaria(development: pd.DataFrame) -> dict[str, object]:
    """Ajusta somente média/desvio das duas features aprovadas no desenvolvimento."""
    if development.empty or 'split' not in development or not development['split'].astype(str).eq('development').all():
        raise ValueError('Padronização só pode usar o desenvolvimento definitivo.')
    delta_report = validar_delta_dcb_menos_inn(development)
    constants = []
    for name, direction, weight in PRIMARY_FEATURES:
        values = pd.to_numeric(development[name], errors='raise').astype(float)
        if not np.isfinite(values).all():
            raise ValueError(f'Feature primária ausente/não finita: {name}')
        mean, std = float(values.mean()), float(values.std(ddof=0))
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError(f'Feature primária sem variação: {name}')
        constants.append({
            'name': name, 'direction': direction, 'weight': weight,
            'development_mean': mean, 'development_std_population': std,
        })
    state = {
        'policy_version': PRIMARY_POLICY_VERSION,
        'features': constants,
        'feature_names_exact': [row[0] for row in PRIMARY_FEATURES],
        'directions_exact': [row[1] for row in PRIMARY_FEATURES],
        'weights_exact': [row[2] for row in PRIMARY_FEATURES],
        'standardization_scope': 'definitive_development_only',
        'delta_semantics': delta_report,
        'salt_extra_in_primary': False,
        'pubmed_frequency_in_primary': False,
        'o200k_in_primary': False,
        'transductive_feature_in_primary': False,
        'constants_frozen_now': False,
        'constants_reserved_for_freeze2': True,
    }
    state['state_sha256'] = _sha_payload(state)
    validar_estado_politica_primaria(state)
    return state


def validar_estado_politica_primaria(state: Mapping[str, object]) -> None:
    """Valida o contrato científico e o conteúdo antes de confiar no SHA declarado.

    Recalcular o hash de pesos não autorizados não os torna admissíveis.
    Constantes matematicamente válidas continuam exigindo proveniência do
    desenvolvimento: o hash é integridade, não comprovação de sua origem.
    """
    if not isinstance(state, Mapping):
        raise ValueError('Estado da política deve ser um mapeamento.')
    if state.get('policy_version') != PRIMARY_POLICY_VERSION:
        raise ValueError('Versão da política não autorizada.')
    if state.get('standardization_scope') != 'definitive_development_only':
        raise ValueError('Escopo de padronização não autorizado.')
    features = state.get('features')
    if not isinstance(features, list) or len(features) != len(PRIMARY_FEATURES):
        raise ValueError('Estado exige exatamente duas features na ordem aprovada.')
    for feature, (name, direction, weight) in zip(features, PRIMARY_FEATURES):
        if not isinstance(feature, Mapping) or feature.get('name') != name:
            raise ValueError('Nomes/ordem das features divergem do contrato aprovado.')
        for key, expected in (('direction', direction), ('weight', weight)):
            value = feature.get(key)
            if isinstance(value, bool) or not isinstance(value, Real) or value != expected:
                raise ValueError(f'{key} não autorizado para {name}.')
        for key in ('development_mean', 'development_std_population'):
            value = feature.get(key)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f'Constante {key} deve ser numérica finita.')
        if feature['development_std_population'] <= 0:
            raise ValueError('Desvio populacional deve ser estritamente positivo.')
    for key, expected in (
        ('feature_names_exact', [x[0] for x in PRIMARY_FEATURES]),
        ('directions_exact', [x[1] for x in PRIMARY_FEATURES]),
        ('weights_exact', [x[2] for x in PRIMARY_FEATURES]),
    ):
        if state.get(key) != expected:
            raise ValueError(f'{key} diverge do contrato aprovado.')
    for key in ('salt_extra_in_primary', 'pubmed_frequency_in_primary',
                'o200k_in_primary', 'transductive_feature_in_primary'):
        if state.get(key) is not False:
            raise ValueError(f'{key} deve ser false.')
    digest = state.get('state_sha256')
    payload = {key: value for key, value in state.items() if key != 'state_sha256'}
    try:
        observed = _sha_payload(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError('Estado não possui conteúdo serializável válido.') from exc
    if not isinstance(digest, str) or digest != observed:
        raise ValueError('Hash do estado da política diverge do conteúdo.')


def carregar_estado_politica_primaria(payload: str) -> dict[str, object]:
    state = json.loads(payload)
    validar_estado_politica_primaria(state)
    return state


def pontuar_politica_primaria(rows: pd.DataFrame, state: Mapping[str, object]) -> pd.Series:
    validar_estado_politica_primaria(state)
    score = pd.Series(0.0, index=rows.index)
    for feature in state['features']:
        values = pd.to_numeric(rows[feature['name']], errors='raise').astype(float)
        if not np.isfinite(values).all():
            raise ValueError(f'Feature primária ausente/não finita: {feature["name"]}')
        z = (values - feature['development_mean']) / feature['development_std_population']
        score += feature['direction'] * feature['weight'] * z
    return score.rename('risk_score')


def classificar_estrato_morfologico(rows: pd.DataFrame) -> pd.Series:
    column = 'nomenclature_delta_morphology_units_pair'
    values = pd.to_numeric(rows[column], errors='raise').astype(float)
    if not np.isfinite(values).all():
        raise ValueError('Estratificação morfológica exige delta finito.')
    return pd.Series(
        np.where(values.eq(0), 'morphology_delta_zero', 'morphology_delta_nonzero'),
        index=rows.index, name='morphology_stratum',
    )


def aplicar_politica_global_com_estratos(
    rows: pd.DataFrame,
    state: Mapping[str, object],
    *,
    coverage_star: float,
    tie_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Aplica uma única política/limiar global e só depois resume os estratos."""
    if coverage_star not in COVERAGE_GRID:
        raise ValueError(f'coverage_star deve pertencer à grade {COVERAGE_GRID}.')
    if 'eligible' not in rows or 'pair_id' not in rows:
        raise ValueError('Aplicação exige eligible e pair_id.')
    result = rows.copy()
    result['risk_score'] = pontuar_politica_primaria(result, state)
    result['morphology_stratum'] = classificar_estrato_morfologico(result)
    eligible = _normal_bool(result['eligible'])
    indices = list(result.index[eligible])
    ordered = sorted(indices, key=lambda index: (
        float(result.at[index, 'risk_score']),
        hashlib.sha256(f'{tie_seed}\x1f{result.at[index, "pair_id"]}'.encode()).hexdigest(),
    ))
    k = int(math.floor(len(ordered) * coverage_star + 0.5))
    accepted = set(ordered[:k])
    result['accepted'] = [index in accepted for index in result.index]
    threshold = float(result.loc[list(accepted), 'risk_score'].max()) if accepted else math.nan
    result['global_threshold'] = threshold
    result['coverage_star_global'] = coverage_star
    result['policy_state_sha256'] = state['state_sha256']
    result['stratum_recalibrated'] = False
    summaries = []
    for stratum, group in result.groupby('morphology_stratum', sort=True):
        group_eligible = _normal_bool(group['eligible'])
        summaries.append({
            'morphology_stratum': stratum,
            'n_total': len(group), 'n_eligible': int(group_eligible.sum()),
            'n_accepted': int(group['accepted'].sum()),
            'observed_coverage_with_global_policy': (
                float(group.loc[group_eligible, 'accepted'].mean()) if group_eligible.any()
                else math.nan
            ),
            'coverage_star_global': coverage_star,
            'global_threshold': threshold,
            'policy_state_sha256': state['state_sha256'],
            'constants_reestimated_in_stratum': False,
            'coverage_reselected_in_stratum': False,
            'threshold_recalibrated_in_stratum': False,
            'counts_recalculate_after_90_mappings_in_freeze2': True,
        })
    report = {
        'same_policy_both_strata': True,
        'same_constants_both_strata': True,
        'same_global_coverage_target_both_strata': True,
        'same_global_threshold_both_strata': True,
        'stratum_specific_recalibration': False,
        'global_eligible_count': len(ordered), 'global_accepted_count': k,
        'global_observed_coverage': k / len(ordered) if ordered else math.nan,
        'coverage_star': coverage_star, 'threshold': threshold, 'tie_seed': tie_seed,
    }
    return result, pd.DataFrame(summaries), report


def selecionar_coverage_star(power_by_coverage: pd.DataFrame) -> dict[str, object]:
    """Regra Freeze 1; o valor numérico só é materializado no Freeze 2."""
    required = {'coverage', 'operational_success_rate', 'referral_rate'}
    if missing := sorted(required - set(power_by_coverage.columns)):
        raise ValueError(f'Tabela de cobertura sem colunas: {missing}')
    table = power_by_coverage.copy()
    table['coverage'] = pd.to_numeric(table['coverage'], errors='raise').astype(float)
    if set(table['coverage']) != set(COVERAGE_GRID) or table['coverage'].duplicated().any():
        raise ValueError('Regra exige uma linha para cada valor da grade congelada.')
    for column in ('operational_success_rate', 'referral_rate'):
        table[column] = pd.to_numeric(table[column], errors='raise').astype(float)
        if not np.isfinite(table[column]).all() or not table[column].between(0, 1).all():
            raise ValueError(f'{column} deve ser finito e estar em [0,1].')
    if 'power' in table:
        raise ValueError('Campo power ambíguo: use operational_success_rate sobre todas as réplicas.')
    adequate = table[table['operational_success_rate'] >= 0.80]
    if len(adequate):
        # Maior cobertura minimiza a carga de encaminhamento entre opções com poder adequado.
        chosen = adequate.sort_values(
            ['coverage', 'referral_rate'], ascending=[False, True]
        ).iloc[0]
        status = 'selected_by_frozen_rule_power_adequate'
    else:
        # Caminho nulo predefinido: mantém rastreabilidade, mas o poder segue insuficiente.
        chosen = table.sort_values(
            ['operational_success_rate', 'coverage', 'referral_rate'], ascending=[False, False, True]
        ).iloc[0]
        status = 'selected_by_fallback_but_power_insufficient'
    return {
        'rule_version': COVERAGE_RULE_VERSION,
        'coverage_grid_frozen_in_freeze1': list(COVERAGE_GRID),
        'selection_rule_frozen_in_freeze1': True,
        'numeric_coverage_star': float(chosen['coverage']),
        'numeric_value_belongs_to_freeze2': True,
        'selection_scope': 'stabilized_definitive_development_before_C',
        'power_threshold': 0.80,
        'power_measure': 'operational_success_rate',
        'power_denominator': 'all_simulation_replicates_including_invalid',
        'selection_status': status,
        'C_consulted': False,
        'performance_based_model_or_feature_selection': False,
    }


def classificar_deriva_cobertura(
    source_coverage: float, target_coverage: float,
    *, tolerance: float = COVERAGE_DRIFT_TOLERANCE,
) -> dict[str, object]:
    drift = float(target_coverage - source_coverage)
    within = abs(drift) <= tolerance + 1e-12
    return {
        'transfer_rule_version': TRANSFER_RULE_VERSION,
        'source_family': 'openai-gpt-4.1', 'target_family': 'openai-gpt-5.4',
        'source_coverage': float(source_coverage), 'target_coverage': float(target_coverage),
        'coverage_drift_pp': drift * 100,
        'absolute_tolerance_pp': tolerance * 100,
        'tolerance_basis': 'engineering_decision',
        'within_tolerance': within,
        'operational_interpretation': (
            'within_engineering_tolerance' if within
            else 'operationally_inconclusive_due_to_coverage_drift'
        ),
    }


def inventario_sensibilidades() -> dict[str, object]:
    return {
        'secondary_sensitivities_only': list(SENSITIVITY_FEATURE_GROUPS),
        'salt_extra_primary': False,
        'pubmed_frequency_primary': False,
        'pubmed_missing_means_rare': False,
        'pubmed_missing_causes_automatic_abstention': False,
        'o200k_primary': False,
        'o200k_base_relevance': {
            'GPT-4o': 'pertinent', 'GPT-4.1': 'pertinent',
            'GPT-5.4': 'not_confirmed', 'Anthropic': 'not_equivalent',
        },
        'transductive_primary': False,
    }


def registro_familias() -> pd.DataFrame:
    rows = [{
        'model_id': model, 'model_family': family,
        'role': 'approved_family_assignment', 'assignment_status': 'approved_methodologically',
        'exact_snapshot_id': True,
    } for model, family in MODEL_FAMILIES.items()]
    rows.extend({
        'model_id': model, 'model_family': '', 'role': role,
        'assignment_status': 'historical_secondary_reference', 'exact_snapshot_id': True,
    } for model, role in HISTORICAL_REFERENCE_MODELS.items())
    return pd.DataFrame(rows).sort_values('model_id').reset_index(drop=True)


def _bh(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('Valores-p secundários devem ser finitos e estar em [0,1].')
    order = np.argsort(p, kind='mergesort')
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(len(p), dtype=float)
    adjusted[order] = np.clip(ranked, 0, 1)
    return adjusted


def aplicar_multiplicidade(
    rows: pd.DataFrame, *, strict_literal_members: bool = True
) -> pd.DataFrame:
    required = {'family_id', 'member_id', 'p_raw'}
    if missing := sorted(required - set(rows.columns)):
        raise ValueError(f'Multiplicidade sem colunas: {missing}')
    if rows[['family_id', 'member_id']].duplicated().any():
        raise ValueError('Membro duplicado em família de multiplicidade.')
    primary = rows[rows['family_id'].eq('PRIMARY')]
    if len(primary) != 1 or primary.iloc[0]['member_id'] != PRIMARY_MEMBER:
        raise ValueError('Deve existir exatamente o contraste primário literal.')
    if strict_literal_members:
        for family, members in SECONDARY_FAMILY_MEMBERS.items():
            observed = set(rows.loc[rows['family_id'].eq(family), 'member_id'])
            if observed != set(members):
                raise ValueError(f'Membros literais divergentes em {family}.')
        unknown = set(rows['family_id']) - {'PRIMARY', *SECONDARY_FAMILY_MEMBERS}
        if unknown:
            raise ValueError(f'Famílias não registradas: {sorted(unknown)}')
    result = rows.copy()
    result['p_adjusted'] = np.nan
    result['adjustment_method'] = ''
    result['fdr'] = np.nan
    result.loc[primary.index, 'p_adjusted'] = pd.to_numeric(
        primary['p_raw'], errors='raise'
    )
    result.loc[primary.index, 'adjustment_method'] = 'none_single_primary_contrast'
    for family, group in result[~result['family_id'].eq('PRIMARY')].groupby('family_id'):
        result.loc[group.index, 'p_adjusted'] = _bh(group['p_raw'])
        result.loc[group.index, 'adjustment_method'] = 'Benjamini-Hochberg'
        result.loc[group.index, 'fdr'] = 0.05
    result['holm_used'] = False
    result['multiplicity_version'] = MULTIPLICITY_VERSION
    return result


def classificar_resultado(
    *, estimate: float, ci95_low: float, ci95_high: float,
    power_adequate: bool, gate_saturated: bool,
    coverage_drift_absolute: float = 0.0,
) -> dict[str, object]:
    crosses_zero = ci95_low <= 0 <= ci95_high
    unfavorable = estimate > 0
    drift_exceeded = coverage_drift_absolute > COVERAGE_DRIFT_TOLERANCE + 1e-12
    if gate_saturated:
        primary = 'not_informative_gate_saturation'
    elif not power_adequate:
        primary = 'inconclusive_insufficient_power'
    elif drift_exceeded:
        primary = 'operationally_inconclusive_coverage_drift_above_5pp'
    elif unfavorable:
        primary = 'possible_harm_report_with_uncertainty'
    elif crosses_zero:
        primary = 'no_evidence_of_superiority_not_equivalence'
    elif ci95_high < 0:
        primary = 'evidence_compatible_with_superiority'
    else:
        primary = 'no_superiority_unfavorable_interval'
    return {
        'null_path_version': NULL_PATH_VERSION,
        'primary_interpretation': primary,
        'ci95_crosses_zero': crosses_zero,
        'absence_of_superiority_is_equivalence': False,
        'power_insufficient': not power_adequate,
        'gate_saturated': gate_saturated,
        'coverage_drift_above_5pp': drift_exceeded,
        'point_estimate_unfavorable': unfavorable,
        'possible_harm_must_be_reported_with_uncertainty': unfavorable,
        'estimate': estimate, 'ci95_low': ci95_low, 'ci95_high': ci95_high,
    }
