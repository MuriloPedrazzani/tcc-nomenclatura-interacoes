# -*- coding: utf-8 -*-
"""Transferência operacional entre famílias com limiar congelado na fonte."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from datetime import datetime

import pandas as pd

from predicao_seletiva import PoliticaLimiar


MODEL_FAMILY_REGISTRY_VERSION = 'model-family-registry-v1.2'
FAMILY_TRANSFER_VERSION = 'frozen-source-family-transfer-v1.2'
REGISTRY_STATUSES = frozenset({'confirmed', 'pending_team_approval', 'rejected'})
FORBIDDEN_COLUMNS = frozenset({
    'severity_label', 'severity', 'severidade', 'truth', 'verdade', 'label',
    'prediction', 'predicao', 'correct', 'correto', 'performance', 'desempenho',
    'underestimation', 'subestimacao',
})


def validar_registro_familias_legacy_v1_2(registry: pd.DataFrame) -> None:
    required = {
        'model_id', 'model_family', 'candidate_model_family',
        'assignment_status', 'evidence_source', 'registry_version',
        'approved_by', 'approved_at', 'provisional',
    }
    if missing := sorted(required - set(registry.columns)):
        raise ValueError(f'Registro de famílias sem colunas: {missing}')
    if registry['model_id'].isna().any() or registry['model_id'].astype(str).str.strip().eq('').any():
        raise ValueError('model_id é obrigatório no registro de famílias.')
    if registry['model_id'].duplicated().any():
        raise ValueError('Registro possui model_id duplicado.')
    invalid = sorted(set(registry['assignment_status'].astype(str)) - REGISTRY_STATUSES)
    if invalid:
        raise ValueError(f'assignment_status inválido: {invalid}')
    confirmed = registry['assignment_status'].eq('confirmed')
    if (
        confirmed
        & (registry['model_family'].isna() | registry['model_family'].astype(str).str.strip().eq(''))
    ).any():
        raise ValueError('Família confirmada exige model_family.')
    if (
        confirmed
        & (registry['approved_by'].isna() | registry['approved_by'].astype(str).str.strip().eq(''))
    ).any():
        raise ValueError('Família confirmada exige aprovação explícita da equipe.')
    if (
        confirmed
        & (registry['evidence_source'].isna() | registry['evidence_source'].astype(str).str.strip().eq(''))
    ).any():
        raise ValueError('Família confirmada exige fonte de evidência explícita.')
    if registry['registry_version'].isna().any() or registry['registry_version'].astype(str).str.strip().eq('').any():
        raise ValueError('registry_version é obrigatória para todas as linhas.')
    for value in registry.loc[confirmed, 'approved_at'].astype(str):
        try:
            parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError as exc:
            raise ValueError('Família confirmada exige approved_at ISO-8601 válido.') from exc
        if parsed.tzinfo is None:
            raise ValueError('Família confirmada exige approved_at com fuso horário.')


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')
    ).hexdigest()


def _validate_sha(name: str, value: str) -> str:
    value = str(value).lower()
    if len(value) != 64 or any(x not in '0123456789abcdef' for x in value):
        raise ValueError(f'{name} deve ser SHA-256 hexadecimal.')
    return value


def executar_transferencia_familias_legacy_v1_2(
    scores: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    source_families: Sequence[str],
    target_families: Sequence[str],
    target_coverage: float,
    coverage_basis: str,
    feature_version: str,
    feature_sha256: str,
    weights_version: str,
    weights_sha256: str,
    denominator_version: str,
    denominator_sha256: str,
    threshold_version: str = FAMILY_TRANSFER_VERSION,
    calibration_split: str = 'calibration',
    target_splits: Sequence[str] = ('test', 'transfer'),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibra apenas na família-fonte e aplica o mesmo limiar nas famílias-alvo.

    A função não aceita rótulos ou predições clínicas. Alterar escores de uma
    família-alvo não pode modificar o limiar já calibrado na fonte.
    """
    validar_registro_familias_legacy_v1_2(registry)
    if forbidden := sorted(set(scores.columns).intersection(FORBIDDEN_COLUMNS)):
        raise ValueError(f'Transferência recebeu colunas clínicas proibidas: {forbidden}')
    required = {
        'prediction_id', 'pair_id', 'model_id', 'condition', 'run_id', 'split',
        'eligible', 'risk_score',
    }
    if missing := sorted(required - set(scores.columns)):
        raise ValueError(f'scores sem colunas: {missing}')
    if scores['prediction_id'].duplicated().any():
        raise ValueError('prediction_id deve ser único na transferência.')
    if coverage_basis not in {'global', 'eligible_after_gate'}:
        raise ValueError('coverage_basis inválida.')
    if not 0 <= float(target_coverage) <= 1:
        raise ValueError('target_coverage deve estar entre 0 e 1.')
    source = tuple(sorted(set(map(str, source_families))))
    target = tuple(sorted(set(map(str, target_families))))
    if not source or not target or set(source).intersection(target):
        raise ValueError('Famílias fonte/alvo devem ser não vazias e disjuntas.')
    confirmed = registry[registry['assignment_status'].eq('confirmed')]
    known_families = set(confirmed['model_family'].astype(str))
    missing_families = sorted((set(source) | set(target)) - known_families)
    if missing_families:
        raise ValueError(
            'Transferência exige famílias confirmadas e aprovadas; pendentes: '
            f'{missing_families}'
        )
    family_by_model = confirmed.set_index('model_id')['model_family'].astype(str)
    unknown_models = sorted(set(scores['model_id'].astype(str)) - set(family_by_model.index))
    if unknown_models:
        raise ValueError(f'Scores possuem modelos sem família confirmada: {unknown_models}')
    base = scores.copy()
    base['model_family'] = base['model_id'].astype(str).map(family_by_model)
    base['risk_score'] = pd.to_numeric(base['risk_score'], errors='raise')
    if not base['risk_score'].map(math.isfinite).all():
        raise ValueError('risk_score deve ser finito.')
    for name, value in (
        ('feature_sha256', feature_sha256), ('weights_sha256', weights_sha256),
        ('denominator_sha256', denominator_sha256),
    ):
        _validate_sha(name, value)

    threshold_rows: list[dict[str, object]] = []
    application_rows: list[dict[str, object]] = []
    for condition, condition_data in base.groupby('condition', sort=True):
        calibration = condition_data[
            condition_data['model_family'].isin(source)
            & condition_data['split'].astype(str).eq(calibration_split)
        ]
        eligible_cal = calibration[calibration['eligible'].astype(bool)]
        if calibration.empty or eligible_cal.empty:
            raise ValueError(f'Calibração fonte vazia para condição {condition}.')
        if coverage_basis == 'global':
            desired = min(
                len(eligible_cal),
                int(math.floor(len(calibration) * float(target_coverage) + 0.5)),
            )
            calibration_coverage = desired / len(eligible_cal)
        else:
            calibration_coverage = float(target_coverage)
        policy = PoliticaLimiar.calibrar(eligible_cal['risk_score'], calibration_coverage)
        threshold_payload = {
            'transfer_version': FAMILY_TRANSFER_VERSION,
            'threshold_version': threshold_version,
            'condition': str(condition),
            'source_families': list(source),
            'target_families': list(target),
            'calibration_split': calibration_split,
            'coverage_basis': coverage_basis,
            'target_coverage': float(target_coverage),
            'threshold': float(policy.limiar),
            'source_prediction_ids_sha256': _sha(sorted(calibration['prediction_id'].astype(str))),
            'feature_version': feature_version,
            'feature_sha256': feature_sha256,
            'weights_version': weights_version,
            'weights_sha256': weights_sha256,
            'denominator_version': denominator_version,
            'denominator_sha256': denominator_sha256,
        }
        threshold_sha = _sha(threshold_payload)
        threshold_id = f'FAMTHR_{threshold_sha[:20]}'
        threshold_rows.append({
            'threshold_id': threshold_id,
            **threshold_payload,
            'threshold_sha256': threshold_sha,
            'calibration_global_count': int(len(calibration)),
            'calibration_gate_eligible_count': int(len(eligible_cal)),
            'calibration_used_clinical_labels': False,
            'frozen_before_target_application': True,
            'target_recalibration_allowed': False,
            'provisional': True,
        })
        targets = condition_data[
            condition_data['model_family'].isin(target)
            & condition_data['split'].astype(str).isin(set(target_splits))
        ]
        for group_key, group in targets.groupby(
            ['model_family', 'model_id', 'run_id', 'split'], sort=True
        ):
            family, model_id, run_id, split = map(str, group_key)
            eligible = group['eligible'].astype(bool)
            accepted = eligible & group['risk_score'].le(policy.limiar)
            n_global = int(len(group))
            n_eligible = int(eligible.sum())
            n_accepted = int(accepted.sum())
            observed_global = n_accepted / n_global if n_global else 0.0
            observed_conditional = n_accepted / n_eligible if n_eligible else 0.0
            application_rows.append({
                'transfer_application_id': f'FAMAPP_{_sha([threshold_id, family, model_id, run_id, split])[:20]}',
                'threshold_id': threshold_id,
                'condition': str(condition),
                'source_families': '|'.join(source),
                'target_family': family,
                'target_model_id': model_id,
                'target_run_id': run_id,
                'target_split': split,
                'coverage_basis': coverage_basis,
                'target_coverage_from_source_calibration': float(target_coverage),
                'observed_coverage_global': float(observed_global),
                'observed_coverage_eligible_after_gate': float(observed_conditional),
                'coverage_deviation_global': float(observed_global - target_coverage),
                'coverage_deviation_eligible_after_gate': float(
                    observed_conditional - target_coverage
                ),
                'global_count': n_global,
                'gate_eligible_count': n_eligible,
                'accepted_count': n_accepted,
                'threshold_recalibrated_on_target': False,
                'target_labels_used': False,
                'coverage_forced_equal_to_source': False,
                'provisional': True,
            })
    return pd.DataFrame(threshold_rows), pd.DataFrame(application_rows)


APPROVED_REGISTRY_CONTRACT = 'approved-model-registry-v1.3.1-administrative-v1.3.2'
PAIRED_TRANSFER_VERSION = 'paired-source-snapshot-transfer-v1.3.3'
APPROVED_SNAPSHOT_PAIRS = (
    ('gpt-4.1-2025-04-14', 'gpt-5.4-2026-03-05'),
    ('gpt-4.1-mini-2025-04-14', 'gpt-5.4-mini-2026-03-17'),
    ('gpt-4.1-nano-2025-04-14', 'gpt-5.4-nano-2026-03-17'),
)
EXECUTION_PAIR_COLUMNS = (
    'source_model_id', 'target_model_id', 'source_run_id', 'target_run_id',
)


def validar_registro_familias(registry: pd.DataFrame) -> None:
    """Valida o registro vigente sem fabricar evidência/identificadores antigos."""
    from metodos_freeze1_v1_3_1 import registro_familias

    expected = registro_familias()
    required = set(expected.columns)
    if missing := sorted(required - set(registry.columns)):
        raise ValueError(f'Registro vigente sem colunas: {missing}')
    if registry['model_id'].isna().any() or registry['model_id'].duplicated().any():
        raise ValueError('Registro vigente possui model_id ausente ou duplicado.')
    if set(registry['model_id']) != set(expected['model_id']):
        raise ValueError('Registro deve conter os nove snapshots aprovados e a referência histórica.')
    observed = registry.set_index('model_id')
    for row in expected.itertuples(index=False):
        actual = observed.loc[row.model_id]
        for column in ('model_family', 'role', 'assignment_status'):
            value = '' if pd.isna(actual[column]) else str(actual[column])
            if value != getattr(row, column):
                raise ValueError(f'Associação/papel divergente para {row.model_id}: {column}')
        if actual['exact_snapshot_id'] not in (True, 1, 'True', 'true'):
            raise ValueError('Registro exige IDs exatos de snapshot.')


def _execution_pair_records(execution_pairs: pd.DataFrame) -> list[dict[str, str]]:
    if missing := sorted(set(EXECUTION_PAIR_COLUMNS) - set(execution_pairs.columns)):
        raise ValueError(f'Mapeamento explícito de execuções sem colunas: {missing}')
    if execution_pairs.empty:
        raise ValueError('Transferência exige mapeamento explícito de execuções.')
    table = execution_pairs[list(EXECUTION_PAIR_COLUMNS)].copy()
    for column in EXECUTION_PAIR_COLUMNS:
        if table[column].isna().any() or table[column].astype(str).str.strip().eq('').any():
            raise ValueError(f'Mapeamento de execuções com identificador ausente: {column}')
        table[column] = table[column].astype(str).str.strip()
    if table.duplicated().any() or table[['target_model_id', 'target_run_id']].duplicated().any():
        raise ValueError('Execução-alvo duplicada ou associada a mais de uma fonte.')
    for row in table.itertuples(index=False):
        if (row.source_model_id, row.target_model_id) not in APPROVED_SNAPSHOT_PAIRS:
            raise ValueError('Pareamento inválido: exige os pares literais GPT-4.1→GPT-5.4 por porte.')
    return table.sort_values(list(EXECUTION_PAIR_COLUMNS)).to_dict(orient='records')


def executar_transferencia_familias(
    scores: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    execution_pairs: pd.DataFrame,
    target_coverage: float,
    coverage_basis: str,
    policy_state_sha256: str,
    feature_version: str,
    feature_sha256: str,
    weights_version: str,
    weights_sha256: str,
    denominator_version: str,
    denominator_sha256: str,
    source_families: Sequence[str] = ('openai-gpt-4.1',),
    target_families: Sequence[str] = ('openai-gpt-5.4',),
    calibration_split: str = 'calibration',
    target_splits: Sequence[str] = ('C',),
    threshold_version: str = PAIRED_TRANSFER_VERSION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transporta limiares por snapshot e execução explicitamente associados.

    A associação dos run_id é entrada, nunca inferida de resultados nem formada
    por pooling. Sua materialização definitiva pertence ao Freeze 2. Escores
    devem vir de estado previamente validado; o hash vincula suas constantes.
    Retorna manifestos de limiar e resumos de aplicação, incluindo os IDs aceitos.
    """
    from metodos_freeze1_v1_3_1 import classificar_deriva_cobertura

    validar_registro_familias(registry)
    mappings = _execution_pair_records(execution_pairs)
    if tuple(source_families) != ('openai-gpt-4.1',) or tuple(target_families) != ('openai-gpt-5.4',):
        raise ValueError('Direção aprovada é somente openai-gpt-4.1→openai-gpt-5.4.')
    if coverage_basis not in {'global', 'eligible_after_gate'}:
        raise ValueError('coverage_basis inválida.')
    if not math.isfinite(float(target_coverage)) or not 0 <= float(target_coverage) <= 1:
        raise ValueError('target_coverage deve estar entre 0 e 1.')
    if calibration_split != 'calibration' or tuple(target_splits) != ('C',):
        raise ValueError('Contrato vigente usa calibração na fonte e conjunto C no alvo.')
    required = {'prediction_id', 'pair_id', 'model_id', 'condition', 'run_id',
                'split', 'eligible', 'risk_score'}
    if missing := sorted(required - set(scores.columns)):
        raise ValueError(f'scores sem colunas: {missing}')
    forbidden = FORBIDDEN_COLUMNS | {
        'true_severity', 'predicted_severity', 'severity_true', 'severity_pred',
        'severe_eligible', 'severe_underprediction_potential', 'u_delta1', 'u_delta2',
        'accepted', 'decision',
    }
    if present := sorted(forbidden & set(scores.columns)):
        raise ValueError(f'Transferência recebeu desfechos/decisões proibidos: {present}')
    base = scores.copy()
    id_columns = ['prediction_id', 'pair_id', 'model_id', 'condition', 'run_id', 'split']
    for column in id_columns:
        if base[column].isna().any() or base[column].astype(str).str.strip().eq('').any():
            raise ValueError(f'Identificador ausente em scores: {column}')
        base[column] = base[column].astype(str).str.strip()
    if base['prediction_id'].duplicated().any() or base[
        ['model_id', 'condition', 'run_id', 'pair_id']
    ].duplicated().any():
        raise ValueError('Observação duplicada na transferência.')
    allowed_models = {mapping[key] for mapping in mappings
                      for key in ('source_model_id', 'target_model_id')}
    if unknown := sorted(set(base['model_id']) - allowed_models):
        raise ValueError(f'Modelos não autorizados pelo pareamento; referência histórica não é alvo: {unknown}')
    if base['eligible'].isna().any() or not base['eligible'].isin([True, False, 0, 1]).all():
        raise ValueError('eligible deve conter somente indicadores binários completos.')
    base['eligible'] = base['eligible'].astype(bool)
    base['risk_score'] = pd.to_numeric(base['risk_score'], errors='raise').astype(float)
    if not base['risk_score'].map(math.isfinite).all():
        raise ValueError('risk_score deve ser finito.')
    hashes = {name: _validate_sha(name, value) for name, value in (
        ('policy_state_sha256', policy_state_sha256), ('feature_sha256', feature_sha256),
        ('weights_sha256', weights_sha256), ('denominator_sha256', denominator_sha256),
    )}
    if 'policy_state_sha256' in base and not base['policy_state_sha256'].eq(hashes['policy_state_sha256']).all():
        raise ValueError('Escores não correspondem ao estado congelado informado.')
    versions = {'feature_version': feature_version, 'weights_version': weights_version,
                'denominator_version': denominator_version, 'threshold_version': threshold_version}
    if any(not isinstance(value, str) or not value.strip() for value in versions.values()):
        raise ValueError('Versões de configuração/proveniência são obrigatórias.')
    registry_records = registry.sort_values('model_id')[
        ['model_id', 'model_family', 'role', 'assignment_status']
    ].fillna('').to_dict(orient='records')
    configuration = {
        'transfer_version': PAIRED_TRANSFER_VERSION,
        'registry_contract': APPROVED_REGISTRY_CONTRACT,
        'registry_sha256': _sha(registry_records),
        'snapshot_pairs': [list(pair) for pair in APPROVED_SNAPSHOT_PAIRS],
        'execution_pairs': mappings,
        'execution_pairs_sha256': _sha(mappings),
        'calibration_split': calibration_split, 'target_splits': list(target_splits),
        'coverage_basis': coverage_basis, 'target_coverage': float(target_coverage),
        'coverage_drift_tolerance_pp': 5.0,
        **hashes, **versions,
    }
    configuration_sha256 = _sha(configuration)
    threshold_rows, applications = [], []
    used_indices: set[int] = set()
    base = base.reset_index(drop=True)
    for mapping in mappings:
        source = base[
            base['model_id'].eq(mapping['source_model_id'])
            & base['run_id'].eq(mapping['source_run_id'])
            & base['split'].eq(calibration_split)
        ]
        target = base[
            base['model_id'].eq(mapping['target_model_id'])
            & base['run_id'].eq(mapping['target_run_id'])
            & base['split'].isin(target_splits)
        ]
        if source.empty or target.empty:
            raise ValueError('Cada pareamento exige calibração fonte e dados-alvo no conjunto C.')
        if set(source['pair_id']) & set(target['pair_id']):
            raise ValueError('Calibração e C compartilham pair_id; partições devem ser separadas.')
        if set(source['condition']) != set(target['condition']):
            raise ValueError('Condições de fonte/alvo devem corresponder explicitamente.')
        used_indices.update(source.index)
        used_indices.update(target.index)
        for condition in sorted(set(source['condition'])):
            calibration = source[source['condition'].eq(condition)].sort_values('pair_id')
            cal_eligible = calibration[calibration['eligible']]
            if cal_eligible.empty:
                raise ValueError('Calibração fonte sem elegíveis: limiar não identificável.')
            desired = int(math.floor(len(calibration) * target_coverage + 0.5))
            cal_coverage = (min(desired, len(cal_eligible)) / len(cal_eligible)
                            if coverage_basis == 'global' else float(target_coverage))
            policy = PoliticaLimiar.calibrar(cal_eligible['risk_score'], cal_coverage)
            accepted_cal = int(cal_eligible['risk_score'].le(policy.limiar).sum())
            source_global = accepted_cal / len(calibration)
            source_conditional = accepted_cal / len(cal_eligible)
            source_saturated = (coverage_basis == 'global' and desired >= len(cal_eligible)
                                and accepted_cal == len(cal_eligible))
            payload = {
                'transfer_version': PAIRED_TRANSFER_VERSION, 'threshold_version': threshold_version,
                **mapping, 'condition': condition, 'threshold': float(policy.limiar),
                'selection_mode': 'frozen_calibration_threshold',
                'configuration_sha256': configuration_sha256,
                'registry_sha256': configuration['registry_sha256'],
                'execution_pairs_sha256': configuration['execution_pairs_sha256'],
                'source_calibration_observations_sha256': _sha(calibration[
                    ['prediction_id', 'pair_id', 'eligible', 'risk_score']
                ].to_dict(orient='records')),
                **hashes, **versions,
            }
            threshold_sha = _sha(payload)
            threshold_id = f'FAMTHR_{threshold_sha[:20]}'
            threshold_rows.append({
                **payload, 'threshold_id': threshold_id, 'threshold_sha256': threshold_sha,
                'calibration_global_count': len(calibration),
                'calibration_gate_eligible_count': len(cal_eligible),
                'calibration_accepted_count': accepted_cal,
                'observed_source_coverage_global': source_global,
                'observed_source_coverage_eligible_after_gate': source_conditional,
                'coverage_basis': coverage_basis, 'target_coverage': float(target_coverage),
                'source_gate_saturated': source_saturated,
                'source_selector_active': accepted_cal < len(cal_eligible),
                'source_total_abstention': accepted_cal == 0,
                'calibration_used_clinical_labels': False,
                'frozen_before_target_application': True, 'target_recalibration_allowed': False,
                'approval_evidence_required': False, 'approval_identifiers_required': False,
                'standardization_constants_linked_by_policy_state_sha256': True,
                'provisional': True,
            })
            group = target[target['condition'].eq(condition)].sort_values('pair_id')
            eligible = group['eligible']
            accepted = eligible & group['risk_score'].le(policy.limiar)
            n_global, n_eligible, n_accepted = len(group), int(eligible.sum()), int(accepted.sum())
            observed_global = n_accepted / n_global
            observed_conditional = n_accepted / n_eligible if n_eligible else math.nan
            source_coverage = source_global if coverage_basis == 'global' else source_conditional
            target_observed = observed_global if coverage_basis == 'global' else observed_conditional
            drift = classificar_deriva_cobertura(source_coverage, target_observed)
            target_saturated = (coverage_basis == 'global'
                                and int(math.floor(n_global * target_coverage + 0.5)) >= n_eligible
                                and n_accepted == n_eligible)
            saturated = bool(source_saturated or target_saturated)
            informative = n_eligible > 0 and not saturated
            interpretation = ('noninformative_gate_saturation' if saturated
                              else 'noninformative_no_eligible_cases' if not n_eligible
                              else drift['operational_interpretation'])
            applications.append({
                **mapping, 'condition': condition, 'target_split': 'C',
                'threshold_id': threshold_id, 'threshold_sha256': threshold_sha,
                'threshold': float(policy.limiar), 'configuration_sha256': configuration_sha256,
                'policy_state_sha256': hashes['policy_state_sha256'],
                'selection_mode': 'frozen_calibration_threshold', 'coverage_basis': coverage_basis,
                'target_coverage_from_source_calibration': float(target_coverage),
                'observed_source_coverage_global': source_global,
                'observed_source_coverage_eligible_after_gate': source_conditional,
                'observed_coverage_global': observed_global,
                'observed_coverage_eligible_after_gate': observed_conditional,
                'coverage_drift_global_pp': 100 * (observed_global - source_global),
                'coverage_drift_eligible_after_gate_pp': 100 * (observed_conditional - source_conditional),
                'coverage_drift_pp': drift['coverage_drift_pp'],
                'absolute_tolerance_pp': 5.0,
                'coverage_drift_exceeds_tolerance': bool(n_eligible and not drift['within_tolerance']),
                'within_tolerance': bool(n_eligible and drift['within_tolerance']),
                'source_gate_saturated': bool(source_saturated),
                'target_gate_saturated': bool(target_saturated), 'gate_saturated': saturated,
                'source_selector_active': accepted_cal < len(cal_eligible),
                'target_selector_active': n_eligible > 0 and n_accepted < n_eligible,
                'source_total_abstention': accepted_cal == 0,
                'target_total_abstention': n_accepted == 0,
                'transfer_informative': informative,
                'operational_interpretation': interpretation,
                'global_count': n_global, 'gate_eligible_count': n_eligible,
                'accepted_count': n_accepted,
                'accepted_prediction_ids': group.loc[accepted, 'prediction_id'].tolist(),
                'applied_prediction_ids_sha256': _sha(group['prediction_id'].tolist()),
                'threshold_recalibrated_on_target': False, 'target_labels_used': False,
                'coverage_forced_equal_to_source': False, 'provisional': True,
            })
    if used_indices != set(base.index):
        raise ValueError('Scores contêm observações fora dos pareamentos/splits explícitos.')
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.attrs['configuration'] = configuration
    return thresholds, pd.DataFrame(applications)
