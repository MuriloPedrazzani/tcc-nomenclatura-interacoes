"""Regra candidata completa de split; aplicação habilitada somente para sintéticos."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Sequence

import pandas as pd

from split_duas_particoes_v1_3 import (
    TwoSideSplitConfig, gerar_candidatos_duas_particoes,
)


SPLIT_RULE_VERSION = 'first-viable-two-side-split-v1.3.3'
SPLIT_RULE_STATUS = 'PROPOSED_PENDING_TEAM_APPROVAL'


@dataclass(frozen=True)
class ProposedSplitConfig(TwoSideSplitConfig):
    """A ordem da tupla é a ordem de avaliação, nunca uma classificação empírica."""

    def __post_init__(self) -> None:
        if not isinstance(self.seeds, tuple):
            raise ValueError('A sequência de sementes deve ser uma tupla sem duplicatas.')
        if any(isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0
               for seed in self.seeds):
            raise ValueError('Sementes devem ser inteiros não negativos.')
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError('A sequência de sementes deve ser uma tupla sem duplicatas.')
        super().__post_init__()
        for field in ('min_development_pairs', 'min_calibration_pairs', 'min_c_pairs'):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f'{field} deve ser inteiro não negativo.')


def especificacao_split_candidato(
    config: ProposedSplitConfig | None = None,
) -> dict[str, object]:
    config = config or ProposedSplitConfig()
    payload = {
        'rule_version': SPLIT_RULE_VERSION,
        'rule_status': SPLIT_RULE_STATUS,
        **asdict(config),
        'seeds': [int(seed) for seed in config.seeds],
        'population': 'complete_stabilized_unique_identity_and_pair_universe_before_split',
        'identity_rule': 'canonical_nonempty_string_ids_no_silent_merge_or_deduplication',
        'pair_rule': 'unique_unordered_distinct_known_endpoints',
        'missing_identity_or_endpoint': 'fail_before_candidate_generation',
        'diagnostic_missingness': 'report_missing_no_imputation_no_selection_effect',
        'allocation': 'sha256(seed,drug-side,drug_id); round_half_up_C_capacity',
        'allocation_tie_break': 'lexicographic_drug_id',
        'training_pair_assignment': 'first_16_sha256_hex_divided_by_16_power_16_below_calibration_proportion',
        'viability': 'nonempty_entity_sides_and_all_three_pair_count_minimums',
        'candidate_selection': 'first_viable_in_explicit_seed_sequence',
        'candidate_tie_break': 'sequence_position_is_unique_no_secondary_ranking',
        'balance_diagnostics_used_for_selection': False,
        'no_viable_candidate': 'NO_VIABLE_CANDIDATE_FAIL_CLOSED_no_expansion_or_relaxation',
        'real_application_enabled': False,
        'definitive_seed': None,
        'definitive_split': None,
    }
    # Use JSON round-trip so returned content is exactly what was hashed.
    payload = json.loads(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    payload['configuration_sha256'] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
    ).encode('utf-8')).hexdigest()
    return payload


def _validate_ids(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    for column in columns:
        if column not in frame:
            raise ValueError(f'{name}: coluna obrigatória ausente: {column}')
        if not frame[column].map(
            lambda value: isinstance(value, str) and bool(value)
            and value == value.strip() and not any(c in value for c in ('\t', '\n', '\r', '\x1f'))
        ).all():
            raise ValueError(f'{name}.{column}: IDs devem ser strings canônicas não vazias.')


def _validate_population(drugs: pd.DataFrame, pairs: pd.DataFrame,
                         features: pd.DataFrame, balance_features: Sequence[str]) -> None:
    if len(drugs) < 2 or pairs.empty:
        raise ValueError('População exige pelo menos dois medicamentos e um par.')
    _validate_ids(drugs, ('drug_id',), 'drugs')
    _validate_ids(pairs, ('pair_id', 'drug_id_a', 'drug_id_b'), 'pairs')
    _validate_ids(features, ('pair_id',), 'features')
    for frame, column in ((drugs, 'drug_id'), (pairs, 'pair_id'), (features, 'pair_id')):
        if frame[column].duplicated().any():
            raise ValueError(f'Identificador duplicado: {column}; nenhuma deduplicação silenciosa.')
    if drugs['drug_id'].str.casefold().duplicated().any():
        raise ValueError('drug_id ambíguo após casefold usado pela inferência diádica.')
    if pairs['drug_id_a'].eq(pairs['drug_id_b']).any():
        raise ValueError('Autoarestas não são permitidas.')
    dyads = pairs.apply(lambda row: tuple(sorted((row.drug_id_a, row.drug_id_b))), axis=1)
    if dyads.duplicated().any():
        raise ValueError('Díade duplicada, inclusive endpoints invertidos.')
    if set(features['pair_id']) != set(pairs['pair_id']):
        raise ValueError('Features devem representar exatamente o universo de pair_id.')
    if len(balance_features) != len(set(balance_features)) or 'pair_id' in balance_features:
        raise ValueError('Features de diagnóstico devem ser distintas e não incluir pair_id.')


def propor_split_duas_particoes(
    drugs: pd.DataFrame, pairs: pd.DataFrame, features: pd.DataFrame, *,
    balance_features: Sequence[str] = (),
    config: ProposedSplitConfig | None = None,
    dataset_scope: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Demonstra mecanicamente a proposta sem selecionar split definitivo real.

    A aprovação futura exige alteração explícita deste bloqueio de aplicação real;
    a aprovação administrativa v1.3.2 não autoriza esta nova regra.
    """
    if dataset_scope != 'synthetic':
        raise ValueError('Regra proposta permite apenas sintéticos; aplicação real aguarda aprovação/Freeze 2.')
    config = config or ProposedSplitConfig()
    if not isinstance(config, ProposedSplitConfig):
        raise TypeError('Use ProposedSplitConfig para a regra candidata versionada.')
    _validate_population(drugs, pairs, features, balance_features)
    specification = especificacao_split_candidato(config)
    diagnostic_features = features.copy()
    for name in balance_features:
        if name not in diagnostic_features:
            raise ValueError(f'Feature de diagnóstico ausente: {name}')
        numeric = pd.to_numeric(diagnostic_features[name], errors='coerce')
        diagnostic_features[name] = numeric.where(numeric.map(math.isfinite))
    candidates = gerar_candidatos_duas_particoes(
        drugs.sort_values('drug_id'), pairs.sort_values('pair_id'),
        diagnostic_features.sort_values('pair_id'),
        balance_features=balance_features, config=config,
    )
    candidate_ids = {
        row.candidate_id: f'CAND_{specification["configuration_sha256"][:16]}_{int(row.seed)}'
        for row in candidates['candidate_summary'].itertuples(index=False)
    }
    for table in candidates.values():
        if 'candidate_id' in table:
            table['candidate_id'] = table['candidate_id'].map(candidate_ids)
            table['configuration_sha256'] = specification['configuration_sha256']
            table['rule_version'] = SPLIT_RULE_VERSION
            table['rule_status'] = SPLIT_RULE_STATUS
    candidates['generator_configuration'] = candidates['configuration']
    candidates['configuration'] = pd.DataFrame([specification])
    summary = candidates['candidate_summary'].copy()
    summary['sequence_position'] = summary['seed'].map(
        {seed: index + 1 for index, seed in enumerate(config.seeds)}
    )
    summary['operationally_usable'] &= (
        summary['training_drug_count'].gt(0) & summary['c_drug_count'].gt(0)
    )
    def failure_reasons(row: pd.Series) -> str:
        reasons = []
        for count_column, minimum, reason in (
            ('training_drug_count', 1, 'empty_training_entities'),
            ('c_drug_count', 1, 'empty_C_entities'),
            ('development_pair_count', config.min_development_pairs, 'development_pairs_below_minimum'),
            ('calibration_pair_count', config.min_calibration_pairs, 'calibration_pairs_below_minimum'),
            ('c_pair_count', config.min_c_pairs, 'C_pairs_below_minimum'),
        ):
            if row[count_column] < minimum:
                reasons.append(reason)
        return '|'.join(reasons)
    summary['nonviability_reasons'] = summary.apply(failure_reasons, axis=1)
    summary = summary.sort_values('sequence_position').reset_index(drop=True)
    viable = summary[summary['operationally_usable']]
    chosen = viable.iloc[0] if len(viable) else None
    summary['selected_for_synthetic_demonstration'] = (
        summary['candidate_id'].eq(chosen['candidate_id']) if chosen is not None else False
    )
    summary['rule_status'] = SPLIT_RULE_STATUS
    summary['configuration_sha256'] = specification['configuration_sha256']
    candidates['candidate_summary'] = summary
    report = {
        'rule_version': SPLIT_RULE_VERSION,
        'rule_status': SPLIT_RULE_STATUS,
        'configuration': specification,
        'configuration_sha256': specification['configuration_sha256'],
        'dataset_scope': dataset_scope,
        'selection_status': ('SYNTHETIC_FIRST_VIABLE_CANDIDATE' if chosen is not None
                             else 'NO_VIABLE_CANDIDATE_FAIL_CLOSED'),
        'synthetic_candidate_id': None if chosen is None else str(chosen['candidate_id']),
        'synthetic_candidate_seed': None if chosen is None else int(chosen['seed']),
        'evaluated_seed_sequence': list(config.seeds),
        'viable_candidate_count': len(viable),
        'automatic_search_expansion': False,
        'minimums_relaxed': False,
        'definitive_seed': None,
        'definitive_split': None,
        'FREEZE2_STARTED': False,
    }
    return candidates, report
