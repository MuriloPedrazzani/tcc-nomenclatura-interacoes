# -*- coding: utf-8 -*-
"""Candidatos de duas partições de entidades para o pré-Freeze 1 v1.3."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


SPLIT_VERSION = 'two-side-drug-disjoint-candidate-v1.3'
SEED_GENERATOR_VERSION = 'explicit-seed-list-sha256-v1.3'
DEFAULT_SEEDS = (1301, 1302, 1303)
FORBIDDEN = frozenset({
    'severity', 'severidade', 'truth', 'verdade', 'label', 'rotulo', 'target',
    'prediction', 'predicao', 'performance', 'desempenho', 'error', 'erro',
    'underestimation', 'subestimacao', 'significance', 'significancia',
})


@dataclass(frozen=True)
class TwoSideSplitConfig:
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    c_drug_proportion: float = 0.25
    calibration_pair_proportion_within_training: float = 0.20
    min_development_pairs: int = 500
    min_calibration_pairs: int = 100
    min_c_pairs: int = 100

    def __post_init__(self) -> None:
        if len(set(self.seeds)) < 3:
            raise ValueError('Exige pelo menos três sementes candidatas distintas.')
        if not 0 < self.c_drug_proportion < 1:
            raise ValueError('Proporção de medicamentos em C deve estar em (0,1).')
        if not 0 < self.calibration_pair_proportion_within_training < 1:
            raise ValueError('Proporção de calibração deve estar em (0,1).')


def _sha(*parts: object) -> str:
    return hashlib.sha256('\x1f'.join(str(x) for x in parts).encode('utf-8')).hexdigest()


def _reject_labels(frame: pd.DataFrame, context: str) -> None:
    flagged = []
    for column in frame.columns:
        tokens = set(str(column).casefold().replace('-', '_').split('_'))
        if tokens.intersection(FORBIDDEN):
            flagged.append(str(column))
    if flagged:
        raise ValueError(f'{context} contém colunas proibidas: {flagged}')


def _atc_group(value: object) -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        return 'missing'
    groups = sorted({part.strip()[0].upper() for part in str(value).replace(',', '|').split('|') if part.strip()})
    return '|'.join(groups) or 'missing'


def gerar_candidatos_duas_particoes(
    drugs: pd.DataFrame,
    pairs: pd.DataFrame,
    features: pd.DataFrame,
    *,
    balance_features: Sequence[str],
    config: TwoSideSplitConfig | None = None,
) -> dict[str, pd.DataFrame]:
    config = config or TwoSideSplitConfig()
    required_drugs = {'drug_id', 'atc_codes'}
    required_pairs = {'pair_id', 'drug_id_a', 'drug_id_b'}
    if missing := sorted(required_drugs - set(drugs.columns)):
        raise ValueError(f'drugs sem colunas: {missing}')
    if missing := sorted(required_pairs - set(pairs.columns)):
        raise ValueError(f'pairs sem colunas: {missing}')
    if missing := sorted({'pair_id', *balance_features} - set(features.columns)):
        raise ValueError(f'features sem colunas: {missing}')
    _reject_labels(drugs, 'drugs')
    _reject_labels(pairs, 'pairs')
    _reject_labels(features[['pair_id', *balance_features]], 'features de balanceamento')
    if drugs['drug_id'].duplicated().any() or pairs['pair_id'].duplicated().any():
        raise ValueError('drug_id e pair_id devem ser únicos.')
    drug_ids = sorted(drugs['drug_id'].astype(str))
    known = set(drug_ids)
    endpoints = set(pairs['drug_id_a'].astype(str)) | set(pairs['drug_id_b'].astype(str))
    if missing := sorted(endpoints - known):
        raise ValueError(f'Pares com medicamentos órfãos: {missing[:10]}')

    drug_rows, pair_rows, summaries, feature_rows, atc_rows = [], [], [], [], []
    n_c = int(math.floor(len(drug_ids) * config.c_drug_proportion + 0.5))
    drug_meta = drugs[['drug_id', 'atc_codes']].copy()
    drug_meta['atc_group'] = drug_meta['atc_codes'].map(_atc_group)
    overall_atc = drug_meta['atc_group'].value_counts(normalize=True)

    for seed in config.seeds:
        candidate_id = f'TWOSIDE_{_sha(SPLIT_VERSION, seed)[:20]}'
        ordered_drugs = sorted(drug_ids, key=lambda value: _sha(seed, 'drug-side', value))
        c_set = set(ordered_drugs[:n_c])
        side = {drug: ('C' if drug in c_set else 'training_side') for drug in drug_ids}
        assignment_payload = '\n'.join(f'{drug}\t{side[drug]}' for drug in drug_ids)
        assignment_sha = hashlib.sha256(assignment_payload.encode('utf-8')).hexdigest()
        for drug in drug_ids:
            drug_rows.append({
                'candidate_id': candidate_id, 'seed': seed, 'drug_id': drug,
                'entity_side': side[drug], 'assignment_sha256': assignment_sha,
                'assignment_method': 'capacity_balanced_sha256_without_pair_outcomes',
                'seed_generator_version': SEED_GENERATOR_VERSION,
                'uses_labels_or_model_performance': False,
                'definitive': False, 'provisional': True,
            })

        provisional_pairs = []
        for row in pairs[['pair_id', 'drug_id_a', 'drug_id_b']].itertuples(index=False):
            pair_id, a, b = str(row.pair_id), str(row.drug_id_a), str(row.drug_id_b)
            side_a, side_b = side[a], side[b]
            if side_a != side_b:
                partition, reason = 'cross_partition', 'endpoints_cross_training_and_C'
            elif side_a == 'C':
                partition, reason = 'C', 'both_endpoints_in_C'
            else:
                uniform = int(_sha(seed, 'training-pair', pair_id)[:16], 16) / 16**16
                partition = (
                    'calibration'
                    if uniform < config.calibration_pair_proportion_within_training
                    else 'development'
                )
                reason = 'pair_hash_within_training_side'
            item = {
                'candidate_id': candidate_id, 'seed': seed, 'pair_id': pair_id,
                'drug_id_a': a, 'drug_id_b': b,
                'entity_side_a': side_a, 'entity_side_b': side_b,
                'pair_partition': partition, 'assignment_reason': reason,
                'retained': partition != 'cross_partition',
                'uses_labels_or_model_performance': False,
                'definitive': False, 'provisional': True,
            }
            pair_rows.append(item)
            provisional_pairs.append(item)
        pair_frame = pd.DataFrame(provisional_pairs)
        counts = pair_frame['pair_partition'].value_counts()
        training_drugs = known - c_set
        usable = (
            counts.get('development', 0) >= config.min_development_pairs
            and counts.get('calibration', 0) >= config.min_calibration_pairs
            and counts.get('C', 0) >= config.min_c_pairs
        )
        summaries.append({
            'candidate_id': candidate_id, 'seed': seed,
            'training_drug_count': len(training_drugs), 'c_drug_count': len(c_set),
            'development_pair_count': int(counts.get('development', 0)),
            'calibration_pair_count': int(counts.get('calibration', 0)),
            'c_pair_count': int(counts.get('C', 0)),
            'cross_partition_pair_count': int(counts.get('cross_partition', 0)),
            'retained_pair_count': int(pair_frame['retained'].sum()),
            'total_pair_count': len(pair_frame),
            'cross_partition_loss_rate': float((~pair_frame['retained']).mean()),
            'training_c_shared_drug_count': 0,
            'development_calibration_drug_sharing_allowed': True,
            'transfer_entity_partition_created': False,
            'operationally_usable': bool(usable),
            'uses_labels_or_model_performance': False,
            'auto_selected': False, 'definitive': False, 'provisional': True,
        })

        assigned_drugs = drug_meta.assign(
            entity_side=drug_meta['drug_id'].astype(str).map(side)
        )
        for partition in ('training_side', 'C'):
            local = assigned_drugs.loc[
                assigned_drugs['entity_side'].eq(partition), 'atc_group'
            ].value_counts(normalize=True)
            groups = sorted(set(local.index) | set(overall_atc.index))
            distance = 0.5 * sum(
                abs(float(local.get(group, 0)) - float(overall_atc.get(group, 0)))
                for group in groups
            )
            atc_rows.append({
                'candidate_id': candidate_id, 'seed': seed, 'partition': partition,
                'atc_total_variation_distance_vs_all': distance,
                'uses_labels_or_model_performance': False,
            })

        partition_by_pair = pair_frame.set_index('pair_id')['pair_partition']
        feature_base = features[['pair_id', *balance_features]].copy()
        feature_base['pair_partition'] = feature_base['pair_id'].astype(str).map(partition_by_pair)
        for feature in balance_features:
            all_values = pd.to_numeric(features[feature], errors='coerce')
            all_mean, all_std = float(all_values.mean()), float(all_values.std(ddof=1))
            for partition in ('development', 'calibration', 'C'):
                values = pd.to_numeric(
                    feature_base.loc[feature_base['pair_partition'].eq(partition), feature],
                    errors='coerce',
                ).dropna()
                mean = float(values.mean()) if len(values) else math.nan
                feature_rows.append({
                    'candidate_id': candidate_id, 'seed': seed,
                    'partition': partition, 'feature': feature,
                    'n_nonmissing': len(values), 'mean': mean,
                    'std': float(values.std(ddof=1)) if len(values) > 1 else math.nan,
                    'absolute_standardized_mean_difference_vs_all': (
                        abs(mean - all_mean) / all_std if len(values) and all_std > 0 else math.nan
                    ),
                    'uses_labels_or_model_performance': False,
                })

    config_payload = {
        'split_version': SPLIT_VERSION,
        'seed_generator_version': SEED_GENERATOR_VERSION,
        'seeds': list(config.seeds),
        'c_drug_proportion': config.c_drug_proportion,
        'calibration_pair_proportion_within_training': config.calibration_pair_proportion_within_training,
        'tie_breaking': 'lexicographic_sha256',
        'balance_criteria': ['ATC', *balance_features],
        'minimums': {
            'development_pairs': config.min_development_pairs,
            'calibration_pairs': config.min_calibration_pairs,
            'C_pairs': config.min_c_pairs,
        },
        'uses_labels_or_model_performance': False,
        'generated_on_current_mapping_for_demonstration_only': True,
        'regenerate_after_90_mappings_stabilized': True,
        'candidate_not_formally_frozen': True,
    }
    config_payload['config_sha256'] = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True).encode('utf-8')
    ).hexdigest()
    return {
        'drug_assignments': pd.DataFrame(drug_rows),
        'pair_assignments': pd.DataFrame(pair_rows),
        'candidate_summary': pd.DataFrame(summaries),
        'feature_balance': pd.DataFrame(feature_rows),
        'atc_balance': pd.DataFrame(atc_rows),
        'configuration': pd.DataFrame([{
            **config_payload,
            'seeds': '|'.join(map(str, config.seeds)),
            'balance_criteria': '|'.join(['ATC', *balance_features]),
            'minimums': json.dumps(config_payload['minimums'], sort_keys=True),
        }]),
    }
