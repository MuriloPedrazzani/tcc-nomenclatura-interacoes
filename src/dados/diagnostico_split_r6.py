"""Diagnóstico estrutural real da regra aprovada, sem selecionar split definitivo.

Reutiliza o gerador histórico; a interface sintética e sua configuração permanecem
inalteradas. A interface desta rodada não aceita sementes, mínimos ou features.
"""
from __future__ import annotations

import pandas as pd

from split_candidato_v1_3_3 import (
    ProposedSplitConfig, _validate_population, especificacao_split_candidato,
)
from split_duas_particoes_v1_3 import _reject_labels, gerar_candidatos_duas_particoes


DIAGNOSTIC_VERSION = 'real-structural-split-diagnostic-r6-v1'
APPROVED_CONFIGURATION_SHA256 = 'd5fdcf18ccbee740d9c879fc2e93e9849ba96f9031c6bbc2b1ac81a732ef8083'
PARTITIONS = ('development', 'calibration', 'C', 'cross_partition')


def auditar_candidato_estrutural(
    drugs: pd.DataFrame, pairs: pd.DataFrame,
    drug_assignments: pd.DataFrame, pair_assignments: pd.DataFrame,
) -> dict[str, object]:
    """Confere a conservação das entradas e a separação efetivamente produzida."""
    violations = []
    known = set(drugs.drug_id)
    expected_pairs = set(pairs.pair_id)
    da, pa = drug_assignments, pair_assignments
    if da.drug_id.duplicated().any() or set(da.drug_id) != known:
        violations.append('entity_universe_not_preserved')
    if pa.pair_id.duplicated().any() or set(pa.pair_id) != expected_pairs:
        violations.append('pair_universe_not_preserved')
    if not da.entity_side.isin(['training_side', 'C']).all():
        violations.append('invalid_entity_side')
    if not pa.pair_partition.isin(PARTITIONS).all():
        violations.append('invalid_pair_partition')
    side = da.drop_duplicates('drug_id').set_index('drug_id').entity_side.to_dict()
    original = pairs.set_index('pair_id')[['drug_id_a', 'drug_id_b']].to_dict('index')
    wrong_endpoints, wrong_sides, wrong_partitions = [], [], []
    for row in pa.itertuples(index=False):
        source = original.get(row.pair_id)
        if source is None or (row.drug_id_a, row.drug_id_b) != (
            source['drug_id_a'], source['drug_id_b']
        ):
            wrong_endpoints.append(row.pair_id)
        a, b = side.get(row.drug_id_a), side.get(row.drug_id_b)
        if a is None or b is None or (row.entity_side_a, row.entity_side_b) != (a, b):
            wrong_sides.append(row.pair_id)
        allowed = ({'cross_partition'} if a != b else
                   {'C'} if a == 'C' else {'development', 'calibration'})
        if row.pair_partition not in allowed:
            wrong_partitions.append(row.pair_id)
    if wrong_endpoints:
        violations.append('pair_endpoints_changed')
    if wrong_sides:
        violations.append('pair_entity_sides_inconsistent')
    if wrong_partitions:
        violations.append('pair_partition_inconsistent_with_entity_sides')
    c_entities = set(da.loc[da.entity_side.eq('C'), 'drug_id'])
    training_entities = set(da.loc[da.entity_side.eq('training_side'), 'drug_id'])
    if shared := c_entities & training_entities:
        violations.append('C_shares_entities_with_training')
    dev = pa[pa.pair_partition.eq('development')]
    calib = pa[pa.pair_partition.eq('calibration')]
    pair_overlap = set(dev.pair_id) & set(calib.pair_id)
    if pair_overlap:
        violations.append('development_calibration_pair_overlap')
    dev_entities = set(dev.drug_id_a) | set(dev.drug_id_b)
    calib_entities = set(calib.drug_id_a) | set(calib.drug_id_b)
    c_rows = pa[pa.pair_partition.eq('C')]
    c_pair_entities = set(c_rows.drug_id_a) | set(c_rows.drug_id_b)
    endpoint_overlap = c_pair_entities & (dev_entities | calib_entities)
    if endpoint_overlap:
        violations.append('C_pair_endpoints_share_entities_with_training_pairs')
    incident = set(pairs.drug_id_a) | set(pairs.drug_id_b)
    counts = {p: int(pa.pair_partition.eq(p).sum()) for p in PARTITIONS}
    return {
        'violations': violations,
        'violation_count': len(violations),
        'entity_count': len(da), 'pair_count': len(pa),
        'partition_counts': counts,
        'isolated_entities_preserved': sorted(known - incident),
        'isolated_entities_in_C': sorted((known - incident) & c_entities),
        'isolated_entities_in_training': sorted((known - incident) & training_entities),
        'training_C_shared_entity_ids': sorted(shared),
        'training_C_shared_pair_endpoint_ids': sorted(endpoint_overlap),
        'development_calibration_shared_pair_ids': sorted(pair_overlap),
        'development_calibration_shared_entity_ids': sorted(dev_entities & calib_entities),
        'development_calibration_entity_sharing_allowed': True,
        'changed_endpoint_pair_ids': sorted(set(wrong_endpoints)),
        'wrong_side_pair_ids': sorted(set(wrong_sides)),
        'wrong_partition_pair_ids': sorted(set(wrong_partitions)),
    }


def diagnosticar_split_estrutural(
    drugs: pd.DataFrame, pairs: pd.DataFrame, *, population_id: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Aplica a sequência aprovada a IDs reais; nenhuma seleção fica congelada."""
    if not isinstance(population_id, str) or not population_id.strip():
        raise ValueError('Identificação explícita da população versionada é obrigatória.')
    for frame, allowed, name in (
        (drugs, {'drug_id', 'atc_codes'}, 'drugs'),
        (pairs, {'pair_id', 'drug_id_a', 'drug_id_b'}, 'pairs'),
    ):
        _reject_labels(frame, name)
        if extra := sorted(set(frame.columns) - allowed):
            raise ValueError(f'{name}: interface estrutural não recebe outras colunas: {extra}')
    entities = drugs.copy()
    if 'atc_codes' not in entities:
        entities['atc_codes'] = None
    feature_keys = pairs[['pair_id']].copy() if 'pair_id' in pairs else pd.DataFrame()
    _validate_population(entities, pairs, feature_keys, ())
    config = ProposedSplitConfig()
    specification = especificacao_split_candidato(config)
    if specification['configuration_sha256'] != APPROVED_CONFIGURATION_SHA256:
        raise ValueError('A configuração histórica divergiu do hash aprovado; diagnóstico bloqueado.')
    generated = gerar_candidatos_duas_particoes(
        entities.sort_values('drug_id'), pairs.sort_values('pair_id'),
        feature_keys.sort_values('pair_id'), balance_features=(), config=config,
    )
    candidate_ids = {
        int(seed): f'CAND_{APPROVED_CONFIGURATION_SHA256[:16]}_{int(seed)}'
        for seed in config.seeds
    }
    tables = {name: generated[name].copy() for name in (
        'drug_assignments', 'pair_assignments', 'candidate_summary', 'atc_balance'
    )}
    old_ids = dict(zip(generated['candidate_summary'].candidate_id,
                       generated['candidate_summary'].seed.map(candidate_ids), strict=True))
    for table in tables.values():
        table['candidate_id'] = table.candidate_id.map(old_ids)
        table['diagnostic_version'] = DIAGNOSTIC_VERSION
        table['dataset_scope'] = 'real_structural_diagnostic'
        table['configuration_sha256'] = APPROVED_CONFIGURATION_SHA256
    audits = {}
    summary = tables['candidate_summary'].copy()
    summary['sequence_position'] = summary.seed.map({s: i + 1 for i, s in enumerate(config.seeds)})
    summary['nonviability_reasons'] = ''
    summary['violation_count'] = 0
    for index, row in summary.iterrows():
        seed = int(row.seed)
        audit = auditar_candidato_estrutural(
            entities, pairs,
            tables['drug_assignments'][tables['drug_assignments'].seed.eq(seed)],
            tables['pair_assignments'][tables['pair_assignments'].seed.eq(seed)],
        )
        audits[str(seed)] = audit
        if audit['violations']:
            raise ValueError(f'Violação estrutural na semente {seed}: {audit["violations"]}')
        reasons = []
        for column, minimum, reason in (
            ('training_drug_count', 1, 'empty_training_entities'),
            ('c_drug_count', 1, 'empty_C_entities'),
            ('development_pair_count', config.min_development_pairs, 'development_pairs_below_minimum'),
            ('calibration_pair_count', config.min_calibration_pairs, 'calibration_pairs_below_minimum'),
            ('c_pair_count', config.min_c_pairs, 'C_pairs_below_minimum'),
        ):
            if row[column] < minimum:
                reasons.append(reason)
        summary.at[index, 'nonviability_reasons'] = '|'.join(reasons)
        summary.at[index, 'operationally_usable'] = not reasons
    summary = summary.sort_values('sequence_position').reset_index(drop=True)
    viable = summary[summary.operationally_usable]
    first = viable.iloc[0] if len(viable) else None
    summary['first_viable_diagnostic'] = summary.seed.eq(int(first.seed)) if first is not None else False
    tables['candidate_summary'] = summary
    report = {
        'diagnostic_version': DIAGNOSTIC_VERSION,
        'dataset_scope': 'real_structural_diagnostic',
        'population_id': population_id,
        'configuration_sha256': APPROVED_CONFIGURATION_SHA256,
        'historical_configuration_unmodified': specification,
        'approval_record': 'docs/protocolo/STATUS_FREEZE1_V1_3_5.md; aprovação v1.3.4',
        'rule_status_in_this_diagnostic': 'APPROVED_BY_TEAM',
        'diagnostic_status': 'FIRST_VIABLE_DIAGNOSTIC' if first is not None else 'NO_VIABLE_CANDIDATE_FAIL_CLOSED',
        'first_viable_diagnostic_seed': int(first.seed) if first is not None else None,
        'first_viable_diagnostic_candidate_id': str(first.candidate_id) if first is not None else None,
        'evaluated_seed_sequence': list(config.seeds),
        'viable_candidate_count': len(viable),
        'entity_count': len(entities), 'candidate_pair_count': len(pairs),
        'automatic_search_expansion': False, 'minimums_relaxed': False,
        'features_used_for_selection': False, 'outcomes_read': False,
        'definitive_seed': None, 'definitive_split': None,
        'FREEZE2_STARTED': False, 'DEFINITION_OF_DONE': False,
        'audits': audits,
    }
    return tables, report
