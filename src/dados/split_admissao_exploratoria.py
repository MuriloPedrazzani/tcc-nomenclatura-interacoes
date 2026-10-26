"""Materialização autorizada do split real na passagem exploratória de metadados.

Reutiliza integralmente a regra estrutural aprovada. A seleção documentada aqui
não congela entradas nem libera parsing, gravidades ou avaliação. Interfaces
sintéticas e registros históricos permanecem intactos.
"""
from __future__ import annotations

import pandas as pd

from diagnostico_split_r6 import diagnosticar_split_estrutural


ADMISSION_SPLIT_VERSION = 'real-metadata-first-viable-split-v1'
DATASET_SCOPE = 'real_metadata_admission'


def materializar_split_admissao(
    drugs: pd.DataFrame, pairs: pd.DataFrame, *, population_id: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Seleciona o primeiro viável sem parâmetros ajustáveis ou dados de resposta.

    ``drugs`` recebe apenas ``drug_id`` e ``pairs`` apenas os três IDs. Todas as
    entidades e todos os pares fornecidos são conservados; isolados e cruzados
    continuam explícitos. Sem candidato viável, não existe seleção materializada.
    """
    for frame, allowed, name in (
        (drugs, {'drug_id'}, 'drugs'),
        (pairs, {'pair_id', 'drug_id_a', 'drug_id_b'}, 'pairs'),
    ):
        if extra := sorted(set(frame.columns) - allowed):
            raise ValueError(f'{name}: admissão recebe somente IDs; outras colunas: {extra}')
    diagnostics, diagnostic_report = diagnosticar_split_estrutural(
        drugs, pairs, population_id=population_id,
    )
    seed = diagnostic_report['first_viable_diagnostic_seed']
    candidate_id = diagnostic_report['first_viable_diagnostic_candidate_id']
    tables = {name: frame.copy() for name, frame in diagnostics.items()}
    for frame in tables.values():
        frame['admission_split_version'] = ADMISSION_SPLIT_VERSION
        frame['dataset_scope'] = DATASET_SCOPE
        frame['selected_for_admission'] = (
            frame.candidate_id.eq(candidate_id) if candidate_id is not None else False
        )
        if 'auto_selected' in frame:
            frame['auto_selected'] = frame['selected_for_admission']
    for name in ('drug_assignments', 'pair_assignments'):
        tables[f'selected_{name}'] = tables[name].loc[
            tables[name].selected_for_admission
        ].copy().reset_index(drop=True)
    report = {
        **diagnostic_report,
        'admission_split_version': ADMISSION_SPLIT_VERSION,
        'dataset_scope': DATASET_SCOPE,
        'selection_status': ('SELECTED_FIRST_VIABLE_FOR_METADATA_ADMISSION'
                             if seed is not None else 'NO_VIABLE_CANDIDATE_FAIL_CLOSED'),
        'selection_materialized': seed is not None,
        'selected_seed': seed,
        'selected_candidate_id': candidate_id,
        'selected_pair_count': len(tables['selected_pair_assignments']),
        'selected_entity_count': len(tables['selected_drug_assignments']),
        'selection_rule': 'first_viable_in_explicit_seed_sequence',
        'approval_records': [
            'docs/protocolo/STATUS_FREEZE1_V1_3_5.md',
            'docs/protocolo/EMENDA_ENQUADRAMENTO_EXPLORATORIO_2026-10-03.md',
            'docs/estado/PLANO_ADMISSAO_HISTORICOS_EXPLORATORIO_2026-10-03.md',
        ],
        'execution_authority': 'user_request_for_first_metadata_admission_pass',
        'historical_configuration_record_preserved': True,
        'definitive_seed': None,
        'definitive_split': None,
        'FREEZE2_STARTED': False,
        'FREEZE2_COMPLETE': False,
        'DEFINITION_OF_DONE': False,
    }
    return tables, report
