# -*- coding: utf-8 -*-
"""Contratos tabulares e identificadores estaveis da arquitetura modular.

O modulo nao le nem escreve arquivos. Validadores recebem ``DataFrame`` e
falham cedo, antes que um artefato provisório possa ser confundido com uma
versao confirmatoria.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd


SCHEMA_VERSION = 'modular-schema-v1.3.3'

MAPPING_STATUSES = frozenset({'pending', 'validated', 'rejected', 'uncertain'})
COVERAGE_STATUSES = frozenset({'covered', 'uncovered', 'label_missing', 'unknown'})
EVIDENCE_STATUSES = frozenset(
    {'sufficient', 'insufficient', 'conflicting', 'not_assessed'}
)
PARSE_STATUSES = frozenset({'valid', 'invalid', 'missing'})
REVIEW_STATUSES = frozenset({'pending', 'completed', 'adjudication_required'})
SPLITS = frozenset({'development', 'calibration', 'C', 'test', 'transfer'})
SPLIT_STATES = frozenset({'S0', 'S1', 'S2', 'S3', 'unassigned'})
ELIGIBILITY_STATUSES = frozenset({'eligible', 'ineligible'})
ABSTENTION_REASONS = frozenset(
    {
        'accepted',
        'abstain_schema',
        'abstain_mapping',
        'abstain_coverage',
        'abstain_insufficient_evidence',
        'abstain_nomenclature',
        'abstain_policy',
    }
)
POLICY_DECISIONS = frozenset({'accepted', 'abstain'})
SELECTION_MODES = frozenset({'batch_fixed_quota', 'frozen_calibration_threshold'})
SEVERITY_LABELS = frozenset({'leve', 'moderado', 'grave'})

_TARGET_NAMES = frozenset(
    {
        'verdade', 'truth', 'severity', 'severidade', 'severity_label',
        'rotulo', 'label', 'target', 'y', 'erro', 'erro_observado',
        'observed_error', 'subestimacao', 'underestimation', 'acerto',
        'correto', 'correct',
    }
)


@dataclass(frozen=True)
class TableContract:
    required: tuple[str, ...]
    non_null: tuple[str, ...]
    unique: tuple[tuple[str, ...], ...] = ()
    vocabularies: Mapping[str, frozenset[str]] | None = None
    forbidden_targets: bool = False


CONTRACTS: dict[str, TableContract] = {
    'drugs': TableContract(
        required=(
            'drug_id', 'source_drug_key', 'dcb', 'inn', 'dcb_base',
            'atc_codes', 'atc_version', 'mapping_status', 'mapping_method',
            'mapping_version', 'coverage_status', 'source_id',
            'source_version', 'provisional',
        ),
        non_null=(
            'drug_id', 'source_drug_key', 'mapping_status', 'mapping_version',
            'coverage_status', 'source_id', 'source_version', 'provisional',
        ),
        unique=(('drug_id',), ('source_id', 'source_drug_key')),
        vocabularies={
            'mapping_status': MAPPING_STATUSES,
            'coverage_status': COVERAGE_STATUSES,
        },
    ),
    'pairs': TableContract(
        required=(
            'pair_id', 'legacy_pair_id', 'drug_id_a', 'drug_id_b', 'source_id',
            'sampling_stratum', 'eligibility_status', 'selection_probability',
            'construction_version', 'provisional',
        ),
        non_null=(
            'pair_id', 'drug_id_a', 'drug_id_b', 'source_id',
            'eligibility_status', 'construction_version', 'provisional',
        ),
        unique=(('pair_id',), ('drug_id_a', 'drug_id_b')),
        vocabularies={'eligibility_status': ELIGIBILITY_STATUSES},
        forbidden_targets=True,
    ),
    'annotations': TableContract(
        required=(
            'annotation_id', 'pair_id', 'source_id', 'annotator_id',
            'evidence_status', 'severity_label', 'confidence', 'source_version',
            'review_status', 'annotated_at', 'provisional',
        ),
        non_null=(
            'annotation_id', 'pair_id', 'source_id', 'evidence_status',
            'source_version', 'review_status', 'provisional',
        ),
        unique=(('annotation_id',),),
        vocabularies={
            'evidence_status': EVIDENCE_STATUSES,
            'severity_label': SEVERITY_LABELS,
            'review_status': REVIEW_STATUSES,
        },
    ),
    'adjudications': TableContract(
        required=(
            'adjudication_id', 'pair_id', 'annotation_ids', 'adjudicator_id',
            'evidence_status', 'severity_label', 'confidence',
            'disagreement_reason', 'protocol_version', 'adjudicated_at',
            'provisional',
        ),
        non_null=(
            'adjudication_id', 'pair_id', 'annotation_ids', 'evidence_status',
            'protocol_version', 'provisional',
        ),
        unique=(('adjudication_id',),),
        vocabularies={
            'evidence_status': EVIDENCE_STATUSES,
            'severity_label': SEVERITY_LABELS,
        },
    ),
    'predictions': TableContract(
        required=(
            'prediction_id', 'pair_id', 'provider', 'model', 'model_snapshot',
            'condition', 'run_id', 'prediction', 'confidence', 'parse_status',
            'prompt_hash', 'parameters_hash', 'cache_key',
            'collection_version', 'collected_at', 'provisional',
        ),
        non_null=(
            'prediction_id', 'pair_id', 'provider', 'model', 'condition',
            'run_id', 'parse_status', 'collection_version', 'provisional',
        ),
        unique=(('prediction_id',), ('pair_id', 'model', 'condition', 'run_id')),
        vocabularies={
            'parse_status': PARSE_STATUSES,
            'prediction': SEVERITY_LABELS,
        },
        forbidden_targets=True,
    ),
    'features': TableContract(
        required=(
            'feature_id', 'pair_id', 'feature_version', 'tokenizer_version',
            'morphology_version', 'feature_source_scope', 'frequency_source_scope',
            'source_version', 'available_before_decision', 'provisional',
        ),
        non_null=(
            'feature_id', 'pair_id', 'feature_version', 'tokenizer_version',
            'morphology_version', 'feature_source_scope', 'frequency_source_scope',
            'source_version', 'available_before_decision', 'provisional',
        ),
        unique=(('feature_id',), ('pair_id', 'feature_version')),
        forbidden_targets=True,
    ),
    'split_manifest': TableContract(
        required=(
            'split_assignment_id', 'pair_id', 'split', 'seed', 'split_hash',
            'state', 'seen_drug_a', 'seen_drug_b', 'reason', 'version',
            'locked', 'provisional',
        ),
        non_null=(
            'split_assignment_id', 'pair_id', 'split', 'seed', 'split_hash',
            'state', 'version', 'locked', 'provisional',
        ),
        unique=(('split_assignment_id',), ('pair_id',)),
        vocabularies={'split': SPLITS, 'state': SPLIT_STATES},
    ),
    'eligibility_checks': TableContract(
        required=(
            'check_id', 'pair_id', 'prediction_id', 'eligible',
            'eligibility_status', 'gate_type', 'abstention_reason',
            'secondary_reasons', 'coverage_status', 'rule_version',
            'checked_at', 'provisional',
        ),
        non_null=(
            'check_id', 'pair_id', 'prediction_id', 'eligible', 'eligibility_status',
            'gate_type', 'abstention_reason', 'coverage_status',
            'rule_version', 'provisional',
        ),
        unique=(('check_id',),),
        vocabularies={
            'eligibility_status': ELIGIBILITY_STATUSES,
            'abstention_reason': ABSTENTION_REASONS,
            'coverage_status': COVERAGE_STATUSES,
        },
        forbidden_targets=True,
    ),
    'policy_evaluations': TableContract(
        required=(
            'policy_evaluation_id', 'pair_id', 'prediction_id', 'policy_name',
            'policy_version', 'model', 'condition', 'run_id', 'split',
            'evaluation_group_id', 'selection_mode', 'coverage_basis',
            'coverage_target_feasible', 'selector_active',
            'gate_type', 'eligibility_status',
            'abstention_reason', 'risk_score', 'threshold', 'target_coverage',
            'observed_coverage_global', 'observed_coverage_eligible_after_gate',
            'coverage_target_deviation_global',
            'coverage_target_deviation_eligible_after_gate',
            'gate_eligible_count', 'selector_accepted_count', 'final_accepted_count',
            'decision', 'label_source', 'label_version', 'provisional',
        ),
        non_null=(
            'policy_evaluation_id', 'pair_id', 'prediction_id', 'policy_name', 'policy_version',
            'model', 'condition', 'run_id', 'split', 'evaluation_group_id',
            'selection_mode', 'coverage_basis', 'coverage_target_feasible',
            'selector_active', 'eligibility_status', 'abstention_reason',
            'target_coverage', 'observed_coverage_global',
            'observed_coverage_eligible_after_gate',
            'coverage_target_deviation_global',
            'coverage_target_deviation_eligible_after_gate', 'gate_eligible_count',
            'selector_accepted_count', 'final_accepted_count', 'decision', 'provisional',
        ),
        unique=(('policy_evaluation_id',),),
        vocabularies={
            'eligibility_status': ELIGIBILITY_STATUSES,
            'abstention_reason': ABSTENTION_REASONS,
            'decision': POLICY_DECISIONS,
            'coverage_basis': frozenset({'global', 'eligible_after_gate'}),
            'selection_mode': SELECTION_MODES,
            'split': SPLITS,
        },
    ),
    'source_versions': TableContract(
        required=(
            'source_version_id', 'source_id', 'version', 'path', 'sha256',
            'license_status', 'retrieved_at', 'recorded_at', 'provisional',
        ),
        non_null=(
            'source_version_id', 'source_id', 'version', 'path', 'sha256',
            'license_status', 'recorded_at', 'provisional',
        ),
        unique=(('source_version_id',), ('source_id', 'version', 'sha256')),
    ),
}


def normalizar_chave(valor: object) -> str:
    """Normaliza uma chave-fonte sem alterar o valor armazenado na tabela."""
    texto = unicodedata.normalize('NFC', str(valor)).casefold().strip()
    return re.sub(r'\s+', ' ', texto)


def identificador_estavel(prefixo: str, *partes: object, tamanho: int = 20) -> str:
    """Gera ID opaco a partir de chaves imutaveis explicitamente fornecidas."""
    if not prefixo or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', prefixo):
        raise ValueError('Prefixo de identificador invalido.')
    normalizadas = [normalizar_chave(x) for x in partes]
    if not normalizadas or any(not x for x in normalizadas):
        raise ValueError('Identificador estavel exige partes nao vazias.')
    digest = hashlib.sha256('\x1f'.join(normalizadas).encode('utf-8')).hexdigest()
    return f'{prefixo}_{digest[:tamanho]}'


def drug_id_estavel(source_id: str, source_drug_key: str) -> str:
    """ID de fármaco; corrigir DCB/INN nao o altera se a chave for preservada."""
    return identificador_estavel('DRUG', source_id, source_drug_key)


def par_canonico(drug_id_a: str, drug_id_b: str) -> tuple[str, str, str]:
    """Retorna endpoints ordenados e um ``pair_id`` invariante a A/B."""
    a, b = sorted((str(drug_id_a).strip(), str(drug_id_b).strip()))
    if not a or not b:
        raise ValueError('Par exige dois drug_id nao vazios.')
    if a == b:
        raise ValueError('Auto-par de medicamento nao e permitido.')
    return a, b, identificador_estavel('PAIR', a, b)


def _ausente(serie: pd.Series) -> pd.Series:
    return serie.isna() | serie.astype(str).str.strip().eq('')


def colunas_alvo_presentes(colunas: Sequence[object]) -> list[str]:
    """Detecta nomes-alvo exatos, inclusive variantes com prefixo/sufixo."""
    encontrados: list[str] = []
    for original in colunas:
        nome = normalizar_chave(original).replace(' ', '_')
        tokens = set(re.split(r'[^a-z0-9]+', nome))
        if nome in _TARGET_NAMES or tokens.intersection(_TARGET_NAMES):
            encontrados.append(str(original))
    return encontrados


def validar_tabela(nome: str, tabela: pd.DataFrame) -> None:
    """Valida uma tabela isolada contra o contrato versionado."""
    if nome not in CONTRACTS:
        raise KeyError(f'Tabela modular desconhecida: {nome}')
    if not isinstance(tabela, pd.DataFrame):
        raise TypeError(f'{nome} deve ser um pandas.DataFrame.')
    contrato = CONTRACTS[nome]
    faltantes = sorted(set(contrato.required) - set(tabela.columns))
    if faltantes:
        raise ValueError(f'{nome}: colunas obrigatorias ausentes: {faltantes}')
    for coluna in contrato.non_null:
        if _ausente(tabela[coluna]).any():
            linhas = tabela.index[_ausente(tabela[coluna])].tolist()[:5]
            raise ValueError(f'{nome}.{coluna}: valor ausente nas linhas {linhas}')
    for chave in contrato.unique:
        duplicada = tabela.duplicated(list(chave), keep=False)
        if duplicada.any():
            exemplos = tabela.loc[duplicada, list(chave)].head(5).to_dict('records')
            raise ValueError(f'{nome}: chave duplicada {chave}: {exemplos}')
    for coluna, vocabulario in (contrato.vocabularies or {}).items():
        serie = tabela[coluna]
        # Uma resposta com parsing invalido permanece bruta/invalida; seu texto
        # nao e promovido a classe, mas tambem nao impede o registro historico.
        if nome == 'predictions' and coluna == 'prediction':
            serie = serie[tabela['parse_status'].eq('valid')]
        valores = serie.dropna().astype(str).str.strip()
        valores = valores[valores.ne('')]
        invalidos = sorted(set(valores) - set(vocabulario))
        if invalidos:
            raise ValueError(
                f'{nome}.{coluna}: valores fora do vocabulario: {invalidos[:10]}'
            )
    if contrato.forbidden_targets:
        protegidas = set(contrato.required)
        alvos = colunas_alvo_presentes([c for c in tabela.columns if c not in protegidas])
        if alvos:
            raise ValueError(f'{nome}: colunas-alvo proibidas: {alvos}')

    if nome == 'pairs' and len(tabela):
        a = tabela['drug_id_a'].astype(str)
        b = tabela['drug_id_b'].astype(str)
        if (a >= b).any():
            raise ValueError('pairs: endpoints devem ser distintos e estar em ordem canonica.')
    if nome in {'annotations', 'adjudications'} and len(tabela):
        suficiente = tabela['evidence_status'].eq('sufficient')
        sem_rotulo = _ausente(tabela['severity_label'])
        if (suficiente & sem_rotulo).any():
            raise ValueError(f'{nome}: evidencia suficiente exige severity_label.')
    if nome == 'predictions' and len(tabela):
        valido = tabela['parse_status'].eq('valid')
        if (valido & _ausente(tabela['prediction'])).any():
            raise ValueError('predictions: parse valido exige prediction.')
    if nome == 'eligibility_checks' and len(tabela):
        esperado = tabela['eligible'].astype(bool).map(
            {True: 'eligible', False: 'ineligible'}
        )
        if not esperado.equals(tabela['eligibility_status'].astype(str)):
            raise ValueError('eligibility_checks: eligible e eligibility_status divergem.')
        motivo_aceito = tabela['abstention_reason'].eq('accepted')
        if not motivo_aceito.equals(tabela['eligible'].astype(bool)):
            raise ValueError('eligibility_checks: accepted deve equivaler a eligible=true.')
    if nome == 'policy_evaluations' and len(tabela):
        aceito = tabela['decision'].eq('accepted')
        motivo_aceito = tabela['abstention_reason'].eq('accepted')
        if not aceito.equals(motivo_aceito):
            raise ValueError('policy_evaluations: decisao e motivo primario divergem.')
        for coluna in ('coverage_target_feasible', 'selector_active'):
            if not tabela[coluna].map(lambda x: isinstance(x, bool)).all():
                raise ValueError(f'policy_evaluations.{coluna} deve ser booleano.')
        frozen = tabela['selection_mode'].eq('frozen_calibration_threshold')
        if (frozen & _ausente(tabela['threshold'])).any():
            raise ValueError('policy_evaluations: limiar congelado exige threshold.')
    if nome == 'source_versions' and len(tabela):
        hashes = tabela['sha256'].astype(str)
        if not hashes.str.fullmatch(r'[0-9a-fA-F]{64}').all():
            raise ValueError('source_versions.sha256 deve conter SHA-256 hexadecimal.')
        paths = tabela['path'].astype(str).str.replace('\\', '/', regex=False)
        if paths.str.match(r'^(?:[A-Za-z]:/|/)').any():
            raise ValueError('source_versions.path deve ser relativo à raiz do projeto.')


def validar_arquitetura(tabelas: Mapping[str, pd.DataFrame]) -> None:
    """Valida contratos e referências entre todas as camadas presentes."""
    for nome, tabela in tabelas.items():
        validar_tabela(nome, tabela)
    if 'drugs' in tabelas and 'pairs' in tabelas:
        drugs = set(tabelas['drugs']['drug_id'].astype(str))
        refs = set(tabelas['pairs']['drug_id_a'].astype(str)) | set(
            tabelas['pairs']['drug_id_b'].astype(str)
        )
        faltam = sorted(refs - drugs)
        if faltam:
            raise ValueError(f'pairs referencia drug_id inexistente: {faltam[:10]}')
    if 'pairs' in tabelas:
        pair_ids = set(tabelas['pairs']['pair_id'].astype(str))
        for nome in (
            'annotations', 'adjudications', 'predictions', 'features',
            'split_manifest', 'eligibility_checks', 'policy_evaluations',
        ):
            if nome in tabelas:
                refs = set(tabelas[nome]['pair_id'].astype(str))
                faltam = sorted(refs - pair_ids)
                if faltam:
                    raise ValueError(f'{nome} referencia pair_id inexistente: {faltam[:10]}')
    if 'annotations' in tabelas and 'adjudications' in tabelas:
        annotations = set(tabelas['annotations']['annotation_id'].astype(str))
        for valor in tabelas['adjudications']['annotation_ids'].dropna():
            refs = {x.strip() for x in str(valor).split('|') if x.strip()}
            faltam = sorted(refs - annotations)
            if faltam:
                raise ValueError(
                    f'adjudications referencia annotation_id inexistente: {faltam[:10]}'
                )
    if 'predictions' in tabelas:
        predictions = tabelas['predictions']
        pred_pair = predictions.set_index('prediction_id')['pair_id'].astype(str)
        for nome in ('eligibility_checks', 'policy_evaluations'):
            if nome not in tabelas:
                continue
            refs = tabelas[nome][['prediction_id', 'pair_id']].astype(str)
            faltam = sorted(set(refs['prediction_id']) - set(pred_pair.index))
            if faltam:
                raise ValueError(
                    f'{nome} referencia prediction_id inexistente: {faltam[:10]}'
                )
            esperado = refs['prediction_id'].map(pred_pair)
            divergentes = refs.loc[~refs['pair_id'].eq(esperado), 'prediction_id'].tolist()
            if divergentes:
                raise ValueError(
                    f'{nome}: pair_id diverge da predição correspondente: '
                    f'{divergentes[:10]}'
                )
        if 'policy_evaluations' in tabelas:
            policy = tabelas['policy_evaluations']
            prediction_meta = predictions.set_index('prediction_id')[
                ['model', 'condition', 'run_id']
            ].astype(str)
            refs = policy[['prediction_id', 'model', 'condition', 'run_id']].astype(str)
            for coluna in ('model', 'condition', 'run_id'):
                esperado = refs['prediction_id'].map(prediction_meta[coluna])
                divergentes = refs.loc[~refs[coluna].eq(esperado), 'prediction_id'].tolist()
                if divergentes:
                    raise ValueError(
                        f'policy_evaluations.{coluna} diverge da predição referenciada: '
                        f'{divergentes[:10]}'
                    )
            group_meta = [
                'policy_name', 'policy_version', 'selection_mode', 'model',
                'condition', 'run_id', 'split', 'coverage_basis', 'target_coverage',
            ]
            for group_id, group in policy.groupby('evaluation_group_id', sort=True):
                if group.duplicated(['prediction_id', 'pair_id']).any():
                    raise ValueError(
                        f'policy_evaluations: decisões duplicadas no grupo {group_id}.'
                    )
                for coluna in group_meta:
                    if group[coluna].nunique(dropna=False) != 1:
                        raise ValueError(
                            f'policy_evaluations: {coluna} varia no grupo {group_id}.'
                        )
                primeira = group.iloc[0]
                esperado_id = identificador_estavel(
                    'EVALGRP', primeira['policy_name'], primeira['policy_version'],
                    primeira['selection_mode'], primeira['model'], primeira['condition'],
                    primeira['run_id'], primeira['split'], primeira['coverage_basis'],
                )
                if str(group_id) != esperado_id:
                    raise ValueError(
                        f'policy_evaluations: evaluation_group_id incoerente: {group_id}.'
                    )
                total = len(group)
                elegiveis = int(group['eligibility_status'].eq('eligible').sum())
                finais = int(group['decision'].eq('accepted').sum())
                if finais > elegiveis:
                    raise ValueError(
                        f'policy_evaluations: aceitos excedem elegíveis no grupo {group_id}.'
                    )
                global_observada = float(finais / total) if total else 0.0
                condicional = float(finais / elegiveis) if elegiveis else 0.0
                target = float(primeira['target_coverage'])
                if primeira['coverage_basis'] == 'global':
                    desejado = int(math.floor(total * target + 0.5))
                    feasible = desejado <= elegiveis
                else:
                    feasible = True
                active = bool(elegiveis > 0 and finais < elegiveis)
                checks = {
                    'gate_eligible_count': elegiveis,
                    'selector_accepted_count': finais,
                    'final_accepted_count': finais,
                }
                for coluna, calculado in checks.items():
                    valores = pd.to_numeric(group[coluna], errors='raise')
                    if not valores.eq(calculado).all():
                        raise ValueError(
                            f'policy_evaluations.{coluna} incompatível com decisões '
                            f'no grupo {group_id}.'
                        )
                floats = {
                    'observed_coverage_global': global_observada,
                    'observed_coverage_eligible_after_gate': condicional,
                    'coverage_target_deviation_global': global_observada - target,
                    'coverage_target_deviation_eligible_after_gate': condicional - target,
                }
                for coluna, calculado in floats.items():
                    valores = pd.to_numeric(group[coluna], errors='raise')
                    if not valores.map(lambda x: math.isclose(
                        float(x), calculado, rel_tol=0.0, abs_tol=1e-12
                    )).all():
                        raise ValueError(
                            f'policy_evaluations.{coluna} incompatível com decisões '
                            f'no grupo {group_id}.'
                        )
                if not group['coverage_target_feasible'].eq(bool(feasible)).all():
                    raise ValueError(
                        'policy_evaluations.coverage_target_feasible incompatível '
                        f'no grupo {group_id}.'
                    )
                if not group['selector_active'].eq(active).all():
                    raise ValueError(
                        f'policy_evaluations.selector_active incompatível no grupo {group_id}.'
                    )
        if 'source_versions' in tabelas and len(predictions):
            versions = set(tabelas['source_versions']['version'].astype(str))
            faltam = sorted(set(predictions['collection_version'].astype(str)) - versions)
            if faltam:
                raise ValueError(
                    'predictions.collection_version sem source_versions correspondente: '
                    f'{faltam[:10]}'
                )


def tabela_vazia(nome: str) -> pd.DataFrame:
    """Cria uma tabela vazia com todas as colunas obrigatorias do contrato."""
    if nome not in CONTRACTS:
        raise KeyError(f'Tabela modular desconhecida: {nome}')
    return pd.DataFrame(columns=list(CONTRACTS[nome].required))
