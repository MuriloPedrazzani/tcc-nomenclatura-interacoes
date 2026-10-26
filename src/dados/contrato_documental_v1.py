"""Aceitação documental para pesquisa, separada da validação profissional.

Contrato opt-in da emenda de setembro/2026. Funções puras: não leem fontes,
credenciais, pareceres, gravidades ou respostas e não alteram mapping_status.
Referências identificáveis são ponteiros auditáveis, não comprovação automática
do seu conteúdo. A atribuição documental deve ser sustentada no pacote de origem.
Este módulo avalia somente o mapeamento nominal, nunca a cobertura clínica.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from esquemas_modulares import colunas_alvo_presentes, normalizar_chave


CONTRACT_VERSION = 'documentary-acceptance-v1'
DOCUMENTARY_GATE_VERSION = 'hard-gate-documentary-v2'
PROFESSIONAL_STATUS = 'not_independently_validated'
ACCEPTANCE_COLUMNS = (
    'drug_id', 'contract_version', 'research_acceptance',
    'professional_validation_status', 'decision_method', 'source_refs_json',
    'required_condition_ids_json', 'conflict_status',
)
CONDITION_COLUMNS = ('pair_id', 'drug_id', 'condition_id', 'satisfied', 'evidence_refs_json')
RESULT_COLUMNS = ('pair_id', 'mapping_accepted', 'reasons', 'contract_version')
REFERENCE_COLUMNS = ('pair_id','source_id','source_version','applicability_status','evidence_refs_json')


def _validar_colunas(tabela, required, nome):
    if not isinstance(tabela, pd.DataFrame):
        raise TypeError(f'{nome}: esperado pandas.DataFrame.')
    if tabela.columns.duplicated().any():
        raise ValueError(f'{nome}: nomes de coluna duplicados.')
    faltam = sorted(set(required) - set(tabela.columns))
    if faltam:
        raise ValueError(f'{nome}: colunas ausentes: {faltam}')
    prohibited = colunas_alvo_presentes(tabela.columns)
    extra_targets = {'confidence', 'confianca', 'prediction', 'predicao', 'accuracy',
                     'acuracia', 'performance', 'desempenho', 'probability', 'probabilidade'}
    for name in tabela.columns:
        normalized = normalizar_chave(name).replace(' ', '_')
        if set(re.split(r'[^a-z0-9]+', normalized)) & extra_targets:
            # selection_probability é a probabilidade de amostragem do
            # contrato histórico de pares; não é confiança do modelo.
            if normalized != 'selection_probability':
                prohibited.append(str(name))
    if prohibited:
        raise ValueError(f'{nome}: colunas-alvo/confiança proibidas: {sorted(set(prohibited))}')


def _texto(value, *, campo):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{campo}: texto identificável não vazio obrigatório.')
    return value


def _lista_json(value, *, campo, referencias=False):
    if not isinstance(value, str):
        raise ValueError(f'{campo}: esperado texto JSON com lista de strings.')
    try:
        values = json.loads(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{campo}: JSON inválido.') from exc
    if not isinstance(values, list) or any(not isinstance(x, str) or not x.strip() for x in values):
        raise ValueError(f'{campo}: esperado JSON de localizadores/IDs não vazios.')
    if len(values) != len(set(values)):
        raise ValueError(f'{campo}: referências/IDs duplicados.')
    if referencias:
        for ref in values:
            if not any(mark in ref for mark in ('/', '\\', '#', '://', 'sha256:')) and not ref.startswith(('SRC_', 'DOC_')):
                raise ValueError(f'{campo}: fonte sem localizador ou identificador documental verificável.')
    return values


def validar_aceitacoes_documentais(acceptances):
    """Valida o contrato sem consultar nem derivar validação profissional."""
    _validar_colunas(acceptances, ACCEPTANCE_COLUMNS, 'documentary_acceptance')
    if acceptances['drug_id'].duplicated().any():
        raise ValueError('documentary_acceptance: drug_id duplicado.')
    for row in acceptances.to_dict('records'):
        _texto(row['drug_id'], campo='drug_id')
        for field in ('contract_version', 'professional_validation_status', 'research_acceptance', 'conflict_status'):
            _texto(row[field], campo=field)
        if row['contract_version'] != CONTRACT_VERSION:
            raise ValueError('documentary_acceptance: versão de contrato não suportada.')
        if row['professional_validation_status'] != PROFESSIONAL_STATUS:
            raise ValueError('Este contrato não atribui validação profissional independente.')
        _texto(row['decision_method'], campo='decision_method')
        state = row['research_acceptance']
        if state not in {'accepted', 'conditional', 'uncertain'}:
            raise ValueError('research_acceptance inválida.')
        if row['conflict_status'] not in {'none', 'relevant', 'unknown'}:
            raise ValueError('conflict_status inválido.')
        sources = _lista_json(row['source_refs_json'], campo='source_refs_json', referencias=True)
        required = _lista_json(row['required_condition_ids_json'], campo='required_condition_ids_json')
        if state in {'accepted', 'conditional'}:
            if not sources:
                raise ValueError('Aceitação documental requer fonte identificável, não confiança da IA.')
            if row['conflict_status'] != 'none':
                raise ValueError('Aceitação documental exige conflict_status=none; conflito permanece uncertain.')
        if state == 'accepted' and required:
            raise ValueError('Aceitação com condições deve ser conditional.')
        if state == 'conditional' and not required:
            raise ValueError('Conditional requer IDs de condições explícitas.')


def condicoes_documentais_vazias():
    return pd.DataFrame(columns=CONDITION_COLUMNS)


def validar_presenca_documental(presence):
    if not presence['has_label'].map(lambda x:isinstance(x,(bool,np.bool_))).all():
        raise ValueError('has_label documental exige booleano real; texto não comprova presença.')


def validar_aplicabilidade_documental(reference,pairs):
    """Validate an explicit reference-scope index, never infer it from labels."""
    if reference is None: return {}
    _validar_colunas(reference,REFERENCE_COLUMNS,'documentary_reference_applicability')
    if reference.pair_id.duplicated().any(): raise ValueError('Referência documental: pair_id duplicado.')
    pair_ids=set(pairs.pair_id)
    result={}
    for row in reference.to_dict('records'):
        for key in ('pair_id','source_id','source_version'): _texto(row[key],campo=key)
        if row['pair_id'] not in pair_ids: raise ValueError('Referência documental de par órfão.')
        if row['applicability_status'] not in {'demonstrated','not_assessed','uncertain','inapplicable'}:
            raise ValueError('Situação de aplicabilidade documental inválida.')
        refs=_lista_json(row['evidence_refs_json'],campo='reference.evidence_refs_json',referencias=True)
        result[row['pair_id']]={**row,'scope_demonstrated':row['applicability_status']=='demonstrated' and bool(refs)}
    return result


def avaliar_mapeamento_documental(pairs, acceptances, conditions=None):
    """Aceita cada par apenas se ambos os mapeamentos e condições passarem.

    Todas as entidades/linhas de pares são preservadas. Uma condição precisa
    de bool True real e de referência identificável específica para o par e
    medicamento. Não há comparação por gravidade, confiança ou desempenho.
    Candidatos de referência em campos extras nunca são ativados ou copiados.
    """
    _validar_colunas(pairs, ('pair_id', 'drug_id_a', 'drug_id_b'), 'pairs_documentary')
    validar_aceitacoes_documentais(acceptances)
    if pairs['pair_id'].duplicated().any():
        raise ValueError('pairs_documentary: pair_id duplicado.')
    pair_endpoints = {}
    for row in pairs.to_dict('records'):
        pair_id = _texto(row['pair_id'], campo='pair_id')
        a = _texto(row['drug_id_a'], campo='drug_id_a')
        b = _texto(row['drug_id_b'], campo='drug_id_b')
        if a == b:
            raise ValueError('pairs_documentary: endpoints devem ser distintos.')
        pair_endpoints[pair_id] = (a, b)
    if conditions is None:
        conditions = condicoes_documentais_vazias()
    _validar_colunas(conditions, CONDITION_COLUMNS, 'documentary_conditions')
    if conditions.duplicated(subset=['pair_id', 'drug_id', 'condition_id']).any():
        raise ValueError('documentary_conditions: condição duplicada.')
    acceptance_index = acceptances.set_index('drug_id', verify_integrity=True).to_dict('index')
    condition_index = {}
    for row in conditions.to_dict('records'):
        pair_id = _texto(row['pair_id'], campo='condition.pair_id')
        drug_id = _texto(row['drug_id'], campo='condition.drug_id')
        condition_id = _texto(row['condition_id'], campo='condition.condition_id')
        if pair_id not in pair_endpoints or drug_id not in pair_endpoints[pair_id]:
            raise ValueError('documentary_conditions: par/medicamento órfão ou condição atribuída ao endpoint errado.')
        if drug_id not in acceptance_index:
            raise ValueError('documentary_conditions: aceitação do medicamento ausente.')
        required = _lista_json(acceptance_index[drug_id]['required_condition_ids_json'], campo='required_condition_ids_json')
        if condition_id not in required:
            raise ValueError('documentary_conditions: condição não pertence aos requisitos deste medicamento.')
        if not isinstance(row['satisfied'], (bool, np.bool_)):
            raise ValueError('documentary_conditions: satisfied exige booleano real, nunca texto ou número.')
        refs = _lista_json(row['evidence_refs_json'], campo='evidence_refs_json', referencias=True)
        condition_index[(pair_id, drug_id, condition_id)] = (bool(row['satisfied']), refs)

    result = []
    for pair_id, endpoints in pair_endpoints.items():
        reasons = []
        for drug_id in endpoints:
            acceptance = acceptance_index.get(drug_id)
            if acceptance is None:
                reasons.append(f'{drug_id}:documentary_acceptance_missing')
                continue
            if acceptance['research_acceptance'] == 'uncertain':
                reasons.append(f'{drug_id}:identity_uncertain')
                if acceptance['conflict_status'] != 'none':
                    reasons.append(f'{drug_id}:conflict_{acceptance["conflict_status"]}')
                continue
            if acceptance['research_acceptance'] == 'conditional':
                required = _lista_json(acceptance['required_condition_ids_json'], campo='required_condition_ids_json')
                for condition_id in required:
                    observation = condition_index.get((pair_id, drug_id, condition_id))
                    if observation is None:
                        reasons.append(f'{drug_id}:{condition_id}:condition_missing')
                    elif not observation[0]:
                        reasons.append(f'{drug_id}:{condition_id}:condition_not_satisfied')
                    elif not observation[1]:
                        reasons.append(f'{drug_id}:{condition_id}:condition_evidence_missing')
        result.append(dict(pair_id=pair_id, mapping_accepted=not reasons,
                           reasons='|'.join(reasons), contract_version=CONTRACT_VERSION))
    return pd.DataFrame(result, columns=RESULT_COLUMNS)
