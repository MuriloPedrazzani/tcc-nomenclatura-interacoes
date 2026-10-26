# -*- coding: utf-8 -*-
"""Extracao feature-only de sinais de risco de nomenclatura.

Por contrato, a funcao publica recebe somente ``drugs`` e ``pairs``. Ela nao
aceita anotacoes, predicoes ou rotulos clinicos.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from esquemas_modulares import identificador_estavel, validar_tabela


FEATURE_VERSION = 'nomenclature-features-v0.2'
MORPHOLOGY_VERSION = 'unicode-wordpunct-morphology-v1'


@dataclass(frozen=True)
class FeatureConfig:
    feature_version: str = FEATURE_VERSION
    tokenizer_version: str = 'not_available'
    morphology_version: str = MORPHOLOGY_VERSION
    feature_source_scope: str = 'pair_entities_with_optional_offline_references'
    frequency_source_scope: str = 'evaluation_pairs_transductive_only'
    tokenizer_family_limitations: str = 'no_llm_tokenizer_measurement_available'
    source_version: str = 'provisional-current-adapter-v0'
    missing_policy: str = 'preserve_nan_and_emit_missing_flag'


def _texto(valor: object) -> str | None:
    if valor is None or pd.isna(valor):
        return None
    normalizado = unicodedata.normalize('NFC', str(valor)).strip().casefold()
    return re.sub(r'\s+', ' ', normalizado) or None


def _morphology_units(valor: object) -> tuple[str, ...] | None:
    texto = _texto(valor)
    if texto is None:
        return None
    return tuple(re.findall(r'\w+|[^\w\s]', texto, flags=re.UNICODE))


def _n_morphology_units(valor: object) -> float:
    units = _morphology_units(valor)
    return float(len(units)) if units is not None else math.nan


def _n_palavras(valor: object) -> float:
    texto = _texto(valor)
    return float(len(re.findall(r'\w+', texto))) if texto is not None else math.nan


def _n_caracteres(valor: object) -> float:
    texto = _texto(valor)
    return float(len(texto)) if texto is not None else math.nan


def _similaridade(a: object, b: object) -> float:
    aa, bb = _texto(a), _texto(b)
    if aa is None or bb is None:
        return math.nan
    return float(SequenceMatcher(None, aa, bb, autojunk=False).ratio())


def _nivel_atc(valor: object) -> float:
    texto = _texto(valor)
    if texto is None:
        return math.nan
    codigo = re.sub(r'[^A-Z0-9]', '', texto.upper())
    if len(codigo) >= 7:
        return 5.0
    if len(codigo) >= 5:
        return 4.0
    if len(codigo) >= 4:
        return 3.0
    if len(codigo) >= 3:
        return 2.0
    if len(codigo) >= 1:
        return 1.0
    return math.nan


def _soma(a: float, b: float) -> float:
    return float(a + b) if np.isfinite(a) and np.isfinite(b) else math.nan


def _minimo(a: float, b: float) -> float:
    return float(min(a, b)) if np.isfinite(a) and np.isfinite(b) else math.nan


def _maximo(a: float, b: float) -> float:
    return float(max(a, b)) if np.isfinite(a) and np.isfinite(b) else math.nan


def _lado(
    drug: pd.Series,
    evaluation_pair_frequency: int,
    independent_frequency: float,
) -> dict[str, object]:
    inn_units = _n_morphology_units(drug['inn'])
    dcb_units = _n_morphology_units(drug['dcb'])
    base_units = _n_morphology_units(drug['dcb_base'])
    extra_sal = (
        max(0.0, dcb_units - base_units)
        if np.isfinite(dcb_units) and np.isfinite(base_units)
        else math.nan
    )
    score = drug.get('mapping_score', np.nan)
    try:
        score = float(score) if not pd.isna(score) and str(score).strip() else math.nan
    except (TypeError, ValueError) as exc:
        raise ValueError(f"mapping_score invalido para {drug['drug_id']}") from exc
    return {
        'morphology_units_inn': inn_units,
        'morphology_units_dcb': dcb_units,
        'delta_morphology_units': (
            dcb_units - inn_units
            if np.isfinite(dcb_units) and np.isfinite(inn_units)
            else math.nan
        ),
        'chars_inn': _n_caracteres(drug['inn']),
        'chars_dcb': _n_caracteres(drug['dcb']),
        'words_inn': _n_palavras(drug['inn']),
        'words_dcb': _n_palavras(drug['dcb']),
        'salt_extra_morphology_units': extra_sal,
        'salt_affix_present': (
            bool(extra_sal > 0) if np.isfinite(extra_sal) else pd.NA
        ),
        'lexical_similarity': _similaridade(drug['dcb'], drug['inn']),
        'evaluation_pair_frequency': float(evaluation_pair_frequency),
        'evaluation_pair_rarity_inverse': 1.0 / float(evaluation_pair_frequency),
        'independent_frequency': independent_frequency,
        'independent_log1p_frequency': (
            float(math.log1p(independent_frequency))
            if np.isfinite(independent_frequency) else math.nan
        ),
        'independent_frequency_missing': not np.isfinite(independent_frequency),
        'mapping_method': str(drug['mapping_method']),
        'mapping_status': str(drug['mapping_status']),
        'mapping_score': score,
        'atc_available': bool(_texto(drug['atc_codes'])),
        'atc_level': _nivel_atc(drug['atc_codes']),
        'name_missing': bool(_texto(drug['dcb']) is None or _texto(drug['inn']) is None),
    }


def extrair_features_nomenclatura(
    drugs: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    config: FeatureConfig | None = None,
    token_measurements: pd.DataFrame | None = None,
    frequency_reference: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Retorna features sem rotulos e metadados de proveniencia."""
    config = config or FeatureConfig()
    validar_tabela('drugs', drugs)
    validar_tabela('pairs', pairs)
    by_id = drugs.set_index('drug_id', verify_integrity=True)
    referencias = pd.concat([pairs['drug_id_a'], pairs['drug_id_b']])
    frequencias = referencias.value_counts().to_dict()
    independent_frequency_by_dcb: dict[str, float] = {}
    if frequency_reference is not None:
        required_frequency = {'dcb', 'freq_pubmed'}
        missing = sorted(required_frequency - set(frequency_reference.columns))
        if missing:
            raise ValueError(f'frequency_reference sem colunas: {missing}')
        freq = frequency_reference.copy()
        freq['dcb_key_normalized'] = freq['dcb'].map(_texto)
        if freq['dcb_key_normalized'].duplicated().any():
            raise ValueError('frequency_reference possui DCB duplicada após normalização.')
        for row in freq.itertuples(index=False):
            value = pd.to_numeric(pd.Series([row.freq_pubmed]), errors='coerce').iloc[0]
            independent_frequency_by_dcb[row.dcb_key_normalized] = (
                float(value) if pd.notna(value) and float(value) >= 0 else math.nan
            )
    token_by_legacy: dict[str, dict[str, float]] = {}
    if token_measurements is not None:
        required_tokens = {'id', 'dcb', 'dcb_base', 'inn', 'd_dcb', 'd_base'}
        missing = sorted(required_tokens - set(token_measurements.columns))
        if missing:
            raise ValueError(f'token_measurements sem colunas: {missing}')
        if token_measurements['id'].astype(str).duplicated().any():
            raise ValueError('token_measurements possui id duplicado.')
        for row in token_measurements.itertuples(index=False):
            token_by_legacy[str(row.id)] = {
                'dcb': float(row.dcb), 'dcb_base': float(row.dcb_base),
                'inn': float(row.inn), 'd_dcb': float(row.d_dcb),
                'd_base': float(row.d_base),
            }
    faltam = sorted(set(referencias.astype(str)) - set(by_id.index.astype(str)))
    if faltam:
        raise ValueError(f'Features referenciam drug_id inexistente: {faltam[:10]}')

    linhas: list[dict[str, object]] = []
    for pair in pairs.itertuples(index=False):
        da = by_id.loc[pair.drug_id_a]
        db = by_id.loc[pair.drug_id_b]
        freq_a = independent_frequency_by_dcb.get(_texto(da['dcb']), math.nan)
        freq_b = independent_frequency_by_dcb.get(_texto(db['dcb']), math.nan)
        a = _lado(da, int(frequencias[pair.drug_id_a]), freq_a)
        b = _lado(db, int(frequencias[pair.drug_id_b]), freq_b)
        token_row = token_by_legacy.get(str(pair.legacy_pair_id))
        linha: dict[str, object] = {
            'feature_id': identificador_estavel(
                'FEAT', pair.pair_id, config.feature_version
            ),
            'pair_id': pair.pair_id,
            'feature_version': config.feature_version,
            'tokenizer_version': config.tokenizer_version,
            'morphology_version': config.morphology_version,
            'feature_source_scope': config.feature_source_scope,
            'frequency_source_scope': config.frequency_source_scope,
            'source_version': config.source_version,
            'available_before_decision': True,
            'provisional': bool(
                pair.provisional
                or da['provisional']
                or db['provisional']
                or a['mapping_status'] != 'validated'
                or b['mapping_status'] != 'validated'
            ),
        }
        for sufixo, valores in (('a', a), ('b', b)):
            for nome, valor in valores.items():
                linha[f'nomenclature_{nome}_{sufixo}'] = valor
        linha.update(
            {
                'nomenclature_delta_morphology_units_pair': _soma(
                    float(a['delta_morphology_units']), float(b['delta_morphology_units'])
                ),
                'nomenclature_morphology_units_dcb_pair': _soma(
                    float(a['morphology_units_dcb']), float(b['morphology_units_dcb'])
                ),
                'nomenclature_morphology_units_inn_pair': _soma(
                    float(a['morphology_units_inn']), float(b['morphology_units_inn'])
                ),
                'nomenclature_salt_extra_morphology_units_pair': _soma(
                    float(a['salt_extra_morphology_units']),
                    float(b['salt_extra_morphology_units'])
                ),
                'nomenclature_lexical_similarity_min': _minimo(
                    float(a['lexical_similarity']), float(b['lexical_similarity'])
                ),
                'nomenclature_lexical_dissimilarity_max': (
                    1.0 - _minimo(
                        float(a['lexical_similarity']), float(b['lexical_similarity'])
                    )
                    if np.isfinite(
                        _minimo(
                            float(a['lexical_similarity']),
                            float(b['lexical_similarity']),
                        )
                    )
                    else math.nan
                ),
                'nomenclature_evaluation_pair_frequency_min': _minimo(
                    float(a['evaluation_pair_frequency']),
                    float(b['evaluation_pair_frequency']),
                ),
                'nomenclature_evaluation_pair_rarity_max': _maximo(
                    float(a['evaluation_pair_rarity_inverse']),
                    float(b['evaluation_pair_rarity_inverse']),
                ),
                'nomenclature_evaluation_pair_frequency_transductive': True,
                'nomenclature_evaluation_pair_frequency_primary_policy_eligible': False,
                'nomenclature_independent_frequency_min': _minimo(
                    float(a['independent_frequency']), float(b['independent_frequency'])
                ),
                'nomenclature_independent_log1p_frequency_min': _minimo(
                    float(a['independent_log1p_frequency']),
                    float(b['independent_log1p_frequency'])
                ),
                'nomenclature_independent_frequency_available_both': bool(
                    not a['independent_frequency_missing']
                    and not b['independent_frequency_missing']
                ),
                # Disponibilidade não autoriza uso na política primária.
                # Freeze 1 v1.3.1 restringe frequência às sensibilidades.
                'nomenclature_independent_frequency_primary_policy_eligible': False,
                'nomenclature_o200k_tokens_dcb_pair': (
                    token_row['dcb'] if token_row else math.nan
                ),
                'nomenclature_o200k_tokens_dcb_base_pair': (
                    token_row['dcb_base'] if token_row else math.nan
                ),
                'nomenclature_o200k_tokens_inn_pair': (
                    token_row['inn'] if token_row else math.nan
                ),
                'nomenclature_o200k_delta_tokens_pair': (
                    token_row['d_dcb'] if token_row else math.nan
                ),
                'nomenclature_o200k_delta_base_tokens_pair': (
                    token_row['d_base'] if token_row else math.nan
                ),
                'nomenclature_o200k_measurement_available': token_row is not None,
                'nomenclature_o200k_family_limitation': config.tokenizer_family_limitations,
                'nomenclature_mapping_pending_any': bool(
                    a['mapping_status'] == 'pending' or b['mapping_status'] == 'pending'
                ),
                'nomenclature_mapping_uncertain_any': bool(
                    a['mapping_status'] == 'uncertain'
                    or b['mapping_status'] == 'uncertain'
                ),
                'nomenclature_mapping_validated_both': bool(
                    a['mapping_status'] == 'validated'
                    and b['mapping_status'] == 'validated'
                ),
                'nomenclature_atc_available_both': bool(
                    a['atc_available'] and b['atc_available']
                ),
                'nomenclature_missing_any': bool(a['name_missing'] or b['name_missing']),
            }
        )
        linhas.append(linha)

    features = pd.DataFrame(linhas)
    if features.empty:
        features = pd.DataFrame(
            columns=[
                'feature_id', 'pair_id', 'feature_version', 'tokenizer_version',
                'morphology_version', 'feature_source_scope', 'frequency_source_scope',
                'source_version', 'available_before_decision', 'provisional',
            ]
        )
    validar_tabela('features', features)
    config_json = json.dumps(asdict(config), sort_keys=True, ensure_ascii=False)
    metadata = {
        'feature_version': config.feature_version,
        'tokenizer_version': config.tokenizer_version,
        'morphology_version': config.morphology_version,
        'feature_source_scope': config.feature_source_scope,
        'frequency_source_scope': config.frequency_source_scope,
        'tokenizer_family_limitations': config.tokenizer_family_limitations,
        'source_version': config.source_version,
        'missing_policy': config.missing_policy,
        'config_sha256': hashlib.sha256(config_json.encode('utf-8')).hexdigest(),
        'rows': int(len(features)),
        'uses_labels': False,
        'symmetric_aggregates': [
            'nomenclature_delta_morphology_units_pair',
            'nomenclature_morphology_units_dcb_pair',
            'nomenclature_morphology_units_inn_pair',
            'nomenclature_salt_extra_morphology_units_pair',
            'nomenclature_lexical_similarity_min',
            'nomenclature_lexical_dissimilarity_max',
            'nomenclature_evaluation_pair_frequency_min',
            'nomenclature_evaluation_pair_rarity_max',
            'nomenclature_independent_frequency_min',
            'nomenclature_o200k_delta_tokens_pair',
        ],
        'pair_frequency_is_transductive': True,
        'pair_frequency_primary_policy_eligible': False,
        'independent_frequency_rows': int(len(frequency_reference))
        if frequency_reference is not None else 0,
        'offline_token_measurement_rows': int(len(token_measurements))
        if token_measurements is not None else 0,
    }
    return features, metadata
