# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'dados'))

from esquemas_modulares import drug_id_estavel, par_canonico  # noqa: E402
from features_nomenclatura import (  # noqa: E402
    FeatureConfig,
    extrair_features_nomenclatura,
)


class FeaturesNomenclaturaTest(unittest.TestCase):
    def setUp(self):
        ids = [drug_id_estavel('fixture', x) for x in ('a', 'b', 'c')]
        self.drugs = pd.DataFrame(
            [
                {
                    'drug_id': ids[0], 'source_drug_key': 'a', 'dcb': 'cloridrato de alfa',
                    'inn': 'Alpha', 'dcb_base': 'alfa', 'atc_codes': 'A01AA01',
                    'atc_version': 'fixture', 'mapping_status': 'pending',
                    'mapping_method': 'fuzzy', 'mapping_version': 'v0',
                    'coverage_status': 'covered', 'source_id': 'fixture',
                    'source_version': 'v0', 'provisional': True,
                },
                {
                    'drug_id': ids[1], 'source_drug_key': 'b', 'dcb': 'beta',
                    'inn': 'Beta', 'dcb_base': 'beta', 'atc_codes': 'B01AA01',
                    'atc_version': 'fixture', 'mapping_status': 'validated',
                    'mapping_method': 'exact', 'mapping_version': 'v0',
                    'coverage_status': 'covered', 'source_id': 'fixture',
                    'source_version': 'v0', 'provisional': True,
                },
                {
                    'drug_id': ids[2], 'source_drug_key': 'c', 'dcb': 'gama',
                    'inn': pd.NA, 'dcb_base': 'gama', 'atc_codes': pd.NA,
                    'atc_version': 'fixture', 'mapping_status': 'uncertain',
                    'mapping_method': 'sem_par', 'mapping_version': 'v0',
                    'coverage_status': 'unknown', 'source_id': 'fixture',
                    'source_version': 'v0', 'provisional': True,
                },
            ]
        )
        linhas = []
        for legacy, x, y in [('X1', ids[0], ids[1]), ('X2', ids[0], ids[2])]:
            a, b, pid = par_canonico(x, y)
            linhas.append({
                'pair_id': pid, 'legacy_pair_id': legacy, 'drug_id_a': a,
                'drug_id_b': b, 'source_id': 'fixture',
                'sampling_stratum': 'fixture', 'eligibility_status': 'eligible',
                'selection_probability': 1.0, 'construction_version': 'v0',
                'provisional': True,
            })
        self.pairs = pd.DataFrame(linhas)

    def test_extrator_nao_recebe_rotulos_e_marca_pending(self):
        features, meta = extrair_features_nomenclatura(
            self.drugs, self.pairs, config=FeatureConfig(source_version='fixture')
        )
        self.assertFalse(meta['uses_labels'])
        self.assertNotIn('severity_label', features.columns)
        self.assertTrue(features['nomenclature_mapping_pending_any'].any())
        self.assertTrue(features['provisional'].all())

    def test_ausencia_permanece_nan_com_flag(self):
        features, meta = extrair_features_nomenclatura(self.drugs, self.pairs)
        linha = features.loc[features['nomenclature_missing_any']].iloc[0]
        lados = [
            linha['nomenclature_morphology_units_inn_a'],
            linha['nomenclature_morphology_units_inn_b'],
        ]
        self.assertTrue(any(np.isnan(float(x)) for x in lados))
        self.assertIn('preserve_nan', meta['missing_policy'])
        self.assertEqual(meta['tokenizer_version'], 'not_available')
        self.assertIn('morphology', meta['morphology_version'])

    def test_ordem_das_tabelas_nao_altera_agregados(self):
        a, _ = extrair_features_nomenclatura(self.drugs, self.pairs)
        b, _ = extrair_features_nomenclatura(
            self.drugs.iloc[::-1], self.pairs.iloc[::-1]
        )
        cols = [
            'pair_id', 'nomenclature_delta_morphology_units_pair',
            'nomenclature_lexical_similarity_min',
            'nomenclature_evaluation_pair_frequency_min',
        ]
        pd.testing.assert_frame_equal(
            a[cols].sort_values('pair_id').reset_index(drop=True),
            b[cols].sort_values('pair_id').reset_index(drop=True),
        )

    def test_reutiliza_o200k_offline_e_frequencia_independente(self):
        measurements = pd.DataFrame([
            {'id': 'X1', 'dcb': 8, 'dcb_base': 4, 'inn': 3,
             'd_dcb': 5, 'd_base': 1},
            {'id': 'X2', 'dcb': 9, 'dcb_base': 5, 'inn': 4,
             'd_dcb': 5, 'd_base': 1},
        ])
        frequency = pd.DataFrame([
            {'dcb': 'cloridrato de alfa', 'freq_pubmed': 100},
            {'dcb': 'beta', 'freq_pubmed': 25},
            {'dcb': 'gama', 'freq_pubmed': 4},
        ])
        cfg = FeatureConfig(
            tokenizer_version='o200k_base@tiktoken==0.8.0',
            frequency_source_scope='pubmed_cached_independent_population',
            tokenizer_family_limitations='known_gpt4o_gpt41_proxy_other_families',
        )
        features, meta = extrair_features_nomenclatura(
            self.drugs, self.pairs, config=cfg,
            token_measurements=measurements, frequency_reference=frequency,
        )
        x1 = features.loc[
            features['pair_id'] == self.pairs.iloc[0]['pair_id']
        ].iloc[0]
        self.assertEqual(x1['nomenclature_o200k_delta_tokens_pair'], 5)
        self.assertEqual(x1['nomenclature_independent_frequency_min'], 25)
        self.assertTrue(x1['nomenclature_independent_frequency_available_both'])
        self.assertFalse(x1['nomenclature_independent_frequency_primary_policy_eligible'])
        self.assertEqual(meta['tokenizer_version'], 'o200k_base@tiktoken==0.8.0')
        self.assertEqual(meta['offline_token_measurement_rows'], 2)

    def test_frequencia_dos_pares_e_transdutiva_e_excluida(self):
        features, meta = extrair_features_nomenclatura(self.drugs, self.pairs)
        self.assertTrue(
            features['nomenclature_evaluation_pair_frequency_transductive'].all()
        )
        self.assertFalse(
            features[
                'nomenclature_evaluation_pair_frequency_primary_policy_eligible'
            ].any()
        )
        self.assertTrue(meta['pair_frequency_is_transductive'])
        self.assertFalse(meta['pair_frequency_primary_policy_eligible'])


if __name__ == '__main__':
    unittest.main()
