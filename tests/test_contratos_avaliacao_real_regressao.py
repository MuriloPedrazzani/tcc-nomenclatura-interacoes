"""Independent tiny fixtures for reference identities and predecision types."""
import pandas as pd
import pytest

from test_contratos_avaliacao_real import fixture
from src.analise.contratos_avaliacao_real import validar_entradas,pontuar_c
from src.analise.metodos_freeze1_v1_3_1 import ajustar_politica_primaria


@pytest.mark.parametrize('failure',['empty','missing','same_reference_endpoint'])
def test_usable_reference_requires_two_literal_distinct_reference_endpoints(failure):
    masks,predictions,references,registry,contexts=fixture()
    if failure=='empty':
        references.loc[0,['ref_id_a','ref_id_b']]=''
        contexts.loc[0,['ref_id_a','ref_id_b']]=''
    elif failure=='missing':
        references.loc[0,'ref_id_a']=None
        contexts.loc[0,'ref_id_a']=None
    else:
        references.loc[0,'ref_id_b']=references.loc[0,'ref_id_a']
        contexts.loc[0,'ref_id_b']=contexts.loc[0,'ref_id_a']
    # Matching two projections is insufficient if the joined reference itself
    # cannot represent a queryable pair of distinct benchmark entities.
    with pytest.raises(ValueError):
        validar_entradas(masks,predictions,references,registry,contexts)


@pytest.mark.parametrize('feature',['nomenclature_delta_morphology_units_pair',
                                    'nomenclature_lexical_dissimilarity_max'])
def test_scoring_rejects_boolean_primary_feature_before_numeric_coercion(feature):
    features=pd.DataFrame(dict(pair_id=['p1','p2'],nomenclature_delta_morphology_units_pair=[0.,1.],
                               nomenclature_lexical_dissimilarity_max=[.1,.3]))
    development=features.assign(split='development',nomenclature_morphology_units_dcb_pair=[2.,3.],
                                nomenclature_morphology_units_inn_pair=[2.,2.])
    state=ajustar_politica_primaria(development)
    features[feature]=features[feature].astype(object)
    features.loc[0,feature]=True
    with pytest.raises(ValueError):pontuar_c(features,fixture()[0],state)
