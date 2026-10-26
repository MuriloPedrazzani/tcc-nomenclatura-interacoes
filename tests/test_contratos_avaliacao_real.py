"""Invented fixtures for future real adapters; never read project outcomes."""
import copy
import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/analise'))
from src.analise.contratos_avaliacao_real import (
    MASK_COLUMNS,PREDICTION_COLUMNS,BINDING,validar_entradas,avaliar_d01_real,pontuar_c)
from src.analise.preparacao_analitica_real import build_analytic_masks
from test_preparacao_analitica_real import fixture as status_fixture
from src.analise.metodos_freeze1_v1_3_1 import ajustar_politica_primaria


def fixture():
    args=status_fixture()
    masks=build_analytic_masks(*args).loc[lambda d:d.pair_partition.eq('C'),sorted(MASK_COLUMNS)].copy()
    predictions=[]
    for row in masks.to_dict('records'):
        p={k:row[k] for k in PREDICTION_COLUMNS if k!='category'}
        p['category']='moderado' if row['pair_id']=='p1' and row['condition']=='dcb' else 'grave'
        predictions.append(p)
    refs=[];contexts=[]
    for row in masks.drop_duplicates('pair_id').to_dict('records'):
        ctx={k:row[k] for k in ['pair_id','drug_id_a','drug_id_b','context_id']}
        ctx.update(ref_id_a=row['drug_id_a']+'ref',ref_id_b=row['drug_id_b']+'ref');contexts.append(ctx)
        refs.append({k:ctx[k] for k in ['pair_id','ref_id_a','ref_id_b','context_id']}|
            dict(source_id=row['reference_source_id'],source_version=row['reference_source_version'],quality_status='usable',canonical_label='grave'))
    registry=args[4].copy()
    return [masks,pd.DataFrame(predictions),pd.DataFrame(refs),registry,pd.DataFrame(contexts)]


def test_join_and_d01_sign_no_selector_and_absent_configuration():
    rows=validar_entradas(*fixture());result=avaliar_d01_real(rows,snapshots=['central','absent'])
    first,absent=result['snapshots'].to_dict('records')
    assert first['denominator_common_severe']==2 and first['estimate_dcb_minus_inn']==.5
    assert absent['estimate_dcb_minus_inn'] is None and absent['status']=='no_planned_configuration'
    assert not result['metadata']['selector_applied'] and not first['equivalence_claim']


def test_line_order_invariance_and_no_input_mutation():
    args=fixture();before=[x.copy(deep=True) for x in args]
    original=validar_entradas(*args);shuffled=validar_entradas(*[x.iloc[::-1] for x in args])
    pd.testing.assert_frame_equal(original,shuffled)
    for a,b in zip(args,before):pd.testing.assert_frame_equal(a,b)


@pytest.mark.parametrize('index',[0,1,2,3,4])
def test_duplicates_rejected(index):
    args=fixture();args[index]=pd.concat([args[index],args[index].iloc[[0]]],ignore_index=True)
    with pytest.raises(ValueError):validar_entradas(*args)


@pytest.mark.parametrize('index',[1,2,4])
def test_orphans_and_missing_rejected(index):
    args=fixture();args[index]=args[index].iloc[1:].copy()
    with pytest.raises(ValueError):validar_entradas(*args)


@pytest.mark.parametrize('field',['pair_id','model','condition','execution_slot','pair_partition','request_id',
                                  'context_id','payload_fingerprint','text_binding_sha256','planned_observation_id'])
def test_prediction_join_mismatch(field):
    args=fixture();args[1].loc[0,field]='wrong'
    with pytest.raises(ValueError):validar_entradas(*args)


@pytest.mark.parametrize('field',['context_id','ref_id_a','source_version','source_id'])
def test_reference_join_mismatch(field):
    args=fixture();args[2].loc[0,field]='wrong'
    with pytest.raises(ValueError):validar_entradas(*args)


def test_zero_grave_denominator_is_not_zero_effect():
    args=fixture();args[2]['canonical_label']='leve'
    result=avaliar_d01_real(validar_entradas(*args),snapshots=['central'])['snapshots'].iloc[0]
    assert result.denominator_common_severe==0 and result.estimate_dcb_minus_inn is None


def test_unavailable_position_remains_planned_not_model_error():
    args=fixture();mask=(args[0].pair_id=='p1')&(args[0].condition=='inn');oid=args[0].loc[mask,'observation_id'].item()
    for k in ['available','structural_response_valid','eligible','eligible_d01_response']:args[0].loc[mask,k]=False
    args[0].loc[mask,'admission_status']='missing_current_cache_key';args[1]=args[1].loc[args[1].observation_id.ne(oid)]
    rows=validar_entradas(*args);assert len(rows)==4 and rows.category.isna().sum()==1
    report=avaliar_d01_real(rows,snapshots=['central'])['snapshots'].iloc[0]
    assert report.planned_pairs==2 and report.denominator_common_severe==1


def test_d01e_three_states_and_main_unaffected():
    args=fixture();rows=validar_entradas(*args)
    assert avaliar_d01_real(rows,snapshots=['central'],documentary_only=True)['snapshots'].iloc[0].status=='documentary_sensitivity_not_evaluable'
    rows.loc[rows.pair_id.eq('p1'),'d01e_eligible']=True
    rows.loc[rows.pair_id.eq('p1'),'d01e_documentary_state']='demonstrated_support'
    rows.loc[rows.pair_id.eq('p2'),'d01e_eligible']=False
    rows.loc[rows.pair_id.eq('p2'),'d01e_documentary_state']='demonstrated_nonfulfillment'
    restricted=avaliar_d01_real(rows,snapshots=['central'],documentary_only=True)['snapshots'].iloc[0]
    assert restricted.denominator_common_severe==1 and restricted.estimate_dcb_minus_inn==1
    assert avaliar_d01_real(rows,snapshots=['central'])['snapshots'].iloc[0].denominator_common_severe==2


def test_unknown_reference_not_invented():
    args=fixture();args[2].loc[0,'quality_status']='conflict';args[2].loc[0,'canonical_label']=''
    mask=args[0].pair_id.eq(args[2].iloc[0].pair_id)
    for k in ['reference_available','eligible','eligible_d01_response']:args[0].loc[mask,k]=False
    rows=validar_entradas(*args);assert rows.severity_label.isna().sum()==2
    assert avaliar_d01_real(rows,snapshots=['central'])['snapshots'].iloc[0].denominator_common_severe==1


@pytest.mark.parametrize('change',['label','status','provenance','eligible','d01e','hash','extra_column'])
def test_invalid_contract_rejected(change):
    args=fixture()
    if change=='label':args[1].loc[0,'category']='minor'
    elif change=='status':args[1].loc[0,'structural_response_valid']=False
    elif change=='provenance':args[0].loc[0,'response_execution_attribution_verified']=True
    elif change=='eligible':args[0].loc[0,'eligible_d01_response']=False
    elif change=='d01e':args[0].loc[0,'d01e_eligible']=True
    elif change=='hash':args[0].loc[0,'request_fingerprint']='not-a-hash'
    else:args[1]['bruto']='forbidden'
    with pytest.raises(ValueError):validar_entradas(*args)


def test_paired_fingerprint_mismatch_rejected_after_validated_join():
    rows=validar_entradas(*fixture());rows.loc[0,'text_binding_sha256']='a'*64
    with pytest.raises(ValueError):avaliar_d01_real(rows,snapshots=['central'])


def test_scoring_frozen_constants_does_not_fit_or_use_labels():
    f=pd.DataFrame(dict(pair_id=['p1','p2'],nomenclature_delta_morphology_units_pair=[0.,1.],
        nomenclature_lexical_dissimilarity_max=[.1,.3]))
    development=f.assign(split='development',nomenclature_morphology_units_dcb_pair=[2,3],nomenclature_morphology_units_inn_pair=[2,2])
    state=ajustar_politica_primaria(development);before=copy.deepcopy(state)
    out=pontuar_c(f,fixture()[0],state)
    assert out.risk_score.tolist()==pytest.approx([-2.,2.]) and before==state
    with pytest.raises(ValueError):pontuar_c(f.iloc[:1],fixture()[0],state)
