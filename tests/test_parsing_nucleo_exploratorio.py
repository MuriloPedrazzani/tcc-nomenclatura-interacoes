"""Small complete pipeline with invented cache values, including planned losses."""
import hashlib
import json
import sqlite3

import pytest

from src.dados.admissao_metadados_exploratoria import construir_matriz_admissao, parear_admissao
from src.dados.leitor_cache_delimitado import RestrictCacheReader
from src.dados.parsing_nucleo_exploratorio import process_queue, inventories


def fixture(tmp_path):
    snapshots=[dict(model='fixture',provider='openai',family='fixture',role='fixture')]
    population=[dict(pair_id=p,drug_id_a=p+'a',drug_id_b=p+'b',benchmark_candidate=True,
                     pair_partition='C',context_id=p+'ctx') for p in ['p1','p2','p3']]
    observations=[]
    for pair in population:
        for arm in ['dcb','inn']:
            oid=pair['pair_id']+arm
            observations.append(dict(observation_id=oid,request_id=oid,pair_id=pair['pair_id'],model='fixture',
                condition=arm,execution_slot='historico_a_identificar',context_id=pair['context_id'],
                request_fingerprint='f'*64,payload_fingerprint='e'*64,text_binding_sha256='b'*64,
                legacy_key=hashlib.sha256(oid.encode()).hexdigest(),binding_verified=True,cache_audited=True,
                cache_current_present=pair['pair_id']!='p3',classification='indeterminada',
                historical_use_allowed_with_caveat=True,submission_execution_ids=[],
                response_execution_id_observed=None,complete_configuration_observed=False))
    matrix=construir_matriz_admissao(population,snapshots,observations)
    queue=[dict(**{k:r[k] for k in ['observation_id','pair_id','model','condition','pair_partition','execution_slot']},
                cache_key=r['legacy_key']) for r in matrix if r['parsing_candidate']]
    path=tmp_path/'fixture.sqlite'
    c=sqlite3.connect(path);c.execute('create table r(k text primary key,v text)')
    for r in matrix:
        if not r['parsing_candidate']: continue
        raw=json.dumps(['leve',json.dumps({'severidade':'leve'})])
        if r['observation_id']=='p2inn': raw=json.dumps([None,'ERRO: invented detail'])
        c.execute('insert into r values(?,?)',(r['legacy_key'],raw))
    c.commit();c.close()
    return matrix,queue,path,hashlib.sha256(path.read_bytes()).hexdigest()


def test_pipeline_retains_invalid_unpaired_and_unread_planned_rows(tmp_path):
    matrix,queue,path,digest=fixture(tmp_path)
    with RestrictCacheReader(path,[r['cache_key'] for r in queue],digest) as reader:
        result=process_queue(reader,queue,{r['observation_id']:r for r in matrix})
    full,summaries,pairs,pairsummary=inventories(matrix,parear_admissao(matrix),result)
    assert len(result)==4 and sum(x['structural_response_valid'] for x in result)==3
    assert len(full)==6 and sum(x['parsed_this_round'] for x in full)==4
    assert sum(x['invalid'] for x in summaries)==1
    assert pairsummary[0]['pairs_before']==2 and pairsummary[0]['pairs_after']==1
    assert pairsummary[0]['single_valid']==1 and pairsummary[0]['neither_valid']==1
    assert all(x['historical_caveat_preserved'] and not x['analytical_eligibility_final'] for x in result)
    assert 'invented detail' not in json.dumps([result,full,summaries,pairs,pairsummary])
    assert all('category' not in r for r in full+summaries+pairs+pairsummary)
    assert reader.report()['attempted_unique_keys']==4 and reader.report()['cache_unchanged']


@pytest.mark.parametrize('change',['duplicate','wrong_key','ambiguous','unverified'])
def test_bad_queue_stops_before_any_value_read(tmp_path,change):
    matrix,queue,path,digest=fixture(tmp_path)
    if change=='duplicate': queue.append(queue[0].copy())
    if change=='wrong_key': queue[-1]['cache_key']='c'*64
    if change=='ambiguous': matrix[-3]['submission_execution_ids']=['x','y']
    if change=='unverified': matrix[0]['binding_verified']=False
    class NoRead:
        def read_value(self,key): raise AssertionError('Preflight failure must precede value read')
    with pytest.raises(ValueError): process_queue(NoRead(),queue,{r['observation_id']:r for r in matrix})


def test_absent_entry_is_not_reclassified_as_provider_failure(tmp_path):
    matrix,queue,path,digest=fixture(tmp_path)
    connection=sqlite3.connect(path)
    connection.execute('delete from r where k=?',(queue[0]['cache_key'],)); connection.commit(); connection.close()
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with RestrictCacheReader(path,[x['cache_key'] for x in queue],digest) as reader:
        result=process_queue(reader,queue,{x['observation_id']:x for x in matrix})
    missing=result[0]
    assert missing['parsing']['status']=='cache_entry_absent'
    assert missing['parsing']['history_absence'] and not missing['parsing']['collector_failure_observed']
    assert missing['category'] is None and not missing['structural_response_valid']
