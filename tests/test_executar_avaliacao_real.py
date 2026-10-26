"""Future file runner exercised only in isolated invented fixture directories."""
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.analise import executar_avaliacao_real as runner
from src.analise.contratos_avaliacao_real import PREDICTION_COLUMNS, REFERENCE_COLUMNS
from test_avaliacao_real_exploratoria import fixture, unavailable


def write_json(path, value):
    path.write_text(json.dumps(runner._plain(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_lines(path, rows):
    path.write_text(''.join(json.dumps(runner._plain(r), ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')


def pin(root, path):
    return dict(path=path.relative_to(root).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def fixture_files(tmp_path):
    root = tmp_path / 'invented-project'
    root.mkdir()
    freeze = root / 'freeze_v1'
    freeze.mkdir()
    args = fixture()
    # Preserve planned missing positions and source category exclusion.
    unavailable(args, args['masks'].model.eq('gpt-4o-mini-2024-07-18'))
    write_lines(freeze / 'mascaras_C_contrato.jsonl', args['masks'].to_dict('records'))
    for name, key in [('REGISTRO_SNAPSHOTS.json', 'registry'), ('CONTEXTOS_C.json', 'contexts'),
                      ('FEATURES_C_SEM_ESCORE.json', 'features'), ('LIMIARES_VINCULADOS_GLOBAL_24.json', 'thresholds'),
                      ('CONTEXTOS_CALIBRACAO_SEM_CATEGORIAS.json', 'source_pair_metadata')]:
        write_json(freeze / name, args[key])
    write_json(freeze / 'CONSTANTES_CANDIDATAS.json', {'state': args['state']})
    with (freeze / 'entrada_calibracao_sem_categorias.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(args['source_scores']), delimiter=';')
        writer.writeheader()
        writer.writerows(args['source_scores'].to_dict('records'))
    config = dict(split_seed=1302, tie_seed=20261003, coverage_grid=[.6, .7, .8, .9],
        policy_bases=args['policy_bases'], transfer_source_basis='global', transfer_target_basis='global',
        transfer_comparator='<=', transfer_tolerance_pp=5, transfer_tolerance_inclusive=True,
        no_inference=True, no_coverage_winner=True, no_independent_historical_replicates=True, DEFINITION_OF_DONE=False)
    write_json(freeze / 'CONFIGURACAO_FIXADA.json', config)
    predictions = root / 'REGISTROS_NORMALIZADOS.jsonl'
    # This outsider lacks all clinical/binding columns except identity. Projecting
    # it before filtering would fail. No clinical interpretation is needed.
    invented_predictions = args['predictions'].to_dict('records') + [
        dict(observation_id='fixture-non-C-calibration', category={'do_not_project': True})]
    for row in invented_predictions:
        row['unused_raw_like_field'] = 'invented ignored extra field'
    write_lines(predictions, invented_predictions)
    references = root / 'reference_labels.csv'
    fields = list(args['references']) + ['unused_evidence_field']
    with references.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=';')
        writer.writeheader()
        writer.writerows([dict(**r, unused_evidence_field='ignored') for r in args['references'].to_dict('records')])
        writer.writerow(dict(pair_id='fixture-outside-C', canonical_label='not-a-label', unused_evidence_field='ignored'))
    approval = root / runner.APPROVAL_NAME
    approval.write_text('Invented explicit approval fixture.\n', encoding='utf-8')
    clinical_pins = [pin(root, predictions), pin(root, references)]
    contract = dict(version='future-real-exploratory-reading-v1', evaluation_authorized=False,
        metadata_sources=runner.METADATA_SOURCES, prediction_fields=sorted(PREDICTION_COLUMNS),
        reference_fields=sorted(REFERENCE_COLUMNS), source_categories_required=False,
        source_reference_severities_required=False, additional_cache_keys=0,
        clinical_files=clinical_pins, exact_planned_C_observations=len(args['masks']),
        exact_available_C_observations=int(args['masks'].available.sum()), exact_C_pairs=args['masks'].pair_id.nunique())
    write_json(freeze / 'CONTRATO_LEITURA_FUTURA.json', contract)
    # Binding uses the actual approval file digest, so rebuild only fixture
    # bindings. No data are tagged synthetic to bypass a production safeguard.
    approval_sha = pin(root, approval)['sha256']
    bound = args['thresholds']
    from src.analise.transferencia_real_exploratoria import bind_thresholds
    new_bound = bind_thresholds(pd.DataFrame(bound.candidate_payload.tolist(), dtype=object),
        pd.DataFrame(bound.source_calculation_payload.tolist(), dtype=object),
        approval_sha256=approval_sha, candidate_artifact_sha256='c' * 64,
        source_calculation_artifact_sha256='e' * 64)['thresholds']
    write_json(freeze / 'LIMIARES_VINCULADOS_GLOBAL_24.json', new_bound)
    env = dict(root=root, freeze=freeze, status=root / 'STATUS_FREEZE2.json', output=root / 'future-results',
               approval=approval, predictions=predictions, references=references)
    seal(env)
    return env


def seal(env):
    root, freeze = env['root'], env['freeze']
    clinical_pins = [pin(root, env['predictions']), pin(root, env['references'])]
    contract_path = freeze / 'CONTRATO_LEITURA_FUTURA.json'
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    contract['clinical_files'] = clinical_pins
    write_json(contract_path, contract)
    artifacts = [pin(root, p) for p in sorted(freeze.iterdir()) if p.name != 'MANIFESTO_CONGELAMENTO.json']
    manifest = dict(version='exploratory-freeze2-inputs-v1', inputs=[pin(root, env['approval'])],
        artifacts=artifacts, clinical_pins=clinical_pins, not_self_authorizing_evaluation=True,
        DEFINITION_OF_DONE=False)
    manifest_path = freeze / 'MANIFESTO_CONGELAMENTO.json'
    write_json(manifest_path, manifest)
    write_json(env['status'], dict(FREEZE2_COMPLETE=True, freeze_manifest_sha256=pin(root, manifest_path)['sha256'],
        approval_sha256=pin(root, env['approval'])['sha256'], DEFINITION_OF_DONE=False))


def run(env, **kwargs):
    return runner.executar_avaliacao(env['root'], env['freeze'], env['status'], env['output'],
        authorize_real_evaluation=kwargs.pop('authorize_real_evaluation', True), **kwargs)


def test_complete_file_fixture_executes_actual_kernel_and_records_narrow_projection(tmp_path):
    env = fixture_files(tmp_path)
    result = run(env)
    assert result['files'] == 3
    saved = json.loads((env['output'] / 'RESULTADO_EXPLORATORIO.json').read_text(encoding='utf-8'))
    assert len(saved['member_inventory']) == 37
    assert len(saved['transfer']['configurations']) == 24
    assert saved['metadata']['DEFINITION_OF_DONE'] is False
    assert result['audit']['predictions'] == dict(records_physically_decoded=91,
        records_projected=90, outside_C_records_decoded_not_projected=1)
    assert result['audit']['references']['records_projected'] == 5
    assert result['audit']['references']['outside_C_records_decoded_not_projected'] == 1
    assert result['audit']['source_categories_projected'] == result['audit']['cache_files_read'] == 0
    assert set(result['audit']['prediction_fields']) == PREDICTION_COLUMNS
    assert set(result['audit']['reference_fields']) == REFERENCE_COLUMNS
    missing = [r for r in saved['d01']['snapshots'] if r['model'] == 'gpt-4o-mini-2024-07-18'][0]
    assert missing['denominator_common_severe'] == 0 and missing['estimate_dcb_minus_inn'] is None
    manifest = json.loads((env['output'] / 'MANIFESTO_AVALIACAO.json').read_text(encoding='utf-8'))
    assert len(manifest['artifacts']) == 2
    for entry in manifest['artifacts']:
        assert pin(env['root'], env['root'] / entry['path']) == entry


@pytest.mark.parametrize('authorization', [False, 1, 'true', None])
def test_no_authorization_blocks_every_read_before_even_missing_paths(tmp_path, monkeypatch, authorization):
    def forbidden(*args, **kwargs):
        pytest.fail('File read started without authorization')
    monkeypatch.setattr(runner, '_json', forbidden)
    monkeypatch.setattr(runner, '_sha', forbidden)
    with pytest.raises(ValueError, match='not authorized'):
        runner.executar_avaliacao(tmp_path, 'missing-freeze', 'missing-status', 'new-output',
            authorize_real_evaluation=authorization)
    assert list(tmp_path.iterdir()) == []


def test_cli_default_refuses_without_touching_inputs(tmp_path):
    with pytest.raises(ValueError, match='not authorized'):
        runner.main(['--root', str(tmp_path), '--freeze-dir', 'absent', '--status', 'absent-status', '--output-dir', 'new'])
    assert list(tmp_path.iterdir()) == []


def test_existing_output_is_rejected_before_input_reads(tmp_path, monkeypatch):
    env = fixture_files(tmp_path)
    env['output'].mkdir()
    marker = env['output'] / 'preserve.txt'
    marker.write_text('must remain', encoding='utf-8')
    monkeypatch.setattr(runner, '_json', lambda *a: pytest.fail('Read before overwrite rejection'))
    with pytest.raises(ValueError, match='overwrite'):
        run(env)
    assert marker.read_text(encoding='utf-8') == 'must remain'


@pytest.mark.parametrize('target', ['metadata', 'prediction', 'reference', 'approval', 'manifest'])
def test_hash_corruption_stops_before_outcome_projection_or_kernel(tmp_path, monkeypatch, target):
    env = fixture_files(tmp_path)
    path = {'metadata': env['freeze'] / 'FEATURES_C_SEM_ESCORE.json', 'prediction': env['predictions'],
        'reference': env['references'], 'approval': env['approval'],
        'manifest': env['freeze'] / 'MANIFESTO_CONGELAMENTO.json'}[target]
    with path.open('a', encoding='utf-8') as stream:
        stream.write(' ')
    monkeypatch.setattr(runner, '_project_predictions', lambda *a: pytest.fail('Outcome projection before integrity'))
    monkeypatch.setattr(runner, 'avaliar_estudo_exploratorio', lambda **k: pytest.fail('Analysis before integrity'))
    with pytest.raises(ValueError, match='changed|hash'):
        run(env)
    assert not env['output'].exists()


@pytest.mark.parametrize('fault', ['status_incomplete', 'wrong_approval', 'extra_source_category', 'string_boolean',
    'contract_count', 'contract_fields', 'config_seed', 'duplicate_selected_prediction', 'missing_selected_prediction',
    'duplicate_selected_reference', 'reference_header', 'mask_wrong_partition', 'prediction_bad_category'])
def test_schema_and_scope_errors_rejected_without_output(tmp_path, fault):
    env = fixture_files(tmp_path)
    if fault in {'status_incomplete', 'wrong_approval'}:
        status = json.loads(env['status'].read_text(encoding='utf-8'))
        status['FREEZE2_COMPLETE' if fault == 'status_incomplete' else 'approval_sha256'] = False if fault == 'status_incomplete' else 'f' * 64
        write_json(env['status'], status)
    else:
        if fault in {'extra_source_category', 'string_boolean'}:
            path = env['freeze'] / 'entrada_calibracao_sem_categorias.csv'
            rows, fields = runner._csv(path)
            if fault == 'extra_source_category':
                fields.append('category')
                for row in rows: row['category'] = 'grave'
            else: rows[0]['available'] = 'true'
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, delimiter=';')
                writer.writeheader(); writer.writerows(rows)
        elif fault.startswith('contract_'):
            path = env['freeze'] / 'CONTRATO_LEITURA_FUTURA.json'
            data = json.loads(path.read_text(encoding='utf-8'))
            if fault == 'contract_count': data['exact_C_pairs'] += 1
            else: data['prediction_fields'].append('unused_raw_like_field')
            write_json(path, data)
        elif fault == 'config_seed':
            path = env['freeze'] / 'CONFIGURACAO_FIXADA.json'
            data = json.loads(path.read_text(encoding='utf-8')); data['tie_seed'] = 1302
            write_json(path, data)
        elif fault in {'duplicate_selected_prediction', 'missing_selected_prediction', 'prediction_bad_category'}:
            rows = [json.loads(s) for s in env['predictions'].read_text(encoding='utf-8').splitlines()]
            if fault == 'duplicate_selected_prediction': rows.append(rows[0])
            elif fault == 'missing_selected_prediction': rows = rows[1:]
            else: rows[0]['category'] = 'undefined-label'
            write_lines(env['predictions'], rows)
        elif fault == 'duplicate_selected_reference':
            lines = env['references'].read_text(encoding='utf-8').splitlines()
            env['references'].write_text('\n'.join(lines + [lines[1]]) + '\n', encoding='utf-8')
        elif fault == 'reference_header':
            text = env['references'].read_text(encoding='utf-8').replace('canonical_label', 'wrong_label', 1)
            env['references'].write_text(text, encoding='utf-8')
        elif fault == 'mask_wrong_partition':
            path = env['freeze'] / 'mascaras_C_contrato.jsonl'
            rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
            rows[0]['pair_partition'] = 'calibration'
            write_lines(path, rows)
        seal(env)
    with pytest.raises(ValueError):
        run(env)
    assert not env['output'].exists()


@pytest.mark.parametrize('relative', ['cache/invented.sqlite', 'metadata/invented.sqlite-wal',
    'metadata/invented.sqlite-shm', 'metadata/invented.sqlite-journal', 'metadata/cache.db-journal',
    '.local/invented.json', 'metadata/.private/invented.json'])
def test_cache_pin_is_blocked_even_when_its_hash_is_valid(tmp_path, monkeypatch, relative):
    env = fixture_files(tmp_path)
    cache = env['root'] / relative
    cache.parent.mkdir(parents=True); cache.write_bytes(b'invented-not-a-database')
    manifest_path = env['freeze'] / 'MANIFESTO_CONGELAMENTO.json'
    data = json.loads(manifest_path.read_text(encoding='utf-8'))
    data['inputs'].append(pin(env['root'], cache))
    write_json(manifest_path, data)
    status = json.loads(env['status'].read_text(encoding='utf-8'))
    status['freeze_manifest_sha256'] = pin(env['root'], manifest_path)['sha256']
    write_json(env['status'], status)
    original = runner._sha
    def safe_sha(path):
        assert path != cache, 'Cache bytes were opened'
        return original(path)
    monkeypatch.setattr(runner, '_sha', safe_sha)
    with pytest.raises(ValueError, match='Cache'):
        run(env)
    assert not env['output'].exists()


@pytest.mark.parametrize('relative', ['.local/state.json', 'nested/.private/state.json', '.settings', '.metadata.json'])
@pytest.mark.parametrize('absolute', [False, True])
def test_hidden_component_pin_rejected_before_file_access(tmp_path, monkeypatch, relative, absolute):
    root = tmp_path.resolve()
    value = str(root / relative) if absolute else relative
    def forbidden(*args, **kwargs):
        pytest.fail('Forbidden path reached a file check or read')
    monkeypatch.setattr(Path, 'is_file', forbidden)
    monkeypatch.setattr(runner, '_sha', forbidden)
    with pytest.raises(ValueError, match='outside this reading contract'):
        runner._pin(root, dict(path=value, sha256='a' * 64, bytes=0))


@pytest.mark.parametrize('argument', ['freeze_dir', 'status_path', 'output_directory'])
def test_hidden_entrypoint_argument_rejected_before_reads_or_creation(tmp_path, monkeypatch, argument):
    def forbidden(*args, **kwargs):
        pytest.fail('Hidden path reached a file read')
    monkeypatch.setattr(runner, '_json', forbidden)
    monkeypatch.setattr(runner, '_sha', forbidden)
    paths = dict(freeze_dir='freeze', status_path='status.json', output_directory='results')
    paths[argument] = '.local'
    with pytest.raises(ValueError, match='outside this reading contract'):
        runner.executar_avaliacao(tmp_path, **paths, authorize_real_evaluation=True)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('relative', ['metadata.v1/state.json', 'src/dados/leitor_cache_delimitado.py', 'docs/approval.md'])
def test_common_paths_and_explicit_project_root_remain_allowed(tmp_path, relative):
    for root in (tmp_path.resolve(), (tmp_path / '.selected-project').resolve()):
        assert runner._safe_path(root, relative) == root / relative
        assert runner._safe_path(root, str(root / relative)) == root / relative


def test_resolved_escape_from_project_root_still_rejected(tmp_path):
    with pytest.raises(ValueError, match='escapes'):
        runner._safe_path(tmp_path.resolve(), '../outside/state.json')


def test_source_csv_float_parser_preserves_round_trip_and_blank_none(tmp_path):
    path = tmp_path / 'source.csv'
    fields = sorted(runner.SOURCE_FIELDS)
    row = dict(observation_id='fixture-o', pair_id='fixture-p', model='fixture-m', condition='dcb',
        execution_slot='historico_a_identificar', pair_partition='calibration', available=True, eligible=True,
        risk_score=repr(0.12345678912345678))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=';')
        writer.writeheader(); writer.writerow(row)
        writer.writerow(dict(row, observation_id='fixture-absent', available=False, eligible=False, risk_score=''))
    result = runner._source_scores(path)
    assert result.iloc[0].risk_score.hex() == float(row['risk_score']).hex()
    assert result.iloc[1].risk_score is None
    assert result.iloc[0].available is True and result.iloc[1].available is False
