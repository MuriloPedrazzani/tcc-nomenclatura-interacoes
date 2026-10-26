"""Future file entrypoint for explicitly authorized exploratory evaluation.

This round validates invented fixture files only. The command defaults to
refusing evaluation before any artifact or clinical-file read. A later explicit
authorization permits physical reads of the two pinned normalized clinical
files, never cache/raw responses. JSONL and CSV records outside C must be
decoded to inspect their identities; their outcome fields are not projected,
classified, counted or supplied to the analytical kernel.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

# Legacy analytical modules use sibling absolute imports. This merely makes
# installed repository modules resolvable; it grants no data-reading scope.
_CODE_ROOT = Path(__file__).resolve().parents[2]
for _folder in (_CODE_ROOT, _CODE_ROOT / 'src/analise', _CODE_ROOT / 'src/dados'):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

from src.analise.avaliacao_real_exploratoria import avaliar_estudo_exploratorio
from src.analise.contratos_avaliacao_real import (
    MASK_COLUMNS, PREDICTION_COLUMNS, REFERENCE_COLUMNS, table,
)
from src.analise.transferencia_real_exploratoria import SOURCE_FIELDS


VERSION = 'authorized-exploratory-file-entrypoint-v1'
METADATA_SOURCES = dict(masks='mascaras_C_contrato.jsonl', registry='REGISTRO_SNAPSHOTS.json',
    contexts='CONTEXTOS_C.json', features='FEATURES_C_SEM_ESCORE.json',
    state='CONSTANTES_CANDIDATAS.json:state', thresholds='LIMIARES_VINCULADOS_GLOBAL_24.json',
    source_scores='entrada_calibracao_sem_categorias.csv',
    source_pair_metadata='CONTEXTOS_CALIBRACAO_SEM_CATEGORIAS.json',
    policy_bases='CONFIGURACAO_FIXADA.json:policy_bases')
METADATA_FILES = {v.split(':')[0] for v in METADATA_SOURCES.values()} | {'CONTRATO_LEITURA_FUTURA.json'}
CLINICAL_NAMES = {'REGISTROS_NORMALIZADOS.jsonl', 'reference_labels.csv'}
APPROVAL_NAME = 'APROVACAO_FECHAMENTO_EXPLORATORIO_2026-10-03.md'


def _digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _safe_path(root, value):
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('Artifact path escapes the explicitly selected project root')
    # Cache/database/credential access is forbidden even if a malformed manifest
    # attempts to pin it. Source code referring to cache is not a cache file.
    parts = [p.casefold() for p in resolved.relative_to(root).parts]
    database_or_sidecar = re.search(r'\.(?:db|db3|sqlite|sqlite3)(?:-(?:wal|shm|journal))?$', resolved.name.casefold())
    # Dot-prefixed components are excluded relative to the selected root,
    # including the final component when it names a not-yet-created directory.
    if (any(p.startswith('.') for p in parts)
            or any(p in {'cache', 'caches', '__pycache__', 'credentials'}
            or p.startswith('cache_') for p in parts[:-1])
            or database_or_sidecar is not None
            or resolved.name.casefold().startswith('.env')
            or resolved.name.casefold() in {'credentials.json', 'auth.json'}):
        raise ValueError('Cache, database, credential or hidden path is outside this reading contract')
    return resolved


def _json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _pin(root, entry):
    if (not isinstance(entry, dict) or not {'path', 'sha256', 'bytes'} <= set(entry)
            or not isinstance(entry['path'], str) or not _digest(entry['sha256'])
            or type(entry['bytes']) is not int or entry['bytes'] < 0):
        raise ValueError('Invalid pinned artifact identity, digest or size')
    path = _safe_path(root, entry['path'])
    if not path.is_file() or path.stat().st_size != entry['bytes'] or _sha(path) != entry['sha256']:
        raise ValueError('Pinned artifact changed: ' + entry['path'])
    return path


def _csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter=';')
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('Missing or duplicated CSV columns')
        rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError('CSV record has inconsistent column cardinality')
    return rows, reader.fieldnames


def _source_scores(path):
    rows, fields = _csv(path)
    if set(fields) != SOURCE_FIELDS:
        raise ValueError('Source CSV requires the exact category-free nine-column projection')
    for row in rows:
        for key in ('available', 'eligible'):
            if row[key] not in {'True', 'False'}:
                raise ValueError('Source booleans must be literal True/False')
            row[key] = row[key] == 'True'
        row['risk_score'] = None if row['risk_score'] == '' else float(row['risk_score'])
    return pd.DataFrame(rows, columns=fields, dtype=object)


def _project_predictions(path, allowed_ids):
    projected, seen = [], set()
    physical = 0
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            physical += 1
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('Normalized JSONL must contain objects')
            oid = row.get('observation_id')
            # Do not access any outcome field until membership is established.
            if oid not in allowed_ids:
                continue
            if oid in seen or not PREDICTION_COLUMNS <= set(row):
                raise ValueError('Selected prediction is duplicated or missing binding columns')
            seen.add(oid)
            projected.append({key: row[key] for key in PREDICTION_COLUMNS})
    if seen != allowed_ids:
        raise ValueError('Normalized file does not cover the exact admitted C observations')
    return pd.DataFrame(projected, columns=sorted(PREDICTION_COLUMNS), dtype=object), dict(
        records_physically_decoded=physical, records_projected=len(projected),
        outside_C_records_decoded_not_projected=physical-len(projected))


def _project_references(path, allowed_pairs):
    projected, seen = [], set()
    physical = 0
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter=';')
        if (reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames)
                or not REFERENCE_COLUMNS <= set(reader.fieldnames)):
            raise ValueError('Reference CSV lacks explicit identity and label projection')
        for row in reader:
            physical += 1
            pid = row.get('pair_id')
            if pid not in allowed_pairs:
                continue
            if pid in seen or None in row or any(row[k] is None for k in REFERENCE_COLUMNS):
                raise ValueError('Selected reference is duplicated or structurally invalid')
            seen.add(pid)
            projected.append({key: row[key] for key in REFERENCE_COLUMNS})
    if seen != allowed_pairs:
        raise ValueError('Reference file does not cover exact C pair identities')
    return pd.DataFrame(projected, columns=sorted(REFERENCE_COLUMNS), dtype=object), dict(
        records_physically_decoded=physical, records_projected=len(projected),
        outside_C_records_decoded_not_projected=physical-len(projected))


def _plain(value):
    if isinstance(value, pd.DataFrame):
        return _plain(value.to_dict('records'))
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if value is pd.NA or isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(_plain(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def executar_avaliacao(root, freeze_dir, status_path, output_directory, *, authorize_real_evaluation=False):
    """Verify the complete frozen envelope, project only C, then call the kernel.

    No output directory is created on authorization, integrity or schema failure.
    The two clinical files are physically hashed and read only after explicit
    authorization. No raw/cache access, source-label projection or extra keys.
    """
    if authorize_real_evaluation is not True:
        raise ValueError('Real evaluation is not authorized; no input file was read')
    root = Path(root).resolve()
    freeze = _safe_path(root, freeze_dir)
    status_file = _safe_path(root, status_path)
    out = _safe_path(root, output_directory)
    if out.exists():
        raise ValueError('Refuse to overwrite an existing output directory')
    if out == freeze or out.is_relative_to(freeze) or freeze.is_relative_to(out):
        raise ValueError('Evaluation output must be separate from the frozen package')
    status = _json(status_file)
    if (status.get('FREEZE2_COMPLETE') is not True or not _digest(status.get('freeze_manifest_sha256'))
            or not _digest(status.get('approval_sha256')) or status.get('DEFINITION_OF_DONE') is not False):
        raise ValueError('Explicit completed formal freeze with unchanged project completion state is required')
    manifest_file = freeze / 'MANIFESTO_CONGELAMENTO.json'
    if _sha(manifest_file) != status['freeze_manifest_sha256']:
        raise ValueError('Freeze manifest hash differs from formal status')
    manifest = _json(manifest_file)
    if (manifest.get('version') != 'exploratory-freeze2-inputs-v1'
            or manifest.get('not_self_authorizing_evaluation') is not True
            or manifest.get('DEFINITION_OF_DONE') is not False):
        raise ValueError('Unsupported or self-authorizing freeze manifest')
    pins, paths = {}, {}
    for section in ('inputs', 'artifacts', 'clinical_pins'):
        if not isinstance(manifest.get(section), list) or not manifest[section]:
            raise ValueError('Incomplete freeze manifest sections')
        for entry in manifest[section]:
            path = _safe_path(root, entry['path'])
            if path in pins and any(pins[path][k] != entry[k] for k in ('sha256', 'bytes')):
                raise ValueError('Conflicting pins for the same artifact')
            pins[path] = entry
            paths[path] = _pin(root, entry)
    for name in METADATA_FILES:
        if freeze / name not in paths:
            raise ValueError('Required frozen metadata file is not pinned: ' + name)
    approvals = [e for p, e in pins.items() if p.name == APPROVAL_NAME]
    if len(approvals) != 1 or approvals[0]['sha256'] != status['approval_sha256']:
        raise ValueError('Status approval does not match the approved pinned document')
    clinical = {}
    for entry in manifest['clinical_pins']:
        path = _safe_path(root, entry['path'])
        if path.name not in CLINICAL_NAMES or path.name in clinical:
            raise ValueError('Only the two pinned normalized clinical files may be projected')
        clinical[path.name] = path
    if set(clinical) != CLINICAL_NAMES:
        raise ValueError('Incomplete clinical pin contract')
    contract = _json(freeze / 'CONTRATO_LEITURA_FUTURA.json')
    if (contract.get('version') != 'future-real-exploratory-reading-v1'
            or contract.get('evaluation_authorized') is not False
            or contract.get('metadata_sources') != METADATA_SOURCES
            or contract.get('prediction_fields') != sorted(PREDICTION_COLUMNS)
            or contract.get('reference_fields') != sorted(REFERENCE_COLUMNS)
            or contract.get('source_categories_required') is not False
            or contract.get('source_reference_severities_required') is not False
            or contract.get('additional_cache_keys') != 0):
        raise ValueError('Future reading contract differs from the explicit implemented scope')
    if {(e['path'], e['sha256'], e['bytes']) for e in contract.get('clinical_files', [])} != {
            (e['path'], e['sha256'], e['bytes']) for e in manifest['clinical_pins']}:
        raise ValueError('Reading contract clinical pins differ from the frozen manifest')
    with (freeze / 'mascaras_C_contrato.jsonl').open(encoding='utf-8-sig') as stream:
        masks = pd.DataFrame([json.loads(line) for line in stream], dtype=object)
    table(masks, MASK_COLUMNS, ('observation_id',), 'frozen C masks')
    if not masks.pair_partition.eq('C').all() or not masks.available.map(lambda v: type(v) is bool).all():
        raise ValueError('Frozen masks must contain explicit C positions and strict availability')
    allowed_ids = set(masks.loc[masks.available, 'observation_id'])
    allowed_pairs = set(masks.pair_id)
    if (len(masks) != contract.get('exact_planned_C_observations')
            or len(allowed_ids) != contract.get('exact_available_C_observations')
            or len(allowed_pairs) != contract.get('exact_C_pairs')):
        raise ValueError('Frozen C counts differ from the authorized reading contract')
    config = _json(freeze / 'CONFIGURACAO_FIXADA.json')
    expected = dict(split_seed=1302, tie_seed=20261003, coverage_grid=[.6, .7, .8, .9],
        transfer_source_basis='global', transfer_target_basis='global', transfer_comparator='<=',
        transfer_tolerance_pp=5, transfer_tolerance_inclusive=True,
        no_inference=True, no_coverage_winner=True, no_independent_historical_replicates=True,
        DEFINITION_OF_DONE=False)
    if any(config.get(k) != v or type(config.get(k)) is not type(v) for k, v in expected.items()):
        raise ValueError('Frozen methodological configuration differs from the implemented contract')
    inputs = dict(masks=masks,
        registry=pd.DataFrame(_json(freeze / 'REGISTRO_SNAPSHOTS.json'), dtype=object),
        contexts=pd.DataFrame(_json(freeze / 'CONTEXTOS_C.json'), dtype=object),
        features=pd.DataFrame(_json(freeze / 'FEATURES_C_SEM_ESCORE.json'), dtype=object),
        state=_json(freeze / 'CONSTANTES_CANDIDATAS.json')['state'],
        thresholds=pd.DataFrame(_json(freeze / 'LIMIARES_VINCULADOS_GLOBAL_24.json'), dtype=object),
        source_scores=_source_scores(freeze / 'entrada_calibracao_sem_categorias.csv'),
        source_pair_metadata=pd.DataFrame(_json(freeze / 'CONTEXTOS_CALIBRACAO_SEM_CATEGORIAS.json'), dtype=object),
        policy_bases=config['policy_bases'])
    inputs['predictions'], prediction_audit = _project_predictions(clinical['REGISTROS_NORMALIZADOS.jsonl'], allowed_ids)
    inputs['references'], reference_audit = _project_references(clinical['reference_labels.csv'], allowed_pairs)
    result = avaliar_estudo_exploratorio(**inputs,
        freeze_manifest_sha256=status['freeze_manifest_sha256'], approval_sha256=status['approval_sha256'],
        evaluation_authorized=True, formal_freeze_complete=True)
    # A second complete check catches changed inputs before anything is saved.
    for entry in pins.values():
        _pin(root, entry)
    if _sha(manifest_file) != status['freeze_manifest_sha256'] or _json(status_file) != status:
        raise ValueError('Freeze envelope changed during evaluation')
    audit = dict(version=VERSION, authorization_explicit=True, verified_pins=len(pins),
        all_pins_verified_before_and_after=True, cache_files_read=0, raw_responses_read=0,
        clinical_files_physically_read_whole=True,
        limitation='Whole normalized JSONL/CSV files are physically decoded to select C IDs; outside-C outcome fields are never projected or supplied to analysis.',
        predictions=prediction_audit, references=reference_audit, source_categories_projected=0,
        prediction_fields=sorted(PREDICTION_COLUMNS), reference_fields=sorted(REFERENCE_COLUMNS),
        freeze_manifest_sha256=status['freeze_manifest_sha256'], approval_sha256=status['approval_sha256'],
        DEFINITION_OF_DONE=False)
    out.mkdir(parents=True, exist_ok=False)
    _save(out / 'RESULTADO_EXPLORATORIO.json', result)
    _save(out / 'LEITURA_E_INTEGRIDADE.json', audit)
    artifacts = [dict(path=p.relative_to(root).as_posix(), sha256=_sha(p), bytes=p.stat().st_size)
                 for p in sorted(out.iterdir())]
    _save(out / 'MANIFESTO_AVALIACAO.json', dict(version=VERSION, inputs=list(pins.values()),
        artifacts=artifacts, freeze_manifest_sha256=status['freeze_manifest_sha256'],
        approval_sha256=status['approval_sha256'], DEFINITION_OF_DONE=False))
    return dict(output_directory=str(out), files=3, audit=audit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--freeze-dir', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--authorize-real-evaluation', choices=('false', 'true'), nargs='?', const='true', default='false')
    args = parser.parse_args(argv)
    result = executar_avaliacao(args.root, args.freeze_dir, args.status, args.output_dir,
                               authorize_real_evaluation=args.authorize_real_evaluation == 'true')
    print(json.dumps(dict(output_directory=result['output_directory'], files=result['files']), ensure_ascii=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
