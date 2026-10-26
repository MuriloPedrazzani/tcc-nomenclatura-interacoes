"""Explicit benchmark admission, independent of universal clinical applicability.

Pure metadata functions. Proof bindings are validated; substantive identity
claims must be curated from documentary evidence. No severity or model output
is accepted for identity selection. Historical gates remain unchanged.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contrato_documental_v1 import _validar_colunas, _lista_json

CONTRACT_VERSION = 'benchmark-ddinter-entity-v1'
ENTITY_COLUMNS = ('drug_id', 'dcb', 'experimental_name', 'ref_id', 'snapshot_name', 'active_reference_ids_json')
ASSESSMENT_COLUMNS = ('drug_id', 'status', 'material_conflict', 'required_qualifiers_json',
                      'demonstrated_qualifiers_json', 'evidence_ids_json', 'reason')
EVIDENCE_COLUMNS = ('evidence_id', 'drug_id', 'dcb', 'experimental_name', 'ref_id', 'snapshot_name',
                    'source_id', 'source_version', 'kind', 'locator', 'claim')
PAIR_COLUMNS = ('pair_id', 'drug_id_a', 'drug_id_b', 'ref_id_a', 'ref_id_b', 'context_id')
RECORD_COLUMNS = ('pair_id', 'ref_id_a', 'ref_id_b', 'context_id', 'source_id', 'source_version',
                  'record_present', 'quality_status', 'record_evidence_refs_json')
PRIOR_COLUMNS = ('pair_id', 'source_id', 'source_version', 'applicability_status', 'evidence_refs_json')
QUALITY_STATES = {'usable', 'unknown', 'missing', 'invalid', 'conflict', 'not_located', 'not_queryable'}


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field}: nonempty text required')
    return value


def _bool(value, field):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{field}: strict boolean required')
    return bool(value)


def _list(value, field, refs=False):
    return _lista_json(value, campo=field, referencias=refs)


def _table(frame, columns, name, keys):
    _validar_colunas(frame, columns, name)
    if frame.duplicated(list(keys)).any():
        raise ValueError(f'{name}: duplicate key')
    for key in keys:
        for value in frame[key]:
            _text(value, f'{name}.{key}')


def _version(contract_version, source_id, source_version):
    if contract_version != CONTRACT_VERSION:
        raise ValueError('Explicit benchmark-ddinter-entity-v1 contract required')
    _text(source_id, 'source_id')
    _text(source_version, 'source_version')


def avaliar_entidades_benchmark(entities, assessments, evidence, *, source_id, source_version, contract_version):
    """Validate positive documentary declarations for every original entity.

    Missing/insufficient evidence remains unresolved. Wrong bindings and
    duplicate/orphan proof rows are structural errors, not silent exclusions.
    Absence of universal RENAME proof is deliberately not a requirement here.
    """
    _version(contract_version, source_id, source_version)
    _table(entities, ENTITY_COLUMNS, 'entities', ('drug_id',))
    _table(assessments, ASSESSMENT_COLUMNS, 'assessments', ('drug_id',))
    _table(evidence, EVIDENCE_COLUMNS, 'evidence', ('evidence_id',))
    by_entity = entities.set_index('drug_id').to_dict('index')
    if set(assessments.drug_id) - set(by_entity) or set(evidence.drug_id) - set(by_entity):
        raise ValueError('Orphan entity assessment/evidence')
    by_proof = evidence.set_index('evidence_id').to_dict('index')
    binding_fields = ('dcb', 'experimental_name', 'ref_id', 'snapshot_name')
    for proof in evidence.to_dict('records'):
        entity = by_entity[proof['drug_id']]
        for field in binding_fields:
            _text(proof[field], 'evidence.' + field)
            if proof[field] != entity[field]:
                raise ValueError('Identity proof mismatch: ' + field)
        if proof['source_id'] != source_id or proof['source_version'] != source_version:
            raise ValueError('Identity proof source/version mismatch')
        for field in ('kind', 'locator', 'claim'):
            _text(proof[field], 'evidence.' + field)
        _list(json.dumps([proof['locator']]), 'evidence.locator', refs=True)
    by_assessment = {}
    for row in assessments.to_dict('records'):
        if row['status'] not in {'supported', 'unresolved', 'conflict'}:
            raise ValueError('Unknown assessment status')
        _bool(row['material_conflict'], 'material_conflict')
        _text(row['reason'], 'assessment.reason')
        for field in ('required_qualifiers_json', 'demonstrated_qualifiers_json', 'evidence_ids_json'):
            _list(row[field], field)
        for evidence_id in _list(row['evidence_ids_json'], 'evidence_ids'):
            if evidence_id not in by_proof or by_proof[evidence_id]['drug_id'] != row['drug_id']:
                raise ValueError('Evidence absent or assigned to another entity')
        by_assessment[row['drug_id']] = row
    rows = []
    for did, entity in by_entity.items():
        reasons = []
        active = _list(entity['active_reference_ids_json'], 'active_reference_ids_json')
        if not entity['dcb'] or not entity['experimental_name'] or not entity['snapshot_name']:
            reasons.append('nominal_identity_incomplete')
        if not entity['ref_id'] or active != [entity['ref_id']]:
            reasons.append('unique_active_reference_not_resolved')
        assessment = by_assessment.get(did)
        proof_ids = []
        status = 'unresolved'
        if assessment is None:
            reasons.append('documentary_assessment_missing')
        else:
            status = assessment['status']
            if status != 'supported':
                reasons.append('identity_' + status)
            if assessment['material_conflict']:
                reasons.append('material_conflict_unresolved')
            required = set(_list(assessment['required_qualifiers_json'], 'required_qualifiers'))
            demonstrated = set(_list(assessment['demonstrated_qualifiers_json'], 'demonstrated_qualifiers'))
            if required - demonstrated:
                reasons.append('material_qualifier_not_demonstrated')
            proof_ids = _list(assessment['evidence_ids_json'], 'evidence_ids')
            if not any(by_proof[eid]['kind'] == 'documentary_identity_correspondence' for eid in proof_ids):
                reasons.append('positive_identity_proof_missing')
        rows.append(dict(drug_id=did, dcb=entity['dcb'], experimental_name=entity['experimental_name'],
            ref_id=entity['ref_id'], snapshot_name=entity['snapshot_name'],
            source_id=source_id, source_version=source_version, contract_version=CONTRACT_VERSION,
            entity_admissible=not reasons, documentary_status=status,
            evidence_ids_json=json.dumps(proof_ids, ensure_ascii=False), reasons='|'.join(reasons),
            universal_clinical_applicability_required=False, professional_validation=False))
    return pd.DataFrame(rows)


def admitir_pares_benchmark(pairs, mapping, entities, records, previous_reference, *, source_id, source_version, contract_version):
    """Keep identity admission, record availability and usable labels separate.

    records carries quality *status*, never severity categories. It must bind
    the two reference IDs in the same canonical drug orientation as pairs.
    Unknown or absent reference remains in the operational candidate flow.
    """
    _version(contract_version, source_id, source_version)
    _table(pairs, PAIR_COLUMNS, 'pairs', ('pair_id',))
    _table(mapping, ('pair_id', 'mapping_accepted'), 'mapping', ('pair_id',))
    _table(entities, ('drug_id','ref_id','source_id','source_version','contract_version','entity_admissible','reasons'), 'entity_results', ('drug_id',))
    _table(records, RECORD_COLUMNS, 'records', ('pair_id',))
    _table(previous_reference, PRIOR_COLUMNS, 'previous_reference', ('pair_id',))
    pair_ids = set(pairs.pair_id)
    for name, table in (('mapping',mapping),('records',records),('previous_reference',previous_reference)):
        if set(table.pair_id) != pair_ids:
            raise ValueError(name + ': missing/orphan pair')
    eby = entities.set_index('drug_id').to_dict('index')
    mby = mapping.set_index('pair_id').to_dict('index')
    rby = records.set_index('pair_id').to_dict('index')
    pby = previous_reference.set_index('pair_id').to_dict('index')
    for entity in eby.values():
        if (entity['source_id'],entity['source_version'],entity['contract_version']) != (source_id,source_version,CONTRACT_VERSION):
            raise ValueError('Entity result source/version/contract mismatch')
        _bool(entity['entity_admissible'], 'entity_admissible')
    rows=[]
    for pair in pairs.to_dict('records'):
        pid=pair['pair_id']
        if pair['drug_id_a']==pair['drug_id_b'] or any(pair['drug_id_'+side] not in eby for side in ('a','b')):
            raise ValueError('Invalid/orphan pair endpoints')
        _text(pair['context_id'], 'context_id')
        record, previous = rby[pid], pby[pid]
        if (record['source_id'],record['source_version']) != (source_id,source_version):
            raise ValueError('Record source/version mismatch')
        if record['context_id'] != pair['context_id']:
            raise ValueError('Interaction record context mismatch')
        if (previous['source_id'],previous['source_version']) != (source_id,source_version):
            raise ValueError('Historical audit source/version mismatch')
        if previous['applicability_status'] not in {'uncertain','demonstrated','inapplicable','not_assessed'}:
            raise ValueError('Unknown historical applicability status')
        _list(previous['evidence_refs_json'], 'previous.evidence_refs_json', refs=True)
        for side in ('a','b'):
            if record['ref_id_'+side] != pair['ref_id_'+side]:
                raise ValueError('Interaction record assigned to wrong endpoint')
        present = _bool(record['record_present'], 'record_present')
        refs = _list(record['record_evidence_refs_json'], 'record_evidence_refs_json', refs=True)
        quality = record['quality_status']
        if quality not in QUALITY_STATES:
            raise ValueError('Invalid reference quality status')
        if present and (not refs or not pair['ref_id_a'] or not pair['ref_id_b'] or pair['ref_id_a']==pair['ref_id_b']):
            raise ValueError('Present record needs a proven link to distinct reference entities')
        if (present and quality in {'not_located','not_queryable'}) or (not present and quality not in {'not_located','not_queryable'}):
            raise ValueError('Contradictory reference presence/quality')
        map_ok = _bool(mby[pid]['mapping_accepted'], 'mapping_accepted')
        reasons = [] if map_ok else ['previous_mapping_blocked']
        for side in ('a','b'):
            did=pair['drug_id_'+side]
            entity=eby[did]
            if not entity['entity_admissible']:
                reasons.append(did+':'+entity['reasons'])
            if entity['ref_id'] != pair['ref_id_'+side] or not pair['ref_id_'+side]:
                reasons.append(did+':pair_reference_unresolved_or_different')
        if pair['ref_id_a'] and pair['ref_id_a']==pair['ref_id_b']:
            reasons.append('distinct_drugs_share_reference_entity')
        identity_ok = not reasons
        candidate = identity_ok and present
        rows.append(dict(pair_id=pid, contract_version=CONTRACT_VERSION, source_id=source_id, source_version=source_version,
            context_id=pair['context_id'], mapping_accepted=map_ok, benchmark_identity_admissible=identity_ok,
            operational_candidate=identity_ok, record_present=present, benchmark_candidate=candidate,
            quality_status=quality, reference_usable_for_outcome=candidate and quality=='usable',
            identity_reasons='|'.join(reasons), reference_reasons='' if quality=='usable' else quality,
            prior_clinical_applicability=previous['applicability_status'],
            prior_clinical_evidence_refs_json=previous['evidence_refs_json'],
            prior_clinical_reasons=previous.get('reasons',''),
            clinical_evaluation_authorized=False, api_calls_authorized=False, freeze2_started=False))
    return pd.DataFrame(rows)
