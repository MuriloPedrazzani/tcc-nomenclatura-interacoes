"""Classify literal experimental names; never infer identity or admission.

Only case and whitespace may differ for a positive literal-name proof.
The caller supplies curated evidence and a versioned source registry. This
module validates bindings; it does not fetch, invent or independently certify
the source passage supplied by the curator.
"""
from __future__ import annotations

import json
from urllib.parse import urlparse

VERSION = 'literal-name-evidence-v1'
PROOF_KINDS = {'who_recommended_inn', 'who_proposed_inn',
               'reference_node_title', 'official_documented_name'}
RELATIONS = {'exact_literal', 'case_whitespace_only', 'different_literal'}


def _records(table, label):
    if hasattr(table, 'to_dict'):
        table = table.to_dict('records')
    if not isinstance(table, (list, tuple)) or any(not isinstance(x, dict) for x in table):
        raise ValueError(f'{label}: expected records')
    return [dict(x) for x in table]


def _text(row, field, label, *, allow_empty=False):
    value = row.get(field)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f'{label}: missing or invalid {field}')
    return value


def _unique(rows, key, label):
    result = {}
    for row in rows:
        value = _text(row, key, label)
        if value in result:
            raise ValueError(f'{label}: duplicate {key}: {value}')
        result[value] = row
    return result


def _literal(value):
    return ' '.join(value.split()).casefold()


def avaliar_nomes(entities, evidence, *, source_registry):
    """Return one naming-status record per entity, preserving its literal text.

    Required entity columns: drug_id, experimental_name. Evidence columns:
    evidence_id, drug_id, experimental_name, documented_name, source_id,
    source_version, locator, proof_kind, relation. Registry columns: source_id,
    source_version, publisher, source_kind, url, locator. Missing evidence
    gives an unresolved result. Malformed, duplicate, orphan or incompatible
    evidence fails explicitly instead of being silently discarded.

    A 'different_literal' passage is retained for audit but cannot certify the
    experimental text, including salts, routes, combinations and formulations.
    'outra_denominacao_documentada' means only that the current proof supports
    a documented name; it is not a finding that the name is not an INN.
    """
    entities = _records(entities, 'entities')
    evidence = _records(evidence, 'evidence')
    sources = _records(source_registry, 'source_registry')
    entity_index = _unique(entities, 'drug_id', 'entities')
    _unique(evidence, 'evidence_id', 'evidence')
    source_index = _unique(sources, 'source_id', 'source_registry')
    for row in entities:
        _text(row, 'experimental_name', 'entities', allow_empty=True)
    for row in sources:
        for field in ['source_version', 'publisher', 'source_kind', 'url', 'locator']:
            _text(row, field, 'source_registry')
        if row['source_kind'] not in PROOF_KINDS:
            raise ValueError('source_registry: unsupported source kind')
        if row['source_kind'].startswith('who_'):
            domain = (urlparse(row['url']).hostname or '').lower()
            if row['publisher'] != 'WHO' or not (domain == 'who.int' or domain.endswith('.who.int')):
                raise ValueError('source_registry: WHO evidence requires a WHO source')
    grouped = {key: [] for key in entity_index}
    seen_proofs = set()
    for row in evidence:
        for field in ['drug_id', 'experimental_name', 'documented_name', 'source_id',
                      'source_version', 'locator', 'proof_kind', 'relation']:
            _text(row, field, 'evidence')
        entity = entity_index.get(row['drug_id'])
        if entity is None or row['experimental_name'] != entity['experimental_name']:
            raise ValueError('evidence: orphan or wrong experimental name')
        source = source_index.get(row['source_id'])
        if source is None or source['source_version'] != row['source_version']:
            raise ValueError('evidence: unknown source or wrong source version')
        if row['proof_kind'] not in PROOF_KINDS or row['proof_kind'] != source['source_kind']:
            raise ValueError('evidence: incompatible proof kind')
        if row['relation'] not in RELATIONS:
            raise ValueError('evidence: unsupported relation')
        if row['relation'] == 'exact_literal' and row['experimental_name'] != row['documented_name']:
            raise ValueError('evidence: false exact-literal relation')
        if row['relation'] == 'case_whitespace_only' and _literal(row['experimental_name']) != _literal(row['documented_name']):
            raise ValueError('evidence: relation removes or changes literal qualifiers')
        if row['relation'] == 'different_literal' and _literal(row['experimental_name']) == _literal(row['documented_name']):
            raise ValueError('evidence: false different-literal relation')
        signature = tuple(row[x] for x in ['drug_id', 'source_id', 'source_version',
                         'locator', 'proof_kind', 'documented_name'])
        if signature in seen_proofs:
            raise ValueError('evidence: duplicate proof under another identifier')
        seen_proofs.add(signature)
        grouped[row['drug_id']].append(row)
    output = []
    for entity in entities:
        all_proofs = grouped[entity['drug_id']]
        proofs = [x for x in all_proofs if x['relation'] != 'different_literal']
        official = [x for x in proofs if x['proof_kind'] == 'who_recommended_inn']
        if official:
            status, reasons = 'inn_oficial_documentada', 'who_recommended_literal_name'
        elif proofs:
            status, reasons = 'outra_denominacao_documentada', 'documented_name_not_a_negative_inn_claim'
        else:
            status, reasons = 'natureza_nao_resolvida', 'no_supported_literal_name_evidence'
        if all_proofs and len(proofs) != len(all_proofs):
            reasons += '|different_literal_source_does_not_certify_experimental_text'
        output.append(dict(drug_id=entity['drug_id'],
                           experimental_name=entity['experimental_name'],
                           naming_status=status, literal_inn_documented=bool(official),
                           evidence_ids_json=json.dumps(sorted(x['evidence_id'] for x in all_proofs), ensure_ascii=False),
                           literal_support_ids_json=json.dumps(sorted(x['evidence_id'] for x in proofs), ensure_ascii=False),
                           reasons=reasons, version=VERSION))
    return output
