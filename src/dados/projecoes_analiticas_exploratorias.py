"""Pure preparation from existing documentary and reference-status projections.

No file access, outcome categories, model responses or statistical evaluation.
Callers must project the exact allowlists and parse CSV booleans explicitly.
Availability is distinct from identity admission and documentary strength.
"""
from __future__ import annotations

import json

import pandas as pd


VERSION = 'exploratory-documentary-status-projections-v1'
CONTRACT_VERSION = 'benchmark-ddinter-entity-v1'
UNIVERSE_COLUMNS = (
    'pair_id', 'contract_version', 'drug_id_a', 'drug_id_b', 'context_id',
    'source_id', 'source_version', 'benchmark_identity_admissible',
    'benchmark_candidate', 'record_present', 'quality_status',
    'reference_usable_for_outcome', 'literal_inn_to_dcb_supported',
    'both_international_names_documented_inn',
)
REFERENCE_COLUMNS = (
    'pair_id', 'ref_id_a', 'ref_id_b', 'context_id', 'source_id',
    'source_version', 'record_present', 'quality_status',
    'record_evidence_refs_json',
)
CONTEXT_COLUMNS = (
    'pair_id', 'drug_id_a', 'drug_id_b', 'ref_id_a', 'ref_id_b', 'context_id',
)
QUALITY_STATUSES = frozenset({
    'usable', 'unknown', 'missing', 'invalid', 'conflict',
    'not_located', 'not_queryable',
})
UNIVERSE_BOOLEAN_COLUMNS = (
    'benchmark_identity_admissible', 'benchmark_candidate', 'record_present',
    'reference_usable_for_outcome', 'literal_inn_to_dcb_supported',
    'both_international_names_documented_inn',
)
OUTPUT_COLUMNS = (
    'pair_id', 'drug_id_a', 'drug_id_b', 'context_id',
    'reference_source_id', 'reference_source_version', 'identity_admitted',
    'benchmark_candidate', 'reference_record_present', 'reference_available',
    'reference_quality_status', 'literal_inn_to_dcb_supported',
    'both_international_names_documented_inn', 'd01e_documentary_state',
    'd01e_reason',
)


def _text(value, field, *, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f'{field}: text metadata required')
    if value and value != value.strip():
        raise ValueError(f'{field}: noncanonical outer whitespace')
    return value


def _table(frame, columns, name):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f'{name}: DataFrame required')
    if frame.columns.duplicated().any() or set(frame.columns) != set(columns):
        raise ValueError(f'{name}: only exact projected columns are permitted')
    if frame.empty:
        raise ValueError(f'{name}: empty population')
    if frame['pair_id'].duplicated().any():
        raise ValueError(f'{name}: duplicate pair_id')
    rows = frame.to_dict('records')
    for row in rows:
        _text(row['pair_id'], f'{name}.pair_id')
    return rows


def _boolean(value, field):
    # DataFrame.to_dict normalizes native/numpy bool storage to Python bool.
    # Strings, integers and missing values must never pass by truthiness.
    if type(value) is not bool:
        raise ValueError(f'{field}: strict boolean metadata required')
    return value


def _evidence(value):
    try:
        refs = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError('reference evidence: JSON list required') from exc
    if not isinstance(refs, list) or any(
        not isinstance(item, str) or not item.strip() for item in refs
    ):
        raise ValueError('reference evidence: nonempty string locators required')
    return refs


def documentaire_projections(universe_df, reference_df, contexts_df):
    """Validate one population and preserve every row in its original order.

    ``reference_available`` comes exclusively from an existing index's presence
    and quality status, not from a severity/category reader. It may be true for
    an excluded identity, which does not make that identity eligible. The prior
    universe outcome-availability flag is checked against both dimensions.

    Existing identity and literal-INN flags do not encode the stronger D01-E
    direct-complete documentary criterion. Its state remains not_assessed for
    candidates, outside_population otherwise; neither means an incorrect map.
    No missing proof, context or clinical category is invented by this function.
    """
    universe = _table(universe_df, UNIVERSE_COLUMNS, 'universe')
    references = _table(reference_df, REFERENCE_COLUMNS, 'reference')
    contexts = _table(contexts_df, CONTEXT_COLUMNS, 'context')
    ids = {row['pair_id'] for row in universe}
    if ids != {row['pair_id'] for row in references} or ids != {
        row['pair_id'] for row in contexts
    }:
        raise ValueError('Projected population ID sets differ: missing/orphan pair')
    rby = {row['pair_id']: row for row in references}
    cby = {row['pair_id']: row for row in contexts}
    output = []
    source_versions = set()
    for pair in universe:
        pid = pair['pair_id']
        reference, context = rby[pid], cby[pid]
        if pair['contract_version'] != CONTRACT_VERSION:
            raise ValueError('Explicit benchmark-ddinter-entity-v1 contract required')
        for field in UNIVERSE_BOOLEAN_COLUMNS:
            _boolean(pair[field], 'universe.' + field)
        present = _boolean(reference['record_present'], 'reference.record_present')
        for field in ('source_id', 'source_version', 'context_id'):
            _text(pair[field], 'universe.' + field)
            _text(reference[field], 'reference.' + field)
            if pair[field] != reference[field]:
                raise ValueError(f'Incoherent reference binding: {field}')
        source_versions.add((pair['source_id'], pair['source_version']))
        if context['context_id'] != pair['context_id']:
            raise ValueError('Incoherent pair context binding')
        for side in ('a', 'b'):
            drug = 'drug_id_' + side
            ref = 'ref_id_' + side
            _text(pair[drug], 'universe.' + drug)
            _text(context[drug], 'context.' + drug)
            _text(context[ref], 'context.' + ref, allow_empty=True)
            _text(reference[ref], 'reference.' + ref, allow_empty=True)
            if pair[drug] != context[drug] or context[ref] != reference[ref]:
                raise ValueError(f'Incoherent endpoint binding: {side}')
        if pair['drug_id_a'] == pair['drug_id_b']:
            raise ValueError('A pair requires two distinct drug IDs')
        status = reference['quality_status']
        if not isinstance(status, str) or status not in QUALITY_STATUSES:
            raise ValueError('Unknown reference quality status')
        if pair['quality_status'] != status or pair['record_present'] != present:
            raise ValueError('Incoherent reference presence/quality projection')
        refs = _evidence(reference['record_evidence_refs_json'])
        queryable = bool(reference['ref_id_a'] and reference['ref_id_b'] and
                         reference['ref_id_a'] != reference['ref_id_b'])
        if present:
            if not queryable or status in {'not_located', 'not_queryable'}:
                raise ValueError('Present reference has incoherent IDs/status')
            if not refs:
                raise ValueError('Present reference evidence is required')
        elif status != ('not_located' if queryable else 'not_queryable') or refs:
            raise ValueError('Absent reference has incoherent IDs/status/evidence')
        identity = pair['benchmark_identity_admissible']
        candidate = pair['benchmark_candidate']
        available = present and status == 'usable'
        if candidate != (identity and present):
            raise ValueError('Candidate flag contradicts identity/reference presence')
        if pair['reference_usable_for_outcome'] != (candidate and available):
            raise ValueError('Outcome availability flag contradicts status projection')
        literal = pair['literal_inn_to_dcb_supported']
        official = pair['both_international_names_documented_inn']
        if literal != (identity and official):
            raise ValueError('Literal INN claim contradicts identity/name documentation')
        output.append(dict(
            pair_id=pid, drug_id_a=pair['drug_id_a'], drug_id_b=pair['drug_id_b'],
            context_id=pair['context_id'], reference_source_id=pair['source_id'],
            reference_source_version=pair['source_version'], identity_admitted=identity,
            benchmark_candidate=candidate, reference_record_present=present,
            reference_available=available, reference_quality_status=status,
            literal_inn_to_dcb_supported=literal,
            both_international_names_documented_inn=official,
            d01e_documentary_state='not_assessed' if candidate else 'outside_population',
            d01e_reason=(
                'direct_complete_strength_both_entities_and_pair_context_not_assessed'
                if candidate else 'outside_current_documentary_benchmark_population'
            ),
        ))
    if len(source_versions) != 1:
        raise ValueError('Population contains mixed reference source/version bindings')
    return pd.DataFrame(output, columns=OUTPUT_COLUMNS)
