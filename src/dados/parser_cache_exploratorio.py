"""Offline adapter for the historical ``r.v = JSON [pred, bruto]`` contract.

This module neither opens files nor imports collectors. A caller must enforce
the authorized key scope before supplying a value. Successful structural
parsing does not prove request compatibility, execution attribution, clinical
correctness, or independent replication. No response text is returned.
"""
from __future__ import annotations

import json
import math


VERSION = 'exploratory-historical-cache-parser-v1'
LABELS = frozenset({'leve', 'moderado', 'grave'})
PROVIDERS = frozenset({'openai', 'anthropic'})


class _DuplicateKey(ValueError):
    pass


class _NonFinite(ValueError):
    pass


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey()
        result[key] = value
    return result


def _non_finite(_value):
    raise _NonFinite()


def _finite_float(value):
    converted = float(value)
    if not math.isfinite(converted):
        raise _NonFinite()
    return converted


def _decode(text):
    try:
        return json.loads(text, object_pairs_hook=_object, parse_constant=_non_finite,
                          parse_float=_finite_float), None
    except _DuplicateKey:
        return None, 'duplicate_json_keys'
    except _NonFinite:
        return None, 'non_finite_json_number'
    except (json.JSONDecodeError, RecursionError, ValueError):
        # Exception messages can contain response fragments; do not return them.
        return None, 'malformed_json'


def _label(value):
    return type(value) is str and value in LABELS


def _marker(raw):
    """Recognize collector tokens, never search free text for a category."""
    if raw == 'SEM_RESPOSTA':
        return 'historical_absence'
    if raw == 'ERRO' or raw.startswith('ERRO:'):
        return 'collector_error'
    if raw == 'SEM_TOOL_USE' or raw.startswith('SEM_TOOL_USE:'):
        return 'missing_tool_use'
    if raw.startswith('BATCH_') and len(raw) > len('BATCH_'):
        return 'batch_failure'
    return None


def parse_cache_value(value, *, provider):
    """Parse one already authorized SQLite value; return sanitized diagnostics.

    Accepted envelope: a string encoding a JSON array with exactly two members.
    ``pred`` is null or an exact historical label. ``bruto`` is null or a string
    encoding a JSON object, as serialized by both historical collectors.
    A category is released only when the valid structured category and the
    valid legacy prediction agree. Missing/invalid predictions are not repaired.

    The ``sev`` OpenAI schema allows only ``severidade``. The Anthropic schema
    permits additional fields; only their count is returned, never their keys
    or values. Strict JSON rejects duplicate keys and nonfinite constants.
    Collector failure markers are separate from absent historical content.
    """
    if type(provider) is not str or provider not in PROVIDERS:
        raise ValueError('Provider must be openai or anthropic')
    result = dict(
        parser_version=VERSION, provider=provider, schema_id='sev',
        status='invalid', valid=False, category=None, reasons=[],
        history_absence=False, collector_failure_observed=False,
        error_marker_kind=None, envelope_decoded=False,
        response_json_decoded=False, legacy_prediction_present=None,
        legacy_prediction_category_valid=None, raw_category_valid=None,
        legacy_agreement=None, additional_fields_count=None,
        historical_caveat=True, complete_compatibility_proven=False,
        execution_attribution_proven=False, independent_replication_created=False,
    )
    reasons = result['reasons']
    if value is None:
        result.update(status='historical_absence', history_absence=True)
        reasons.append('historical_cache_value_null')
        return result
    if type(value) is not str:
        reasons.append('cache_value_not_text')
        return result
    envelope, error = _decode(value)
    if error:
        reasons.append('envelope_' + error)
        return result
    if type(envelope) is not list or len(envelope) != 2:
        reasons.append('envelope_not_two_member_array')
        return result
    result['envelope_decoded'] = True
    pred, raw = envelope
    result['legacy_prediction_present'] = pred is not None
    result['legacy_prediction_category_valid'] = _label(pred)
    if pred is not None and not _label(pred):
        reasons.append('invalid_legacy_prediction')
    if raw is None or (type(raw) is str and not raw.strip()):
        result['history_absence'] = True
        reasons.append('historical_response_content_absent')
        if pred is not None:
            reasons.append('legacy_prediction_without_response_content')
        elif len(reasons) == 1:
            result['status'] = 'historical_absence'
        return result
    if type(raw) is not str:
        reasons.append('response_content_not_text_or_null')
        return result
    marker = _marker(raw)
    if marker:
        result['error_marker_kind'] = marker
        if marker == 'historical_absence':
            result['history_absence'] = True
            reasons.append('historical_missing_response_marker')
            if pred is None and len(reasons) == 1:
                result['status'] = 'historical_absence'
        else:
            result['collector_failure_observed'] = True
            reasons.append(marker)
            if pred is None and len(reasons) == 1:
                result['status'] = 'collector_failure'
        if pred is not None:
            reasons.append('legacy_prediction_conflicts_with_marker')
        return result
    body, error = _decode(raw)
    if error:
        reasons.append('response_' + error)
        return result
    result['response_json_decoded'] = True
    if type(body) is not dict:
        reasons.append('response_json_not_object')
        return result
    extras = len(set(body) - {'severidade'})
    result['additional_fields_count'] = extras
    if provider == 'openai' and extras:
        reasons.append('openai_sev_additional_fields_forbidden')
    if 'severidade' not in body:
        result['raw_category_valid'] = False
        reasons.append('structured_category_missing')
        return result
    category = body['severidade']
    result['raw_category_valid'] = _label(category)
    if not _label(category):
        reasons.append('invalid_structured_category')
        return result
    result['legacy_agreement'] = _label(pred) and pred == category
    if pred is None:
        reasons.append('legacy_prediction_missing_no_rescue')
    elif _label(pred) and pred != category:
        reasons.append('legacy_structured_category_contradiction')
    if not reasons:
        result.update(status='valid', valid=True, category=category)
    return result
