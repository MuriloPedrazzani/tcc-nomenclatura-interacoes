"""First-pass metadata admission; no response, severity, score or threshold inputs.

Parsing support is a prospective metadata decision, never observed compatibility
or response eligibility. All planned slots remain visible, including absent arms.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json


VERSION = 'exploratory-metadata-admission-v2'
SLOT = 'historico_a_identificar'
PARTITIONS = ('development', 'calibration', 'C', 'cross_partition', 'documentary_excluded')
PAIR_FIELDS = {'pair_id', 'drug_id_a', 'drug_id_b', 'benchmark_candidate', 'pair_partition', 'context_id'}
MODEL_FIELDS = {'model', 'provider', 'family', 'role'}
OBS_FIELDS = {'observation_id', 'request_id', 'pair_id', 'model', 'condition', 'execution_slot',
              'context_id', 'request_fingerprint', 'payload_fingerprint', 'text_binding_sha256',
              'legacy_key', 'binding_verified', 'cache_audited', 'cache_current_present',
              'classification', 'historical_use_allowed_with_caveat', 'submission_execution_ids',
              'response_execution_id_observed', 'complete_configuration_observed'}
OPTIONAL_OBS_FIELDS = {'response_execution_evidence'}
EXECUTION_EVIDENCE_FIELDS = {'source_path', 'source_sha256', 'source_locator',
                             'execution_id', 'legacy_key', 'request_fingerprint'}
SUPPORTED = {'supported_with_historical_caveat', 'supported_observed_compatible'}


def _id(value):
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def _execution_attribution(row, verifier):
    """Validate metadata coherence before delegating source verification.

    A verifier is a trusted, explicit boundary: it must check the pinned source
    hash, locator and unique attribution, not merely return an assertion from
    the supplied dictionary. This module performs no source or response IO.
    """
    execution = row['response_execution_id_observed']
    evidence = row.get('response_execution_evidence')
    if evidence is not None:
        if (not isinstance(evidence, dict) or set(evidence) != EXECUTION_EVIDENCE_FIELDS
                or not all(_id(value) for value in evidence.values())):
            raise ValueError('Strict traceable execution evidence fields required')
        digest = evidence['source_sha256']
        if len(digest) != 64 or any(char not in '0123456789abcdef' for char in digest):
            raise ValueError('Execution evidence requires a SHA-256 source digest')
    if execution is None:
        if evidence is not None:
            raise ValueError('Execution evidence without an execution claim')
        return False, 'execution_attribution_unknown'
    if execution not in row['submission_execution_ids']:
        return False, 'execution_not_among_documented_submissions'
    if evidence is None:
        return False, 'missing_execution_attribution_evidence'
    if (evidence['execution_id'] != execution or evidence['legacy_key'] != row['legacy_key']
            or evidence['request_fingerprint'] != row['request_fingerprint']):
        return False, 'execution_attribution_evidence_mismatch'
    if verifier is None:
        return False, 'execution_attribution_source_not_verified'
    if verifier(dict(evidence)) is not True:
        return False, 'execution_attribution_source_verification_failed'
    return True, None


def construir_matriz_admissao(membership, snapshots, observations, *, response_execution_verifier=None):
    """Outer product of population, registry and arms; no observed-only join.

    Legacy metadata remain valid inputs. Execution identifiers are claims until
    coherent, traceable evidence is checked by the explicit optional verifier.
    Omitting that verifier cannot establish observed execution attribution.
    """
    if response_execution_verifier is not None and not callable(response_execution_verifier):
        raise ValueError('Execution source verifier must be callable or absent')
    pairs, models, grouped, cache_links = {}, {}, defaultdict(list), defaultdict(set)
    for row in membership:
        if set(row) != PAIR_FIELDS or not all(_id(row[k]) for k in PAIR_FIELDS-{'benchmark_candidate'}):
            raise ValueError('Only explicit population metadata fields are allowed')
        if row['pair_id'] in pairs or row['drug_id_a'] == row['drug_id_b']:
            raise ValueError('Duplicate pair or self edge')
        if type(row['benchmark_candidate']) is not bool or row['pair_partition'] not in PARTITIONS:
            raise ValueError('Invalid population/split flags')
        if row['benchmark_candidate'] == (row['pair_partition'] == 'documentary_excluded'):
            raise ValueError('Documentary membership conflicts with partition')
        pairs[row['pair_id']] = dict(row)
    for row in snapshots:
        if set(row) != MODEL_FIELDS or not all(_id(v) for v in row.values()) or row['model'] in models:
            raise ValueError('Invalid or duplicate registered snapshot')
        models[row['model']] = dict(row)
    if not pairs or not models:
        raise ValueError('Empty population/registry')
    seen = set()
    for row in observations:
        if not OBS_FIELDS <= set(row) or set(row) - OBS_FIELDS - OPTIONAL_OBS_FIELDS:
            raise ValueError('Only projected metadata accepted; outcomes are forbidden')
        ids = OBS_FIELDS-{'binding_verified', 'cache_audited', 'cache_current_present',
                        'historical_use_allowed_with_caveat', 'complete_configuration_observed',
                        'submission_execution_ids', 'response_execution_id_observed'}
        if not all(_id(row[k]) for k in ids):
            raise ValueError('Missing metadata identity')
        if row['observation_id'] in seen:
            raise ValueError('Duplicate observation identifier')
        seen.add(row['observation_id'])
        if row['pair_id'] not in pairs or row['model'] not in models:
            raise ValueError('Orphan observation')
        if row['condition'] not in {'dcb', 'inn'} or row['execution_slot'] != SLOT:
            raise ValueError('Unsupported arm or invented execution slot')
        if row['context_id'] != pairs[row['pair_id']]['context_id']:
            raise ValueError('Context mismatch')
        for key in ('binding_verified', 'cache_audited', 'cache_current_present',
                    'historical_use_allowed_with_caveat', 'complete_configuration_observed'):
            if type(row[key]) is not bool:
                raise ValueError('Strict metadata booleans required')
        if row['classification'] not in {'compativel', 'indeterminada', 'incompativel', 'ausente'}:
            raise ValueError('Unknown compatibility classification')
        subs = row['submission_execution_ids']
        if not isinstance(subs, list) or not all(_id(x) for x in subs) or len(set(subs)) != len(subs):
            raise ValueError('Invalid or duplicated submission identities')
        execution = row['response_execution_id_observed']
        if execution is not None and not _id(execution):
            raise ValueError('Invalid response execution identity')
        verified, attribution_reason = _execution_attribution(row, response_execution_verifier)
        key = (row['pair_id'], row['model'], row['condition'])
        projected = dict(row)
        projected['response_execution_attribution_verified'] = verified
        projected['response_execution_attribution_reason'] = attribution_reason
        grouped[key].append(projected)
        cache_links[row['legacy_key']].add(key)
    output = []
    for pid in sorted(pairs):
        pair = pairs[pid]
        for model in sorted(models):
            for condition in ('dcb', 'inn'):
                records = grouped[(pid, model, condition)]
                meta = records[0] if len(records) == 1 else None
                reasons = []
                if not records:
                    reasons.append('missing_request_metadata')
                elif len(records) > 1:
                    reasons.append('ambiguous_request_metadata')
                else:
                    if not meta['binding_verified']: reasons.append('unverified_text_context_binding')
                    if meta['classification'] == 'incompativel': reasons.append('material_metadata_conflict')
                    if not meta['cache_audited']: reasons.append('cache_access_unverified')
                    elif not meta['cache_current_present']: reasons.append('missing_current_cache_key')
                    if len(meta['submission_execution_ids']) > 1 and not meta['response_execution_attribution_verified']:
                        reasons.append('ambiguous_submission_attribution')
                    if len(cache_links[meta['legacy_key']]) > 1:
                        reasons.append('shared_cache_key_requires_link_rule')
                    if not meta['historical_use_allowed_with_caveat']:
                        reasons.append('historical_use_not_supported')
                complete = bool(meta and meta['classification'] == 'compativel'
                                and meta['complete_configuration_observed']
                                and meta['response_execution_attribution_verified'] and not reasons)
                status = reasons[0] if reasons else ('supported_observed_compatible' if complete
                                                     else 'supported_with_historical_caveat')
                partition = pair['pair_partition']
                scope_reason = ('outside_documentary_population' if not pair['benchmark_candidate'] else
                                'outside_analysis_cross_partition' if partition == 'cross_partition' else None)
                admission = scope_reason or status
                result = dict(pair, **{k: models[model][k] for k in ('model','provider','family')},
                    condition=condition, execution_slot=SLOT, execution_slot_provenance='reconstructed_logical_plan',
                    planned_observation_id='PLAN_'+_digest([pid,model,condition,SLOT]),
                    request_metadata_count=len(records),
                    source_observation_ids=sorted(x['observation_id'] for x in records),
                    link_status=status, metadata_reasons=reasons, admission_status=admission,
                    scope_exclusion_reason=scope_reason,
                    metadata_link_supported=status in SUPPORTED,
                    parsing_candidate=scope_reason is None and status in SUPPORTED,
                    complete_observed_compatibility=complete,
                    response_validity_checked=False, response_parsing_authorized=False,
                    empirical_analysis_authorized=False, new_independent_replications=0,
                    dataset_scope='real_metadata_admission', contract_version=VERSION)
                for field in sorted(OBS_FIELDS-{'pair_id','model','condition','execution_slot','context_id'}):
                    result[field] = meta[field] if meta else None
                result['submission_execution_ids'] = sorted(meta['submission_execution_ids']) if meta else []
                attributed = bool(meta and meta['response_execution_attribution_verified'])
                result['response_execution_id_claimed'] = meta['response_execution_id_observed'] if meta else None
                result['response_execution_id_observed'] = meta['response_execution_id_observed'] if attributed else None
                result['response_execution_attribution_verified'] = attributed
                result['response_execution_attribution_reason'] = (meta['response_execution_attribution_reason']
                    if meta else 'unique_request_metadata_unavailable')
                result['response_execution_evidence'] = (dict(meta['response_execution_evidence'])
                    if meta and meta.get('response_execution_evidence') is not None else None)
                result['shared_cache_key_observations'] = len(cache_links[meta['legacy_key']]) if meta else 0
                output.append(result)
    return output


def resumir_configuracoes(matrix, snapshots):
    groups = defaultdict(list)
    for row in matrix:
        groups[(row['model'], row['condition'], row['pair_partition'])].append(row)
    output = []
    for snapshot in sorted(snapshots, key=lambda x:x['model']):
        for condition in ('dcb','inn'):
            for partition in PARTITIONS:
                rows = groups[(snapshot['model'],condition,partition)]
                output.append(dict(model=snapshot['model'],family=snapshot['family'],condition=condition,
                    execution_slot=SLOT,pair_partition=partition,planned_observations=len(rows),
                    current_cache_keys_present=sum(x['cache_current_present'] is True for x in rows),
                    metadata_link_supported=sum(x['metadata_link_supported'] for x in rows),
                    parsing_candidates=sum(x['parsing_candidate'] for x in rows),
                    observed_compatible=sum(x['complete_observed_compatibility'] for x in rows),
                    admission_counts=dict(Counter(x['admission_status'] for x in rows)),
                    link_status_counts=dict(Counter(x['link_status'] for x in rows))))
    return output


def parear_admissao(matrix):
    groups = defaultdict(dict)
    for row in matrix:
        key = (row['pair_id'],row['model'],row['execution_slot'])
        if row['condition'] in groups[key]: raise ValueError('Duplicate planned arm')
        groups[key][row['condition']] = row
    output=[]
    for (pid,model,slot),arms in sorted(groups.items()):
        if set(arms) != {'dcb','inn'}: raise ValueError('Both planned arms must remain present')
        left,right=arms['dcb'],arms['inn']
        fingerprints_equal=bool(left['text_binding_sha256']
                                and left['text_binding_sha256']==right['text_binding_sha256'])
        contexts_equal=bool(left['context_id'] and left['context_id']==right['context_id'])
        both_individually_verified=left.get('binding_verified') is True and right.get('binding_verified') is True
        same_binding=fingerprints_equal and contexts_equal and both_individually_verified
        output.append(dict(pair_id=pid,model=model,execution_slot=slot,pair_partition=left['pair_partition'],
            dcb_observation_id=left['observation_id'],inn_observation_id=right['observation_id'],
            dcb_admission_status=left['admission_status'],inn_admission_status=right['admission_status'],
            both_current_cache_keys_present=left['cache_current_present'] is True and right['cache_current_present'] is True,
            both_cache_keys_absent=left['cache_current_present'] is not True and right['cache_current_present'] is not True,
            paired_parsing_candidate=left['parsing_candidate'] and right['parsing_candidate'] and same_binding,
            binding_fingerprints_equal=fingerprints_equal,contexts_equal=contexts_equal,
            both_individual_bindings_verified=both_individually_verified,
            paired_binding_verified=same_binding,
            physical_pair_execution_observed=False,
            pair_execution_provenance='reconstructed_logical_cache_link_not_synchronous_execution_proof',
            response_validity_checked=False,new_independent_replications=0))
    return output
