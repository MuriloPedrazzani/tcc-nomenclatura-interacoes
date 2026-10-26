"""Metadata-to-parser adapter; no clinical reference or evaluator dependency."""
from __future__ import annotations

from collections import Counter, defaultdict

from src.dados.parser_cache_exploratorio import parse_cache_value


def process_queue(reader, queue, metadata):
    """Process the provided, previously validated allowlist without dropping rows.

    The real entrypoint separately pins the exact 2,859-entry authorized queue.
    No category summary is constructed here or in the reports.
    """
    ids, keys, result = set(), set(), []
    for item in queue:
        oid, key = item['observation_id'], item['cache_key']
        if oid in ids or key in keys:
            raise ValueError('Duplicate queue identity or cache key')
        ids.add(oid); keys.add(key)
        row = metadata[oid]
        if (not row['parsing_candidate'] or not row['binding_verified']
                or len(row['submission_execution_ids']) > 1
                or key != row['legacy_key']
                or any(item[k] != row[k] for k in ('pair_id','model','condition','pair_partition','execution_slot'))):
            raise ValueError('Queue is not supported by hardened metadata')
    for item in queue:
        row = metadata[item['observation_id']]
        fetched = reader.read_value(item['cache_key'])
        if fetched.status == 'absent':
            parsed = dict(status='cache_entry_absent', valid=False, category=None,
                          reasons=['cache_entry_absent'], history_absence=True,
                          collector_failure_observed=False)
        elif fetched.status == 'present':
            parsed = parse_cache_value(fetched.value, provider=row['provider'])
        else:
            raise ValueError('Unexpected cache cardinality status')
        result.append(dict(
            **{k: row[k] for k in ('observation_id','planned_observation_id','pair_id','model','provider',
                                  'family','condition','execution_slot','pair_partition','request_id',
                                  'request_fingerprint','payload_fingerprint','text_binding_sha256',
                                  'context_id','admission_status','submission_execution_ids')},
            cache_key=item['cache_key'], cache_read_status=fetched.status,
            parsing=parsed, category=parsed['category'] if parsed['valid'] else None,
            structural_processing_eligible=True,
            structural_response_valid=parsed['valid'],
            analytical_eligibility_final=False,
            historical_caveat_preserved=True,
            complete_observed_compatibility=False,
            response_execution_id_observed=None,
            response_submission_attribution='unknown_not_resolved_by_parsing',
            request_configuration_provenance='reconstructed_not_observed',
            dataset_scope='real_historical_parsing_adapter',
            new_independent_replications=0,
            reference_labels_read=False,
        ))
    return result


def inventories(planned_rows, paired_metadata, normalized):
    """Full outer inventory of the requested partitions, including exclusions."""
    by_id = {x['observation_id']: x for x in normalized}
    if len(by_id) != len(normalized): raise ValueError('Duplicate parsed observation')
    if not set(by_id) <= {r['observation_id'] for r in planned_rows}:
        raise ValueError('Parsed observation outside planned scope')
    inventory, groups, pair_groups = [], defaultdict(list), defaultdict(dict)
    for row in planned_rows:
        parsed = by_id.get(row['observation_id'])
        value = dict(**{k: row[k] for k in ('planned_observation_id','observation_id','pair_id','model','provider',
            'family','condition','pair_partition','execution_slot','admission_status')},
            metadata_support=row['parsing_candidate'],
            authorized_in_this_round=parsed is not None,
            parsed_this_round=parsed is not None,
            cache_value_available=bool(parsed and parsed['cache_read_status']=='present'),
            parsing_status=parsed['parsing']['status'] if parsed else 'not_parsed_metadata_exclusion',
            structural_response_valid=bool(parsed and parsed['structural_response_valid']),
            collector_failure_observed=bool(parsed and parsed['parsing']['collector_failure_observed']),
            historical_information_absent=bool(parsed and parsed['parsing']['history_absence']),
            reasons=parsed['parsing']['reasons'] if parsed else row['metadata_reasons'],
            analytical_eligibility_final=False)
        inventory.append(value)
        groups[(row['pair_partition'],row['model'],row['condition'])].append(value)
        key=(row['pair_partition'],row['pair_id'],row['model'],row['execution_slot'])
        if row['condition'] in pair_groups[key]: raise ValueError('Duplicate planned arm')
        pair_groups[key][row['condition']]=value
    summaries=[]
    for (partition,model,condition),rows in sorted(groups.items()):
        summaries.append(dict(pair_partition=partition,model=model,condition=condition,
            planned=len(rows),metadata_supported=sum(x['metadata_support'] for x in rows),
            authorized=sum(x['authorized_in_this_round'] for x in rows),
            cache_values_available=sum(x['cache_value_available'] for x in rows),
            valid=sum(x['structural_response_valid'] for x in rows),
            invalid=sum(x['parsed_this_round'] and not x['structural_response_valid'] for x in rows),
            not_parsed=sum(not x['parsed_this_round'] for x in rows),
            status_counts=dict(Counter(x['parsing_status'] for x in rows)),
            metadata_exclusion_counts=dict(Counter(x['admission_status'] for x in rows if not x['parsed_this_round']))))
    bindings={(r['pair_id'],r['model'],r['execution_slot']):r for r in paired_metadata}
    paired=[]
    for (partition,pid,model,slot),arms in sorted(pair_groups.items()):
        if set(arms) != {'dcb','inn'}: raise ValueError('Missing planned arm')
        a,b=arms['dcb'],arms['inn']; meta=bindings[(pid,model,slot)]
        valid=all(x['structural_response_valid'] for x in (a,b))
        paired.append(dict(pair_partition=partition,pair_id=pid,model=model,execution_slot=slot,
            dcb_observation_id=a['observation_id'],inn_observation_id=b['observation_id'],
            paired_before_parsing=meta['paired_parsing_candidate'],
            paired_binding_verified=meta['paired_binding_verified'],
            dcb_valid=a['structural_response_valid'],inn_valid=b['structural_response_valid'],
            both_structurally_valid=valid,
            paired_after_parsing=bool(valid and meta['paired_parsing_candidate'] and meta['paired_binding_verified']),
            dcb_status=a['parsing_status'],inn_status=b['parsing_status'],
            physical_pair_execution_observed=False,analytical_eligibility_final=False,
            historical_caveat_preserved=True,new_independent_replications=0))
    pair_summary=[]
    keys=sorted({(r['pair_partition'],r['model']) for r in paired})
    for partition,model in keys:
        rows=[r for r in paired if r['pair_partition']==partition and r['model']==model]
        pair_summary.append(dict(pair_partition=partition,model=model,planned_pairs=len(rows),
            pairs_before=sum(x['paired_before_parsing'] for x in rows),
            pairs_after=sum(x['paired_after_parsing'] for x in rows),
            both_valid=sum(x['both_structurally_valid'] for x in rows),
            single_valid=sum(x['dcb_valid'] != x['inn_valid'] for x in rows),
            neither_valid=sum(not x['dcb_valid'] and not x['inn_valid'] for x in rows)))
    return inventory,summaries,paired,pair_summary
