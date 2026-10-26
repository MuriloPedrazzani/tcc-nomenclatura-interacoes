"""Pure future real-data transfer adapter; no files, cache, fitting or inference.

Only small invented fixtures exercise outcomes during Freeze 2 preparation.
Supplied artifact hashes bind contracts; the caller verifies the actual files
and formal freeze envelope. This module never certifies their provenance.
"""
from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
import hashlib
import json
import math
from numbers import Real
import re

import numpy as np
import pandas as pd

from transferencia_familias import APPROVED_SNAPSHOT_PAIRS
from metodos_freeze1_v1_3_1 import MODEL_FAMILIES


VERSION = 'exploratory-real-transfer-global-v1'
SLOT = 'historico_a_identificar'
COVERAGES = (.6, .7, .8, .9)
MODEL_MAP = dict(APPROVED_SNAPSHOT_PAIRS)
_KNOWN_FAMILIES = {**MODEL_FAMILIES, 'gpt-4o-mini-2024-07-18': 'referência histórica secundária'}
HASH_FIELDS = {'policy_state_sha256', 'features_sha256', 'split_sha256',
               'eligibility_sha256', 'source_projection_sha256'}
VERSION_FIELDS = {'feature_version', 'denominator_version', 'provenance_version'}
CANDIDATE_FIELDS = {
    'threshold_id', 'threshold_sha256', 'version', 'model', 'condition',
    'execution_slot', 'pair_partition', 'target_coverage', 'coverage_basis',
    'source_configuration_sha256', 'planned_count', 'available_count', 'eligible_count',
    'unavailable_count', 'ineligible_available_count', 'threshold', 'threshold_status',
    'non_estimable_reason', 'desired_quota_raw', 'nominal_eligible_quota',
    'calibration_quantile_fraction', 'calibration_accepted_count',
    'observed_source_coverage_global', 'observed_source_coverage_eligible_after_gate',
    'strictly_below_threshold_count', 'at_threshold_count', 'additional_accepted_due_to_ties',
    'source_gate_saturated', 'source_selector_active', 'requested_global_quota_feasible',
    'comparator', 'floating_tolerance_added', 'tie_seed', 'tie_seed_status',
    'source_only', 'target_application_performed', 'threshold_formally_frozen',
    'clinical_labels_used', 'DEFINITION_OF_DONE',
} | HASH_FIELDS | VERSION_FIELDS
ROW_FIELDS = {
    'observation_id', 'pair_id', 'model', 'provider', 'family', 'condition',
    'execution_slot', 'pair_partition', 'drug_id_a', 'drug_id_b', 'context_id',
    'available', 'eligible', 'eligible_d01_response', 'reference_available',
    'morphology_stratum', 'd01e_eligible', 'category', 'severity_label',
}
LABELS = {'leve', 'moderado', 'grave'}
SOURCE_FIELDS = {'observation_id','pair_id','model','condition','execution_slot',
                 'pair_partition','available','eligible','risk_score'}
COMMON_COPY_FIELDS = {
    'model','condition','execution_slot','pair_partition','target_coverage','threshold',
    'planned_count','available_count','eligible_count','unavailable_count',
    'ineligible_available_count','nominal_eligible_quota','calibration_accepted_count',
    'observed_source_coverage_global','observed_source_coverage_eligible_after_gate',
    'strictly_below_threshold_count','at_threshold_count','additional_accepted_due_to_ties',
    'source_selector_active','comparator','floating_tolerance_added',
}
COMMON_FIELDS = COMMON_COPY_FIELDS | {
    'threshold_id','threshold_sha256','threshold_status','non_estimable_reason',
    'operational_coverage_basis','operational_basis_decision','calculation_bases',
    'candidate_numeric_invariant_to_current_bases','base_equivalence_proof',
    'provenance_hashes','provenance_versions','tie_seed','tie_seed_required_for_threshold',
    'thresholds_formally_frozen','target_application_performed','DEFINITION_OF_DONE',
}
PROOF_FIELDS = {'model','condition','execution_slot','coverage','same_threshold',
                'same_source_accepted_ids','source_accepted_ids_sha256',
                'eligible_calculation_sha256','global_calculation_sha256'}


def _plain(value):
    if isinstance(value, dict): return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(v) for v in value]
    if value is pd.NA: return None
    if isinstance(value, np.generic): return _plain(value.item())
    if isinstance(value, float) and math.isnan(value): return None
    return value


def _sha(value):
    return hashlib.sha256(json.dumps(_plain(value), ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _id(value):
    return isinstance(value, str) and bool(value) and value == value.strip()


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def _frame(table, columns, name):
    if (not isinstance(table, pd.DataFrame) or table.columns.duplicated().any()
            or set(table.columns) != set(columns)):
        raise ValueError(name + ': exact explicit schema required')


def _candidate_rows(candidates):
    _frame(candidates, CANDIDATE_FIELDS, 'candidates')
    rows = _plain(candidates.to_dict('records'))
    expected = {(s, arm, SLOT, c) for s in MODEL_MAP for arm in ('dcb', 'inn') for c in COVERAGES}
    keys = [(r['model'], r['condition'], r['execution_slot'], r['target_coverage']) for r in rows]
    if len(keys) != 24 or len(set(keys)) != 24 or set(keys) != expected:
        raise ValueError('Exactly the 24 approved source configurations are required')
    if len({r['threshold_id'] for r in rows}) != 24:
        raise ValueError('Duplicate threshold identity')
    for r in rows:
        original = {k: v for k, v in r.items() if k not in {'threshold_id', 'threshold_sha256'}}
        digest = _sha(original)
        if r['threshold_sha256'] != digest or r['threshold_id'] != 'CANDTHR_' + digest[:24]:
            raise ValueError('Candidate content differs from its immutable identity/hash')
        if (r['pair_partition'] != 'calibration' or r['comparator'] != 'score <= threshold'
                or r['source_only'] is not True or r['tie_seed'] is not None
                or r['tie_seed_status'] != 'not_required_inclusive_threshold'
                or r['coverage_basis'] not in {'global', 'eligible_after_gate'}):
            raise ValueError('Invalid source-only inclusive candidate contract')
        if any(r[k] is not False for k in ('target_application_performed', 'threshold_formally_frozen',
                                         'clinical_labels_used', 'floating_tolerance_added', 'DEFINITION_OF_DONE')):
            raise ValueError('Candidate contains applied, frozen or clinical claims')
        if any(not _digest(r[k]) for k in HASH_FIELDS | {'source_configuration_sha256'}):
            raise ValueError('Invalid candidate provenance digest')
        if any(not _id(r[k]) for k in VERSION_FIELDS | {'version'}):
            raise ValueError('Missing candidate version')
        for k in ('planned_count', 'available_count', 'eligible_count', 'unavailable_count', 'ineligible_available_count'):
            if type(r[k]) is not int or r[k] < 0:
                raise ValueError('Invalid candidate operational count')
        planned, n, e = r['planned_count'], r['available_count'], r['eligible_count']
        if not planned >= n >= e or r['unavailable_count'] != planned-n or r['ineligible_available_count'] != n-e:
            raise ValueError('Incoherent candidate operational counts')
        if r['threshold_status'] == 'not_estimable':
            reason = ('configuration_absent' if not planned else 'no_available_observations' if not n
                      else 'no_eligible_observations' if not e else None)
            if reason is None or r['threshold'] is not None or r['non_estimable_reason'] != reason:
                raise ValueError('Unsupported non-estimable candidate')
        elif r['threshold_status'] == 'calculated_candidate':
            if not n or not e or not _finite(r['threshold']) or r['non_estimable_reason'] is not None:
                raise ValueError('Invalid calculable threshold')
            a, below, at = (r[k] for k in ('calibration_accepted_count', 'strictly_below_threshold_count', 'at_threshold_count'))
            if (any(type(v) is not int for v in (a, below, at)) or not 0 < a <= e
                    or below < 0 or at < 1 or below+at != a):
                raise ValueError('Inconsistent inclusive threshold counts')
            if (r['observed_source_coverage_global'] != a/n
                    or r['observed_source_coverage_eligible_after_gate'] != a/e):
                raise ValueError('Source operational denominator mismatch')
            original_raw = math.floor((n if r['coverage_basis'] == 'global' else e)*r['target_coverage']+.5)
            global_raw = math.floor(n*r['target_coverage']+.5)
            if r['desired_quota_raw'] != original_raw or r['nominal_eligible_quota'] != min(original_raw, e):
                raise ValueError('Candidate quota contradicts its source contract')
            if min(global_raw, e) != min(original_raw, e):
                raise ValueError('Global binding is not rank-equivalent; preserve candidate and stop')
        else:
            raise ValueError('Unknown candidate status')
    return sorted(rows, key=lambda r: (r['model'], r['condition'], r['target_coverage']))


def bind_thresholds(common_candidates, source_calculations, *, approval_sha256,
                    candidate_artifact_sha256, source_calculation_artifact_sha256):
    """Bind the unchanged 24 values/IDs to the approved GLOBAL denominator.

    The candidate payload and hash remain intact. This is additive metadata,
    never calibration or a formal freeze. If changing the basis changes the
    candidate order statistic, fail instead of inventing another threshold.
    """
    if not all(_digest(x) for x in (approval_sha256, candidate_artifact_sha256, source_calculation_artifact_sha256)):
        raise ValueError('Explicit approval and both source artifact digests required')
    _frame(common_candidates, COMMON_FIELDS, 'common candidates')
    originals = _candidate_rows(source_calculations)
    if any(r['coverage_basis'] != 'global' for r in originals):
        raise ValueError('The supporting calculation must use the approved global basis')
    raw_by_key = {(r['model'],r['condition'],r['execution_slot'],r['target_coverage']):r for r in originals}
    commons = _plain(common_candidates.to_dict('records'))
    keys = [(r['model'],r['condition'],r['execution_slot'],r['target_coverage']) for r in commons]
    if len(keys) != 24 or len(set(keys)) != 24 or set(keys) != set(raw_by_key):
        raise ValueError('Exactly the 24 common candidates and corresponding calculations are required')
    if len({r['threshold_id'] for r in commons}) != 24:
        raise ValueError('Duplicated common candidate identity')
    output = []
    for c in sorted(commons,key=lambda r:(r['model'],r['condition'],r['target_coverage'])):
        common_hash = _sha({k:v for k,v in c.items() if k not in {'threshold_id','threshold_sha256'}})
        if c['threshold_sha256'] != common_hash or c['threshold_id'] != 'CANDTHR_COMMON_'+common_hash[:24]:
            raise ValueError('Common candidate hash or identity changed')
        r = raw_by_key[(c['model'],c['condition'],c['execution_slot'],c['target_coverage'])]
        if (r['threshold_status'] != 'calculated_candidate' or not _finite(c['threshold'])
                or c['threshold_status'] != 'calculated_candidate_common_to_both_bases'
                or c['non_estimable_reason'] is not None or c['operational_coverage_basis'] is not None
                or c['operational_basis_decision'] != 'pending_transfer_scope_decision'
                or c['calculation_bases'] != ['eligible_after_gate','global']
                or c['candidate_numeric_invariant_to_current_bases'] is not True
                or c['tie_seed'] is not None):
            raise ValueError('Unsupported common candidate preparation state')
        if any(c[k] is not False for k in ('tie_seed_required_for_threshold','thresholds_formally_frozen',
                                           'target_application_performed','DEFINITION_OF_DONE')):
            raise ValueError('Common candidate is not an untouched preparation artifact')
        if any(c[k] != r[k] for k in COMMON_COPY_FIELDS):
            raise ValueError('Common candidate disagrees with its global source calculation')
        if c['provenance_hashes'] != {k:r[k] for k in HASH_FIELDS} or c['provenance_versions'] != {k:r[k] for k in VERSION_FIELDS}:
            raise ValueError('Common/source provenance mismatch')
        proof = c['base_equivalence_proof']
        if not isinstance(proof,dict) or set(proof) != PROOF_FIELDS:
            raise ValueError('Exact common/global proof required')
        if (any(proof[k] != c[k] for k in ('model','condition','execution_slot'))
                or proof['coverage'] != c['target_coverage'] or proof['same_threshold'] is not True
                or proof['same_source_accepted_ids'] is not True
                or proof['global_calculation_sha256'] != r['threshold_sha256']
                or any(not _digest(proof[k]) for k in ('global_calculation_sha256','eligible_calculation_sha256','source_accepted_ids_sha256'))):
            raise ValueError('Incoherent common/global equivalence proof')
        payload = dict(version=VERSION, threshold_id=c['threshold_id'], threshold=c['threshold'],
                       candidate_threshold_sha256=c['threshold_sha256'], candidate_payload=c,
                       source_calculation_payload=r, source_calculation_sha256=r['threshold_sha256'],
                       source_model=r['model'], target_model=MODEL_MAP[r['model']], condition=r['condition'],
                       execution_slot=r['execution_slot'], target_coverage=r['target_coverage'],
                       coverage_basis='global', approval_sha256=approval_sha256,
                       candidate_artifact_sha256=candidate_artifact_sha256,
                       source_calculation_artifact_sha256=source_calculation_artifact_sha256)
        output.append(dict(**payload, binding_sha256=_sha(payload)))
    return {'thresholds': pd.DataFrame(output, dtype=object), 'metadata': dict(
        version=VERSION, threshold_count=24, source_configuration_count=6, coverage_basis='global',
        candidate_ids_and_values_preserved=True, recalibration_performed=False,
        thresholds_formally_frozen=False, target_application_performed=False,
        approval_sha256=approval_sha256, candidate_artifact_sha256=candidate_artifact_sha256,
        source_calculation_artifact_sha256=source_calculation_artifact_sha256,
        bindings_sha256=_sha([r['binding_sha256'] for r in output]),
        supplied_hashes_independently_verified=False, DEFINITION_OF_DONE=False)}


def _bound_rows(thresholds, approval_sha256):
    if not isinstance(thresholds, pd.DataFrame) or thresholds.empty or thresholds.columns.duplicated().any():
        raise ValueError('Explicit bound threshold inventory required')
    rows = _plain(thresholds.to_dict('records'))
    if any(not isinstance(r.get('candidate_payload'), Mapping) for r in rows):
        raise ValueError('Missing preserved source candidate')
    artifacts = {r.get('candidate_artifact_sha256') for r in rows}
    source_artifacts = {r.get('source_calculation_artifact_sha256') for r in rows}
    if len(artifacts) != 1 or len(source_artifacts) != 1: raise ValueError('Mixed candidate artifact versions')
    expected = bind_thresholds(pd.DataFrame([r['candidate_payload'] for r in rows], dtype=object),
                               pd.DataFrame([r['source_calculation_payload'] for r in rows],dtype=object),
                               approval_sha256=approval_sha256,
                               candidate_artifact_sha256=next(iter(artifacts)),
                               source_calculation_artifact_sha256=next(iter(source_artifacts)))['thresholds']
    by_id = {r['threshold_id']: r for r in rows}
    if len(by_id) != 24 or any(by_id.get(r['threshold_id']) != r for r in expected.to_dict('records')):
        raise ValueError('Bound threshold content or approval differs from the frozen binding')
    return expected.to_dict('records')


def _validate_rows(rows, scores):
    _frame(rows, ROW_FIELDS, 'rows')
    _frame(scores, {'pair_id', 'risk_score'}, 'scores')
    ids = ROW_FIELDS - {'available', 'eligible', 'eligible_d01_response', 'reference_available',
                        'd01e_eligible', 'category', 'severity_label'}
    for field in ids:
        if not rows[field].map(_id).all(): raise ValueError('Invalid literal row identity')
    if (rows.observation_id.duplicated().any()
            or rows[['pair_id', 'model', 'condition', 'execution_slot', 'pair_partition']].duplicated().any()):
        raise ValueError('Duplicated observation or composite binding')
    if scores.pair_id.duplicated().any() or not scores.pair_id.map(_id).all():
        raise ValueError('Invalid or duplicated score identity')
    if set(scores.pair_id) != set(rows.pair_id): raise ValueError('Missing or orphan score pair')
    if not scores.risk_score.map(_finite).all(): raise ValueError('Finite feature-only scores required')
    for field in ('available', 'eligible', 'eligible_d01_response', 'reference_available'):
        if not rows[field].map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise ValueError('Explicit mask booleans required')
    if not rows.d01e_eligible.map(lambda x: x is None or x is pd.NA or isinstance(x, (bool, np.bool_))).all():
        raise ValueError('D01-E requires an explicit tri-state mask')
    # A local typed copy keeps boolean sums numeric, including np.bool_ in object columns.
    rows = rows.astype({field: bool for field in (
        'available', 'eligible', 'eligible_d01_response', 'reference_available')})
    if (not rows.condition.isin(['inn', 'dcb']).all() or not rows.execution_slot.eq(SLOT).all()
            or not rows.pair_partition.eq('C').all()):
        raise ValueError('Unapproved condition, slot or partition')
    # Avoid Python bool bitwise inversion when validated masks have object dtype.
    if (rows.eligible & rows.available.eq(False)).any() or (rows.eligible_d01_response & rows.available.eq(False)).any():
        raise ValueError('Eligible without admitted availability')
    if not rows.morphology_stratum.isin(['morphology_delta_zero', 'morphology_delta_nonzero']).all():
        raise ValueError('Invalid morphology stratum')
    for r in rows.to_dict('records'):
        if (r['model'] not in _KNOWN_FAMILIES or r['family'] != _KNOWN_FAMILIES[r['model']]
                or r['provider'] != ('anthropic' if r['model'].startswith('claude-') else 'openai')):
            raise ValueError('Snapshot/provider/family provenance mismatch')
        if r['drug_id_a'] == r['drug_id_b']: raise ValueError('Self pair')
        if r['category'] is not None and r['category'] not in LABELS: raise ValueError('Invalid prediction category')
        if r['severity_label'] is not None and r['severity_label'] not in LABELS: raise ValueError('Invalid reference category')
        if r['eligible'] and (r['category'] not in LABELS or not r['reference_available']):
            raise ValueError('Eligible outcome without valid response/reference availability')
        if bool(r['reference_available']) != (r['severity_label'] in LABELS):
            raise ValueError('Reference availability and category disagree')
        if not r['available'] and r['category'] is not None:
            raise ValueError('Unprocessed position cannot contain a prediction')
    for _, group in rows.groupby('pair_id', sort=False):
        for field in ('drug_id_a', 'drug_id_b', 'context_id', 'pair_partition', 'morphology_stratum',
                      'reference_available', 'severity_label', 'd01e_eligible'):
            if len({_sha(_plain(v)) for v in group[field]}) != 1:
                raise ValueError('Pair-level identity/context/reference join is incoherent')
    return rows.merge(scores, on='pair_id', validate='many_to_one').sort_values('observation_id').reset_index(drop=True)


def _validate_source(source_scores, target_rows, source_pair_metadata):
    _frame(source_scores,SOURCE_FIELDS,'source scores without categories')
    source = source_scores.copy()
    for field in SOURCE_FIELDS-{'available','eligible','risk_score'}:
        if not source[field].map(_id).all():raise ValueError('Invalid source identity')
    if (source.observation_id.duplicated().any()
            or source[['pair_id','model','condition','execution_slot']].duplicated().any()):
        raise ValueError('Duplicate source observation or binding')
    if (not source.model.isin(MODEL_MAP).all() or not source.condition.isin(['inn','dcb']).all()
            or not source.execution_slot.eq(SLOT).all() or not source.pair_partition.eq('calibration').all()):
        raise ValueError('Unapproved source snapshot, slot, condition or partition')
    for field in ('available','eligible'):
        if not source[field].map(lambda x:isinstance(x,(bool,np.bool_))).all():
            raise ValueError('Explicit source mask booleans required')
    source = source.astype({'available': bool, 'eligible': bool})
    if (source.eligible & source.available.eq(False)).any():raise ValueError('Source eligible without availability')
    for r in source.to_dict('records'):
        if not _finite(r['risk_score']) and not (not r['eligible'] and _plain(r['risk_score']) is None):
            raise ValueError('Eligible source score must be finite')
    if set(source.pair_id) & set(target_rows.pair_id):raise ValueError('Source and target share a pair')
    if set(source.observation_id) & set(target_rows.observation_id):raise ValueError('Source/target observation collision')
    checked = False
    if source_pair_metadata is not None:
        _frame(source_pair_metadata,{'pair_id','drug_id_a','drug_id_b','context_id'},'source pair metadata')
        if source_pair_metadata.pair_id.duplicated().any() or set(source_pair_metadata.pair_id) != set(source.pair_id):
            raise ValueError('Source endpoint metadata are duplicated or orphaned')
        if not all(source_pair_metadata[col].map(_id).all() for col in source_pair_metadata.columns):
            raise ValueError('Invalid source endpoint metadata')
        if source_pair_metadata.drug_id_a.eq(source_pair_metadata.drug_id_b).any():raise ValueError('Source self pair')
        if (set(source_pair_metadata.drug_id_a)|set(source_pair_metadata.drug_id_b)) & (set(target_rows.drug_id_a)|set(target_rows.drug_id_b)):
            raise ValueError('Source calibration and C share an entity')
        checked = True
    return source.sort_values(['model','condition','pair_id']).reset_index(drop=True),checked


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _target_summary(group, accepted):
    """Descriptive target mechanisms; undefined clinical denominators stay None."""
    available = group.available
    eligible = available & group.eligible
    known = available & group.reference_available
    severe = known & group.severity_label.eq('grave')
    severe_accepted = severe & accepted
    u1 = severe_accepted & group.category.isin(['leve', 'moderado'])
    u2 = severe_accepted & group.category.eq('leve')
    comparable = known & accepted
    errors = comparable & group.category.ne(group.severity_label)
    p, n, e, a, d = len(group), int(available.sum()), int(eligible.sum()), int(accepted.sum()), int(severe.sum())
    return dict(target_planned=p, target_available=n, target_eligible=e,
        target_accepted=a, target_unavailable=p-n, target_gate_ineligible_available=n-e,
        target_selector_abstentions=e-a, target_operational_abstentions=n-a,
        target_coverage_global=_ratio(a, n), target_coverage_eligible_after_gate=_ratio(a, e),
        known_reference_available=int(known.sum()), unknown_reference_available=int((available & ~known).sum()),
        clinical_denominator=d, clinical_denominator_name='known_severe_available_global',
        severe_accepted=int(severe_accepted.sum()), severe_routed=int((severe & ~accepted).sum()),
        severe_routing_rate=_ratio(int((severe & ~accepted).sum()), d),
        u_delta1_count=int(u1.sum()), u_delta2_count=int(u2.sum()),
        underestimation_rate_delta1=_ratio(int(u1.sum()), d),
        underestimation_rate_delta2=_ratio(int(u2.sum()), d),
        risk_among_severe_accepted_delta1=_ratio(int(u1.sum()), int(severe_accepted.sum())),
        residual_error_count=int(errors.sum()), residual_error_denominator=int(comparable.sum()),
        residual_error_rate=_ratio(int(errors.sum()), int(comparable.sum())),
        target_gate_saturated=bool(n and e < n and a == e),
        target_selector_inactive=bool(e and a == e), total_abstention=bool(n and not a),
        equivalence_claim=False, superiority_claim=False, inferential_claim=False)


def aplicar_transferencia(target_rows, target_scores, thresholds, source_scores, *, freeze_manifest_sha256,
                          approval_sha256, formal_freeze_complete=False, source_pair_metadata=None):
    """Apply frozen inclusive thresholds in a future authorized evaluation only.

    The caller must verify files against the formal freeze envelope before
    invoking this pure function. Passing digests is not such verification.
    No threshold is fitted here and no target quota is imposed.
    """
    if formal_freeze_complete is not True or not _digest(freeze_manifest_sha256) or not _digest(approval_sha256):
        raise ValueError('Explicit formal freeze and artifact bindings required before evaluation')
    bound = _bound_rows(thresholds, approval_sha256)
    joined = _validate_rows(target_rows, target_scores)
    sources, drug_disjoint_checked = _validate_source(source_scores, target_rows, source_pair_metadata)
    configurations, decisions = [], []
    for t in bound:
        candidate = t['source_calculation_payload']
        source = sources.loc[sources.model.eq(t['source_model']) & sources.condition.eq(t['condition'])].sort_values('pair_id')
        target = joined.loc[joined.model.eq(t['target_model']) & joined.condition.eq(t['condition'])
                            & joined.pair_partition.eq('C')].sort_values('pair_id')
        projected = _plain(source.to_dict('records'))
        if _sha(projected) != candidate['source_configuration_sha256']:
            raise ValueError('Source identities, masks or scores differ from the preserved candidate')
        source_n, source_e = int(source.available.sum()), int((source.available & source.eligible).sum())
        if (len(source), source_n, source_e) != (candidate['planned_count'], candidate['available_count'], candidate['eligible_count']):
            raise ValueError('Source denominators differ from the frozen candidate')
        base = {k: t[k] for k in ('threshold_id', 'binding_sha256', 'source_model', 'target_model',
                                 'condition', 'execution_slot', 'target_coverage', 'coverage_basis')}
        base.update(threshold=t['threshold'], source_planned=len(source), source_available=source_n,
                    source_eligible=source_e, source_unavailable=len(source)-source_n,
                    source_coverage_global=candidate['observed_source_coverage_global'],
                    source_coverage_eligible_after_gate=candidate['observed_source_coverage_eligible_after_gate'],
                    recalibrated_on_target=False, quota_forced_on_target=False,
                    comparator='score <= threshold', new_independent_replications=0,
                    historical_caveat_preserved=True, equivalence_claim=False)
        if candidate['threshold_status'] == 'not_estimable':
            configurations.append(dict(**base, status='source_threshold_not_estimable',
                reason=candidate['non_estimable_reason'], target_planned=len(target),
                target_available=int(target.available.sum()), target_eligible=int(target.eligible.sum()),
                target_accepted=None, target_coverage_global=None,
                target_coverage_eligible_after_gate=None, clinical_denominator=None,
                underestimation_rate_delta1=None, underestimation_rate_delta2=None,
                drift_pp=None, drift_within_tolerance=None, equivalence_claim_allowed=False))
            continue
        source_accepted = source.available & source.eligible & source.risk_score.le(t['threshold'])
        if int(source_accepted.sum()) != candidate['calibration_accepted_count']:
            raise ValueError('Source threshold application differs from its original count')
        accepted_source_ids=sorted(source.loc[source_accepted,'observation_id'])
        if _sha(accepted_source_ids) != t['candidate_payload']['base_equivalence_proof']['source_accepted_ids_sha256']:
            raise ValueError('Source accepted set differs from the common-candidate proof')
        accepted = target.available & target.eligible & target.risk_score.le(t['threshold'])
        summary = _target_summary(target, accepted)
        n, a = summary['target_available'], summary['target_accepted']
        drift = Fraction(a, n) - Fraction(candidate['calibration_accepted_count'], source_n) if n and source_n else None
        within = abs(drift) <= Fraction(1, 20) if drift is not None else None
        status = ('target_configuration_absent' if target.empty else 'no_target_available_observations' if not n
                  else 'no_target_eligible_observations' if not summary['target_eligible']
                  else 'operationally_inconclusive_due_to_coverage_drift' if within is False
                  else 'descriptive_no_known_severe_reference' if not summary['clinical_denominator']
                  else 'descriptive_selector_inactive' if summary['target_selector_inactive'] else 'descriptive')
        configurations.append(dict(**base, **{k: v for k, v in summary.items() if k not in base},
            status=status, reason=None, drift_pp=float(drift*100) if drift is not None else None,
            drift_within_tolerance=within, drift_tolerance_pp=5, drift_tolerance_inclusive=True))
        for r, accepted_flag in zip(target.to_dict('records'), accepted.tolist()):
            decisions.append(dict(**{k: r[k] for k in ('observation_id', 'pair_id', 'model', 'condition', 'execution_slot')},
                threshold_id=t['threshold_id'], binding_sha256=t['binding_sha256'], target_coverage=t['target_coverage'],
                available=bool(r['available']), eligible=bool(r['eligible']),
                accepted=bool(accepted_flag) if r['available'] else None,
                decision_status=('unavailable_not_a_model_abstention' if not r['available'] else
                                 'gate_ineligible' if not r['eligible'] else
                                 'accepted' if accepted_flag else 'threshold_abstention'),
                threshold=t['threshold'], new_independent_replications=0))
    return {'configurations': pd.DataFrame(configurations, dtype=object),
            'decisions': pd.DataFrame(decisions, dtype=object), 'metadata': dict(
                version=VERSION, planned_configurations=24, coverage_basis='global',
                drift_comparison_basis='global_origin_and_target', inclusive_threshold=True,
                source_thresholds_recalibrated=False, target_quota_forced=False,
                freeze_manifest_sha256=freeze_manifest_sha256, approval_sha256=approval_sha256,
                formal_freeze_declared_by_caller=True, supplied_hashes_independently_verified=False,
                source_categories_required=False,source_clinical_labels_required=False,
                source_target_drug_disjoint_checked=drug_disjoint_checked,
                source_target_drug_disjoint_note=('checked_from_explicit_endpoint_metadata' if drug_disjoint_checked
                    else 'delegated_to_frozen_split_and_caller_source_endpoints_not_supplied'),
                clinical_evaluation_executed=True, inferential_claims=False,
                historical_caveat_preserved=True, DEFINITION_OF_DONE=False)}
