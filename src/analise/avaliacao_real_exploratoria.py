"""Pure future exploratory orchestration; this module performs no file access.

The caller must separately verify the frozen artifact envelope and obtain
authorization to read outcomes. Digests and booleans supplied here are contract
bindings, not proof that the caller performed those external checks. Preparation
exercises this interface exclusively with invented fixtures.
"""
from __future__ import annotations

from collections.abc import Mapping
import re

import pandas as pd

from .contratos_avaliacao_real import (
    validar_entradas, pontuar_c, avaliar_d01_real, table,
)
from .metodos_freeze1_v1_3_1 import (
    MODEL_FAMILIES, PRIMARY_MEMBER, SECONDARY_FAMILY_MEMBERS,
    validar_estado_politica_primaria,
)
from .politica_real_exploratoria import evaluate_fixed_quota, STRATA
from .transferencia_real_exploratoria import (
    aplicar_transferencia, ROW_FIELDS, SOURCE_FIELDS, MODEL_MAP, SLOT,
)


VERSION = 'exploratory-real-integrated-orchestration-v1'
HISTORICAL_MODEL = 'gpt-4o-mini-2024-07-18'
REGISTRY_FAMILIES = {**MODEL_FAMILIES, HISTORICAL_MODEL: 'referência histórica secundária'}
SNAPSHOTS = tuple(REGISTRY_FAMILIES)
CENTRAL_MODELS = tuple(MODEL_MAP.values())
POLICY_MODELS = tuple(MODEL_FAMILIES)
CENTRAL_SCOPE = 'CENTRAL_GPT54_C'
POLICY_SCOPES = (CENTRAL_SCOPE, *(f'SNAPSHOT_C_{m}' for m in POLICY_MODELS))
FEATURES = ('nomenclature_delta_morphology_units_pair',
            'nomenclature_lexical_dissimilarity_max')
SOURCE_PAIR_COLUMNS = {'pair_id', 'drug_id_a', 'drug_id_b', 'context_id'}
DEFERRED_REASONS = {
    'salt_extra': 'approved_deferred_salt_sensitivity_no_change_to_primary_two_features',
    'pubmed_frequency': 'approved_deferred_external_frequency_collection_not_authorized',
    'o200k': 'approved_deferred_tokenizer_sensitivity_not_required_for_primary_route',
    'transductive_features': 'approved_deferred_transductive_sensitivity_no_target_refitting',
}


def _digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _validate_scope_contract(masks, registry, features, state, thresholds,
                             source_scores, source_pair_metadata, policy_bases):
    if set(registry.model) != set(SNAPSHOTS):
        raise ValueError('The literal ten-snapshot registry is required')
    for r in registry.to_dict('records'):
        if (r['family'] != REGISTRY_FAMILIES[r['model']]
                or r['provider'] != ('anthropic' if r['model'].startswith('claude-') else 'openai')
                or not isinstance(r['role'], str) or not r['role'].strip()):
            raise ValueError('Registry family/provider/role binding is incoherent')
    if masks.empty or not masks.pair_partition.eq('C').all():
        raise ValueError('A nonempty explicit C planned population is required')
    expected = {(p, m, c, SLOT) for p in set(masks.pair_id)
                for m in SNAPSHOTS for c in ('inn', 'dcb')}
    actual = set(masks[['pair_id', 'model', 'condition', 'execution_slot']].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError('Incomplete planned grid: preserve unavailable positions explicitly')
    if (not isinstance(policy_bases, Mapping) or set(policy_bases) != set(POLICY_SCOPES)
            or any(v not in {'global', 'eligible_after_gate'} for v in policy_bases.values())):
        raise ValueError('Each of the ten policy scopes needs its explicit frozen basis')
    if policy_bases[CENTRAL_SCOPE] != 'eligible_after_gate':
        raise ValueError('Central GPT54 basis must remain eligible_after_gate')
    for m in CENTRAL_MODELS:
        if policy_bases[f'SNAPSHOT_C_{m}'] != policy_bases[CENTRAL_SCOPE]:
            raise ValueError('Central and individual GPT54 policy bases disagree')
    # Verify invariant-set requirements without changing any frozen basis. Empty
    # available sets remain explicit missing configurations, not new choices.
    for model in POLICY_MODELS:
        if policy_bases[f'SNAPSHOT_C_{model}'] != 'eligible_after_gate':
            continue
        group = masks.loc[masks.model.eq(model)]
        for flag in ('available', 'eligible'):
            a = set(group.loc[group.condition.eq('inn') & group[flag], 'pair_id'])
            b = set(group.loc[group.condition.eq('dcb') & group[flag], 'pair_id'])
            if a != b:
                raise ValueError('Frozen eligible basis contradicts condition-set invariance')
    table(features, {'pair_id', *FEATURES}, ('pair_id',), 'features')
    if set(features.pair_id) != set(masks.pair_id):
        raise ValueError('Features do not cover exact C planned pairs')
    strata = features.set_index('pair_id')[FEATURES[0]].map(
        lambda value: STRATA[0] if value == 0 else STRATA[1]).to_dict()
    if (not masks.primary_features_complete.all()
            or not masks.apply(lambda r: r.morphology_stratum == strata[r.pair_id], axis=1).all()):
        raise ValueError('Frozen primary completeness/morphology projection disagrees with features')
    validar_estado_politica_primaria(state)
    if not isinstance(thresholds, pd.DataFrame) or 'candidate_payload' not in thresholds:
        raise ValueError('Explicit preserved source candidate bindings are required')
    for candidate in thresholds.candidate_payload:
        if (not isinstance(candidate, Mapping)
                or candidate.get('provenance_hashes', {}).get('policy_state_sha256') != state['state_sha256']):
            raise ValueError('Source thresholds and C scoring must use the same frozen policy state')
    table(source_scores, SOURCE_FIELDS, ('observation_id',), 'category-free source scores')
    table(source_pair_metadata, SOURCE_PAIR_COLUMNS, ('pair_id',), 'source pair metadata')
    if set(source_scores.pair_id) != set(source_pair_metadata.pair_id):
        raise ValueError('Source endpoint metadata do not cover the exact calibration population')
    if set(source_scores.pair_id) & set(masks.pair_id):
        raise ValueError('Calibration and C share a pair')
    source_drugs = set(source_pair_metadata.drug_id_a) | set(source_pair_metadata.drug_id_b)
    target_drugs = set(masks.drug_id_a) | set(masks.drug_id_b)
    if source_drugs & target_drugs:
        raise ValueError('Calibration and C share a drug')


def _inventory(policies, d01, transfer):
    """Keep the literal 37-member inventory; D01-E is an extra sensitivity."""
    inventory, payloads = [], {}

    def add(family, member, payload, location, *, deferred_reason=None):
        key = family + '::' + member
        payloads[key] = payload.copy(deep=True)
        inventory.append(dict(family_id=family, member_id=member,
            status='deferred_approved' if deferred_reason else 'descriptive_output_prepared',
            result_count=len(payload), result_location=location,
            deferred_reason=deferred_reason, inferential_claim=False, equivalence_claim=False))

    central = policies[CENTRAL_SCOPE]
    add('PRIMARY', PRIMARY_MEMBER, central['summary_bycoverage'],
        'policy_scopes.CENTRAL_GPT54_C.summary_bycoverage')
    for family, members in SECONDARY_FAMILY_MEMBERS.items():
        for member in members:
            if family == 'SEC_COVERAGE_GRID':
                payload = central['summary_bycoverage'].loc[
                    central['summary_bycoverage'].target_coverage.eq(float(member.removeprefix('coverage_')))]
                location = 'policy_scopes.CENTRAL_GPT54_C.summary_bycoverage'
            elif family in {'SEC_MODEL_SNAPSHOTS_OPENAI', 'SEC_ANTHROPIC_CONCEPTUAL_REPLICATION'}:
                payload = policies[f'SNAPSHOT_C_{member}']['summary_bycoverage']
                location = f'policy_scopes.SNAPSHOT_C_{member}.summary_bycoverage'
            elif family == 'SEC_MECHANISMS':
                field, column = ('strata', 'morphology_stratum') if member in STRATA else ('mechanisms', 'mechanism')
                payload = central[field].loc[central[field].scope.eq('family') & central[field][column].eq(member)]
                location = f'policy_scopes.CENTRAL_GPT54_C.{field}'
            elif family == 'SEC_TRANSFER_SNAPSHOT_PAIRS':
                source, target = member.split('__to__')
                payload = transfer['configurations'].loc[
                    transfer['configurations'].source_model.eq(source) & transfer['configurations'].target_model.eq(target)]
                location = 'transfer.configurations'
            else:
                add(family, member, pd.DataFrame(), 'deferred_feature_sensitivities',
                    deferred_reason=DEFERRED_REASONS[member])
                continue
            add(family, member, payload, location)
    for model in SNAPSHOTS:
        add('SEC_NOMENCLATURE_DIRECT_D01', model,
            d01['snapshots'].loc[d01['snapshots'].model.eq(model)], 'd01.snapshots')
    if len(inventory) != 37 or len(payloads) != 37:
        raise RuntimeError('Literal 37-member inventory was not preserved')
    return pd.DataFrame(inventory, dtype=object), payloads


def avaliar_estudo_exploratorio(masks, predictions, references, registry, contexts,
        features, state, thresholds, source_scores, source_pair_metadata, *,
        policy_bases, freeze_manifest_sha256, approval_sha256,
        evaluation_authorized=False, formal_freeze_complete=False):
    """Future authorized descriptive evaluation using verified frozen projections.

    No fitting, coverage choice, inference, multiplicity correction, file access
    or provenance promotion. Source scores contain no predictions/references.
    Every planned C pair must retain all ten snapshots and both conditions;
    exclusions are represented by unavailable positions, never dropped rows.
    """
    if evaluation_authorized is not True:
        raise ValueError('Explicit future evaluation authorization is required')
    if (formal_freeze_complete is not True or not _digest(freeze_manifest_sha256)
            or not _digest(approval_sha256)):
        raise ValueError('Formal freeze and valid manifest/approval bindings are required')
    rows = validar_entradas(masks, predictions, references, registry, contexts)
    _validate_scope_contract(masks, registry, features, state, thresholds,
                             source_scores, source_pair_metadata, policy_bases)
    # This is the only target scoring operation. No source constants are fitted.
    scores = pontuar_c(features, masks, state)
    d01 = avaliar_d01_real(rows, snapshots=SNAPSHOTS)
    d01e = avaliar_d01_real(rows, snapshots=SNAPSHOTS, documentary_only=True)
    policies = {}
    scopes = [(CENTRAL_SCOPE, CENTRAL_MODELS),
              *((f'SNAPSHOT_C_{m}', (m,)) for m in POLICY_MODELS)]
    for scope, models in scopes:
        subset = rows.loc[rows.model.isin(models) & rows.condition.eq('dcb')].copy()
        policies[scope] = evaluate_fixed_quota(subset,
            scores.loc[scores.pair_id.isin(subset.pair_id)], models=list(models), basis=policy_bases[scope])
    target = rows.loc[rows.model.isin(CENTRAL_MODELS), sorted(ROW_FIELDS)].copy()
    transfer = aplicar_transferencia(target, scores, thresholds, source_scores,
        source_pair_metadata=source_pair_metadata, freeze_manifest_sha256=freeze_manifest_sha256,
        approval_sha256=approval_sha256, formal_freeze_complete=True)
    inventory, member_outputs = _inventory(policies, d01, transfer)
    return dict(member_inventory=inventory, member_outputs=member_outputs,
        policy_scopes=policies, d01=d01, d01e_sensitivity=d01e, transfer=transfer,
        scores_c=scores, planned_availability=rows.groupby(
            ['model', 'condition'], sort=True).agg(planned=('observation_id', 'size'),
            available=('available', 'sum'), eligible=('eligible', 'sum')).reset_index(),
        metadata=dict(version=VERSION, member_count=37, d01e_is_additional_member=False,
            target_scoring_performed=True, constants_refitted=False, coverage_chosen=False,
            inferential_claims=False, performance_calibration=False,
            historical_caveat_preserved=True, prospective_confirmation_claim=False,
            policy_bases=dict(policy_bases), transfer_basis='global',
            mechanisms_reuse_global_decisions=True, strata_reuse_global_decisions=True,
            freeze_manifest_sha256=freeze_manifest_sha256, approval_sha256=approval_sha256,
            supplied_hashes_independently_verified=False, formal_freeze_declared_by_caller=True,
            evaluation_authorization_declared_by_caller=True, source_categories_required=False,
            DEFINITION_OF_DONE=False))
