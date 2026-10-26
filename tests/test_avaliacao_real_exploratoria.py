"""Integration contract tested only with small invented in-memory observations."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ('analise', 'dados'):
    sys.path.insert(0, str(ROOT / 'src' / sub))

from src.analise import avaliacao_real_exploratoria as integrated
from src.analise.contratos_avaliacao_real import MASK_COLUMNS, PREDICTION_COLUMNS
from src.analise.metodos_freeze1_v1_3_1 import ajustar_politica_primaria
from src.analise.calibracao_preparatoria_exploratoria import (
    calibrate_source_grid, SCORE_COLUMNS, CONFIG_COLUMNS, HASH_NAMES, VERSION_NAMES,
)
from src.analise import transferencia_real_exploratoria as transfer


APPROVAL, FREEZE = 'a' * 64, 'b' * 64


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_fixture(state, n=5):
    """Invented category-free calibration and its two equivalent calculations."""
    source = []
    configs = []
    for model in transfer.MODEL_MAP:
        for condition in ('dcb', 'inn'):
            configs.append((model, condition, transfer.SLOT))
            for i in range(n):
                source.append(dict(observation_id=f'fixture-source-{model}-{condition}-{i}',
                    pair_id=f'fixture-source-{i}', model=model, condition=condition,
                    execution_slot=transfer.SLOT, pair_partition='calibration',
                    available=True, eligible=True, risk_score=float(i-2)))
    source = pd.DataFrame(source, columns=SCORE_COLUMNS, dtype=object)
    configurations = pd.DataFrame(configs, columns=CONFIG_COLUMNS)
    hashes = {k: 'd' * 64 for k in HASH_NAMES}
    hashes['policy_state_sha256'] = state['state_sha256']
    calculations = {}
    for basis in ('global', 'eligible_after_gate'):
        calculations[basis] = calibrate_source_grid(source, configurations,
            coverage_basis_by_configuration={tuple(r): basis for r in configs},
            hashes=hashes, versions={k: 'fixture-v1' for k in VERSION_NAMES})['thresholds']
    common = []
    eligible = calculations['eligible_after_gate'].set_index(['model', 'condition', 'target_coverage'])
    for row in calculations['global'].to_dict('records'):
        other = eligible.loc[(row['model'], row['condition'], row['target_coverage'])]
        accepted = source.loc[source.model.eq(row['model']) & source.condition.eq(row['condition'])
            & source.risk_score.le(row['threshold']), 'observation_id']
        payload = {k: row[k] for k in transfer.COMMON_COPY_FIELDS}
        payload.update(threshold_status='calculated_candidate_common_to_both_bases',
            non_estimable_reason=None, operational_coverage_basis=None,
            operational_basis_decision='pending_transfer_scope_decision',
            calculation_bases=['eligible_after_gate', 'global'],
            candidate_numeric_invariant_to_current_bases=True,
            base_equivalence_proof=dict(model=row['model'], condition=row['condition'],
                execution_slot=transfer.SLOT, coverage=row['target_coverage'], same_threshold=True,
                same_source_accepted_ids=True, source_accepted_ids_sha256=digest(sorted(accepted)),
                eligible_calculation_sha256=other['threshold_sha256'],
                global_calculation_sha256=row['threshold_sha256']),
            provenance_hashes={k: row[k] for k in HASH_NAMES},
            provenance_versions={k: row[k] for k in VERSION_NAMES}, tie_seed=None,
            tie_seed_required_for_threshold=False, thresholds_formally_frozen=False,
            target_application_performed=False, DEFINITION_OF_DONE=False)
        sha = digest(payload)
        common.append(dict(**payload, threshold_id='CANDTHR_COMMON_' + sha[:24], threshold_sha256=sha))
    bound = transfer.bind_thresholds(pd.DataFrame(common, dtype=object), calculations['global'],
        approval_sha256=APPROVAL, candidate_artifact_sha256='c' * 64,
        source_calculation_artifact_sha256='e' * 64)['thresholds']
    source_pairs = pd.DataFrame([dict(pair_id=f'fixture-source-{i}', drug_id_a=f'fixture-s-a{i}',
        drug_id_b=f'fixture-s-b{i}', context_id=f'fixture-s-ctx{i}') for i in range(n)])
    return bound, source, source_pairs


def fixture(n=5):
    rows, predictions, refs, contexts, features = [], [], [], [], []
    registry = pd.DataFrame([dict(model=m, family=f,
        provider='anthropic' if m.startswith('claude-') else 'openai', role='fixture_historical_role')
        for m, f in integrated.REGISTRY_FAMILIES.items()])
    for i in range(n):
        pair = f'fixture-C-{i}'
        ctx = dict(pair_id=pair, drug_id_a=f'fixture-c-a{i}', drug_id_b=f'fixture-c-b{i}',
            context_id=f'fixture-c-ctx{i}', ref_id_a=f'fixture-r-a{i}', ref_id_b=f'fixture-r-b{i}')
        contexts.append(ctx)
        refs.append({k: ctx[k] for k in ('pair_id', 'context_id', 'ref_id_a', 'ref_id_b')} |
            dict(source_id='fixture-reference', source_version='fixture-v1', quality_status='usable', canonical_label='grave'))
        features.append(dict(pair_id=pair, nomenclature_delta_morphology_units_pair=float(i % 2),
            nomenclature_lexical_dissimilarity_max=(i + 1) / 10))
        for record in registry.to_dict('records'):
            for condition in ('dcb', 'inn'):
                oid = f"fixture-{record['model']}-{condition}-{i}"
                row = dict(**record, observation_id=oid, planned_observation_id=oid + '-plan', pair_id=pair,
                    condition=condition, execution_slot=transfer.SLOT, pair_partition='C',
                    request_id=oid + '-request', request_fingerprint=digest(oid), payload_fingerprint=digest(oid + '-payload'),
                    text_binding_sha256=digest(pair), context_id=ctx['context_id'], admission_status='fixture_admitted',
                    drug_id_a=ctx['drug_id_a'], drug_id_b=ctx['drug_id_b'], reference_source_id='fixture-reference',
                    reference_source_version='fixture-v1', available=True, structural_response_valid=True,
                    binding_verified=True, eligible=True, eligible_d01_response=True, reference_available=True,
                    identity_admitted=True, benchmark_candidate=True, primary_features_complete=True,
                    morphology_stratum=integrated.STRATA[i % 2], d01e_eligible=None,
                    d01e_documentary_state='insufficient_information', response_execution_id_observed=None,
                    response_execution_attribution_verified=False, complete_observed_compatibility=False)
                rows.append({k: row[k] for k in MASK_COLUMNS})
                prediction = {k: row[k] for k in PREDICTION_COLUMNS if k != 'category'}
                prediction['category'] = ('leve' if i % 2 == 0 else 'moderado') if condition == 'dcb' else 'grave'
                predictions.append(prediction)
    features = pd.DataFrame(features)
    development = features.assign(split='development', nomenclature_morphology_units_inn_pair=2,
        nomenclature_morphology_units_dcb_pair=features.nomenclature_delta_morphology_units_pair + 2)
    state = ajustar_politica_primaria(development)
    thresholds, source_scores, source_pairs = source_fixture(state)
    bases = {scope: 'eligible_after_gate' for scope in integrated.POLICY_SCOPES}
    bases.update({f'SNAPSHOT_C_{m}': 'global' for m in integrated.POLICY_MODELS if m.startswith('claude-')})
    return dict(masks=pd.DataFrame(rows, dtype=object), predictions=pd.DataFrame(predictions, dtype=object),
        references=pd.DataFrame(refs, dtype=object), registry=registry, contexts=pd.DataFrame(contexts),
        features=features, state=state, thresholds=thresholds, source_scores=source_scores,
        source_pair_metadata=source_pairs, policy_bases=bases, freeze_manifest_sha256=FREEZE,
        approval_sha256=APPROVAL, evaluation_authorized=True, formal_freeze_complete=True)


def run(args=None, **updates):
    args = fixture() if args is None else args
    return integrated.avaliar_estudo_exploratorio(**(args | updates))


def unavailable(args, selector):
    ids = set(args['masks'].loc[selector, 'observation_id'])
    args['masks'].loc[selector, ['available', 'structural_response_valid', 'eligible', 'eligible_d01_response']] = False
    args['masks'].loc[selector, 'admission_status'] = 'fixture_unavailable'
    args['predictions'] = args['predictions'].loc[~args['predictions'].observation_id.isin(ids)].copy()


def test_complete_integration_retains_literal_37_members_and_four_coverages():
    result = run()
    inventory = result['member_inventory']
    assert len(inventory) == len(result['member_outputs']) == 37
    assert inventory.groupby('family_id').size().to_dict() == {
        'PRIMARY': 1, 'SEC_COVERAGE_GRID': 4, 'SEC_MODEL_SNAPSHOTS_OPENAI': 6,
        'SEC_MECHANISMS': 6, 'SEC_TRANSFER_SNAPSHOT_PAIRS': 3,
        'SEC_ANTHROPIC_CONCEPTUAL_REPLICATION': 3, 'SEC_FEATURE_SENSITIVITIES': 4,
        'SEC_NOMENCLATURE_DIRECT_D01': 10}
    assert len(result['policy_scopes']) == 10
    assert len(result['d01']['snapshots']) == 10
    assert len(result['transfer']['configurations']) == 24
    assert result['transfer']['metadata']['source_target_drug_disjoint_checked']
    assert result['metadata']['member_count'] == 37
    assert not result['metadata']['d01e_is_additional_member']
    assert not result['metadata']['DEFINITION_OF_DONE']
    primary = result['member_outputs']['PRIMARY::' + integrated.PRIMARY_MEMBER]
    assert primary.target_coverage.tolist() == [.6, .7, .8, .9]
    deferred = inventory.loc[inventory.family_id.eq('SEC_FEATURE_SENSITIVITIES')]
    assert deferred.status.eq('deferred_approved').all() and deferred.result_count.eq(0).all()
    assert deferred.deferred_reason.notna().all()


def test_explicit_bases_do_not_inherit_global_transfer_and_claude_remains_conceptual():
    result = run()
    assert result['transfer']['configurations'].coverage_basis.eq('global').all()
    for scope, policy in result['policy_scopes'].items():
        expected = 'global' if 'claude-' in scope else 'eligible_after_gate'
        assert policy['summary_bycoverage'].coverage_basis.eq(expected).all()
        if 'claude-' in scope:
            assert policy['metadata']['conceptual_replication']
            assert policy['summary_bycoverage'].protocol_confounding_with_other_providers.all()
    assert not result['metadata']['inferential_claims']
    assert not result['metadata']['prospective_confirmation_claim']


def test_missing_snapshot_retains_planned_rows_and_d0_stays_undefined():
    args = fixture()
    model = integrated.CENTRAL_MODELS[0]
    unavailable(args, args['masks'].model.eq(model))
    args['references']['canonical_label'] = 'moderado'
    result = run(args)
    central = result['policy_scopes'][integrated.CENTRAL_SCOPE]
    assert central['summary_bycoverage'].planned_count.eq(15).all()
    assert central['summary_bycoverage'].available_count.eq(10).all()
    assert central['summary_bycoverage'].clinical_denominator.eq(0).all()
    assert all(x is None for x in central['summary_bycoverage'].contrast)
    missing = central['decisions'].loc[central['decisions'].model.eq(model)]
    assert len(missing) == 20 and missing.decision_status.eq('not_available_metadata_exclusion').all()
    assert not missing.model_abstention_inferred_for_unavailable.any()
    assert result['d01']['snapshots'].denominator_common_severe.eq(0).all()
    assert all(x is None for x in result['d01']['snapshots'].estimate_dcb_minus_inn)
    assert len(result['member_inventory']) == 37


def test_d01e_is_separate_sensitivity_without_selector_or_primary_redefinition():
    original = run()
    assert original['d01e_sensitivity']['snapshots'].status.eq('documentary_sensitivity_not_evaluable').all()
    args = fixture()
    selected = args['masks'].pair_id.eq('fixture-C-0')
    args['masks'].loc[selected, 'd01e_eligible'] = True
    args['masks'].loc[selected, 'd01e_documentary_state'] = 'demonstrated_support'
    result = run(args)
    assert result['d01e_sensitivity']['snapshots'].denominator_common_severe.eq(1).all()
    pd.testing.assert_frame_equal(original['d01']['snapshots'], result['d01']['snapshots'])
    pd.testing.assert_frame_equal(original['policy_scopes'][integrated.CENTRAL_SCOPE]['summary_bycoverage'],
                                  result['policy_scopes'][integrated.CENTRAL_SCOPE]['summary_bycoverage'])
    assert not result['d01e_sensitivity']['metadata']['selector_applied']


def test_mechanisms_and_morphology_use_preexisting_global_decisions():
    result = run()
    central = result['policy_scopes'][integrated.CENTRAL_SCOPE]
    assert central['mechanisms'].global_decisions_reused.all()
    assert central['strata'].global_decisions_reused.all()
    assert not central['strata'].quota_recomputed_in_stratum.any()
    assert not central['strata'].constants_refitted_in_stratum.any()
    for coverage in (.6, .7, .8, .9):
        selected = central['decisions'].loc[central['decisions'].target_coverage.eq(coverage)]
        strata = central['strata'].loc[central['strata'].target_coverage.eq(coverage) & central['strata'].scope.eq('family')]
        assert strata.accepted_count.sum() == selected.accepted.sum()
        assert strata.baseline_expected_u_delta1_count.sum() == pytest.approx(selected.baseline_expected_delta1.sum())


def test_order_invariance_no_input_mutation_and_single_target_scoring(monkeypatch):
    args = fixture()
    before = copy.deepcopy(args)
    original = integrated.pontuar_c
    calls = []
    def score(*values):
        calls.append(1)
        return original(*values)
    monkeypatch.setattr(integrated, 'pontuar_c', score)
    one = run(args)
    assert len(calls) == 1
    for key, value in args.items():
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(value, before[key])
        else:
            assert value == before[key]
    reversed_args = {k: v.iloc[::-1].copy() if isinstance(v, pd.DataFrame) else v for k, v in args.items()}
    two = run(reversed_args)
    pd.testing.assert_frame_equal(one['member_inventory'], two['member_inventory'])
    pd.testing.assert_frame_equal(one['scores_c'], two['scores_c'])
    for member in one['member_outputs']:
        pd.testing.assert_frame_equal(one['member_outputs'][member], two['member_outputs'][member])


@pytest.mark.parametrize('field,value', [('evaluation_authorized', False), ('evaluation_authorized', 1),
    ('formal_freeze_complete', False), ('freeze_manifest_sha256', 'invalid'), ('approval_sha256', '')])
def test_authorization_and_formal_envelope_fail_before_any_join_or_score(monkeypatch, field, value):
    args = fixture()
    def forbidden(*a, **k):
        pytest.fail('Evaluation work began before authorization checks')
    monkeypatch.setattr(integrated, 'validar_entradas', forbidden)
    monkeypatch.setattr(integrated, 'pontuar_c', forbidden)
    with pytest.raises(ValueError):
        run(args, **{field: value})


@pytest.mark.parametrize('fault', ['drop_planned', 'drop_prediction', 'duplicate_prediction', 'context',
    'registry_family', 'registry_missing', 'registry_extra', 'source_pair_shared', 'source_drug_shared',
    'source_metadata_missing', 'source_extra_category', 'source_score_modified', 'missing_policy_basis',
    'central_global', 'invariant_sets_changed', 'morphology_changed', 'features_missing', 'nonfinite_feature',
    'threshold_state_changed', 'threshold_value_changed', 'unknown_reference_label', 'request_hash_changed'])
def test_incoherent_joins_scopes_and_bindings_fail_closed(fault):
    args = fixture()
    if fault == 'drop_planned':
        oid = args['masks'].iloc[0].observation_id
        args['masks'] = args['masks'].iloc[1:]
        args['predictions'] = args['predictions'].loc[args['predictions'].observation_id.ne(oid)]
    elif fault == 'drop_prediction': args['predictions'] = args['predictions'].iloc[1:]
    elif fault == 'duplicate_prediction': args['predictions'] = pd.concat([args['predictions'], args['predictions'].iloc[:1]])
    elif fault == 'context': args['contexts'].loc[0, 'context_id'] = 'fixture-incoherent'
    elif fault == 'registry_family': args['registry'].loc[0, 'family'] = 'fixture-wrong'
    elif fault == 'registry_missing': args['registry'] = args['registry'].iloc[1:]
    elif fault == 'registry_extra':
        args['registry'] = pd.concat([args['registry'], pd.DataFrame([dict(model='fixture-extra',
            provider='openai', family='fixture-family', role='fixture-role')])])
    elif fault == 'source_pair_shared':
        args['source_pair_metadata'].loc[0, 'pair_id'] = args['masks'].iloc[0].pair_id
    elif fault == 'source_drug_shared':
        args['source_pair_metadata'].loc[0, 'drug_id_a'] = args['masks'].iloc[0].drug_id_a
    elif fault == 'source_metadata_missing': args['source_pair_metadata'] = None
    elif fault == 'source_extra_category': args['source_scores']['category'] = 'grave'
    elif fault == 'source_score_modified': args['source_scores'].loc[0, 'risk_score'] = 999.
    elif fault == 'missing_policy_basis': args['policy_bases'].pop(next(iter(args['policy_bases'])))
    elif fault == 'central_global': args['policy_bases'][integrated.CENTRAL_SCOPE] = 'global'
    elif fault == 'invariant_sets_changed':
        unavailable(args, args['masks'].model.eq(integrated.CENTRAL_MODELS[0])
            & args['masks'].condition.eq('inn') & args['masks'].pair_id.eq('fixture-C-0'))
    elif fault == 'morphology_changed': args['masks'].loc[0, 'morphology_stratum'] = 'morphology_delta_nonzero'
    elif fault == 'features_missing': args['features'] = args['features'].iloc[1:]
    elif fault == 'nonfinite_feature': args['features'].loc[0, integrated.FEATURES[0]] = float('inf')
    elif fault == 'threshold_state_changed':
        args['thresholds'].iloc[0].candidate_payload['provenance_hashes']['policy_state_sha256'] = 'f' * 64
    elif fault == 'threshold_value_changed': args['thresholds'].loc[0, 'threshold'] = 999.
    elif fault == 'unknown_reference_label': args['references'].loc[0, 'canonical_label'] = 'undefined-label'
    elif fault == 'request_hash_changed': args['predictions'].loc[0, 'request_fingerprint'] = 'e' * 64
    with pytest.raises(ValueError):
        run(args)
