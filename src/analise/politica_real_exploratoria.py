"""Pure, real-ready descriptive fixed-quota adapter; no I/O or authorization.

The caller must validate frozen inputs and joins before invoking this adapter.
Only small invented fixtures are authorized in the Freeze 2 preparation round.
Importing this module never scores, selects or evaluates actual observations.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from collections.abc import Sequence

import numpy as np
import pandas as pd

from metodos_freeze1_v1_3_1 import COVERAGE_GRID, MODEL_FAMILIES
from predicao_seletiva import _hash_uniforme


VERSION = 'exploratory-real-fixed-quota-description-v1'
APPROVED_TIE_SEED = 20261003
TIE_HASH_VERSION = 'sha256-utf8-seed-unitseparator-pairid-first8-big-div2pow64-float-v1'
TIE_HASH_LITERAL = "int.from_bytes(sha256(f'{seed}\\x1f{pair_id}'.encode('utf-8')).digest()[:8], 'big') / 2**64"
SEVERITIES = frozenset({'leve', 'moderado', 'grave'})
STRATA = ('morphology_delta_zero', 'morphology_delta_nonzero')
ROW_COLUMNS = (
    'observation_id', 'pair_id', 'model', 'provider', 'family', 'condition',
    'execution_slot', 'pair_partition', 'drug_id_a', 'drug_id_b', 'context_id',
    'available', 'eligible', 'eligible_d01_response', 'reference_available',
    'morphology_stratum', 'd01e_eligible', 'category', 'severity_label',
)
BATCH_COLUMNS = ('model', 'execution_slot')
KNOWN_FAMILIES = dict(MODEL_FAMILIES, **{'gpt-4o-mini-2024-07-18': ''})
MECHANISMS = ('severe_underestimation_delta2', 'severe_routing_rate',
              'risk_among_severe_accepted_delta1', 'residual_error_rate')


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _missing(value):
    return value is None or value is pd.NA or (isinstance(value, Real) and math.isnan(value))


def _validate(rows, scores, models, basis, seed):
    if (not isinstance(models, Sequence) or isinstance(models, (str, bytes))
            or not models or len(set(models)) != len(models)
            or any(model not in KNOWN_FAMILIES for model in models)):
        raise ValueError('Lista explícita de snapshots aprovados e únicos requerida.')
    families = {KNOWN_FAMILIES[model] for model in models}
    if len(families) != 1:
        raise ValueError('Não agrupar famílias ou fornecedores diferentes.')
    if basis not in {'eligible_after_gate', 'global'}:
        raise ValueError('Base operacional explícita inválida.')
    if families == {'openai-gpt-5.4'} and basis != 'eligible_after_gate':
        raise ValueError('Base central GPT-5.4/C aprovada: eligible_after_gate.')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed != APPROVED_TIE_SEED:
        raise ValueError('Semente literal aprovada exigida: 20261003; 1302 pertence ao split.')
    if (not isinstance(rows, pd.DataFrame) or rows.columns.duplicated().any()
            or not set(ROW_COLUMNS) <= set(rows.columns)):
        raise ValueError('Projeção de observações incompleta ou duplicada.')
    # Extra verified metadata from the common validator is not interpreted here.
    base = rows[list(ROW_COLUMNS)].copy()
    if (not isinstance(scores, pd.DataFrame) or scores.columns.duplicated().any()
            or set(scores.columns) != {'pair_id', 'risk_score'}):
        raise ValueError('Escores exigem somente pair_id/risk_score.')
    identifiers = ('observation_id', 'pair_id', 'model', 'provider', 'condition',
                   'execution_slot', 'pair_partition', 'drug_id_a', 'drug_id_b', 'context_id')
    for column in identifiers:
        if not base[column].map(lambda x: isinstance(x, str) and bool(x.strip()) and x == x.strip()).all():
            raise ValueError('Identidade literal ausente/inválida: ' + column)
    if not scores.pair_id.map(lambda x: isinstance(x, str) and bool(x.strip()) and x == x.strip()).all():
        raise ValueError('Identidade literal inválida em scores.')
    if base.observation_id.duplicated().any() or base[
        ['pair_id', 'model', 'condition', 'execution_slot', 'pair_partition']
    ].duplicated().any() or scores.pair_id.duplicated().any():
        raise ValueError('Observação, lote ou escore duplicado.')
    if set(scores.pair_id) != set(base.pair_id):
        raise ValueError('Universos de escores e pares devem coincidir exatamente; sem join parcial.')
    if not set(base.model) <= set(models):
        raise ValueError('Snapshot fora do escopo explícito.')
    if not base.pair_partition.eq('C').all() or not base.condition.eq('dcb').all():
        raise ValueError('Política exige projeção C/DCB; nenhuma seleção em origem/calibração.')
    if not base.execution_slot.eq('historico_a_identificar').all():
        raise ValueError('Slot histórico reconstruído deve ser preservado.')
    for field in ('available', 'eligible', 'eligible_d01_response', 'reference_available'):
        if not base[field].map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise ValueError('Booleano explícito requerido: ' + field)
        base[field]=base[field].astype(bool)
    if not base.d01e_eligible.map(lambda x: _missing(x) or isinstance(x, (bool, np.bool_))).all():
        raise ValueError('D01-E exige booleano ou estado não aferido.')
    if (base.eligible & (~base.available | ~base.reference_available | ~base.eligible_d01_response)).any():
        raise ValueError('Elegibilidade contradiz disponibilidade/referência/gate documental.')
    if not set(base.morphology_stratum) <= {*STRATA, 'morphology_not_assessable'}:
        raise ValueError('Estrato morfológico desconhecido.')
    if (base.eligible & ~base.morphology_stratum.isin(STRATA)).any():
        raise ValueError('Elegível para política sem features/estrato morfológico definido.')
    if base.drug_id_a.eq(base.drug_id_b).any():
        raise ValueError('Par com o mesmo medicamento nas duas pontas.')
    for row in base.itertuples(index=False):
        expected_provider = 'anthropic' if row.model.startswith('claude-') else 'openai'
        if row.family != KNOWN_FAMILIES[row.model] or row.provider != expected_provider:
            raise ValueError('Proveniência do snapshot/família/fornecedor incoerente.')
        for column in ('category', 'severity_label'):
            value = getattr(row, column)
            if not _missing(value) and value not in SEVERITIES:
                raise ValueError('Categoria clínica não reconhecida: ' + column)
        if row.eligible and (row.category not in SEVERITIES or row.severity_label not in SEVERITIES):
            raise ValueError('Elegível sem categorias válidas; nenhuma imputação permitida.')
        if not row.available and not _missing(row.category):
            raise ValueError('Posição indisponível não pode conter predição inventada.')
        if not row.reference_available and not _missing(row.severity_label):
            raise ValueError('Referência não disponível não pode possuir rótulo utilizável.')
    # Across snapshots, one pair still denotes the same oriented dyad/context,
    # source reference and predecision morphology. Missing reference values may
    # remain absent; two contradictory known values must never be pooled.
    for _, group in base.groupby('pair_id', sort=True):
        for field in ('drug_id_a', 'drug_id_b', 'context_id', 'morphology_stratum', 'reference_available'):
            if group[field].nunique(dropna=False) != 1:
                raise ValueError('Vínculo por par contraditório entre snapshots: ' + field)
        labels = {value for value in group.severity_label if not _missing(value)}
        if len(labels) > 1:
            raise ValueError('Referências contraditórias para o mesmo par.')
    base = base.merge(scores, on='pair_id', validate='many_to_one')
    for row in base.itertuples(index=False):
        valid = isinstance(row.risk_score, Real) and not isinstance(row.risk_score, (bool, np.bool_)) and math.isfinite(row.risk_score)
        if not valid and not (not row.eligible and _missing(row.risk_score)):
            raise ValueError('Escore elegível ausente/não finito ou escore não numérico.')
    return base.sort_values(['model', 'execution_slot', 'pair_id', 'observation_id']).reset_index(drop=True)


def _decide(base, coverage, basis, seed):
    rows = base.copy()
    rows['accepted'] = False
    rows['rank_eligible'] = pd.Series([None] * len(rows), dtype=object)
    rows['tie_hash_uniform'] = pd.Series([None] * len(rows), dtype=object)
    rows['batch_available_count'] = 0
    rows['batch_eligible_count'] = 0
    rows['batch_quota'] = 0
    rows['batch_desired_raw'] = 0
    rows['batch_gate_saturated'] = False
    rows['baseline_acceptance_probability'] = 0.
    for _, batch in rows.groupby(list(BATCH_COLUMNS), sort=True):
        n, e = int(batch.available.sum()), int(batch.eligible.sum())
        desired = math.floor(coverage * (n if basis == 'global' else e) + .5)
        quota = min(e, desired)
        eligible = batch.loc[batch.eligible].copy()
        eligible['tie_hash_uniform'] = [_hash_uniforme(pid, seed) for pid in eligible.pair_id]
        eligible = eligible.sort_values(['risk_score', 'tie_hash_uniform', 'pair_id'], kind='mergesort')
        rows.loc[batch.index, 'batch_available_count'] = n
        rows.loc[batch.index, 'batch_eligible_count'] = e
        rows.loc[batch.index, 'batch_quota'] = quota
        rows.loc[batch.index, 'batch_desired_raw'] = desired
        rows.loc[batch.index, 'batch_gate_saturated'] = bool(n and basis == 'global' and e > 0 and desired >= e)
        rows.loc[eligible.index, 'rank_eligible'] = list(range(1, e + 1))
        rows.loc[eligible.index, 'tie_hash_uniform'] = eligible.tie_hash_uniform.to_list()
        rows.loc[eligible.index[:quota], 'accepted'] = True
        rows.loc[eligible.index, 'baseline_acceptance_probability'] = quota / e if e else 0.
        if int(rows.loc[batch.index, 'accepted'].sum()) != quota:
            raise RuntimeError('Cota exata não preservada.')
    rows['decision_status'] = np.select(
        [~rows.available, ~rows.eligible, rows.accepted],
        ['not_available_metadata_exclusion', 'gate_ineligible', 'accepted'], default='selector_abstention')
    rows['model_abstention_inferred_for_unavailable'] = False
    rows['known_reference'] = rows.available & rows.reference_available & rows.severity_label.isin(SEVERITIES)
    rows['known_severe'] = rows.known_reference & rows.severity_label.eq('grave')
    rows['severe_eligible'] = rows.known_severe & rows.eligible
    rows['severe_accepted'] = rows.known_severe & rows.accepted
    rows['clinical_denominator_member'] = rows.known_severe & (rows.eligible if basis == 'eligible_after_gate' else True)
    rows['potential_delta1'] = rows.severe_eligible & rows.category.isin(['leve', 'moderado'])
    rows['potential_delta2'] = rows.severe_eligible & rows.category.eq('leve')
    rows['u_delta1'] = rows.accepted & rows.potential_delta1
    rows['u_delta2'] = rows.accepted & rows.potential_delta2
    rows['baseline_expected_delta1'] = rows.potential_delta1.astype(float) * rows.baseline_acceptance_probability
    rows['baseline_expected_delta2'] = rows.potential_delta2.astype(float) * rows.baseline_acceptance_probability
    rows['residual_error'] = False
    known_accepted = rows.accepted & rows.known_reference
    rows.loc[known_accepted, 'residual_error'] = rows.loc[known_accepted, 'category'].ne(rows.loc[known_accepted, 'severity_label'])
    rows['severe_routed'] = rows.known_severe & ~rows.accepted
    rows['target_coverage'] = float(coverage)
    rows['coverage_basis'] = basis
    return rows


def _summarize(rows, coverage, basis):
    planned = len(rows)
    available = rows.loc[rows.available]
    n, e, k = len(available), int(available.eligible.sum()), int(available.accepted.sum())
    d = int(available.clinical_denominator_member.sum())
    u1, u2 = int(available.u_delta1.sum()), int(available.u_delta2.sum())
    b1, b2 = float(available.baseline_expected_delta1.sum()), float(available.baseline_expected_delta2.sum())
    known = int(available.known_reference.sum())
    severe = int(available.known_severe.sum())
    severe_accepted = int(available.severe_accepted.sum())
    residual = int(available.residual_error.sum())
    routed = int(available.severe_routed.sum())
    saturated = bool(available.batch_gate_saturated.any())
    inactive = bool(n and e == k)
    status = ('configuration_absent' if not planned else 'no_available_observations' if not n
              else 'no_known_severe_reference' if not d else 'no_eligible_observations' if not e
              else 'selector_inactive_gate_saturation' if saturated and inactive
              else 'selector_inactive' if inactive else 'total_abstention' if not k else 'descriptive')
    return dict(target_coverage=float(coverage), coverage_basis=basis, status=status,
        planned_count=planned, available_count=n, unavailable_count=planned-n, eligible_count=e, accepted_count=k,
        operational_denominator=n if basis == 'global' else e,
        availability_fraction_of_planned=_ratio(n,planned),
        gate_ineligible_available_count=n-e, selector_abstention_count=e-k,
        available_not_accepted_count=n-k, coverage_global=_ratio(k,n), coverage_eligible_after_gate=_ratio(k,e),
        clinical_denominator=d, clinical_denominator_name=('known_severe_available' if basis=='global' else 'known_severe_eligible'),
        known_reference_count=known, unknown_reference_available_count=n-known, known_severe_count=severe,
        severe_eligible_count=int(available.severe_eligible.sum()), severe_accepted_count=severe_accepted,
        policy_u_delta1_count=u1, baseline_expected_u_delta1_count=b1,
        policy_rate=_ratio(u1,d), baseline_expected_rate=_ratio(b1,d), contrast=_ratio(u1-b1,d),
        policy_u_delta2_count=u2, baseline_expected_u_delta2_count=b2,
        policy_rate_delta2=_ratio(u2,d), baseline_expected_rate_delta2=_ratio(b2,d), contrast_delta2=_ratio(u2-b2,d),
        risk_among_severe_accepted_delta1=_ratio(u1,severe_accepted), risk_among_severe_accepted_denominator=severe_accepted,
        residual_error_count=residual, residual_error_denominator=k, residual_error_rate=_ratio(residual,k),
        residual_error_definition='accepted_category_differs_from_known_reference_any_severity',
        severe_routed_count=routed, severe_routing_denominator=severe, severe_routing_rate=_ratio(routed,severe),
        selector_inactive=inactive, any_batch_gate_saturated=saturated, total_abstention=bool(n and not k),
        baseline='exact_uniform_random_same_batch_quota_expectation', baseline_random_draws=0,
        equivalence_claim=False, inferential_claim=False, coverage_winner_selected=False)


def _mechanism_rows(summary):
    specifications = (
        ('severe_underestimation_delta2','policy_u_delta2_count','clinical_denominator','policy_rate_delta2',
         summary['clinical_denominator_name']),
        ('severe_routing_rate','severe_routed_count','severe_routing_denominator','severe_routing_rate','known_severe_available'),
        ('risk_among_severe_accepted_delta1','policy_u_delta1_count','risk_among_severe_accepted_denominator',
         'risk_among_severe_accepted_delta1','known_severe_accepted'),
        ('residual_error_rate','residual_error_count','residual_error_denominator','residual_error_rate','accepted_known_reference'),
    )
    return [dict(target_coverage=summary['target_coverage'],coverage_basis=summary['coverage_basis'],
        scope=summary['scope'],model=summary['model'],mechanism=name,numerator=summary[numerator],
        denominator=summary[denominator],denominator_name=denominator_name,value=summary[value],
        estimable=bool(summary[denominator]),non_estimable_reason=None if summary[denominator] else 'zero_denominator',
        baseline_expected_numerator=(summary['baseline_expected_u_delta2_count'] if name=='severe_underestimation_delta2' else None),
        baseline_expected_rate=(summary['baseline_expected_rate_delta2'] if name=='severe_underestimation_delta2' else None),
        contrast=(summary['contrast_delta2'] if name=='severe_underestimation_delta2' else None),
        global_decisions_reused=True,quota_recomputed=False,equivalence_claim=False,inferential_claim=False)
        for name,numerator,denominator,value,denominator_name in specifications]


def evaluate_fixed_quota(rows, scores, *, models, basis, seed=APPROVED_TIE_SEED):
    """Four quota descriptions for an explicit, single-family C/DCB scope.

Snapshots are parallel observations, never collapsed into independent repeated
experiments. Caller-supplied models retain missing configurations. Models outside
this scope must be removed explicitly before the call; scopes are not inferred
from favourable outcomes. Strata reuse the global decisions and baseline batch
probabilities, never new quotas or parameter adjustments inside each stratum.
"""
    base = _validate(rows,scores,models,basis,seed)
    summary_rows, snapshot_rows, decisions, mechanisms, strata = [], [], [], [], []
    conceptual = all(model.startswith('claude-') for model in models)
    for coverage in COVERAGE_GRID:
        selected = _decide(base,coverage,basis,int(seed))
        decisions.append(selected)
        scopes = [('family',None,selected)] + [('snapshot',model,selected.loc[selected.model.eq(model)]) for model in sorted(models)]
        for scope,model,subset in scopes:
            summary = dict(_summarize(subset,coverage,basis),scope=scope,model=model,
                           conceptual_replication=conceptual,protocol_confounding_with_other_providers=conceptual,
                           historical_caveat_preserved=True,independent_replications_claimed=False)
            if scope=='family':
                present=set(selected.loc[selected.available,'model'])
                summary.update(expected_snapshots=list(sorted(models)),available_snapshots=list(sorted(present)),
                               missing_available_snapshots=list(sorted(set(models)-present)),family_complete=len(present)==len(models))
                summary_rows.append(summary)
            else:snapshot_rows.append(summary)
            mechanisms.extend(_mechanism_rows(summary))
            for stratum in STRATA:
                strata.append(dict(_summarize(subset.loc[subset.morphology_stratum.eq(stratum)],coverage,basis),
                    scope=scope,model=model,morphology_stratum=stratum,global_decisions_reused=True,
                    quota_recomputed_in_stratum=False,constants_refitted_in_stratum=False,
                    conceptual_replication=conceptual,protocol_confounding_with_other_providers=conceptual,
                    unclassified_morphology_in_parent_count=int(subset.morphology_stratum.eq('morphology_not_assessable').sum())))
    return dict(summary_bycoverage=pd.DataFrame(summary_rows,dtype=object),by_snapshot=pd.DataFrame(snapshot_rows,dtype=object),
        decisions=pd.concat(decisions,ignore_index=True),mechanisms=pd.DataFrame(mechanisms,dtype=object),
        strata=pd.DataFrame(strata,dtype=object),metadata=dict(version=VERSION,scope='C_DCB_explicit_models',
        models=list(sorted(models)),coverage_grid=list(COVERAGE_GRID),basis=basis,tie_seed=int(seed),
        tie_hash_version=TIE_HASH_VERSION,tie_hash_literal=TIE_HASH_LITERAL,
        ordering=['risk_score_ascending','existing_hash_uniform_ascending','canonical_pair_id_lexicographic_ascending'],
        residual_tie_rule='literal_pair_id_python_unicode_codepoint_order',
        comparative_eligible_set_invariance_required_from_caller=True,
        comparative_eligible_set_invariance_verified_in_this_adapter=False,
        provenance_validation_required_before_call=True,entrypoint_does_not_authorize_real_execution=True,
        real_data_execution_authorized_by_this_function=False,retrospective_exploratory=True,
        conceptual_replication=conceptual,protocol_confounding_with_other_providers=conceptual,
        random_replicates_generated=0,independent_replications_claimed=False,inference_performed=False,
        coverage_winner_selected=False,equivalence_claim=False,DEFINITION_OF_DONE=False))
