from __future__ import annotations

import copy
import json
from typing import Any, Dict, List

import altair as alt
import pandas as pd
import streamlit as st

from donor_compass import (
    DEFAULT_PROJECT_DATA,
    EXAMPLE_CUSTOM_WORLDVIEWS,
    allocate_budget,
    vote_borda,
    vote_credence_weighted_custom,
    vote_lexicographic_maximin,
    vote_mec,
    vote_met,
    vote_msa,
    vote_my_favorite_theory,
    vote_nash_bargaining,
    vote_split_cycle,
)

MORAL_WEIGHT_KEYS = [
    "human_life_years",
    "human_ylds",
    "human_income_doublings",
    "chickens_birds",
    "fish",
    "shrimp",
    "non_shrimp_invertebrates",
    "mammals",
]

RISK_PROFILE_LABELS = {
    0: "neutral",
    1: "upside",
    2: "downside",
    3: "combined",
}
RISK_PROFILE_DESCRIPTIONS = {
    0: "Baseline best‑guess numbers.",
    1: "Optimistic estimates for uncertain effects.",
    2: "Cautious estimates for uncertain effects.",
    3: "Dataset’s blended estimate across scenarios.",
}

HELP_CREDENCE = "Relative weight of this worldview. Most methods normalize credences to sum to 1."
HELP_RISK_PROFILE = (
    "This is your stance on uncertainty in impact estimates. The data includes multiple "
    "plausible estimates for each effect. You are choosing which kind of estimate to "
    "trust everywhere: baseline best‑guess (0), optimistic estimates that lean toward higher "
    "impact when evidence is uncertain (1), cautious estimates that lean toward lower impact "
    "to avoid over‑valuing projects (2), or the dataset’s blended estimate across scenarios (3)."
)
HELP_P_EXTINCTION = (
    "Probability of near-term extinction. Non-xrisk projects are scaled by (1 - p_extinction)."
)
HELP_THEORY_TYPE = "Used by MSA only: cardinal uses scores, binary votes on permissibility."
HELP_MORAL_WEIGHT = "Weight for this recipient type; higher increases value for effects on this type."
HELP_DISCOUNT_FACTORS = "Time discount factors t0..t5 applied to effect values."
HELP_XRISK = "If checked, project is near-term xrisk and is not discounted by p_extinction."
HELP_DIMINISHING_RETURNS = "Multiplier applied per $10M step (index 0 = first $10M)."
HELP_EFFECT_RECIPIENT = "Recipient type for this effect; links to the matching moral weight."
HELP_EFFECT_VALUES = (
    "These are alternative scenario estimates. Your risk profile chooses which set of "
    "estimates is used throughout the calculation."
)

METHOD_CONFIGS = [
    {
        "id": "credence_weighted",
        "label": "Marketplace / Credence-weighted",
        "fn": vote_credence_weighted_custom,
        "description": "Each worldview allocates its credence share to its top project.",
    },
    {
        "id": "favorite_theory",
        "label": "My Favorite Theory",
        "fn": vote_my_favorite_theory,
        "description": "Use the single highest-credence worldview at each step.",
    },
    {
        "id": "mec",
        "label": "MEC",
        "fn": vote_mec,
        "description": "Maximize expected choiceworthiness across worldviews.",
    },
    {
        "id": "met",
        "label": "MET",
        "fn": vote_met,
        "description": "Threshold switch between favorite-theory and similarity compromise.",
    },
    {
        "id": "nash",
        "label": "Nash Bargaining",
        "fn": vote_nash_bargaining,
        "description": "Maximize bargaining objective relative to disagreement point.",
    },
    {
        "id": "msa",
        "label": "MSA",
        "fn": vote_msa,
        "description": "Cardinal cluster plus binary permissibility vote.",
    },
    {
        "id": "borda",
        "label": "Borda",
        "fn": vote_borda,
        "description": "Credence-weighted Borda ranking.",
    },
    {
        "id": "split_cycle",
        "label": "Split-Cycle",
        "fn": vote_split_cycle,
        "description": "Pairwise majority graph with cycle handling.",
    },
    {
        "id": "lexicographic_maximin",
        "label": "Lexicographic Maximin",
        "fn": vote_lexicographic_maximin,
        "description": "Choose project that best improves the worst-off weighted utility entries.",
    },
]

METHODS_BY_LABEL = {cfg["label"]: cfg for cfg in METHOD_CONFIGS}

METHOD_EXPLANATIONS = {
    "Marketplace / Credence-weighted": (
        "Each worldview gets a slice of the budget equal to its credence and spends it on its "
        "top project at each step. The final allocation is a weighted mix of worldviews."
    ),
    "My Favorite Theory": (
        "Ignores minority views and follows the single highest‑credence worldview. Each step "
        "goes to that worldview’s top project."
    ),
    "MEC": (
        "Averages project values across worldviews using their credences, then allocates each "
        "step to the project with the highest expected value."
    ),
    "MET": (
        "If one worldview has enough credence (above the threshold), follow it. Otherwise, "
        "choose a compromise worldview that is most representative of the group and allocate "
        "to its top project."
    ),
    "Nash Bargaining": (
        "Treats worldviews as negotiators. Chooses the project that best improves everyone "
        "relative to a disagreement baseline, avoiding outcomes that are very bad for any one view."
    ),
    "MSA": (
        "First combines cardinal worldviews to identify permissible projects, then lets binary "
        "worldviews vote on permissibility. If nothing is permissible by majority, it stops or falls back."
    ),
    "Borda": (
        "Each worldview ranks projects; higher ranks get more points. Credence‑weighted points "
        "determine the winner each step."
    ),
    "Split-Cycle": (
        "Uses pairwise majority comparisons between projects and resolves cycles to find a "
        "stable winner when rankings conflict."
    ),
    "Lexicographic Maximin": (
        "Prioritizes the worst‑off worldview first, then the next worst, and so on. It chooses "
        "the project that most improves the bottom of the distribution of worldview scores."
    ),
}


def _make_project(recipient_type: str, base_value: float, dr_curve: List[float]) -> Dict[str, Any]:
    return {
        "tags": {"near_term_xrisk": False},
        "diminishing_returns": dr_curve,
        "effects": {
            "effect_main": {
                "recipient_type": recipient_type,
                "values": [[float(base_value)] * 4 for _ in range(6)],
            }
        },
    }


def _worldview(
    name: str,
    credence: float,
    human: float = 0.0,
    chickens: float = 0.0,
    fish: float = 0.0,
    theory_type: str | None = None,
) -> Dict[str, Any]:
    worldview: Dict[str, Any] = {
        "name": name,
        "credence": credence,
        "moral_weights": {
            "human_life_years": human,
            "human_ylds": 0.0,
            "human_income_doublings": 0.0,
            "chickens_birds": chickens,
            "fish": fish,
            "shrimp": 0.0,
            "non_shrimp_invertebrates": 0.0,
            "mammals": 0.0,
        },
        "discount_factors": [1.0] * 6,
        "risk_profile": 0,
        "p_extinction": 0.0,
    }
    if theory_type:
        worldview["theory_type"] = theory_type
    return worldview


def build_demo_scenarios() -> Dict[str, Dict[str, Any]]:
    dominance_data = {
        "project_dominant": _make_project("human_life_years", 500.0, [1.0, 0.05, 0.01, 0.01, 0.01, 0.01]),
        "project_b": _make_project("chickens_birds", 120.0, [1.0] * 6),
        "project_c": _make_project("fish", 100.0, [1.0] * 6),
    }
    aligned_worldviews = [_worldview("Aligned", 1.0, human=1.0, chickens=1.0, fish=1.0)]

    comparison_data = {
        "project_human": _make_project(
            "human_life_years",
            220.0,
            [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.12],
        ),
        "project_chicken": _make_project("chickens_birds", 180.0, [1.0] * 11),
        "project_fish": _make_project("fish", 140.0, [1.0] * 11),
    }
    shared_worldviews = [
        _worldview("HumanAnchor", 0.5, human=1.0, chickens=0.1, fish=0.05),
        _worldview("ChickenAnchor", 0.3, human=0.1, chickens=1.0, fish=0.2),
        _worldview("FishAnchor", 0.2, human=0.1, chickens=0.1, fish=1.0),
    ]

    scenarios = {
        "Default dataset + example worldviews": {
            "description": "Full Donor Compass dataset with built-in example worldviews.",
            "data": copy.deepcopy(DEFAULT_PROJECT_DATA),
            "worldviews": copy.deepcopy(EXAMPLE_CUSTOM_WORLDVIEWS),
            "default_budget": 100.0,
            "default_increment": 10.0,
        },
        "Dominant project saturates (E2E Scenario A)": {
            "description": "One project starts dominant, then diminishing returns force a switch.",
            "data": dominance_data,
            "worldviews": aligned_worldviews,
            "default_budget": 60.0,
            "default_increment": 10.0,
        },
        "Shared credence tradeoff (E2E Scenario B)": {
            "description": "Three worldviews with different anchors and a shared tradeoff budget.",
            "data": comparison_data,
            "worldviews": shared_worldviews,
            "default_budget": 100.0,
            "default_increment": 10.0,
        },
    }
    return scenarios


def validate_project_data(data: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(data, dict) or not data:
        return ["Project data must be a non-empty object keyed by project id."]

    for project_id, project_data in data.items():
        if not isinstance(project_data, dict):
            errors.append(f"Project '{project_id}' must be an object.")
            continue
        if "effects" not in project_data:
            errors.append(f"Project '{project_id}' is missing 'effects'.")
        if "diminishing_returns" not in project_data:
            errors.append(f"Project '{project_id}' is missing 'diminishing_returns'.")
        if "tags" not in project_data or "near_term_xrisk" not in project_data.get("tags", {}):
            errors.append(f"Project '{project_id}' tags must include 'near_term_xrisk'.")
    return errors


def validate_worldviews(worldviews: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(worldviews, list):
        return ["Worldviews must be a list."]
    if len(worldviews) == 0:
        return ["At least one worldview is required."]

    for idx, worldview in enumerate(worldviews):
        label = worldview.get("name", f"worldview_{idx}") if isinstance(worldview, dict) else f"worldview_{idx}"
        if not isinstance(worldview, dict):
            errors.append(f"{label}: worldview entry must be an object.")
            continue

        if "credence" not in worldview:
            errors.append(f"{label}: missing 'credence'.")
        else:
            try:
                credence = float(worldview["credence"])
                if credence < 0:
                    errors.append(f"{label}: credence must be non-negative.")
            except (TypeError, ValueError):
                errors.append(f"{label}: credence must be numeric.")

        if "moral_weights" not in worldview or not isinstance(worldview["moral_weights"], dict):
            errors.append(f"{label}: missing 'moral_weights' object.")
        else:
            missing_keys = [k for k in MORAL_WEIGHT_KEYS if k not in worldview["moral_weights"]]
            if missing_keys:
                errors.append(f"{label}: missing moral weight keys: {', '.join(missing_keys)}")

        factors = worldview.get("discount_factors")
        if not isinstance(factors, list) or len(factors) != 6:
            errors.append(f"{label}: 'discount_factors' must be a list of 6 numbers.")
        else:
            for factor in factors:
                if not isinstance(factor, (int, float)):
                    errors.append(f"{label}: all discount factors must be numeric.")
                    break

        risk_profile = worldview.get("risk_profile")
        if not isinstance(risk_profile, int) or risk_profile not in RISK_PROFILE_LABELS:
            errors.append(f"{label}: 'risk_profile' must be an integer in [0, 1, 2, 3].")

        if "p_extinction" not in worldview:
            errors.append(f"{label}: missing 'p_extinction'.")
        else:
            try:
                float(worldview["p_extinction"])
            except (TypeError, ValueError):
                errors.append(f"{label}: 'p_extinction' must be numeric.")

        theory_type = worldview.get("theory_type")
        if theory_type is not None and str(theory_type).lower() not in {"binary", "cardinal"}:
            errors.append(f"{label}: optional 'theory_type' must be 'binary' or 'cardinal'.")

    return errors


def compute_hhi(funding: Dict[str, float]) -> float:
    total = float(sum(funding.values()))
    if total <= 0:
        return 0.0
    return float(sum((value / total) ** 2 for value in funding.values()))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_discount_factors(factors: Any) -> List[float]:
    values = factors if isinstance(factors, list) else []
    normalized = []
    for i in range(6):
        normalized.append(_safe_float(values[i], 1.0) if i < len(values) else 1.0)
    return normalized


def _normalize_values_matrix(values: Any) -> List[List[float]]:
    matrix = []
    for i in range(6):
        row = []
        for j in range(4):
            try:
                row.append(float(values[i][j]))
            except (TypeError, ValueError, IndexError):
                row.append(0.0)
        matrix.append(row)
    return matrix


def _default_worldview_template(index: int) -> Dict[str, Any]:
    return {
        "name": f"New worldview {index}",
        "credence": 0.1,
        "moral_weights": {key: 0.0 for key in MORAL_WEIGHT_KEYS},
        "discount_factors": [1.0] * 6,
        "risk_profile": 0,
        "p_extinction": 0.0,
    }


def render_worldviews_editor(worldviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    control_cols = st.columns([1, 1, 3])
    if control_cols[0].button("Add worldview"):
        worldviews.append(_default_worldview_template(len(worldviews) + 1))
    if control_cols[1].button("Remove last worldview", disabled=len(worldviews) <= 1):
        worldviews.pop()

    edited_worldviews: List[Dict[str, Any]] = []
    risk_options = list(RISK_PROFILE_LABELS.keys())

    for idx, worldview in enumerate(worldviews):
        name_default = worldview.get("name", f"worldview_{idx}")
        header = f"Worldview {idx + 1}: {name_default}"
        with st.expander(header, expanded=(idx == 0)):
            name = st.text_input("Name", value=name_default, key=f"wv_{idx}_name")
            credence = st.number_input(
                "Credence",
                min_value=0.0,
                value=_safe_float(worldview.get("credence", 0.0), 0.0),
                step=0.01,
                help=HELP_CREDENCE,
                key=f"wv_{idx}_credence",
            )
            risk_profile_default = worldview.get("risk_profile", 0)
            risk_profile = st.selectbox(
                "Risk profile",
                options=risk_options,
                index=risk_options.index(risk_profile_default)
                if risk_profile_default in risk_options
                else 0,
                format_func=lambda v: f"{v} ({RISK_PROFILE_LABELS[v]}): {RISK_PROFILE_DESCRIPTIONS[v]}",
                help=HELP_RISK_PROFILE,
                key=f"wv_{idx}_risk_profile",
            )
            st.caption(
                "This choice reflects how you want to handle uncertainty in impact estimates. "
                "Baseline (0) uses the dataset’s best‑guess numbers. Upside (1) uses more "
                "optimistic estimates for uncertain effects, meaning you are comfortable "
                "funding based on the possibility that projects go unusually well. Downside (2) "
                "uses more cautious estimates for uncertain effects, meaning you prioritize "
                "robustness and want to avoid over‑valuing projects if outcomes disappoint. "
                "Combined (3) uses the dataset’s blended estimate across scenarios."
            )
            p_extinction = st.slider(
                "Extinction probability",
                min_value=0.0,
                max_value=1.0,
                value=_safe_float(worldview.get("p_extinction", 0.0), 0.0),
                step=0.01,
                help=HELP_P_EXTINCTION,
                key=f"wv_{idx}_p_extinction",
            )
            theory_type = worldview.get("theory_type", "")
            theory_type_options = ["", "cardinal", "binary"]
            theory_type = st.selectbox(
                "Theory type (optional)",
                options=theory_type_options,
                index=theory_type_options.index(theory_type)
                if theory_type in theory_type_options
                else 0,
                help=HELP_THEORY_TYPE,
                key=f"wv_{idx}_theory_type",
            )

            st.markdown("Moral weights")
            weights = {}
            weight_cols = st.columns(2)
            for i, key in enumerate(MORAL_WEIGHT_KEYS):
                col = weight_cols[i % 2]
                weights[key] = col.number_input(
                    key,
                    value=_safe_float(worldview.get("moral_weights", {}).get(key, 0.0), 0.0),
                    step=0.01,
                    help=HELP_MORAL_WEIGHT,
                    key=f"wv_{idx}_mw_{key}",
                )

            st.markdown("Discount factors")
            factors_src = _normalize_discount_factors(worldview.get("discount_factors"))
            factor_cols = st.columns(3)
            factors = []
            for i in range(6):
                factors.append(
                    factor_cols[i % 3].number_input(
                        f"t{i}",
                        value=factors_src[i],
                        step=0.01,
                        help=HELP_DISCOUNT_FACTORS,
                        key=f"wv_{idx}_df_{i}",
                    )
                )

            edited = {
                "name": name,
                "credence": float(credence),
                "moral_weights": weights,
                "discount_factors": [float(value) for value in factors],
                "risk_profile": int(risk_profile),
                "p_extinction": float(p_extinction),
            }
            if theory_type:
                edited["theory_type"] = theory_type
            edited_worldviews.append(edited)

    return edited_worldviews


def render_project_editor(project_data: Dict[str, Any]) -> Dict[str, Any]:
    edited_projects: Dict[str, Any] = {}
    project_ids = sorted(project_data.keys())

    for project_id in project_ids:
        project = project_data[project_id]
        with st.expander(f"Project: {project_id}", expanded=False):
            near_term_xrisk = st.checkbox(
                "Near-term xrisk",
                value=bool(project.get("tags", {}).get("near_term_xrisk", False)),
                help=HELP_XRISK,
                key=f"proj_{project_id}_xrisk",
            )

            dr_curve = project.get("diminishing_returns", [])
            if not isinstance(dr_curve, list) or not dr_curve:
                dr_curve = [1.0]
            dr_df = pd.DataFrame(
                {"step": list(range(len(dr_curve))), "factor": [_safe_float(v, 0.0) for v in dr_curve]}
            )
            st.markdown("Diminishing returns")
            dr_df = st.data_editor(
                dr_df,
                use_container_width=True,
                num_rows="fixed",
                key=f"proj_{project_id}_dr",
                column_config={
                    "step": st.column_config.NumberColumn(disabled=True),
                    "factor": st.column_config.NumberColumn(help=HELP_DIMINISHING_RETURNS),
                },
            )
            dr_values = [_safe_float(v, 0.0) for v in dr_df["factor"].tolist()]

            effects = project.get("effects", {})
            edited_effects: Dict[str, Any] = {}
            for effect_id, effect in effects.items():
                st.markdown(f"Effect: `{effect_id}`")
                recipient_type = effect.get("recipient_type", MORAL_WEIGHT_KEYS[0])
                if recipient_type not in MORAL_WEIGHT_KEYS:
                    recipient_type = MORAL_WEIGHT_KEYS[0]
                recipient_type = st.selectbox(
                    "Recipient type",
                    options=MORAL_WEIGHT_KEYS,
                    index=MORAL_WEIGHT_KEYS.index(recipient_type),
                    help=HELP_EFFECT_RECIPIENT,
                    key=f"proj_{project_id}_effect_{effect_id}_recipient",
                )

                values_matrix = _normalize_values_matrix(effect.get("values"))
                values_df = pd.DataFrame(
                    values_matrix,
                    columns=[f"risk_{i}" for i in range(4)],
                    index=[f"t{i}" for i in range(6)],
                )
                values_df = st.data_editor(
                    values_df,
                    use_container_width=True,
                    num_rows="fixed",
                    key=f"proj_{project_id}_effect_{effect_id}_values",
                    column_config={
                        "risk_0": st.column_config.NumberColumn(help=HELP_EFFECT_VALUES),
                        "risk_1": st.column_config.NumberColumn(help=HELP_EFFECT_VALUES),
                        "risk_2": st.column_config.NumberColumn(help=HELP_EFFECT_VALUES),
                        "risk_3": st.column_config.NumberColumn(help=HELP_EFFECT_VALUES),
                    },
                )
                effect_out = copy.deepcopy(effect)
                effect_out["recipient_type"] = recipient_type
                effect_out["values"] = values_df.to_numpy().tolist()
                edited_effects[effect_id] = effect_out

            project_out = copy.deepcopy(project)
            project_out["tags"] = dict(project_out.get("tags", {}))
            project_out["tags"]["near_term_xrisk"] = bool(near_term_xrisk)
            project_out["diminishing_returns"] = dr_values
            project_out["effects"] = edited_effects
            edited_projects[project_id] = project_out

    return edited_projects


def build_method_kwargs(
    method_id: str,
    custom_worldviews: List[Dict[str, Any]],
    tie_break: str,
    random_seed: int | None,
    return_debug: bool,
    met_threshold: float,
    disagreement_point: str,
    msa_mode: str,
    msa_top_k: int,
    msa_within_percent: float,
    msa_binary_threshold: float,
    msa_no_permissible_action: str,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"custom_worldviews": custom_worldviews}

    if method_id == "credence_weighted":
        return kwargs

    kwargs["tie_break"] = tie_break
    kwargs["return_debug"] = return_debug
    if random_seed is not None:
        kwargs["random_seed"] = random_seed

    if method_id == "met":
        kwargs["met_threshold"] = met_threshold
    elif method_id == "nash":
        kwargs["disagreement_point"] = disagreement_point
    elif method_id == "msa":
        kwargs["cardinal_permissibility_mode"] = msa_mode
        kwargs["binary_permissibility_threshold"] = msa_binary_threshold
        kwargs["no_permissible_action"] = msa_no_permissible_action
        if msa_mode == "top_k":
            kwargs["cardinal_top_k"] = msa_top_k
        elif msa_mode == "within_percent":
            kwargs["cardinal_within_percent"] = msa_within_percent

    return kwargs


def build_history_frames(allocation: Dict[str, Any], project_ids: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    for entry in allocation.get("history", []):
        row: Dict[str, Any] = {
            "iteration": entry.get("iteration", 0),
            "stopped": bool(entry.get("stopped", False)),
            "reason": entry.get("reason", ""),
            "remaining_budget": float(entry.get("remaining_budget", 0.0)),
        }
        allocations = entry.get("allocations", {})
        for project_id in project_ids:
            row[project_id] = float(allocations.get(project_id, 0.0))
        row["iteration_total"] = float(sum(row[project_id] for project_id in project_ids))
        row["meta"] = json.dumps(entry.get("meta", {}), sort_keys=True) if "meta" in entry else ""
        rows.append(row)

    if not rows:
        empty = pd.DataFrame()
        return empty, empty

    iter_df = pd.DataFrame(rows)
    cumulative_df = iter_df[["iteration"] + project_ids].copy()
    cumulative_df[project_ids] = cumulative_df[project_ids].cumsum()
    return iter_df, cumulative_df


def render_single_result(
    method_label: str,
    allocation: Dict[str, Any],
    project_data: Dict[str, Any],
) -> None:
    funding = allocation["funding"]
    project_ids = list(project_data.keys())
    total_allocated = float(sum(funding.values()))
    hhi = compute_hhi(funding)
    top_project = max(funding, key=funding.get) if funding else "n/a"

    st.subheader("Run Summary")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Method", method_label)
    metric_cols[1].metric("Total allocated ($M)", f"{total_allocated:.2f}")
    metric_cols[2].metric("Top project", top_project)
    metric_cols[3].metric("HHI", f"{hhi:.4f}")

    funding_df = (
        pd.DataFrame({"project": list(funding.keys()), "allocated_m": list(funding.values())})
        .sort_values("allocated_m", ascending=False)
        .reset_index(drop=True)
    )
    if total_allocated > 0:
        funding_df["share_pct"] = 100 * funding_df["allocated_m"] / total_allocated
    else:
        funding_df["share_pct"] = 0.0

    st.subheader("Final Funding")
    funding_chart = (
        alt.Chart(funding_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "project:N",
                sort=funding_df["project"].tolist(),
                axis=alt.Axis(labelAngle=-45),
            ),
            y=alt.Y("allocated_m:Q", title="Allocated ($M)"),
            color=alt.Color("project:N", legend=None),
            tooltip=[
                "project:N",
                alt.Tooltip("allocated_m:Q", format=".2f"),
                alt.Tooltip("share_pct:Q", format=".2f"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(funding_chart, use_container_width=True)
    st.dataframe(
        funding_df,
        use_container_width=True,
        column_config={
            "allocated_m": st.column_config.NumberColumn(format="%.3f"),
            "share_pct": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )

    iter_df, cumulative_df = build_history_frames(allocation, project_ids)
    if not iter_df.empty:
        st.subheader("Iteration History")
        chart_df = cumulative_df.melt(
            id_vars=["iteration"],
            value_vars=project_ids,
            var_name="project",
            value_name="allocated_m",
        )
        chart = (
            alt.Chart(chart_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("iteration:Q", title="Iteration"),
                y=alt.Y("allocated_m:Q", title="Cumulative allocation ($M)"),
                color=alt.Color("project:N", title="Project"),
                tooltip=["iteration:Q", "project:N", "allocated_m:Q"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(iter_df, use_container_width=True)

        debug_rows = [entry for entry in allocation.get("history", []) if "meta" in entry]
        if debug_rows:
            with st.expander("Iteration debug metadata"):
                for row in debug_rows:
                    st.markdown(f"Iteration {row.get('iteration', 0)}")
                    st.json(row.get("meta", {}))


def run_single_method(
    method_cfg: Dict[str, Any],
    project_data: Dict[str, Any],
    worldviews: List[Dict[str, Any]],
    total_budget: float,
    increment_size: float,
    tie_break: str,
    random_seed: int | None,
    return_debug: bool,
    met_threshold: float,
    disagreement_point: str,
    msa_mode: str,
    msa_top_k: int,
    msa_within_percent: float,
    msa_binary_threshold: float,
    msa_no_permissible_action: str,
) -> Dict[str, Any]:
    kwargs = build_method_kwargs(
        method_cfg["id"],
        worldviews,
        tie_break,
        random_seed,
        return_debug,
        met_threshold,
        disagreement_point,
        msa_mode,
        msa_top_k,
        msa_within_percent,
        msa_binary_threshold,
        msa_no_permissible_action,
    )
    return allocate_budget(
        project_data,
        method_cfg["fn"],
        total_budget=total_budget,
        increment_size=increment_size,
        **kwargs,
    )


def run_all_methods(
    project_data: Dict[str, Any],
    worldviews: List[Dict[str, Any]],
    total_budget: float,
    increment_size: float,
    tie_break: str,
    random_seed: int | None,
    return_debug: bool,
    met_threshold: float,
    disagreement_point: str,
    msa_mode: str,
    msa_top_k: int,
    msa_within_percent: float,
    msa_binary_threshold: float,
    msa_no_permissible_action: str,
) -> Dict[str, Any]:
    successful_runs: List[Dict[str, Any]] = []
    failed_runs: List[Dict[str, str]] = []

    for method_cfg in METHOD_CONFIGS:
        try:
            allocation = run_single_method(
                method_cfg,
                project_data,
                worldviews,
                total_budget,
                increment_size,
                tie_break,
                random_seed,
                return_debug,
                met_threshold,
                disagreement_point,
                msa_mode,
                msa_top_k,
                msa_within_percent,
                msa_binary_threshold,
                msa_no_permissible_action,
            )
            funding = allocation["funding"]
            top_project = max(funding, key=funding.get) if funding else "n/a"
            successful_runs.append(
                {
                    "method": method_cfg["label"],
                    "total_allocated_m": float(sum(funding.values())),
                    "top_project": top_project,
                    "hhi": compute_hhi(funding),
                    "funding": funding,
                }
            )
        except Exception as exc:  # pragma: no cover - surfaced directly in app
            failed_runs.append(
                {
                    "method": method_cfg["label"],
                    "error": str(exc),
                }
            )

    return {
        "success": successful_runs,
        "failed": failed_runs,
    }


def _initialize_editor_state(scenarios: Dict[str, Dict[str, Any]], selected_scenario_name: str) -> None:
    scenario = scenarios[selected_scenario_name]
    active = st.session_state.get("active_scenario_name")
    if active != selected_scenario_name:
        st.session_state["active_scenario_name"] = selected_scenario_name
        st.session_state["worldviews_data"] = copy.deepcopy(scenario["worldviews"])
        st.session_state["project_data_data"] = copy.deepcopy(scenario["data"])
        st.session_state["last_run"] = None


def _reset_editors_to_scenario(scenarios: Dict[str, Dict[str, Any]], selected_scenario_name: str) -> None:
    scenario = scenarios[selected_scenario_name]
    st.session_state["worldviews_data"] = copy.deepcopy(scenario["worldviews"])
    st.session_state["project_data_data"] = copy.deepcopy(scenario["data"])


def _parse_optional_seed(raw_seed: str) -> int | None:
    stripped = raw_seed.strip()
    if not stripped:
        return None
    return int(stripped)


def app() -> None:
    st.set_page_config(page_title="Donor Compass Aggregation Demo", layout="wide")
    st.title("Donor Compass Aggregation Demo")
    st.caption("Quick start: choose a scenario in the sidebar, keep defaults, and click Run demo.")

    scenarios = build_demo_scenarios()

    st.sidebar.header("1) Setup")
    view_mode = st.sidebar.radio(
        "View mode",
        ["Single Method Explorer", "All Methods Comparison"],
        help="Single method shows detailed iteration history; comparison runs all methods together.",
    )

    scenario_name = st.sidebar.selectbox(
        "Scenario",
        list(scenarios.keys()),
        help="Starting dataset and example worldviews for the demo.",
    )
    scenario = scenarios[scenario_name]
    _initialize_editor_state(scenarios, scenario_name)

    default_budget = float(scenario["default_budget"])
    default_increment = float(scenario["default_increment"])

    total_budget = st.sidebar.number_input(
        "Total budget ($M)",
        min_value=0.0,
        value=default_budget,
        step=10.0,
        help="Total budget to allocate across projects.",
    )
    increment_size = st.sidebar.number_input(
        "Increment size ($M)",
        min_value=0.1,
        value=default_increment,
        step=1.0,
        help="Allocation step size. Smaller steps allow more switching as returns diminish.",
    )

    method_label = None
    selected_method_id = None
    if view_mode == "Single Method Explorer":
        method_label = st.sidebar.selectbox("Aggregation method", [cfg["label"] for cfg in METHOD_CONFIGS])
        selected_method_id = METHODS_BY_LABEL[method_label]["id"]
        st.sidebar.caption(METHODS_BY_LABEL[method_label]["description"])

    tie_break = "deterministic"
    random_seed_raw = ""
    return_debug = False
    met_threshold = 0.5
    disagreement_point = "zero_spending"
    msa_mode = "winner_take_all"
    msa_top_k = 2
    msa_within_percent = 0.10
    msa_binary_threshold = 0.0
    msa_no_permissible_action = "stop"

    with st.sidebar.expander("Advanced options", expanded=False):
        tie_break = st.selectbox(
            "Tie break",
            ["deterministic", "random"],
            help="How to resolve ties between projects with equal scores.",
        )
        random_seed_raw = st.text_input(
            "Random seed (optional integer)",
            value="",
            help="Used only when tie break is random.",
        )
        return_debug = st.checkbox(
            "Collect per-iteration debug metadata",
            value=False,
            help="Adds per-iteration metadata to the results view.",
        )

        if view_mode == "All Methods Comparison" or selected_method_id == "met":
            st.markdown("MET")
            met_threshold = st.slider(
                "MET threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                help="If max credence >= threshold use favorite theory; else use similarity centroid.",
            )

        if view_mode == "All Methods Comparison" or selected_method_id == "nash":
            st.markdown("Nash")
            disagreement_point = st.selectbox(
                "Nash disagreement point",
                [
                    "zero_spending",
                    "anti_utopia",
                    "random_dictator",
                    "exclusionary_proportional_split",
                ],
                help="Baseline utilities used for Nash bargaining gains.",
            )

        if view_mode == "All Methods Comparison" or selected_method_id == "msa":
            st.markdown("MSA")
            msa_mode = st.selectbox(
                "MSA cardinal permissibility mode",
                ["winner_take_all", "top_k", "within_percent"],
                help="How to convert cardinal scores into a permissible set.",
            )
            if msa_mode == "top_k":
                msa_top_k = st.number_input(
                    "MSA top_k",
                    min_value=1,
                    max_value=20,
                    value=2,
                    step=1,
                    help="Number of top-scoring projects treated as permissible.",
                )
            elif msa_mode == "within_percent":
                msa_within_percent = st.slider(
                    "MSA within-percent",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.10,
                    step=0.01,
                    help="Projects within this percent of the best score are permissible.",
                )
            msa_binary_threshold = st.number_input(
                "MSA binary permissibility threshold",
                value=0.0,
                step=0.01,
                help="Binary worldviews treat projects above this threshold as permissible.",
            )
            msa_no_permissible_action = st.selectbox(
                "MSA no-permissible action",
                ["stop", "fallback_mec"],
                help="What to do if no project exceeds 50% permissibility.",
            )

    st.sidebar.caption("2) Click Run demo in the main panel.")

    st.subheader("2) Inputs")
    st.caption(scenario["description"])
    summary_cols = st.columns(4)
    current_projects = st.session_state.get("project_data_data", scenario["data"])
    current_worldviews = st.session_state.get("worldviews_data", scenario["worldviews"])
    summary_cols[0].metric("Projects", len(current_projects))
    summary_cols[1].metric("Worldviews", len(current_worldviews))
    summary_cols[2].metric("Budget ($M)", f"{float(total_budget):.1f}")
    summary_cols[3].metric("Increment ($M)", f"{float(increment_size):.1f}")

    if st.button("Reset inputs to scenario defaults"):
        _reset_editors_to_scenario(scenarios, scenario_name)
        st.rerun()

    editor_tabs = st.tabs(["Worldviews", "Projects"])
    with editor_tabs[0]:
        worldviews = render_worldviews_editor(st.session_state["worldviews_data"])
    with editor_tabs[1]:
        project_data = render_project_editor(st.session_state["project_data_data"])

    st.session_state["worldviews_data"] = worldviews
    st.session_state["project_data_data"] = project_data

    with st.expander("Method explanations", expanded=False):
        for cfg in METHOD_CONFIGS:
            label = cfg["label"]
            explanation = METHOD_EXPLANATIONS.get(label, cfg["description"])
            st.markdown(f"**{label}**")
            st.write(explanation)

    with st.expander("Definitions & Glossary", expanded=False):
        st.markdown(
            """
- **Credence**: Relative weight for a worldview; most methods normalize credences to sum to 1.
- **Risk profile**: Your stance on uncertainty in impact estimates. Baseline uses best‑guess numbers. Upside leans toward higher estimates (you’re comfortable funding based on potential big wins). Downside leans toward lower estimates (you prioritize robustness and avoid over‑valuing uncertain effects). Combined uses the dataset’s blended estimate across scenarios.
- **Extinction probability**: Scales non-xrisk projects by (1 - p_extinction); xrisk projects are not scaled.
- **Moral weights**: Importance given to each recipient type (higher means more value).
- **Discount factors**: Per-period multipliers t0..t5 applied to effect values.
- **Near-term xrisk**: Marks projects that are exempt from extinction discounting.
- **Diminishing returns**: Multiplier per $10M step that reduces marginal value as funding grows.
- **Effect values**: 6 time periods by 4 risk columns used to compute project value.
- **Risk column mapping**: 0=neutral (baseline), 1=upside (optimistic), 2=downside (pessimistic), 3=combined (mixed, dataset-provided).
- **Increment size**: Budget step size for iterative allocation; smaller steps can switch winners more often.
- **MET threshold**: If max credence >= threshold use favorite theory; else use similarity centroid.
- **Nash disagreement point**: Baseline utilities used when computing bargaining gains.
- **MSA permissibility mode**: How cardinal scores are turned into permissible sets (winner, top_k, within_percent).
            """
        )

    if "last_run" not in st.session_state:
        st.session_state["last_run"] = None

    run_clicked = st.button("Run demo", type="primary", use_container_width=True)

    if run_clicked:
        try:
            random_seed = _parse_optional_seed(random_seed_raw)
        except ValueError:
            st.error("Random seed must be an integer if provided.")
            return

        worldviews = copy.deepcopy(st.session_state.get("worldviews_data", scenario["worldviews"]))
        project_data = copy.deepcopy(st.session_state.get("project_data_data", scenario["data"]))

        worldview_errors = validate_worldviews(worldviews)
        project_data_errors = validate_project_data(project_data)
        validation_errors = worldview_errors + project_data_errors
        if validation_errors:
            st.error("Input validation failed:")
            for err in validation_errors:
                st.write(f"- {err}")
            return

        try:
            if view_mode == "Single Method Explorer":
                method_cfg = METHODS_BY_LABEL[method_label]
                allocation = run_single_method(
                    method_cfg,
                    project_data,
                    worldviews,
                    float(total_budget),
                    float(increment_size),
                    tie_break,
                    random_seed,
                    return_debug,
                    float(met_threshold),
                    disagreement_point,
                    msa_mode,
                    int(msa_top_k),
                    float(msa_within_percent),
                    float(msa_binary_threshold),
                    msa_no_permissible_action,
                )
                st.session_state["last_run"] = {
                    "mode": "single",
                    "method_label": method_cfg["label"],
                    "method_description": method_cfg["description"],
                    "allocation": allocation,
                    "project_data": project_data,
                }
            else:
                comparison = run_all_methods(
                    project_data,
                    worldviews,
                    float(total_budget),
                    float(increment_size),
                    tie_break,
                    random_seed,
                    return_debug,
                    float(met_threshold),
                    disagreement_point,
                    msa_mode,
                    int(msa_top_k),
                    float(msa_within_percent),
                    float(msa_binary_threshold),
                    msa_no_permissible_action,
                )
                st.session_state["last_run"] = {
                    "mode": "comparison",
                    "comparison": comparison,
                }
        except Exception as exc:  # pragma: no cover - surfaced to user
            st.error(f"Allocation run failed: {exc}")
            return

    last_run = st.session_state.get("last_run")
    if not last_run:
        st.info("Ready to run. Click Run demo.")
        return

    if last_run["mode"] == "single":
        st.markdown("---")
        st.subheader(last_run["method_label"])
        st.caption(last_run["method_description"])
        render_single_result(
            last_run["method_label"],
            last_run["allocation"],
            last_run["project_data"],
        )
        return

    comparison = last_run["comparison"]
    successes = comparison["success"]
    failures = comparison["failed"]

    st.markdown("---")
    st.subheader("All Methods Comparison")

    if successes:
        summary_rows = []
        all_projects = sorted({p for row in successes for p in row["funding"].keys()})
        funding_rows = []

        for row in successes:
            summary_rows.append(
                {
                    "method": row["method"],
                    "total_allocated_m": row["total_allocated_m"],
                    "top_project": row["top_project"],
                    "hhi": row["hhi"],
                }
            )
            funding_row = {"method": row["method"]}
            funding_row.update({project_id: float(row["funding"].get(project_id, 0.0)) for project_id in all_projects})
            funding_rows.append(funding_row)

        summary_df = pd.DataFrame(summary_rows).sort_values(["hhi", "method"], ascending=[True, True])
        st.dataframe(
            summary_df,
            use_container_width=True,
            column_config={
                "total_allocated_m": st.column_config.NumberColumn(format="%.3f"),
                "hhi": st.column_config.NumberColumn(format="%.4f"),
            },
        )

        funding_df = pd.DataFrame(funding_rows).set_index("method")
        st.subheader("Funding by method")
        funding_long = funding_df.reset_index().melt(
            id_vars="method",
            var_name="project",
            value_name="allocated_m",
        )
        method_order = list(funding_df.index)
        project_order = list(funding_df.columns)
        funding_chart = (
            alt.Chart(funding_long)
            .mark_bar()
            .encode(
                x=alt.X(
                    "method:N",
                    sort=method_order,
                    axis=alt.Axis(labelAngle=-45),
                ),
                y=alt.Y("allocated_m:Q", title="Allocated ($M)", stack="zero"),
                color=alt.Color(
                    "project:N",
                    sort=project_order,
                    legend=alt.Legend(title=None),
                ),
                tooltip=[
                    "method:N",
                    "project:N",
                    alt.Tooltip("allocated_m:Q", format=".2f"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(funding_chart, use_container_width=True)
        st.dataframe(funding_df, use_container_width=True)

    if failures:
        st.subheader("Methods with errors")
        st.dataframe(pd.DataFrame(failures), use_container_width=True)


if __name__ == "__main__":
    app()
