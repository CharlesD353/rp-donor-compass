# -*- coding: utf-8 -*-
"""End-to-end stress tests for aggregation methods."""

import os

from donor_compass import (
    allocate_budget,
    vote_credence_weighted_custom,
    vote_my_favorite_theory,
    vote_mec,
    vote_met,
    vote_nash_bargaining,
    vote_msa,
    vote_borda,
    vote_split_cycle,
    vote_lexicographic_maximin,
)


def _make_project(recipient_type, base_value, dr_curve):
    return {
        "tags": {"near_term_xrisk": False},
        "diminishing_returns": dr_curve,
        "effects": {
            "effect_main": {
                "recipient_type": recipient_type,
                "values": [[base_value] * 4 for _ in range(6)],
            }
        },
    }


def _worldview(name, credence, human=0.0, chickens=0.0, fish=0.0):
    return {
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


def _find_first_switch_iteration(history, dominant_project):
    for idx, entry in enumerate(history[1:], start=1):
        dominant_allocation = entry["allocations"].get(dominant_project, 0.0)
        other_allocation = sum(
            amount for project_id, amount in entry["allocations"].items() if project_id != dominant_project
        )
        if dominant_allocation < 10.0 and other_allocation > 0:
            return idx
    return None


def _hhi(funding):
    total = sum(funding.values())
    if total <= 0:
        return 0.0
    shares = [amount / total for amount in funding.values()]
    return sum(share * share for share in shares)


def _get_output_mode():
    """
    Output mode for e2e summary.

    Defaults to verbose.
    Set DONOR_COMPASS_E2E_OUTPUT=quiet for compact output when running alongside
    larger test suites.
    """
    raw_mode = os.getenv("DONOR_COMPASS_E2E_OUTPUT", "verbose").strip().lower()
    if raw_mode in {"quiet", "q", "minimal", "min"}:
        return "quiet"
    return "verbose"


def test_end_to_end_aggregation_stress_summary():
    """
    Stress test requested by user:
    1) Make one project dominant and verify all methods pick it first.
    2) Make dominant project saturate quickly and verify methods switch.
    3) Run all methods with the same credences and compare spread/concentration
       (with explicit ordering check on credence-weighted, MEC, and favorite-theory).

    Run with -s to view summary:
        pytest donor_compass/test_aggregation_e2e.py -vv -s

    Quiet mode (for mixed/full test runs):
        DONOR_COMPASS_E2E_OUTPUT=quiet pytest donor_compass/test_aggregation_e2e.py -vv -s
    """
    # -------------------------------------------------------------------------
    # Scenario A: Dominant project should win first, then lose after saturation
    # -------------------------------------------------------------------------
    dominance_data = {
        "project_dominant": _make_project("human_life_years", 500.0, [1.0, 0.05, 0.01, 0.01, 0.01, 0.01]),
        "project_b": _make_project("chickens_birds", 120.0, [1.0] * 6),
        "project_c": _make_project("fish", 100.0, [1.0] * 6),
    }
    aligned_worldviews = [_worldview("Aligned", 1.0, human=1.0, chickens=1.0, fish=1.0)]

    method_configs = [
        ("credence_weighted", vote_credence_weighted_custom, {"custom_worldviews": aligned_worldviews}),
        ("my_favorite_theory", vote_my_favorite_theory, {"custom_worldviews": aligned_worldviews}),
        ("mec", vote_mec, {"custom_worldviews": aligned_worldviews}),
        ("met", vote_met, {"custom_worldviews": aligned_worldviews, "met_threshold": 0.5}),
        ("nash_bargaining", vote_nash_bargaining, {"custom_worldviews": aligned_worldviews}),
        ("msa", vote_msa, {"custom_worldviews": aligned_worldviews, "cardinal_permissibility_mode": "winner_take_all"}),
        ("borda", vote_borda, {"custom_worldviews": aligned_worldviews}),
        ("split_cycle", vote_split_cycle, {"custom_worldviews": aligned_worldviews}),
        ("lexicographic_maximin", vote_lexicographic_maximin, {"custom_worldviews": aligned_worldviews}),
    ]

    scenario_a_rows = []
    for method_name, method_fn, kwargs in method_configs:
        allocation = allocate_budget(
            dominance_data,
            method_fn,
            total_budget=60,
            increment_size=10,
            **kwargs,
        )
        history = allocation["history"]
        first_alloc = history[0]["allocations"]
        assert first_alloc["project_dominant"] > 0, (
            f"{method_name} failed dominant-project first-choice check: {first_alloc}"
        )

        switch_iteration = _find_first_switch_iteration(history, "project_dominant")
        assert switch_iteration is not None, (
            f"{method_name} did not switch away from dominant project after saturation. "
            f"History: {history}"
        )

        non_dominant_total = allocation["funding"]["project_b"] + allocation["funding"]["project_c"]
        assert non_dominant_total > 0, (
            f"{method_name} never allocated to alternatives after saturation. "
            f"Funding: {allocation['funding']}"
        )

        scenario_a_rows.append(
            (
                method_name,
                switch_iteration,
                dict(allocation["funding"]),
                [dict(entry["allocations"]) for entry in history],
            )
        )

    # -------------------------------------------------------------------------
    # Scenario B: Same credences, run all methods and compare spread
    # -------------------------------------------------------------------------
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

    scenario_b_method_configs = [
        ("credence_weighted", vote_credence_weighted_custom, {"custom_worldviews": shared_worldviews}),
        ("my_favorite_theory", vote_my_favorite_theory, {"custom_worldviews": shared_worldviews}),
        ("mec", vote_mec, {"custom_worldviews": shared_worldviews}),
        ("met", vote_met, {"custom_worldviews": shared_worldviews, "met_threshold": 0.5}),
        ("nash_bargaining", vote_nash_bargaining, {"custom_worldviews": shared_worldviews}),
        (
            "msa",
            vote_msa,
            {
                "custom_worldviews": shared_worldviews,
                "cardinal_permissibility_mode": "winner_take_all",
                "no_permissible_action": "fallback_mec",
            },
        ),
        ("borda", vote_borda, {"custom_worldviews": shared_worldviews}),
        ("split_cycle", vote_split_cycle, {"custom_worldviews": shared_worldviews}),
        ("lexicographic_maximin", vote_lexicographic_maximin, {"custom_worldviews": shared_worldviews}),
    ]

    scenario_b_rows = []
    scenario_b_by_name = {}
    for method_name, method_fn, kwargs in scenario_b_method_configs:
        allocation = allocate_budget(
            comparison_data,
            method_fn,
            total_budget=100,
            increment_size=10,
            **kwargs,
        )
        total_allocated = sum(allocation["funding"].values())
        assert abs(total_allocated - 100.0) < 1e-9, (
            f"{method_name} failed to allocate full budget in Scenario B. "
            f"Funding: {allocation['funding']}"
        )

        hhi_value = _hhi(allocation["funding"])
        top_project = max(allocation["funding"], key=allocation["funding"].get)
        row = {
            "method_name": method_name,
            "funding": dict(allocation["funding"]),
            "hhi": hhi_value,
            "top_project": top_project,
        }
        scenario_b_rows.append(row)
        scenario_b_by_name[method_name] = row

    hhi_cw = scenario_b_by_name["credence_weighted"]["hhi"]
    hhi_favorite = scenario_b_by_name["my_favorite_theory"]["hhi"]
    hhi_mec = scenario_b_by_name["mec"]["hhi"]

    assert hhi_cw < hhi_mec < hhi_favorite, (
        "Expected concentration ordering failed (credence-weighted most spread, "
        "favorite most concentrated, MEC in-between). "
        f"HHI: cw={hhi_cw:.4f}, mec={hhi_mec:.4f}, favorite={hhi_favorite:.4f}"
    )

    # -------------------------------------------------------------------------
    # Human-readable summary
    # -------------------------------------------------------------------------
    output_mode = _get_output_mode()
    lines = []
    lines.append("=" * 80)
    lines.append("AGGREGATION STRESS TEST SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Scenario A: Dominant project -> saturation switch (all methods)")
    lines.append("-" * 80)
    for method_name, switch_iteration, funding, history_allocations in scenario_a_rows:
        lines.append(
            f"{method_name:<24} | switch_iter={switch_iteration:<2} | "
            f"funding={funding}"
        )
        if output_mode == "verbose":
            for iteration_idx, allocations in enumerate(history_allocations):
                lines.append(f"  iter {iteration_idx:<2}: {allocations}")

    lines.append("")
    lines.append("Scenario B: Same credences, all methods side-by-side")
    lines.append("-" * 80)
    for row in scenario_b_rows:
        lines.append(
            f"{row['method_name']:<24} | top={row['top_project']:<15} | "
            f"HHI={row['hhi']:.4f} | funding={row['funding']}"
        )
    lines.append("")
    lines.append("Expected ordering validated: HHI(credence_weighted) < HHI(mec) < HHI(my_favorite_theory)")
    lines.append("=" * 80)

    summary = "\n".join(lines)
    if output_mode == "quiet":
        compact = (
            "[donor_compass e2e] Scenario A dominance/saturation checks passed "
            "for all methods; Scenario B HHI ordering validated "
            "(credence_weighted < mec < my_favorite_theory)."
        )
        print("\n" + compact)
    else:
        print("\n" + summary)
