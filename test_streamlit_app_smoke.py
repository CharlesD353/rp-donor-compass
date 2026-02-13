import pytest

pytest.importorskip("streamlit")

from streamlit_app import (
    build_demo_scenarios,
    build_method_kwargs,
    compute_hhi,
    validate_project_data,
    validate_worldviews,
)


def test_scenarios_are_available_and_valid():
    scenarios = build_demo_scenarios()
    assert "Default dataset + example worldviews" in scenarios
    assert "Dominant project saturates (E2E Scenario A)" in scenarios
    assert "Shared credence tradeoff (E2E Scenario B)" in scenarios

    default = scenarios["Default dataset + example worldviews"]
    assert validate_project_data(default["data"]) == []
    assert validate_worldviews(default["worldviews"]) == []


def test_compute_hhi_bounds():
    assert compute_hhi({"a": 0.0, "b": 0.0}) == 0.0
    assert compute_hhi({"a": 50.0, "b": 50.0}) == pytest.approx(0.5)
    assert compute_hhi({"a": 100.0, "b": 0.0}) == pytest.approx(1.0)


def test_method_kwargs_include_method_specific_settings():
    kwargs = build_method_kwargs(
        method_id="met",
        custom_worldviews=[{"name": "w", "credence": 1.0}],
        tie_break="deterministic",
        random_seed=7,
        return_debug=True,
        met_threshold=0.42,
        disagreement_point="zero_spending",
        msa_mode="winner_take_all",
        msa_top_k=2,
        msa_within_percent=0.1,
        msa_binary_threshold=0.0,
        msa_no_permissible_action="stop",
    )
    assert kwargs["met_threshold"] == pytest.approx(0.42)
    assert kwargs["random_seed"] == 7
    assert kwargs["return_debug"] is True
