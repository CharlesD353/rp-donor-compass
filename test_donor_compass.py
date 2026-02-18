# -*- coding: utf-8 -*-
"""
Tests for donor_compass module.

This test suite validates the core functionality that exists in the original
Colab notebook (the non-commented-out/non-DEPRIORITIZED portions).

Tested functionality:
- calculate_single_effect: Computes weighted effect values across time periods
- calculate_project: Aggregates effects for a single project
- calculate_all_projects: Batch calculation across all projects
- adjust_for_extinction_risk: Scales non-xrisk projects by (1 - p_extinction)
- get_diminishing_returns_factor: Looks up DR factor by funding level
- vote_credence_weighted_custom: Allocates increments based on custom worldviews
- allocate_budget: Iteratively allocates total budget in increments
"""

import pytest
import numpy as np
from donor_compass import (
    DEFAULT_PROJECT_DATA,
    calculate_single_effect,
    calculate_project,
    calculate_all_projects,
    adjust_for_extinction_risk,
    INCREMENT_SIZE,
    get_diminishing_returns_factor,
    vote_credence_weighted_custom,
    vote_my_favorite_theory,
    vote_mec,
    vote_met,
    vote_nash_bargaining,
    vote_msa,
    vote_borda,
    vote_split_cycle,
    vote_lexicographic_maximin,
    allocate_budget,
    show_allocation,
    EXAMPLE_CUSTOM_WORLDVIEWS,
)


NASH_TEST_RANDOM_DICTATOR_SEED = 13


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_effect_data():
    """
    Minimal effect data for isolated testing.
    6 time periods x 4 risk profiles, all values = 100.
    """
    return {
        "recipient_type": "human_life_years",
        "values": [[100] * 4 for _ in range(6)]
    }


@pytest.fixture
def varied_effect_data():
    """
    Effect data with varied values for testing weighted calculations.
    Values vary by time period (rows) and risk profile (columns).
    """
    return {
        "recipient_type": "human_life_years",
        "values": [
            [100, 90, 80, 70],   # t0
            [50, 45, 40, 35],    # t1
            [25, 22, 20, 17],    # t2
            [10, 9, 8, 7],       # t3
            [5, 4, 3, 2],        # t4
            [1, 1, 1, 1]         # t5
        ]
    }


@pytest.fixture
def simple_project():
    """Simple project with one effect for isolated testing."""
    return {
        "tags": {"near_term_xrisk": False},
        "diminishing_returns": [1.0, 0.5, 0.25],
        "effects": {
            "effect_1": {
                "recipient_type": "human_life_years",
                "values": [[100] * 4 for _ in range(6)]
            }
        }
    }


@pytest.fixture
def xrisk_project():
    """Project marked as near-term xrisk for extinction adjustment testing."""
    return {
        "tags": {"near_term_xrisk": True},
        "diminishing_returns": [1.0, 0.9, 0.8],
        "effects": {
            "effect_xrisk": {
                "recipient_type": "human_life_years",
                "values": [[1000] * 4 for _ in range(6)]
            }
        }
    }


@pytest.fixture
def simple_data(simple_project, xrisk_project):
    """Combined dataset with both xrisk and non-xrisk projects."""
    return {
        "project_a": simple_project,
        "project_xrisk": xrisk_project
    }


@pytest.fixture
def uniform_discount_factors():
    """All time periods weighted equally."""
    return [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


@pytest.fixture
def declining_discount_factors():
    """Time-discounted factors (near-term prioritized)."""
    return [1.0, 0.8, 0.5, 0.2, 0.05, 0.01]


@pytest.fixture
def simple_worldview():
    """Single worldview for testing vote_credence_weighted_custom."""
    return [{
        "name": "Simple test worldview",
        "credence": 1.0,
        "moral_weights": {
            "human_life_years": 1.0,
            "human_ylds": 0.5,
            "human_income_doublings": 0.1,
            "chickens_birds": 0.01,
            "fish": 0.005,
            "shrimp": 0.0001,
            "non_shrimp_invertebrates": 0.00005,
            "mammals": 0.05
        },
        "discount_factors": [1.0, 0.9, 0.5, 0.2, 0.05, 0.01],
        "risk_profile": 0,
        "p_extinction": 0.0
    }]


# =============================================================================
# Tests for calculate_single_effect
# =============================================================================

class TestCalculateSingleEffect:
    """Tests for the calculate_single_effect function."""

    def test_uniform_values_uniform_discount(self, simple_effect_data, uniform_discount_factors):
        """
        With uniform values (100) and uniform discounts (1.0),
        result = moral_weight * 6 * 100 = moral_weight * 600.
        """
        moral_weight = 1.0
        risk_profile = 0
        result = calculate_single_effect(
            simple_effect_data, moral_weight, uniform_discount_factors, risk_profile
        )
        assert result == 600.0

    def test_moral_weight_scaling(self, simple_effect_data, uniform_discount_factors):
        """Moral weight should scale the result proportionally."""
        risk_profile = 0
        result_weight_1 = calculate_single_effect(
            simple_effect_data, 1.0, uniform_discount_factors, risk_profile
        )
        result_weight_2 = calculate_single_effect(
            simple_effect_data, 2.0, uniform_discount_factors, risk_profile
        )
        assert result_weight_2 == 2 * result_weight_1

    def test_zero_moral_weight(self, simple_effect_data, uniform_discount_factors):
        """Zero moral weight should result in zero value."""
        result = calculate_single_effect(
            simple_effect_data, 0.0, uniform_discount_factors, 0
        )
        assert result == 0.0

    def test_risk_profile_selection(self, varied_effect_data, uniform_discount_factors):
        """Different risk profiles should select different value columns."""
        moral_weight = 1.0
        # Risk profile 0: column 0 values = 100+50+25+10+5+1 = 191
        result_neutral = calculate_single_effect(
            varied_effect_data, moral_weight, uniform_discount_factors, 0
        )
        # Risk profile 3: column 3 values = 70+35+17+7+2+1 = 132
        result_combined = calculate_single_effect(
            varied_effect_data, moral_weight, uniform_discount_factors, 3
        )
        assert result_neutral == 191.0
        assert result_combined == 132.0
        assert result_neutral > result_combined

    def test_discount_factors_applied(self, varied_effect_data):
        """Discount factors should weight each time period."""
        moral_weight = 1.0
        # Custom discount: only first time period counts
        first_only_discount = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = calculate_single_effect(
            varied_effect_data, moral_weight, first_only_discount, 0
        )
        # Only t0 value (100) should be counted
        assert result == 100.0


# =============================================================================
# Tests for calculate_project
# =============================================================================

class TestCalculateProject:
    """Tests for the calculate_project function."""

    def test_single_effect_project(self, simple_project, uniform_discount_factors):
        """Project with one effect should return that effect's value."""
        moral_weights = {"human_life_years": 1.0}
        result = calculate_project(
            simple_project, moral_weights, uniform_discount_factors, 0
        )
        # 6 time periods * 100 value * 1.0 weight = 600
        assert result["total"] == 600.0
        assert "effect_1" in result["breakdown"]
        assert result["breakdown"]["effect_1"] == 600.0

    def test_missing_moral_weight_defaults_to_zero(self, simple_project, uniform_discount_factors):
        """Effect types not in moral_weights should contribute zero."""
        moral_weights = {"some_other_type": 1.0}  # no human_life_years
        result = calculate_project(
            simple_project, moral_weights, uniform_discount_factors, 0
        )
        assert result["total"] == 0.0

    def test_multi_effect_project(self, uniform_discount_factors):
        """Project with multiple effects should sum their values."""
        project = {
            "tags": {"near_term_xrisk": False},
            "diminishing_returns": [1.0],
            "effects": {
                "effect_a": {
                    "recipient_type": "human_life_years",
                    "values": [[100] * 4 for _ in range(6)]
                },
                "effect_b": {
                    "recipient_type": "human_ylds",
                    "values": [[50] * 4 for _ in range(6)]
                }
            }
        }
        moral_weights = {"human_life_years": 1.0, "human_ylds": 0.5}
        result = calculate_project(project, moral_weights, uniform_discount_factors, 0)
        # effect_a: 6 * 100 * 1.0 = 600
        # effect_b: 6 * 50 * 0.5 = 150
        # total: 750
        assert result["total"] == 750.0
        assert result["breakdown"]["effect_a"] == 600.0
        assert result["breakdown"]["effect_b"] == 150.0


# =============================================================================
# Tests for calculate_all_projects
# =============================================================================

class TestCalculateAllProjects:
    """Tests for the calculate_all_projects function."""

    def test_returns_dict_with_all_projects(self, simple_data, uniform_discount_factors):
        """Should return values for all projects in the data."""
        moral_weights = {"human_life_years": 1.0}
        result = calculate_all_projects(
            simple_data, moral_weights, uniform_discount_factors, 0
        )
        assert "project_a" in result
        assert "project_xrisk" in result
        assert len(result) == 2

    def test_calculates_correct_values(self, simple_data, uniform_discount_factors):
        """Each project value should match calculate_project output."""
        moral_weights = {"human_life_years": 1.0}
        result = calculate_all_projects(
            simple_data, moral_weights, uniform_discount_factors, 0
        )
        # project_a: 6 * 100 = 600
        # project_xrisk: 6 * 1000 = 6000
        assert result["project_a"] == 600.0
        assert result["project_xrisk"] == 6000.0


# =============================================================================
# Tests for adjust_for_extinction_risk
# =============================================================================

class TestAdjustForExtinctionRisk:
    """Tests for the adjust_for_extinction_risk function."""

    def test_xrisk_project_unchanged(self, simple_data):
        """xrisk projects should not be scaled by extinction probability."""
        project_values = {"project_a": 100.0, "project_xrisk": 1000.0}
        p_extinction = 0.5
        adjusted = adjust_for_extinction_risk(project_values, simple_data, p_extinction)
        # xrisk project unchanged
        assert adjusted["project_xrisk"] == 1000.0

    def test_non_xrisk_project_scaled(self, simple_data):
        """Non-xrisk projects should be scaled by (1 - p_extinction)."""
        project_values = {"project_a": 100.0, "project_xrisk": 1000.0}
        p_extinction = 0.5
        adjusted = adjust_for_extinction_risk(project_values, simple_data, p_extinction)
        # non-xrisk scaled: 100 * (1 - 0.5) = 50
        assert adjusted["project_a"] == 50.0

    def test_zero_extinction_probability(self, simple_data):
        """Zero extinction probability should leave all values unchanged."""
        project_values = {"project_a": 100.0, "project_xrisk": 1000.0}
        adjusted = adjust_for_extinction_risk(project_values, simple_data, 0.0)
        assert adjusted["project_a"] == 100.0
        assert adjusted["project_xrisk"] == 1000.0

    def test_high_extinction_probability(self, simple_data):
        """High extinction probability dramatically reduces non-xrisk values."""
        project_values = {"project_a": 100.0, "project_xrisk": 1000.0}
        p_extinction = 0.9
        adjusted = adjust_for_extinction_risk(project_values, simple_data, p_extinction)
        # non-xrisk: 100 * 0.1 = 10 (allow floating point tolerance)
        assert abs(adjusted["project_a"] - 10.0) < 0.001
        # xrisk unchanged
        assert adjusted["project_xrisk"] == 1000.0


# =============================================================================
# Tests for get_diminishing_returns_factor
# =============================================================================

class TestGetDiminishingReturnsFactor:
    """Tests for the get_diminishing_returns_factor function."""

    def test_zero_funding_returns_first_factor(self, simple_data):
        """Zero funding should return the first DR factor (index 0)."""
        factor = get_diminishing_returns_factor(simple_data, "project_a", 0)
        assert factor == 1.0

    def test_funding_maps_to_correct_index(self, simple_data):
        """Funding in $10M increments should map to array indices."""
        # project_a has DR [1.0, 0.5, 0.25]
        # $10M -> index 1 -> 0.5
        factor = get_diminishing_returns_factor(simple_data, "project_a", 10)
        assert factor == 0.5
        # $20M -> index 2 -> 0.25
        factor = get_diminishing_returns_factor(simple_data, "project_a", 20)
        assert factor == 0.25

    def test_beyond_array_returns_last_factor(self, simple_data):
        """Funding beyond array length should return last factor."""
        # project_a has only 3 entries, $1000M should still return last (0.25)
        factor = get_diminishing_returns_factor(simple_data, "project_a", 1000)
        assert factor == 0.25

    def test_works_with_default_data(self):
        """Should work correctly with the default project data."""
        # malaria_nets has 90 DR entries
        factor_0 = get_diminishing_returns_factor(DEFAULT_PROJECT_DATA, "malaria_nets", 0)
        assert factor_0 == 1.0
        # Check a middle value
        factor_100 = get_diminishing_returns_factor(DEFAULT_PROJECT_DATA, "malaria_nets", 100)
        assert factor_100 < 1.0  # Should be diminished


# =============================================================================
# Tests for vote_credence_weighted_custom
# =============================================================================

class TestVoteCredenceWeightedCustom:
    """Tests for the vote_credence_weighted_custom function."""

    def test_single_worldview_allocates_to_best(self, simple_data, simple_worldview):
        """Single worldview should allocate entire increment to highest-value project."""
        funding = {p: 0 for p in simple_data}
        increment = 10
        allocations = vote_credence_weighted_custom(
            simple_data, funding, increment, simple_worldview
        )
        # Should allocate to project_xrisk (higher value: 6000 vs 600)
        assert sum(allocations.values()) == increment
        assert allocations["project_xrisk"] == increment

    def test_split_credence_splits_allocation(self, simple_data):
        """Multiple worldviews should split the increment by credence."""
        funding = {p: 0 for p in simple_data}
        increment = 10
        # Two worldviews with 50/50 split, but same preferences
        worldviews = [
            {
                "name": "WV1",
                "credence": 0.5,
                "moral_weights": {"human_life_years": 1.0},
                "discount_factors": [1.0] * 6,
                "risk_profile": 0,
                "p_extinction": 0.0
            },
            {
                "name": "WV2",
                "credence": 0.5,
                "moral_weights": {"human_life_years": 1.0},
                "discount_factors": [1.0] * 6,
                "risk_profile": 0,
                "p_extinction": 0.0
            }
        ]
        allocations = vote_credence_weighted_custom(
            simple_data, funding, increment, worldviews
        )
        # Both prefer project_xrisk, so full allocation goes there
        assert allocations["project_xrisk"] == increment

    def test_conflicting_worldviews_split(self, simple_data):
        """Worldviews with different preferences should split allocation."""
        funding = {p: 0 for p in simple_data}
        increment = 10
        # WV1 prefers xrisk (high human_life_years weight)
        # WV2 zero weight on human_life_years -> zero value for all projects
        # Both should still pick something (highest marginal)
        worldviews = [
            {
                "name": "Prefers xrisk",
                "credence": 0.6,
                "moral_weights": {"human_life_years": 1.0},
                "discount_factors": [1.0] * 6,
                "risk_profile": 0,
                "p_extinction": 0.0
            },
            {
                "name": "Also prefers xrisk",
                "credence": 0.4,
                "moral_weights": {"human_life_years": 0.5},
                "discount_factors": [1.0] * 6,
                "risk_profile": 0,
                "p_extinction": 0.0
            }
        ]
        allocations = vote_credence_weighted_custom(
            simple_data, funding, increment, worldviews
        )
        # Both worldviews prefer xrisk (6000 * weight vs 600 * weight)
        assert allocations["project_xrisk"] == increment

    def test_extinction_risk_affects_choice(self, simple_data):
        """High extinction probability can shift preference to xrisk projects."""
        funding = {p: 0 for p in simple_data}
        increment = 10
        # With p_extinction=0.99, non-xrisk project_a value drops to 1%
        # (600 -> 6), making project_xrisk (6000) clearly dominant
        worldviews = [{
            "name": "High extinction concern",
            "credence": 1.0,
            "moral_weights": {"human_life_years": 1.0},
            "discount_factors": [1.0] * 6,
            "risk_profile": 0,
            "p_extinction": 0.99
        }]
        allocations = vote_credence_weighted_custom(
            simple_data, funding, increment, worldviews
        )
        assert allocations["project_xrisk"] == increment


# =============================================================================
# Tests for allocate_budget
# =============================================================================

class TestAllocateBudget:
    """Tests for the allocate_budget function."""

    def test_allocates_full_budget(self, simple_data, simple_worldview):
        """Total funding should equal total budget."""
        total_budget = 100
        result = allocate_budget(
            simple_data,
            vote_credence_weighted_custom,
            total_budget,
            custom_worldviews=simple_worldview
        )
        total_allocated = sum(result["funding"].values())
        assert total_allocated == total_budget

    def test_respects_increment_size(self, simple_data, simple_worldview):
        """Number of iterations should match budget / increment_size."""
        total_budget = 100
        increment_size = 20
        result = allocate_budget(
            simple_data,
            vote_credence_weighted_custom,
            total_budget,
            increment_size=increment_size,
            custom_worldviews=simple_worldview
        )
        # 100 / 20 = 5 iterations
        assert len(result["history"]) == 5

    def test_history_tracks_iterations(self, simple_data, simple_worldview):
        """History should record each iteration's allocations."""
        total_budget = 30
        result = allocate_budget(
            simple_data,
            vote_credence_weighted_custom,
            total_budget,
            custom_worldviews=simple_worldview
        )
        # Default INCREMENT_SIZE is 10, so 3 iterations
        assert len(result["history"]) == 3
        for i, entry in enumerate(result["history"]):
            assert entry["iteration"] == i
            assert "allocations" in entry

    def test_partial_final_increment(self, simple_data, simple_worldview):
        """Budget not divisible by increment should handle remainder."""
        total_budget = 25  # Not divisible by 10
        result = allocate_budget(
            simple_data,
            vote_credence_weighted_custom,
            total_budget,
            custom_worldviews=simple_worldview
        )
        total_allocated = sum(result["funding"].values())
        assert total_allocated == 25
        # 3 iterations: 10, 10, 5
        assert len(result["history"]) == 3


# =============================================================================
# Tests with DEFAULT_PROJECT_DATA
# =============================================================================

class TestWithDefaultData:
    """Integration tests using the actual DEFAULT_PROJECT_DATA."""

    def test_calculate_all_projects_runs(self):
        """Smoke test that calculation works with default data."""
        moral_weights = {
            "human_life_years": 1.0,
            "human_ylds": 0.5,
            "human_income_doublings": 0.1,
            "chickens_birds": 0.01,
            "fish": 0.005,
            "shrimp": 0.0001,
            "non_shrimp_invertebrates": 0.00005,
            "mammals": 0.05
        }
        discount_factors = [1.0, 0.9, 0.5, 0.2, 0.05, 0.01]
        result = calculate_all_projects(
            DEFAULT_PROJECT_DATA, moral_weights, discount_factors, 0
        )
        # All 6 projects should have values
        assert len(result) == 6
        assert all(v >= 0 for v in result.values())

    def test_vote_credence_weighted_custom_with_example_worldviews(self):
        """Test voting with the provided example worldviews."""
        funding = {p: 0 for p in DEFAULT_PROJECT_DATA}
        allocations = vote_credence_weighted_custom(
            DEFAULT_PROJECT_DATA, funding, 10, EXAMPLE_CUSTOM_WORLDVIEWS
        )
        # Should allocate exactly the increment
        assert sum(allocations.values()) == 10
        # At least one project should receive funding
        assert max(allocations.values()) > 0

    def test_full_allocation_with_example_worldviews(self):
        """Integration test: full budget allocation with example worldviews."""
        result = allocate_budget(
            DEFAULT_PROJECT_DATA,
            vote_credence_weighted_custom,
            total_budget=100,
            custom_worldviews=EXAMPLE_CUSTOM_WORLDVIEWS
        )
        assert sum(result["funding"].values()) == 100
        # Should have 10 iterations (100 / 10)
        assert len(result["history"]) == 10

    def test_show_allocation_does_not_crash(self, capsys):
        """Smoke test that show_allocation runs without error."""
        result = allocate_budget(
            DEFAULT_PROJECT_DATA,
            vote_credence_weighted_custom,
            total_budget=50,
            custom_worldviews=EXAMPLE_CUSTOM_WORLDVIEWS
        )
        show_allocation(result, DEFAULT_PROJECT_DATA)
        captured = capsys.readouterr()
        assert "BUDGET ALLOCATION" in captured.out
        assert "50.0M" in captured.out


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_budget_allocation(self, simple_data, simple_worldview):
        """Zero budget should result in no allocations."""
        result = allocate_budget(
            simple_data,
            vote_credence_weighted_custom,
            total_budget=0,
            custom_worldviews=simple_worldview
        )
        assert all(v == 0 for v in result["funding"].values())
        assert len(result["history"]) == 0

    def test_very_small_increment(self, simple_data, simple_worldview):
        """Very small increments should still work correctly."""
        result = allocate_budget(
            simple_data,
            vote_credence_weighted_custom,
            total_budget=1,
            increment_size=0.1,
            custom_worldviews=simple_worldview
        )
        total = sum(result["funding"].values())
        assert abs(total - 1.0) < 0.001  # Allow for floating point

    def test_all_worldviews_zero_credence(self, simple_data):
        """Worldviews with zero credence should allocate nothing."""
        worldviews = [{
            "name": "Zero credence",
            "credence": 0.0,
            "moral_weights": {"human_life_years": 1.0},
            "discount_factors": [1.0] * 6,
            "risk_profile": 0,
            "p_extinction": 0.0
        }]
        funding = {p: 0 for p in simple_data}
        allocations = vote_credence_weighted_custom(
            simple_data, funding, 10, worldviews
        )
        # Zero credence means zero share allocated
        assert all(v == 0 for v in allocations.values())

    def test_negative_values_in_effects(self):
        """Projects can have negative values (e.g., downside scenarios)."""
        # ai_safety_policy has negative values in risk profiles 2 and 3 for early periods
        moral_weights = {"human_life_years": 1.0}
        discount_factors = [1.0] * 6
        # Risk profile 2 (downside) - check values
        result = calculate_project(
            DEFAULT_PROJECT_DATA["ai_safety_policy"],
            moral_weights,
            discount_factors,
            2  # downside risk profile
        )
        # The t2 value for risk profile 2 is -50, but overall sum should be positive
        # due to large positive values in later periods
        # Just verify it computes without error
        assert isinstance(result["total"], (int, float))


def _make_worldview(name, credence, human=0.0, chickens=0.0, fish=0.0):
    """Build compact worldview inputs for voting-method tests."""
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


@pytest.fixture
def tri_project_data():
    """Three-project dataset with one recipient type per project."""
    return {
        "project_human": {
            "tags": {"near_term_xrisk": False},
            "diminishing_returns": [1.0],
            "effects": {
                "effect_human": {
                    "recipient_type": "human_life_years",
                    "values": [[100] * 4 for _ in range(6)],
                }
            },
        },
        "project_chicken": {
            "tags": {"near_term_xrisk": False},
            "diminishing_returns": [1.0],
            "effects": {
                "effect_chicken": {
                    "recipient_type": "chickens_birds",
                    "values": [[100] * 4 for _ in range(6)],
                }
            },
        },
        "project_fish": {
            "tags": {"near_term_xrisk": False},
            "diminishing_returns": [1.0],
            "effects": {
                "effect_fish": {
                    "recipient_type": "fish",
                    "values": [[100] * 4 for _ in range(6)],
                }
            },
        },
    }


@pytest.fixture
def maximin_data():
    """Dataset where a balanced project should win lexicographic maximin."""
    return {
        "project_human": {
            "tags": {"near_term_xrisk": False},
            "diminishing_returns": [1.0],
            "effects": {
                "effect_human": {
                    "recipient_type": "human_life_years",
                    "values": [[100] * 4 for _ in range(6)],
                }
            },
        },
        "project_balanced": {
            "tags": {"near_term_xrisk": False},
            "diminishing_returns": [1.0],
            "effects": {
                "effect_human_balanced": {
                    "recipient_type": "human_life_years",
                    "values": [[60] * 4 for _ in range(6)],
                },
                "effect_chicken_balanced": {
                    "recipient_type": "chickens_birds",
                    "values": [[60] * 4 for _ in range(6)],
                },
            },
        },
        "project_chicken": {
            "tags": {"near_term_xrisk": False},
            "diminishing_returns": [1.0],
            "effects": {
                "effect_chicken": {
                    "recipient_type": "chickens_birds",
                    "values": [[100] * 4 for _ in range(6)],
                }
            },
        },
    }


@pytest.fixture
def condorcet_worldviews():
    """Worldviews giving project_human a Condorcet win."""
    return [
        _make_worldview("WV1", 0.5, human=3.0, chickens=2.0, fish=1.0),  # H > C > F
        _make_worldview("WV2", 0.3, human=3.0, chickens=1.0, fish=2.0),  # H > F > C
        _make_worldview("WV3", 0.2, human=1.0, chickens=3.0, fish=2.0),  # C > F > H
    ]


class TestVoteMecRegression:
    """Regression checks for MEC behavior."""

    def test_vote_mec_allocates_full_increment(self, simple_data):
        funding = {p: 0 for p in simple_data}

        def build_moral_weights_fn(avg_q1, avg_q2, avg_q3, avg_q4):
            return {
                "human_life_years": avg_q1,
                "human_ylds": 0.0,
                "human_income_doublings": avg_q2,
                "chickens_birds": avg_q3,
                "fish": avg_q3,
                "shrimp": avg_q4,
                "non_shrimp_invertebrates": avg_q4,
                "mammals": avg_q3,
            }

        allocations = vote_mec(
            simple_data,
            funding,
            increment=10,
            q1_cred=[1.0],
            q2_cred=[1.0],
            q3_cred=[1.0],
            q4_cred=[1.0],
            q5_cred=[0.25, 0.25, 0.25, 0.25],
            q6_cred=[1.0, 0.0, 0.0, 0.0],
            q7_cred=[1.0],
            q1_daly_weights=[1.0],
            q2_income_weights=[1.0],
            q3_chicken_multipliers=[1.0],
            q4_shrimp_multipliers=[1.0],
            q5_discount_factors=[[1.0] * 6 for _ in range(4)],
            q7_extinction_probs=[0.0],
            build_moral_weights_fn=build_moral_weights_fn,
        )

        assert sum(allocations.values()) == 10


class TestVoteMet:
    """Tests for MET voting behavior."""

    def test_threshold_switches_to_favorite_theory(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        worldviews = [
            _make_worldview("Human-focused", 0.7, human=1.0),
            _make_worldview("Chicken-focused", 0.3, chickens=1.0),
        ]

        allocations, debug = vote_met(
            tri_project_data,
            funding,
            10,
            worldviews,
            met_threshold=0.6,
            return_debug=True,
        )

        assert debug["strategy"] == "favorite_theory"
        assert allocations["project_human"] == 10

    def test_below_threshold_uses_similarity_centroid(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        worldviews = [
            _make_worldview("Human-focused", 0.49, human=1.0),
            _make_worldview("Chicken-focused", 0.31, chickens=1.0),
            _make_worldview("Fish-focused", 0.20, fish=1.0),
        ]

        allocations, debug = vote_met(
            tri_project_data,
            funding,
            10,
            worldviews,
            met_threshold=0.5,
            return_debug=True,
        )

        assert debug["strategy"] == "similarity_centroid"
        assert sum(allocations.values()) == 10


class TestVoteNashBargaining:
    """Tests for Nash bargaining disagreement-point handling."""

    def test_disagreement_point_switch_changes_baseline(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        worldviews = [
            _make_worldview("Human-focused", 0.6, human=1.0),
            _make_worldview("Chicken-focused", 0.4, chickens=1.0),
        ]

        allocations_zero, debug_zero = vote_nash_bargaining(
            tri_project_data,
            funding,
            10,
            worldviews,
            disagreement_point="zero_spending",
            return_debug=True,
        )
        allocations_random, debug_random = vote_nash_bargaining(
            tri_project_data,
            funding,
            10,
            worldviews,
            disagreement_point="random_dictator",
            random_seed=NASH_TEST_RANDOM_DICTATOR_SEED,
            return_debug=True,
        )

        assert sum(allocations_zero.values()) == 10
        assert sum(allocations_random.values()) == 10
        assert debug_zero["disagreement_utilities"] != debug_random["disagreement_utilities"]

    def test_random_tie_breaking_is_reproducible(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        worldviews = [
            _make_worldview("Human-focused", 0.5, human=1.0),
            _make_worldview("Chicken-focused", 0.5, chickens=1.0),
        ]

        allocations_1, _ = vote_nash_bargaining(
            tri_project_data,
            funding,
            10,
            worldviews,
            disagreement_point="zero_spending",
            tie_break="random",
            random_seed=11,
            return_debug=True,
        )
        allocations_2, _ = vote_nash_bargaining(
            tri_project_data,
            funding,
            10,
            worldviews,
            disagreement_point="zero_spending",
            tie_break="random",
            random_seed=11,
            return_debug=True,
        )

        assert allocations_1 == allocations_2

    def test_random_tie_breaking_without_seed_is_reproducible(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        worldviews = [
            _make_worldview("Human-focused", 0.5, human=1.0),
            _make_worldview("Chicken-focused", 0.5, chickens=1.0),
        ]

        allocations_1, _ = vote_nash_bargaining(
            tri_project_data,
            funding,
            10,
            worldviews,
            disagreement_point="zero_spending",
            tie_break="random",
            return_debug=True,
        )
        allocations_2, _ = vote_nash_bargaining(
            tri_project_data,
            funding,
            10,
            worldviews,
            disagreement_point="zero_spending",
            tie_break="random",
            return_debug=True,
        )

        assert allocations_1 == allocations_2


class TestVoteMsa:
    """Tests for MSA voting, including stop behavior."""

    def test_winner_take_all_selects_cardinal_mec_winner(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        worldviews = [
            _make_worldview("Total Utilitarianism", 0.6, human=1.0),
            _make_worldview("Kantianism", 0.4, chickens=1.0),
        ]

        allocations, _ = vote_msa(
            tri_project_data,
            funding,
            10,
            worldviews,
            cardinal_permissibility_mode="winner_take_all",
            return_debug=True,
        )

        assert allocations["project_human"] == 10

    def test_top_k_mode_respects_tie_breaking(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        worldviews = [
            _make_worldview("Total Utilitarianism", 0.6, human=1.0, chickens=1.0, fish=0.5),
            _make_worldview("Kantianism", 0.4, human=1.0, chickens=1.0),
        ]

        allocations = vote_msa(
            tri_project_data,
            funding,
            10,
            worldviews,
            cardinal_permissibility_mode="top_k",
            cardinal_top_k=2,
            tie_break="deterministic",
        )

        assert allocations["project_chicken"] == 10

    def test_no_permissible_default_stop_signal(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        worldviews = [_make_worldview("Kantianism", 1.0)]  # all project scores are zero

        stop_signal, _ = vote_msa(
            tri_project_data,
            funding,
            10,
            worldviews,
            binary_permissibility_threshold=1.0,
            no_permissible_action="stop",
            return_debug=True,
        )

        assert stop_signal.get("__stop__") is True

    def test_allocate_budget_stops_when_msa_signals_stop(self, tri_project_data):
        worldviews = [_make_worldview("Kantianism", 1.0)]  # all project scores are zero

        allocation = allocate_budget(
            tri_project_data,
            vote_msa,
            total_budget=30,
            custom_worldviews=worldviews,
            binary_permissibility_threshold=1.0,
            no_permissible_action="stop",
        )

        assert sum(allocation["funding"].values()) == 0
        assert allocation["history"][-1]["stopped"] is True


class TestAdditionalVotingMethods:
    """Tests for Borda, Split-Cycle, and Lexicographic Maximin."""

    def test_borda_selects_expected_winner(self, tri_project_data, condorcet_worldviews):
        funding = {p: 0 for p in tri_project_data}
        allocations = vote_borda(tri_project_data, funding, 10, condorcet_worldviews)
        assert allocations["project_human"] == 10

    def test_split_cycle_selects_condorcet_winner(self, tri_project_data, condorcet_worldviews):
        funding = {p: 0 for p in tri_project_data}
        allocations = vote_split_cycle(tri_project_data, funding, 10, condorcet_worldviews)
        assert allocations["project_human"] == 10

    def test_lexicographic_maximin_prefers_balanced_project(self, maximin_data):
        funding = {p: 0 for p in maximin_data}
        worldviews = [
            _make_worldview("Human-focused", 0.5, human=1.0, chickens=0.01),
            _make_worldview("Chicken-focused", 0.5, human=0.01, chickens=1.0),
        ]

        allocations = vote_lexicographic_maximin(maximin_data, funding, 10, worldviews)
        assert allocations["project_balanced"] == 10


@pytest.fixture
def two_project_data():
    """Two-project dataset useful for deterministic tie/fallback assertions."""
    return {
        "project_alpha": {
            "tags": {"near_term_xrisk": False},
            "diminishing_returns": [1.0],
            "effects": {
                "effect_human": {
                    "recipient_type": "human_life_years",
                    "values": [[100] * 4 for _ in range(6)],
                }
            },
        },
        "project_beta": {
            "tags": {"near_term_xrisk": False},
            "diminishing_returns": [1.0],
            "effects": {
                "effect_chicken": {
                    "recipient_type": "chickens_birds",
                    "values": [[100] * 4 for _ in range(6)],
                }
            },
        },
    }


class TestVoteMetExtended:
    """Additional MET edge-case tests."""

    def test_exact_threshold_uses_favorite_theory(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        worldviews = [
            _make_worldview("Human-focused", 0.5, human=1.0),
            _make_worldview("Chicken-focused", 0.5, chickens=1.0),
        ]
        allocations, debug = vote_met(
            tri_project_data,
            funding,
            10,
            worldviews,
            met_threshold=0.5,
            return_debug=True,
        )
        assert debug["strategy"] == "favorite_theory"
        assert allocations["project_human"] == 10

    def test_no_worldviews_returns_zero_allocations_with_debug(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        allocations, debug = vote_met(
            tri_project_data,
            funding,
            10,
            [],
            return_debug=True,
        )
        assert all(v == 0 for v in allocations.values())
        assert debug["strategy"] == "no_worldviews"

    def test_random_tie_breaking_is_reproducible(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        # Favorite-theory branch with a project tie inside the selected worldview
        worldviews = [
            _make_worldview("TieWorldview", 0.8, human=1.0, chickens=1.0),
            _make_worldview("FishWorldview", 0.2, fish=1.0),
        ]
        allocations_1 = vote_met(
            tri_project_data,
            funding,
            10,
            worldviews,
            met_threshold=0.5,
            tie_break="random",
            random_seed=17,
        )
        allocations_2 = vote_met(
            tri_project_data,
            funding,
            10,
            worldviews,
            met_threshold=0.5,
            tie_break="random",
            random_seed=17,
        )
        assert allocations_1 == allocations_2


class TestVoteNashBargainingExtended:
    """Additional Nash bargaining branch coverage."""

    def test_supports_anti_utopia_and_exclusionary_split(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        worldviews = [
            _make_worldview("Human-focused", 0.55, human=1.0),
            _make_worldview("Chicken-focused", 0.45, chickens=1.0),
        ]

        alloc_anti, dbg_anti = vote_nash_bargaining(
            tri_project_data, funding, 10, worldviews, disagreement_point="anti_utopia", return_debug=True
        )
        alloc_excl, dbg_excl = vote_nash_bargaining(
            tri_project_data,
            funding,
            10,
            worldviews,
            disagreement_point="exclusionary_proportional_split",
            return_debug=True,
        )

        assert sum(alloc_anti.values()) == 10
        assert sum(alloc_excl.values()) == 10
        assert dbg_anti["disagreement_point"] == "anti_utopia"
        assert dbg_excl["disagreement_point"] == "exclusionary_proportional_split"

    def test_supports_moral_marketplace(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        worldviews = [
            _make_worldview("Human-focused", 0.55, human=1.0),
            _make_worldview("Chicken-focused", 0.45, chickens=1.0),
        ]

        alloc_market, dbg_market = vote_nash_bargaining(
            tri_project_data,
            funding,
            10,
            worldviews,
            disagreement_point="moral_marketplace",
            return_debug=True,
        )
        alloc_random, dbg_random = vote_nash_bargaining(
            tri_project_data,
            funding,
            10,
            worldviews,
            disagreement_point="random_dictator",
            random_seed=NASH_TEST_RANDOM_DICTATOR_SEED,
            return_debug=True,
        )

        assert sum(alloc_market.values()) == 10
        assert sum(alloc_random.values()) == 10
        assert dbg_market["disagreement_point"] == "moral_marketplace"
        assert dbg_market["disagreement_utilities"] != pytest.approx(
            dbg_random["disagreement_utilities"]
        )

    def test_random_dictator_is_reproducible_with_seed(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        worldviews = [
            _make_worldview("Human-focused", 0.55, human=1.0),
            _make_worldview("Chicken-focused", 0.45, chickens=1.0),
        ]

        alloc_1, dbg_1 = vote_nash_bargaining(
            tri_project_data,
            funding,
            10,
            worldviews,
            disagreement_point="random_dictator",
            random_seed=NASH_TEST_RANDOM_DICTATOR_SEED,
            return_debug=True,
        )
        alloc_2, dbg_2 = vote_nash_bargaining(
            tri_project_data,
            funding,
            10,
            worldviews,
            disagreement_point="random_dictator",
            random_seed=NASH_TEST_RANDOM_DICTATOR_SEED,
            return_debug=True,
        )

        assert alloc_1 == alloc_2
        assert dbg_1["disagreement_utilities"] == pytest.approx(dbg_2["disagreement_utilities"])

    def test_random_dictator_without_seed_is_reproducible(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        worldviews = [
            _make_worldview("Human-focused", 0.55, human=1.0),
            _make_worldview("Chicken-focused", 0.45, chickens=1.0),
        ]

        alloc_1, dbg_1 = vote_nash_bargaining(
            tri_project_data,
            funding,
            10,
            worldviews,
            disagreement_point="random_dictator",
            return_debug=True,
        )
        alloc_2, dbg_2 = vote_nash_bargaining(
            tri_project_data,
            funding,
            10,
            worldviews,
            disagreement_point="random_dictator",
            return_debug=True,
        )

        assert alloc_1 == alloc_2
        assert dbg_1["disagreement_utilities"] == pytest.approx(dbg_2["disagreement_utilities"])

    def test_invalid_disagreement_point_raises(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        worldviews = [_make_worldview("Human-focused", 1.0, human=1.0)]
        with pytest.raises(ValueError, match="Unknown disagreement_point"):
            vote_nash_bargaining(
                tri_project_data,
                funding,
                10,
                worldviews,
                disagreement_point="not_a_mode",
            )

    def test_fallback_objective_path_is_used_when_no_feasible_project(self, two_project_data):
        funding = {p: 0 for p in two_project_data}
        # Opposing preferences + moral marketplace baseline -> no project has non-negative gains for everyone
        worldviews = [
            _make_worldview("AlphaOnly", 0.5, human=1.0),
            _make_worldview("BetaOnly", 0.5, chickens=1.0),
        ]
        allocations, debug = vote_nash_bargaining(
            two_project_data,
            funding,
            10,
            worldviews,
            disagreement_point="moral_marketplace",
            return_debug=True,
        )
        assert debug["objective"] == "sum_gains_fallback"
        assert sum(allocations.values()) == 10

    def test_no_worldviews_returns_zero_allocations_with_debug(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        allocations, debug = vote_nash_bargaining(
            tri_project_data,
            funding,
            10,
            [],
            return_debug=True,
        )
        assert all(v == 0 for v in allocations.values())
        assert debug["strategy"] == "no_worldviews"


class TestVoteMsaExtended:
    """Additional MSA branch and validation coverage."""

    def test_within_percent_mode_marks_multiple_cardinal_permissible(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        worldviews = [_make_worldview("Cardinal", 1.0, human=1.0, chickens=0.95, fish=0.1)]
        allocations, debug = vote_msa(
            tri_project_data,
            funding,
            10,
            worldviews,
            worldview_types={"Cardinal": "cardinal"},
            cardinal_permissibility_mode="within_percent",
            cardinal_within_percent=0.10,
            return_debug=True,
        )
        assert sum(allocations.values()) == 10
        assert "project_human" in debug["cardinal_permissible"]
        assert "project_chicken" in debug["cardinal_permissible"]
        assert debug["threshold_score"] is not None

    def test_fallback_mec_with_cardinal_cluster(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        worldviews = [
            _make_worldview("CardinalA", 0.4, human=1.0),
            _make_worldview("BinaryB", 0.6, chickens=1.0),
        ]
        allocations, debug = vote_msa(
            tri_project_data,
            funding,
            10,
            worldviews,
            worldview_types={"CardinalA": "cardinal", "BinaryB": "binary"},
            binary_permissibility_threshold=1_000_000,
            no_permissible_action="fallback_mec",
            return_debug=True,
        )
        assert debug["fallback_used"] is True
        assert allocations["project_human"] == 10

    def test_fallback_mec_without_cardinal_uses_weighted_scores(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        worldviews = [
            _make_worldview("BinaryA", 0.7, human=1.0),
            _make_worldview("BinaryB", 0.3, chickens=1.0),
        ]
        allocations, debug = vote_msa(
            tri_project_data,
            funding,
            10,
            worldviews,
            worldview_types={"BinaryA": "binary", "BinaryB": "binary"},
            binary_permissibility_threshold=1_000_000,
            no_permissible_action="fallback_mec",
            return_debug=True,
        )
        assert debug["fallback_used"] is True
        assert allocations["project_human"] == 10

    def test_worldview_type_override_changes_behavior(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        worldviews = [
            _make_worldview("WV-A", 0.4, human=1.0),
            _make_worldview("WV-B", 0.6, chickens=1.0),
        ]
        allocations, debug = vote_msa(
            tri_project_data,
            funding,
            10,
            worldviews,
            worldview_types={"WV-A": "cardinal", "WV-B": "binary"},
            binary_permissibility_threshold=1_000_000,
            no_permissible_action="fallback_mec",
            return_debug=True,
        )
        assert debug["fallback_used"] is True
        assert allocations["project_human"] == 10

    def test_invalid_cardinal_mode_raises(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        worldviews = [_make_worldview("Cardinal", 1.0, human=1.0)]
        with pytest.raises(ValueError, match="Unknown cardinal_permissibility_mode"):
            vote_msa(
                tri_project_data,
                funding,
                10,
                worldviews,
                worldview_types={"Cardinal": "cardinal"},
                cardinal_permissibility_mode="bad_mode",
            )

    def test_invalid_no_permissible_action_raises(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        worldviews = [_make_worldview("BinaryOnly", 1.0)]
        with pytest.raises(ValueError, match="Unknown no_permissible_action"):
            vote_msa(
                tri_project_data,
                funding,
                10,
                worldviews,
                worldview_types={"BinaryOnly": "binary"},
                binary_permissibility_threshold=1_000_000,
                no_permissible_action="bad_mode",
            )

    def test_no_worldviews_returns_zero_allocations_with_debug(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        allocations, debug = vote_msa(
            tri_project_data,
            funding,
            10,
            [],
            return_debug=True,
        )
        assert all(v == 0 for v in allocations.values())
        assert debug["strategy"] == "no_worldviews"


class TestAdditionalVotingMethodsExtended:
    """Extended tie and no-worldview coverage for social-choice methods."""

    def test_borda_deterministic_tie_break(self, two_project_data):
        funding = {p: 0 for p in two_project_data}
        worldviews = [
            _make_worldview("Alpha", 0.5, human=1.0),
            _make_worldview("Beta", 0.5, chickens=1.0),
        ]
        allocations = vote_borda(two_project_data, funding, 10, worldviews, tie_break="deterministic")
        assert allocations["project_alpha"] == 10

    def test_borda_random_tie_break_is_reproducible(self, two_project_data):
        funding = {p: 0 for p in two_project_data}
        worldviews = [
            _make_worldview("Alpha", 0.5, human=1.0),
            _make_worldview("Beta", 0.5, chickens=1.0),
        ]
        alloc_1 = vote_borda(
            two_project_data, funding, 10, worldviews, tie_break="random", random_seed=19
        )
        alloc_2 = vote_borda(
            two_project_data, funding, 10, worldviews, tie_break="random", random_seed=19
        )
        assert alloc_1 == alloc_2

    def test_split_cycle_deterministic_tie_break(self, two_project_data):
        funding = {p: 0 for p in two_project_data}
        worldviews = [
            _make_worldview("Alpha", 0.5, human=1.0),
            _make_worldview("Beta", 0.5, chickens=1.0),
        ]
        allocations = vote_split_cycle(
            two_project_data, funding, 10, worldviews, tie_break="deterministic"
        )
        assert allocations["project_alpha"] == 10

    def test_split_cycle_random_tie_break_is_reproducible(self, two_project_data):
        funding = {p: 0 for p in two_project_data}
        worldviews = [
            _make_worldview("Alpha", 0.5, human=1.0),
            _make_worldview("Beta", 0.5, chickens=1.0),
        ]
        alloc_1 = vote_split_cycle(
            two_project_data, funding, 10, worldviews, tie_break="random", random_seed=29
        )
        alloc_2 = vote_split_cycle(
            two_project_data, funding, 10, worldviews, tie_break="random", random_seed=29
        )
        assert alloc_1 == alloc_2

    def test_lexicographic_maximin_deterministic_tie_break(self, two_project_data):
        funding = {p: 0 for p in two_project_data}
        worldviews = [
            _make_worldview("Alpha", 0.5, human=1.0),
            _make_worldview("Beta", 0.5, chickens=1.0),
        ]
        allocations = vote_lexicographic_maximin(
            two_project_data, funding, 10, worldviews, tie_break="deterministic"
        )
        assert allocations["project_alpha"] == 10

    def test_lexicographic_maximin_random_tie_break_is_reproducible(self, two_project_data):
        funding = {p: 0 for p in two_project_data}
        worldviews = [
            _make_worldview("Alpha", 0.5, human=1.0),
            _make_worldview("Beta", 0.5, chickens=1.0),
        ]
        alloc_1 = vote_lexicographic_maximin(
            two_project_data, funding, 10, worldviews, tie_break="random", random_seed=31
        )
        alloc_2 = vote_lexicographic_maximin(
            two_project_data, funding, 10, worldviews, tie_break="random", random_seed=31
        )
        assert alloc_1 == alloc_2

    @pytest.mark.parametrize(
        "method",
        [vote_borda, vote_split_cycle, vote_lexicographic_maximin],
    )
    def test_no_worldviews_returns_zero_allocations_with_debug(self, tri_project_data, method):
        funding = {p: 0 for p in tri_project_data}
        allocations, debug = method(
            tri_project_data,
            funding,
            10,
            [],
            return_debug=True,
        )
        assert all(v == 0 for v in allocations.values())
        assert debug["strategy"] == "no_worldviews"


class TestAllocateBudgetDebugMetadata:
    """Verify tuple return values propagate metadata into allocation history."""

    def test_history_contains_vote_metadata_for_debug_methods(self, tri_project_data):
        worldviews = [
            _make_worldview("Human-focused", 0.7, human=1.0),
            _make_worldview("Chicken-focused", 0.3, chickens=1.0),
        ]
        allocation = allocate_budget(
            tri_project_data,
            vote_met,
            total_budget=10,
            custom_worldviews=worldviews,
            met_threshold=0.6,
            return_debug=True,
        )
        assert len(allocation["history"]) == 1
        assert "meta" in allocation["history"][0]
        assert allocation["history"][0]["meta"]["strategy"] == "favorite_theory"


class TestCredenceValidation:
    """Validation behavior for worldview credences."""

    def test_vote_credence_weighted_custom_requires_sum_to_one(self, simple_data):
        funding = {p: 0 for p in simple_data}
        worldviews = [
            _make_worldview("A", 0.7, human=1.0),
            _make_worldview("B", 0.2, chickens=1.0),
        ]
        with pytest.raises(ValueError, match="sum to 1.0"):
            vote_credence_weighted_custom(simple_data, funding, 10, worldviews)

    def test_vote_credence_weighted_custom_rejects_negative_credence(self, simple_data):
        funding = {p: 0 for p in simple_data}
        worldviews = [
            _make_worldview("A", 1.1, human=1.0),
            _make_worldview("B", -0.1, chickens=1.0),
        ]
        with pytest.raises(ValueError, match="non-negative"):
            vote_credence_weighted_custom(simple_data, funding, 10, worldviews)


class TestDiminishingReturnsBoundaryPrecision:
    """Ensure DR lookup is stable around floating boundaries."""

    def test_near_boundary_rounds_to_next_step(self, simple_data):
        # project_a has [1.0, 0.5, 0.25]
        factor_10ish = get_diminishing_returns_factor(simple_data, "project_a", 9.999999999)
        factor_20ish = get_diminishing_returns_factor(simple_data, "project_a", 19.999999999)
        assert factor_10ish == 0.5
        assert factor_20ish == 0.25

    def test_negative_funding_clamps_to_first_factor(self, simple_data):
        factor = get_diminishing_returns_factor(simple_data, "project_a", -0.5)
        assert factor == 1.0


class TestVoteMecUnifiedInterface:
    """Unified custom_worldviews interface behavior for MEC."""

    def test_custom_worldviews_interface_selects_expected_project(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        worldviews = [
            _make_worldview("Human-focused", 0.7, human=1.0),
            _make_worldview("Chicken-focused", 0.3, chickens=1.0),
        ]
        allocations, debug = vote_mec(
            tri_project_data,
            funding,
            10,
            custom_worldviews=worldviews,
            return_debug=True,
        )
        assert allocations["project_human"] == 10
        assert debug["strategy"] == "custom_worldviews"

    def test_allocate_budget_works_with_custom_worldviews(self, tri_project_data):
        worldviews = [
            _make_worldview("Human-focused", 0.6, human=1.0),
            _make_worldview("Chicken-focused", 0.4, chickens=1.0),
        ]
        allocation = allocate_budget(
            tri_project_data,
            vote_mec,
            total_budget=20,
            custom_worldviews=worldviews,
        )
        assert sum(allocation["funding"].values()) == 20

    def test_legacy_interface_missing_args_raises(self, simple_data):
        funding = {p: 0 for p in simple_data}
        with pytest.raises(ValueError, match="missing required parameters"):
            vote_mec(simple_data, funding, 10)


class TestVoteMyFavoriteTheoryUnifiedInterface:
    """Unified custom_worldviews interface behavior for favorite theory."""

    def test_custom_worldviews_interface(self, tri_project_data):
        funding = {p: 0 for p in tri_project_data}
        worldviews = [
            _make_worldview("Human-focused", 0.8, human=1.0),
            _make_worldview("Chicken-focused", 0.2, chickens=1.0),
        ]
        allocations, debug = vote_my_favorite_theory(
            tri_project_data,
            funding,
            10,
            custom_worldviews=worldviews,
            return_debug=True,
        )
        assert allocations["project_human"] == 10
        assert debug["strategy"] == "custom_worldviews"

    def test_legacy_interface_still_supported(self, simple_data):
        funding = {p: 0 for p in simple_data}
        legacy_worldviews = [
            {"name": "WV-A", "credence": 0.6, "result_idx": 0},
            {"name": "WV-B", "credence": 0.4, "result_idx": 1},
        ]
        legacy_results = [
            {"project_values": {"project_a": 10.0, "project_xrisk": 5.0}},
            {"project_values": {"project_a": 1.0, "project_xrisk": 20.0}},
        ]
        allocations, debug = vote_my_favorite_theory(
            simple_data,
            funding,
            10,
            results=legacy_results,
            worldviews=legacy_worldviews,
            return_debug=True,
        )
        assert allocations["project_a"] == 10
        assert debug["strategy"] == "legacy_precomputed"
