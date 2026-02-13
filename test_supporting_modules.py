# -*- coding: utf-8 -*-
"""Direct tests for supporting aggregation modules."""

import numpy as np

from met_sim_utils import (
    calculate_pairwise_similarities,
    embed_worldviews_in_2d_space,
    calculate_weighted_centroid,
    find_closest_worldview,
)
from multi_stage_aggregation import (
    MoralTheory,
    pure_mec_choose_intervention,
    mec_aggregate_cardinal_theories,
    convert_ordinal_to_binary,
    convert_mec_result_to_binary,
    credence_weighted_vote,
    multistage_aggregation,
    multistage_with_incomparability_handling,
)


class _DummyWorldview:
    def __init__(self, values):
        self.values = values

    def evaluate(self, project):
        return self.values[project]


class TestMetSimilarityUtils:
    """Direct unit tests for met_sim_utils helpers."""

    def test_calculate_pairwise_similarities_shapes_and_diagonal(self):
        projects = ["a", "b", "c"]
        worldviews = [
            _DummyWorldview({"a": 3.0, "b": 2.0, "c": 1.0}),
            _DummyWorldview({"a": 1.0, "b": 3.0, "c": 2.0}),
            _DummyWorldview({"a": 2.0, "b": 1.0, "c": 3.0}),
        ]
        pearson_matrix, rank_matrix = calculate_pairwise_similarities(worldviews, projects)

        assert pearson_matrix.shape == (3, 3)
        assert rank_matrix.shape == (3, 3)
        assert np.allclose(np.diag(pearson_matrix), 1.0)
        assert np.allclose(np.diag(rank_matrix), 1.0)

    def test_embed_worldviews_in_2d_space_returns_expected_shape(self):
        pearson_matrix = np.array(
            [
                [1.0, 0.8, 0.6],
                [0.8, 1.0, 0.7],
                [0.6, 0.7, 1.0],
            ]
        )
        rank_matrix = np.array(
            [
                [1.0, 0.7, 0.5],
                [0.7, 1.0, 0.6],
                [0.5, 0.6, 1.0],
            ]
        )
        positions = embed_worldviews_in_2d_space(pearson_matrix, rank_matrix)
        assert positions.shape == (3, 2)

    def test_calculate_weighted_centroid(self):
        positions = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        weights = np.array([0.5, 0.25, 0.25])
        centroid = calculate_weighted_centroid(positions, weights)
        assert np.allclose(centroid, np.array([0.5, 0.5]))

    def test_calculate_weighted_centroid_zero_weights(self):
        positions = np.array([[1.0, 1.0], [2.0, 2.0]])
        weights = np.array([0.0, 0.0])
        centroid = calculate_weighted_centroid(positions, weights)
        assert np.allclose(centroid, np.array([0.0, 0.0]))

    def test_find_closest_worldview(self):
        positions = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
        target = np.array([1.8, 0.1])
        idx = find_closest_worldview(positions, target)
        assert idx == 1


class TestMultiStageAggregationModule:
    """Direct unit tests for multi_stage_aggregation helpers."""

    def test_moral_theory_value_of(self):
        theory = MoralTheory("t", {"x": 10.0})
        assert theory.value_of("x") == 10.0
        assert theory.value_of("missing") == 0.0

    def test_pure_mec_choose_intervention_and_mec_aggregate(self):
        interventions = ["a", "b"]
        theories = [
            MoralTheory("t1", {"a": 10.0, "b": 0.0}),
            MoralTheory("t2", {"a": 2.0, "b": 3.0}),
        ]
        credences = {"t1": 0.6, "t2": 0.4}

        best, scores = mec_aggregate_cardinal_theories(interventions, theories, credences)
        assert best == "a"
        assert scores["a"] > scores["b"]

        pure_best = pure_mec_choose_intervention(interventions, theories, credences)
        assert pure_best == "a"

    def test_convert_ordinal_to_binary(self):
        theory = MoralTheory("ordinal", {"a": 0.0, "b": 1.0})
        assert convert_ordinal_to_binary("a", theory) is False
        assert convert_ordinal_to_binary("b", theory) is True

    def test_convert_mec_result_to_binary_winner_take_all_and_threshold(self):
        mec_scores = {"a": 100.0, "b": 95.0, "c": 40.0}
        assert convert_mec_result_to_binary("a", "a", mec_scores, threshold_based=False) is True
        assert convert_mec_result_to_binary("b", "a", mec_scores, threshold_based=False) is False

        assert convert_mec_result_to_binary("b", "a", mec_scores, threshold_based=True, threshold=0.9) is True
        assert convert_mec_result_to_binary("c", "a", mec_scores, threshold_based=True, threshold=0.9) is False

    def test_credence_weighted_vote(self):
        interventions = ["a", "b"]
        permissibility_votes = {
            "a": [("t1", True, 0.4), ("t2", False, 0.6)],
            "b": [("t1", True, 0.4), ("t2", True, 0.6)],
        }
        credences = {"t1": 0.4, "t2": 0.6}
        chosen = credence_weighted_vote(interventions, permissibility_votes, credences)
        assert chosen == "b"

    def test_multistage_aggregation_returns_choice_and_debug(self):
        interventions = ["a", "b", "c"]
        cardinal_theories = [
            MoralTheory("card1", {"a": 10.0, "b": 7.0, "c": 1.0}),
            MoralTheory("card2", {"a": 9.0, "b": 8.0, "c": 2.0}),
        ]
        ordinal_theories = [
            MoralTheory("ord1", {"a": 1.0, "b": 1.0, "c": 0.0}),
            MoralTheory("ord2", {"a": 0.0, "b": 1.0, "c": 1.0}),
        ]
        credences = {"card1": 0.3, "card2": 0.3, "ord1": 0.2, "ord2": 0.2}

        chosen, debug = multistage_aggregation(
            interventions,
            cardinal_theories,
            ordinal_theories,
            credences,
            mec_conversion_method="threshold",
            mec_threshold=0.8,
        )
        assert chosen in interventions
        assert "mec_recommendation" in debug
        assert "vote_tallies" in debug

    def test_multistage_with_incomparability_handling(self):
        interventions = ["a", "b"]
        cardinal_theories = [MoralTheory("card", {"a": 10.0, "b": 9.6})]
        ordinal_theories = [MoralTheory("ord", {"a": 1.0, "b": 1.0})]
        credences = {"card": 0.6, "ord": 0.4}

        chosen, debug = multistage_with_incomparability_handling(
            interventions,
            cardinal_theories,
            ordinal_theories,
            credences,
            incomparability_threshold=0.1,
        )
        assert chosen in interventions
        assert "threshold_score" in debug
        assert debug["threshold_score"] <= max(debug["mec_scores"].values())
