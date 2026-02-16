# Donor Compass

A toolkit for calculating and allocating charitable donations across projects
based on moral weights, discount factors, and risk profiles. Includes multiple
aggregation/voting methods and an interactive Streamlit app.

## What It Does

Given a set of "worldviews" (combinations of moral beliefs), Donor Compass:

1. **Calculates project values** based on:
   - **Moral weights** — how much you value different recipients (humans, chickens, shrimp, etc.)
   - **Discount factors** — how much you value each time period (near-term vs far future)
   - **Risk profile** — which effect estimates to use (optimistic, pessimistic, neutral)
   - **Extinction probability** — discounts non-xrisk projects if you believe extinction is likely
2. **Allocates a budget** across projects in configurable increments:
   - Supports multiple aggregation/voting methods
   - Votes can be weighted by worldview credence
   - Diminishing returns are applied as projects accumulate funding

## Streamlit App

An interactive app for exploring and comparing all aggregation methods.

```bash
pip install streamlit pandas numpy scipy scikit-learn
streamlit run streamlit_app.py
```

The app includes:
- Single-method exploration with full run history
- All-method side-by-side comparison
- Scenario presets (default dataset + two E2E stress scenarios)
- Editable worldview JSON and optional project-data override
- CSV exports

## Programmatic Usage

```python
from donor_compass import (
    DEFAULT_PROJECT_DATA,
    vote_credence_weighted_custom,
    allocate_budget,
    show_allocation,
    EXAMPLE_CUSTOM_WORLDVIEWS,
)

# Allocate $100M using the example worldviews
result = allocate_budget(
    DEFAULT_PROJECT_DATA,
    vote_credence_weighted_custom,
    total_budget=100,
    custom_worldviews=EXAMPLE_CUSTOM_WORLDVIEWS
)

show_allocation(result, DEFAULT_PROJECT_DATA)
```

You can swap in other methods using the same `allocate_budget` interface:

```python
from donor_compass import vote_met

result = allocate_budget(
    DEFAULT_PROJECT_DATA,
    vote_met,
    total_budget=100,
    custom_worldviews=EXAMPLE_CUSTOM_WORLDVIEWS,
    met_threshold=0.5,
)
```

## Project Structure

| File | Purpose |
|------|---------|
| `donor_compass.py` | Core module — project data, calculators, all voting methods, budget allocation |
| `met_sim_utils.py` | MET similarity/centroid utilities |
| `multi_stage_aggregation.py` | Multi-Stage Aggregation (MSA) logic |
| `streamlit_app.py` | Streamlit UI |
| `AGGREGATION_METHODS_README.md` | Detailed documentation of each voting method |
| `DonorCompass_Colab_Demo.ipynb` | Standalone Colab-friendly demo notebook |
| `test_donor_compass.py` | Unit tests for core module |
| `test_supporting_modules.py` | Unit tests for met_sim_utils and multi_stage_aggregation |
| `test_aggregation_e2e.py` | End-to-end stress tests across all methods |

## Data Schemas

`DEFAULT_PROJECT_DATA`:

```python
{
  "<project_id>": {
    "tags": {"near_term_xrisk": bool},
    "diminishing_returns": [float, ...],  # indexed by $10M funding steps
    "effects": {
      "<effect_id>": {
        "recipient_type": str,
        "values": [[float, float, float, float], ...]  # 6x4 matrix (time x risk profile)
      }
    }
  }
}
```

`custom_worldviews`:

```python
[
  {
    "name": str,
    "credence": float,
    "moral_weights": {
      "human_life_years": float,
      "human_ylds": float,
      "human_income_doublings": float,
      "chickens_birds": float,
      "fish": float,
      "shrimp": float,
      "non_shrimp_invertebrates": float,
      "mammals": float
    },
    "discount_factors": [float, float, float, float, float, float],
    "risk_profile": int,    # 0=neutral, 1=upside, 2=downside, 3=combined
    "p_extinction": float,  # usually in [0, 1]
    # optional for MSA:
    "theory_type": "binary" | "cardinal"
  }
]
```

## Voting Methods

| Method | Function | Key Parameters |
|--------|----------|----------------|
| Credence-Weighted | `vote_credence_weighted_custom` | `custom_worldviews` |
| MET | `vote_met` | `met_threshold` (default 0.5), `tie_break`, `random_seed` |
| Nash Bargaining | `vote_nash_bargaining` | `disagreement_point` (default `zero_spending`) |
| Multi-Stage Aggregation | `vote_msa` | `cardinal_permissibility_mode`, `binary_permissibility_threshold`, `no_permissible_action` |
| MEC | `vote_mec` | `tie_break`, `random_seed` |
| Borda | `vote_borda` | `tie_break`, `random_seed` |
| Split Cycle | `vote_split_cycle` | `tie_break`, `random_seed` |
| Lexicographic Maximin | `vote_lexicographic_maximin` | `tie_break`, `random_seed` |
| My Favorite Theory | `vote_my_favorite_theory` | `tie_break`, `random_seed` |

See `AGGREGATION_METHODS_README.md` for detailed descriptions, formulas, and fallback behaviour.

## Calculator Functions

- `calculate_single_effect(effect_data, moral_weight, discount_factors, risk_profile)`
- `calculate_project(project_data, moral_weights, discount_factors, risk_profile)`
- `calculate_all_projects(data, moral_weights, discount_factors, risk_profile)`
- `adjust_for_extinction_risk(project_values, data, p_extinction)`
- `allocate_budget(data, voting_method, total_budget, **kwargs)`
- `show_allocation(allocation, data)`

## Domain Primer

- **Worldview**: one moral/empirical stance with a credence and valuation parameters.
- **Credence**: confidence weight assigned to a worldview.
- **MEC**: selects the action with highest expected choiceworthiness across uncertainty.
- **MET**: selects action/theory by expected truthlikeness, with threshold fallback to favorite theory.
- **MSA**: combines cardinal-theory MEC aggregation with binary permissibility voting.
- **Risk profile**: selects which column in each 6x4 effect matrix to use.
- **Extinction probability (`p_extinction`)**: downweights non-xrisk projects.
- **Diminishing returns**: per-project marginal value decay as funding increases in $10M steps.

## Fallback Behaviour

- **Tie-breaking**: default is `deterministic` (alphabetical project-id). `random` uses `random_seed` when provided.
- **MET**: if highest worldview credence >= `met_threshold`, falls back to favorite-theory behaviour.
- **Nash Bargaining**: if no project has non-negative gains for all worldviews, falls back to maximising sum of gains.
- **MSA**: if no project exceeds 50% permissibility, either stops allocation (`no_permissible_action="stop"`) or falls back to MEC (`"fallback_mec"`).
- **`allocate_budget`**: accepts vote methods returning either a dict or a `(dict, metadata)` tuple. If the dict contains `__stop__`, the allocation loop halts early.

## Notes

- `vote_credence_weighted_custom` validates credences strictly: must be non-negative and sum to 1.0.
- `get_diminishing_returns_factor` is hardened for floating-point near-boundary funding values.
- `INCREMENT_SIZE` is $10M by default.

## Tests

```bash
pytest test_donor_compass.py test_supporting_modules.py test_aggregation_e2e.py -vv
```

E2E verbosity:
- Default (verbose summary): `pytest test_aggregation_e2e.py -vv -s`
- Quiet mode: `DONOR_COMPASS_E2E_OUTPUT=quiet pytest test_aggregation_e2e.py -vv -s`
