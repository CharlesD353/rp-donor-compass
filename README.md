# Donor Compass

A module for calculating and allocating charitable donations across projects based on moral weights, discount factors, and risk profiles.

Original Colab notebook: [https://colab.research.google.com/drive/1fTV-fgyitcO3JYrlVlj2Jnm5dCzhzHF8](https://colab.research.google.com/drive/1fTV-fgyitcO3JYrlVlj2Jnm5dCzhzHF8)

## What This Module Does

Given a set of "worldviews" (combinations of moral beliefs), this module:

1. Calculates the value of each project based on:
  - **Moral weights** - how much you value different recipients (humans, chickens, shrimp, etc.)
  - **Discount factors** - how much you value each time period (near-term vs far future)
  - **Risk profile** - which effect estimates to use (optimistic, pessimistic, neutral)
  - **Extinction probability** - discounts non-xrisk projects if you believe extinction is likely
2. Allocates a budget across projects in configurable increments:
  - Supports multiple aggregation/voting methods
  - Votes can be weighted by worldview credence
  - Diminishing returns are applied as projects accumulate funding

## Usage

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

## Streamlit Demo App

An interactive Streamlit app is included for exploring and comparing all aggregation methods.

### Run locally

```bash
pip install streamlit pandas numpy scipy scikit-learn
streamlit run streamlit_app.py
```

If you are in a parent directory, run:

```bash
streamlit run donor_compass/streamlit_app.py
```

The app includes:
- Single-method exploration with full run history
- All-method side-by-side comparison
- Scenario presets (default dataset + two E2E stress scenarios)
- Editable worldview JSON and optional project-data override
- CSV exports for EV and EU across all risk models

## Colab Notebook Demo

A standalone Colab-friendly notebook is included at:

- `DonorCompass_Colab_Demo.ipynb`

### How to use it in Colab

1. Open Google Colab.
2. Upload `DonorCompass_Colab_Demo.ipynb`.
3. Run the notebook top-to-bottom (or use **Runtime -> Run all**).

### Run order in the notebook

1. Dependency setup cell
2. Standalone implementation snapshot cell
3. Single-method demo config + run cells
4. All-method comparison cell
5. E2E Scenario A/B verification cell

Notes:
- The notebook intentionally contains copied logic to stay standalone in Colab.
- It includes only E2E verification scenarios (not the full pytest unit suite).
- Source of truth remains the Python files in this repo (`donor_compass.py`, `met_sim_utils.py`, `multi_stage_aggregation.py`).

## Exclusions from Original Notebook

The original Colab notebook contains code that is either **broken** or **deprioritized** (commented out). This module only includes the functional portions.

### Broken Code (Not Included)


| Function                      | Problem                                                                                              |
| ----------------------------- | ---------------------------------------------------------------------------------------------------- |
| `calculate_all_worldviews`    | Calls `calculate_project_value` which is never defined anywhere in the notebook                      |
| `vote_mec` (original version) | References global `dummy_data`, `build_moral_weights`, and quiz constants that are all commented out |


### Deprioritized Code (Commented Out in Original)

The notebook has a section marked "DEPRIORITIZED CODE BITS" that was intended for a quiz-based approach with 25,600 precomputed worldview combinations. This code is all commented out:


| Item                                         | Description                                 |
| -------------------------------------------- | ------------------------------------------- |
| `build_moral_weights`                        | Constructs moral weights from quiz answers  |
| `q1_daly_weights`, `q2_income_weights`, etc. | Quiz option constants                       |
| `vote_credence_weighted_precomputed`         | Voting using precomputed results            |
| Precomputation loop                          | Generates 25,600 combinations               |
| `compute_worldview_credences`                | Called but never defined (causes NameError) |
| `find_result`                                | Lookup helper for precomputed results       |
| `show_results`                               | Display helper for quiz-based results       |


### What Works

The module preserves the original functional path and now also includes additional
aggregation methods from local Donor Compass specs (`desired_aggregations.rtf`,
`met_summary.txt`, `msa.txt`, `nash_bargaining.txt`).

## Module Contents

### Data

- `DEFAULT_PROJECT_DATA` - 6 projects (malaria_nets, cage_free_campaigns, shrimp_welfare, wild_animal_welfare, fish_welfare, ai_safety_policy)
- `EXAMPLE_CUSTOM_WORLDVIEWS` - 3 sample worldviews for testing

### Data Schemas

`DEFAULT_PROJECT_DATA` schema:

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

`custom_worldviews` schema:

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

### Calculator Functions

- `calculate_single_effect(effect_data, moral_weight, discount_factors, risk_profile)`
- `calculate_project(project_data, moral_weights, discount_factors, risk_profile)`
- `calculate_all_projects(data, moral_weights, discount_factors, risk_profile)`
- `adjust_for_extinction_risk(project_values, data, p_extinction)`

### Voting Methods

- `vote_credence_weighted_custom(data, funding, increment, custom_worldviews)` - Marketplace/Budget by credence
- `vote_met(data, funding, increment, custom_worldviews, ...)` - MET with threshold switch
- `vote_nash_bargaining(data, funding, increment, custom_worldviews, ...)` - Nash with configurable disagreement points
- `vote_msa(data, funding, increment, custom_worldviews, ...)` - Multi-stage aggregation (cardinal cluster + binary permissibility voting)
- `vote_mec(...)` - MEC (unified `custom_worldviews` mode + legacy quiz-input mode)
- `vote_borda(data, funding, increment, custom_worldviews, ...)` - Borda voting
- `vote_split_cycle(data, funding, increment, custom_worldviews, ...)` - Split-Cycle voting
- `vote_lexicographic_maximin(data, funding, increment, custom_worldviews, ...)` - Lexicographic Maximin
- `vote_my_favorite_theory(...)` - Favorite-theory baseline (unified `custom_worldviews` mode + legacy precomputed mode)

### Method-Specific kwargs

- `vote_credence_weighted_custom`
  - Required: `custom_worldviews`
- `vote_met`
  - Required: `custom_worldviews`
  - Optional: `met_threshold` (default `0.5`), `tie_break`, `random_seed`
- `vote_nash_bargaining`
  - Required: `custom_worldviews`
  - Optional: `disagreement_point` (`zero_spending`, `anti_utopia`, `random_dictator`, `exclusionary_proportional_split`) (default `zero_spending`)
- `vote_mec`
  - Unified mode (recommended): pass `custom_worldviews`
  - Legacy mode: pass quiz credences/input tables (`q1_cred` ... `q7_extinction_probs`) and `build_moral_weights_fn`
  - Optional in both modes: `tie_break`, `random_seed`
- `vote_msa`
  - Required: `custom_worldviews`
  - Optional: `worldview_types`, `cardinal_permissibility_mode` (`winner_take_all`, `top_k`, `within_percent`), `cardinal_top_k`, `cardinal_within_percent`, `binary_permissibility_threshold`, `no_permissible_action` (`stop` or `fallback_mec`)
- `vote_my_favorite_theory`
  - Unified mode (recommended): pass `custom_worldviews`
  - Legacy mode: pass `results` + `worldviews` (precomputed format)
  - Optional in both modes: `tie_break`, `random_seed`
- `vote_borda`, `vote_split_cycle`, `vote_lexicographic_maximin`
  - Required: `custom_worldviews`
  - Optional: `tie_break`, `random_seed`

### Notes on defaults and assumptions

- Initial MSA worldview typing defaults to:
  - Binary: `Kantianism`, `Rawlsian Contractarianism`
  - Cardinal: all others (unless overridden via `worldview_types` or worldview metadata)
- If MSA finds no project above 50% permissibility:
  - Default behavior is stop allocation early via an explicit stop signal
- Prior parliament-tool implementations for Split-Cycle/Borda/Lexicographic Maximin were not present in this workspace, so these methods use canonical first-pass algorithm definitions with deterministic tie-breaking by default.
- `vote_credence_weighted_custom` validates credences strictly:
  - credences must be non-negative
  - credences must sum to 1.0 (within tolerance)
- `get_diminishing_returns_factor` is hardened for floating-point near-boundary funding values (e.g. `9.999999999` is treated as a `$10M` step boundary).

### Domain Primer

- **Worldview**: one moral/empirical stance with a credence and valuation parameters.
- **Credence**: confidence weight assigned to a worldview.
- **MEC**: selects the action with highest expected choiceworthiness across uncertainty.
- **MET**: selects action/theory by expected truthlikeness, with threshold fallback to favorite theory.
- **MSA**: combines cardinal-theory MEC aggregation with binary permissibility voting.
- **Risk profile**: selects which column in each 6x4 effect matrix to use.
- **Extinction probability (`p_extinction`)**: downweights non-xrisk projects.
- **Diminishing returns**: per-project marginal value decay as funding increases in `$10M` steps.

### Fallback Behavior

- `tie_break` fallback (global for methods that rank winners):
  - Default is `deterministic` (alphabetical project-id tie break)
  - `random` uses `random_seed` when provided for reproducible ties

- `vote_met`:
  - If highest worldview credence is `>= met_threshold`, method falls back to favorite-theory behavior
  - If highest worldview credence is `< met_threshold`, method uses similarity-centroid selection

- `vote_nash_bargaining`:
  - Primary objective is Nash product over non-negative gains vs disagreement point
  - If no project has non-negative gains for all worldviews, method falls back to maximizing sum of gains (`sum_gains_fallback`)

- `vote_msa`:
  - If no project exceeds 50% permissibility and `no_permissible_action="stop"`:
    - returns stop signal: `{"__stop__": True, "__reason__": "..."}`
  - If no project exceeds 50% permissibility and `no_permissible_action="fallback_mec"`:
    - with cardinal theories: picks cardinal MEC winner
    - with no cardinal theories: picks highest credence-weighted score project
  - Invalid `cardinal_permissibility_mode` or invalid `no_permissible_action` raises `ValueError`

- `allocate_budget` stop handling:
  - Accepts vote method return types:
    - allocation dict, or
    - `(allocation_dict, metadata_dict)` tuple
  - If allocation dict contains `__stop__`, allocation loop halts and history entry includes:
    - `stopped: True`
    - `remaining_budget`
    - optional `reason` (from `__reason__`)
    - optional `meta` (if vote method returned metadata)

### Budget Allocation

- `allocate_budget(data, voting_method, total_budget, **kwargs)`
- `show_allocation(allocation, data)`

### Constants

- `INCREMENT_SIZE` - $10M default step size
- `get_diminishing_returns_factor(data, project_id, current_funding)`

## Tests

```bash
pytest test_donor_compass.py test_supporting_modules.py test_aggregation_e2e.py -vv
```

E2E verbosity modes (`test_aggregation_e2e.py`):

- Default (verbose summary):
  - `pytest test_aggregation_e2e.py -vv -s`
- Quiet mode (useful when running with other tests):
  - `DONOR_COMPASS_E2E_OUTPUT=quiet pytest test_aggregation_e2e.py -vv -s`

The tests cover all functional code in the module, including the new aggregation methods:

`TestCalculateSingleEffect` **(5 tests)**

- Uniform values with uniform discounts produce expected sum
- Moral weight scales results proportionally
- Zero moral weight produces zero value
- Different risk profiles select different value columns
- Discount factors weight each time period correctly

`TestCalculateProject` **(3 tests)**

- Single-effect project returns that effect's value
- Missing moral weight types default to zero contribution
- Multi-effect projects sum all effect values

`TestCalculateAllProjects` **(2 tests)**

- Returns values for all projects in the dataset
- Each project value matches individual `calculate_project` output

`TestAdjustForExtinctionRisk` **(4 tests)**

- xrisk projects remain unchanged regardless of p_extinction
- Non-xrisk projects scaled by (1 - p_extinction)
- Zero extinction probability leaves all values unchanged
- High extinction probability dramatically reduces non-xrisk values

`TestGetDiminishingReturnsFactor` **(4 tests)**

- Zero funding returns the first DR factor
- Funding in $10M increments maps to correct array indices
- Funding beyond array length returns the last factor
- Works correctly with DEFAULT_PROJECT_DATA

`TestVoteCredenceWeightedCustom` **(4 tests)**

- Single worldview allocates entire increment to highest-value project
- Split credence allocates proportionally
- Worldviews with same preferences combine allocations
- High extinction probability shifts preference toward xrisk projects

`TestAllocateBudget` **(4 tests)**

- Total funding equals total budget
- Number of iterations matches budget / increment_size
- History records each iteration's allocations
- Partial final increment handled correctly (budget not divisible by increment)

`TestWithDefaultData` **(4 tests)**

- Smoke test with real project data and moral weights
- Voting with EXAMPLE_CUSTOM_WORLDVIEWS produces valid allocations
- Full allocation integration test (100M budget, 10 iterations)
- `show_allocation` runs without crashing

`TestEdgeCases` **(4 tests)**

- Zero budget results in no allocations
- Very small increments work correctly
- Zero credence worldviews allocate nothing
- Negative effect values (downside scenarios) compute without error

Additional test groups cover:

- Credence validation failures and floating-point DR boundary handling
- Unified interfaces for `vote_mec` and `vote_my_favorite_theory`
- Direct unit tests for `met_sim_utils.py`
- Direct unit tests for `multi_stage_aggregation.py`
- End-to-end stress test across all aggregation methods with printed summary output

