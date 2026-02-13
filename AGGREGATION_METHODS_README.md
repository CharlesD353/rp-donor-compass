# Donor Compass Aggregation Methods

This document summarizes the voting and aggregation methods implemented in `donor_compass.py`, with practical guidance on behavior, inputs, and fallbacks.

For a runnable standalone Colab demo, see `DonorCompass_Colab_Demo.ipynb`.

## Common Execution Model

All methods are designed to work with:

```python
allocate_budget(data, voting_method, total_budget, **kwargs)
```

Each method receives:
- `data`: project dataset
- `funding`: current cumulative funding by project
- `increment`: current budget chunk
- method-specific kwargs (usually `custom_worldviews`)

Each method returns:
- allocation dict `{project_id: amount}` (amounts usually sum to `increment`), or
- `(allocation_dict, metadata_dict)` when `return_debug=True`

Special stop behavior:
- `vote_msa` can return `{"__stop__": True, "__reason__": ...}`.
- `allocate_budget` detects this and halts early, recording a stop entry in history.

## Method Quick Reference

| Method | Function | Core Idea | Typical Inputs |
|---|---|---|---|
| Marketplace / Budget by credence | `vote_credence_weighted_custom` | Split increment by credence; each worldview funds its top project | `custom_worldviews` |
| My Favorite Theory | `vote_my_favorite_theory` | Pick highest-credence worldview and fully follow it | `custom_worldviews` (recommended) |
| MEC | `vote_mec` | Maximize expected choiceworthiness | `custom_worldviews` (recommended) |
| MET | `vote_met` | Thresholded switch between favorite theory and similarity-based compromise | `custom_worldviews`, `met_threshold` |
| Nash Bargaining | `vote_nash_bargaining` | Choose project maximizing bargaining objective relative to disagreement point | `custom_worldviews`, `disagreement_point` |
| MSA | `vote_msa` | Combine cardinal MEC cluster with binary permissibility voting | `custom_worldviews`, MSA options |
| Borda | `vote_borda` | Credence-weighted rank points | `custom_worldviews` |
| Split-Cycle | `vote_split_cycle` | Pairwise majority graph + cycle handling via strongest paths | `custom_worldviews` |
| Lexicographic Maximin | `vote_lexicographic_maximin` | Maximize worst-off (lexicographically over weighted utility vectors) | `custom_worldviews` |

---

## 1) Marketplace / Budget by Credence

Function: `vote_credence_weighted_custom`

How it works:
1. Validate worldview credences.
2. For each worldview, compute marginal project values at current funding.
3. Allocate that worldview's share (`credence * increment`) to its best project.
4. Sum across worldviews.

Validation rules:
- Credences must be numeric and non-negative.
- Credences must sum to `1.0` (tolerance) or all be `0.0`.
- If all credences are zero, method returns zero allocations.

Tie behavior:
- Deterministic alphabetical tie-break for best project.

Use when:
- You want proportional moral compromise where each worldview keeps control of its share.

---

## 2) My Favorite Theory

Function: `vote_my_favorite_theory`

How it works:
- Select worldview with highest credence.
- Allocate all of `increment` to that worldview's top marginal-value project.

Interfaces:
- Recommended: `custom_worldviews`
- Legacy: `results + worldviews` (precomputed notebook format)

Fallback behavior:
- Empty worldviews: zero allocation (`no_worldviews` debug strategy).
- All-zero normalized credence (unified path): zero allocation (`no_positive_credence`).

Use when:
- You want a non-compromise baseline: "act as if most likely worldview is true."

---

## 3) MEC (Maximizing Expected Choiceworthiness)

Function: `vote_mec`

How it works (unified mode):
1. Normalize worldview credences.
2. Compute each worldview's marginal project values.
3. Compute expected marginal score per project (credence-weighted sum).
4. Allocate full `increment` to project with highest expected score.

How it works (legacy quiz mode):
1. Average continuous quiz dimensions by credence.
2. Keep risk profile (`q6`) as a mixture over discrete profiles.
3. For each risk profile slice, pick best project and allocate that slice.

Interfaces:
- Recommended: `custom_worldviews`
- Legacy: `q1_cred ... q7_extinction_probs` + `build_moral_weights_fn`

Use when:
- You want expected-value aggregation across worldview uncertainty.

---

## 4) MET (Maximizing Expected Truthlikeness)

Function: `vote_met`

How it works:
1. Compute normalized credences and worldview marginal scores.
2. If `max_credence >= met_threshold`, use favorite-theory behavior.
3. Else, use similarity compromise:
   - pairwise Pearson + Spearman similarities,
   - 2D MDS embedding,
   - credence-weighted centroid,
   - choose worldview nearest centroid.
4. Allocate full `increment` to selected worldview's top project.

Key kwargs:
- `met_threshold` (default `0.5`)
- `tie_break`, `random_seed`

Use when:
- You want thresholded compromise: strong-confidence behavior near favorite theory, but similarity-based compromise under higher uncertainty.

---

## 5) Nash Bargaining

Function: `vote_nash_bargaining`

How it works:
1. Compute each worldview's project utilities (marginal values).
2. Compute disagreement utilities using selected disagreement point:
   - `zero_spending` (default)
   - `anti_utopia`
   - `random_dictator`
   - `exclusionary_proportional_split`
3. For each project, compute gains over disagreement utilities.
4. If all gains are non-negative for all worldviews, maximize Nash product.
5. Otherwise, fallback to maximizing sum of gains (`sum_gains_fallback`).

Tie behavior:
- Deterministic by default, optional seeded random.

Use when:
- You want an explicit bargaining framing with configurable disagreement baselines.

---

## 6) MSA (Multi-Stage Aggregation)

Function: `vote_msa`

How it works:
1. Classify worldviews into `cardinal` vs `binary`.
2. Stage 1: MEC over cardinal theories to create a cardinal cluster recommendation.
3. Stage 2: Convert to permissibility:
   - `winner_take_all`
   - `top_k`
   - `within_percent`
   plus binary-theory permissibility via `score > binary_permissibility_threshold`.
4. Stage 3: Credence-weighted tally of permissibility votes; choose highest tally.

Classification:
- Explicit via worldview `theory_type` or `worldview_types` map.
- Default binary names: `Kantianism`, `Rawlsian Contractarianism`; others treated as cardinal.

No-permissible behavior:
- `no_permissible_action="stop"`: return stop signal.
- `no_permissible_action="fallback_mec"`:
  - with cardinal theories: choose cardinal MEC best,
  - without cardinal theories: choose best weighted score.

Use when:
- You want mixed treatment of cardinal and binary/constraint-style worldviews.

---

## 7) Borda Voting

Function: `vote_borda`

How it works:
1. Each worldview ranks projects by marginal value.
2. Projects receive Borda points (`n-1` to `0`) per worldview.
3. Points are credence-weighted and summed.
4. Highest total wins.

Use when:
- You want rank-based compromise that uses full preference orderings.

---

## 8) Split-Cycle Voting

Function: `vote_split_cycle`

How it works:
1. Build credence-weighted pairwise preference matrix.
2. Compute pairwise margins.
3. Compute strongest paths (Floyd-Warshall style).
4. Keep defeats that are not neutralized by stronger reverse paths.
5. Select unbeaten candidates.
6. Defensive fallback: net-margin winner if needed.

Complexity note:
- Strongest-path computation is cubic in number of projects (`O(n^3)`).

Use when:
- You want Condorcet-style cycle-resistant social choice.

---

## 9) Lexicographic Maximin

Function: `vote_lexicographic_maximin`

How it works:
1. For each project, compute worldview utilities weighted by credence.
2. Sort each project's utility vector ascending (worst-off first).
3. Compare vectors lexicographically; choose maximal vector.
4. Full increment goes to winner.

Use when:
- You want a worst-off-first fairness criterion under moral uncertainty.

---

## Shared Tie-Break and Debug Conventions

Tie-break:
- `tie_break="deterministic"` (default): alphabetical by project id.
- `tie_break="random"` with optional `random_seed` for reproducibility.

Debug:
- Most methods support `return_debug=True` and return `(allocations, metadata)`.
- `allocate_budget` stores per-iteration `meta` when present.

## Practical Defaults

If you want a robust default setup:
- Use `vote_mec` with `custom_worldviews` for expected-value aggregation.
- Use `vote_met` when you want thresholded compromise behavior.
- Use `vote_msa` when binary-vs-cardinal worldview structure matters.
- Keep deterministic tie-breaks unless you explicitly want stochastic ties.
