# PRISM — Project Memory & Backlog

## Quick Reference
- **Python**: `py -3.11`, venv at `.venv/`
- **Activate**: `.venv/Scripts/activate` (Windows)
- **Tests**: `py -3.11 -m pytest tests/ -v` (97 pass as of Phase 2)
- **Package**: `pip install -e .` (flat `prism/` package)
- **MiniGrid 2.5**: `import minigrid` REQUIRED before `gym.make()`

## Architecture
```
prism/
├── config.py              # SRConfig, MetaSRConfig, ControllerConfig, PRISMConfig
├── env/
│   ├── state_mapper.py    # (x,y) <-> int, 260 accessible cells in FourRooms
│   ├── dynamics_wrapper.py # reward_shift, door_block, door_open + true T matrix
│   └── perturbation_schedule.py  # STUB
├── agent/
│   ├── sr_layer.py        # Tabular SR with TD(0), M init=I, returns delta_M
│   ├── meta_sr.py         # U(s) uncertainty map, C(s) confidence, change detection
│   ├── controller.py      # Adaptive epsilon, V_explore = V + lambda*U, IDK flag
│   └── prism_agent.py     # Full assembled agent, train_episode(), train()
├── baselines/             # ALL STUBS — SRBlind, SRCount, SRBayesian, SB3Baseline
├── analysis/
│   ├── calibration.py     # ECE, reliability diagram, metacognitive index
│   ├── spectral.py        # SR eigenvectors
│   ├── visualization.py   # Heatmaps for SR, value, uncertainty
│   └── metrics.py         # STUBS: bootstrap_ci(), compare_conditions()
└── pedagogy/
    └── toy_grid.py        # Lightweight grid for notebooks (no MiniGrid dep)
```

## Phase Status

### Phase 0: COMPLETE
Setup, env verification, 22 tests pass.

### Phase 1: COMPLETE
StateMapper, SRLayer, DynamicsWrapper, spectral, visualization. CP1 passed.

### Phase 2: COMPLETE
MetaSR, PRISMController, PRISMAgent, calibration. 97 tests pass.

### Phase 3: TODO — Experiments A, B, C

#### Experiment A — Calibration (tests P1+P2)
- ECE < 0.15, MI > 0.5, beats baselines
- Needs: baselines (SRBlind, SRBayesian), PerturbationSchedule, metrics stubs
- Checkpoint: CP2 (hyperparam sweep) + CP3 (calibration results)

#### Experiment B — Exploration (tests P3)
- PRISM vs 7(+1) baselines on goal discovery task
- Needs: all baselines, experiment runner, statistical tests
- See "Exp B Improvements" section below for proposed enhancements
- Checkpoint: CP4

#### Experiment C — Adaptation (tests P4+P5)
- Change detection + re-exploration after perturbations
- Needs: PerturbationSchedule, baselines, adaptation metrics
- Checkpoint: CP5

## Known Issues / Blockers
1. **Controller exploit branch incomplete**: select_action() falls back to random even in exploit — needs true greedy-on-V_explore for Exp B
2. **hosmer_lemeshow_test() missing**: needed for Exp A (CP3)
3. **bootstrap_ci() and compare_conditions() are stubs** in metrics.py
4. **All baseline agents are stubs** — nothing runnable yet
5. **PerturbationSchedule is a stub** — needed for Exp A and C
6. **SR-Oracle baseline not implemented** — needs agent class wrapping DynamicsWrapper.get_true_transition_matrix()

## Exp B Improvements — Decisions (2026-02-16)
Source: `inputs_for_claude/exp_b_improvements.md`

### ACCEPTED: SR-Count-Matched (Amélioration 2)
- 9th condition controlling for temporal decay profile
- Cleanest test of structural vs temporal contribution of U(s)
- **Fix needed**: calibration must collect (n_visits, U) pairs online at each step, not post-hoc at end of training
- Cost: ~0.5-1 day

### ACCEPTED (partial): Enriched Metrics (Amélioration 3)
- **AUC discovery**: yes — strictly better than "steps to 4 goals"
- **Guidance index** (Spearman U vs visit order): yes — most direct test of the thesis
- **Systematicity index**: NO — too sensitive to window size hyperparameter
- **Entropy heatmap**: optional visualization only, not a primary metric
- Cost: ~0.5 day, post-hoc on existing trajectories

### DEFERRED: Multi-geometry (Amélioration 1)
- Reduced to 3 configs (FourRooms + linear corridor + 3x3 grid) instead of 10
- Only after Palier 1+2 are done and results are clear
- Custom MiniGrid envs are non-trivial (2-3 days, not 1)
- Cost: 2-3 days

## Phase 3 Implementation Plan — Exp B

### Palier 1 — Exp B vanilla (MVP for CP4)
1 config (FourRooms 19x19), 8 conditions, 100 runs/condition = 800 runs.

Work items:
1. Infrastructure commune (experiment runner, results I/O)
2. `metrics.py` — bootstrap_ci(), compare_conditions() (Mann-Whitney U + Holm-Bonferroni)
3. Fix controller exploit branch (greedy on V_explore, not random)
4. Implement 7 baselines:
   - SR-Oracle (uses true M* from DynamicsWrapper)
   - SR-ε-greedy (fixed ε=0.1)
   - SR-ε-decay (decaying ε)
   - SR-Count-Bonus (λ/√visits)
   - SR-Norm-Bonus (λ/||M(s,:)||)
   - SR-Posterior (posterior sampling on V)
   - Random (uniform)
5. Experiment runner: `experiments/exp_b_exploration.py`
6. Run 800 runs + analysis + figures for CP4

### Palier 2 — Count-Matched + enriched metrics
- SR-Count-Matched baseline with corrected online calibration
- AUC discovery + guidance index computed post-hoc
- 100 additional runs for condition 9 only

### Palier 3 — Multi-geometry (if time permits)
- 2 custom MiniGrid envs (linear corridor C6, 3x3 grid C8)
- Re-run all 9 conditions on 3 configs
- Mixed-effects analysis

## Known Technical Constraints
- **FourRooms wall positions change per seed**: passages are random. Must use a fixed env seed across all runs (StateMapper built once). Variation comes from goal placement + agent seed only.
- **141 tests pass** as of Palier 1.5 completion (97 original + 44 new).
- **SR-Oracle bonus must include count decay**: Raw `||M-M*||₂` has 1.16x contrast (spatially smooth L2 norm dominated by tail of 260-dim vector). Multiplying by `1/sqrt(visits+1)` fixes it (100% coverage on all 20 seeds). Without decay, agent gets trapped in local attractors (~31% coverage).

## Backlog
- [x] Critical review of Exp B improvements
- [x] Decide on Exp B scope (incremental 3-palier strategy)
- [x] **Palier 1.1**: Experiment infrastructure (runner, results I/O)
- [x] **Palier 1.2**: metrics.py (bootstrap_ci, compare_conditions)
- [x] **Palier 1.3**: Fix controller exploit branch
- [x] **Palier 1.4**: Implement 7 baselines (+ 44 tests)
- [x] **Palier 1.5**: exp_b_exploration.py runner (smoke tested)
- [x] **Doc review**: Coherence audit + fix master.md, checkpoints.md (CP4 columns), prism_plan
- [x] **Notebook 01**: Improved with M* vs M, V_explore, enriched CP1 diagnostic (13 checks)
- [x] **Notebook 02**: Created experiment tracking notebook (8 sections, CP4 diagnostic)
- [x] **Pre-launch fixes**: 6 bug/perf fixes (SRPosterior double-update, MetaSR p99 cache, Oracle epsilon, sentinel, max_steps, agent_seed)
- [x] **Oracle fix**: bonus changed from `||M-M*||` to `||M-M*||/sqrt(v+1)` — fixes exploration trap
- [x] **Palier 1.6**: Exp B vanilla run complete (100 runs × 8 conds, results in `results/exp_b/`)
- [x] **Palier 1.6b**: CP4 figures + notebook analysis (6 interpretation cells, PRISM vs Count-Bonus analysis, Section 8 rewrite)
- [ ] **Palier 2.1**: SR-Count-Matched baseline + corrected calibration
- [ ] **Palier 2.2**: AUC discovery + guidance index metrics
- [ ] **Palier 3.1**: Custom envs (C6 linear, C8 3x3)
- [ ] **Palier 3.2**: Multi-geometry runs + mixed-effects analysis
