# PRISM — Predictive Representation for Introspective Spatial Metacognition

> First computational test of the neuroscientific thesis of the hippocampal meta-map:
> the successor representation as a unified substrate for cognition and metacognition,
> evaluated with the tools of psychophysics.

---

## 1. Literature Review

### 1.1 Successor representations — foundations

The successor representation formalism was introduced by **Dayan (1993)** as a compromise between model-free learning (efficient but rigid) and model-based learning (flexible but costly). The central idea is the decomposition of the value function V(s) = M · R, where M encodes predicted transitions and R encodes rewards, enabling rapid adaptation when one changes independently of the other.

**Stachenfeld, Botvinick & Gershman (2017, Nature Neuroscience)** reformulated the hippocampus as a "predictive map": CA1 place cells do not encode geodesic position but the transition probability toward future positions. Grid cells in the entorhinal cortex emerge as eigenvectors of the SR matrix — a multi-scale spectral compression. This theory predicts and explains the asymmetric expansion of place fields, clustering, reward sensitivity, and time cells.

**Gershman (2018, J. Neuroscience)** provided a synthesis of the computational logic and neural substrates of the SR, establishing that it does not operate in isolation but interacts with model-based and model-free computations.

**Momennejad, Russek et al. (2017, Nature Human Behaviour)** provided the first behavioral evidence in humans: subjects show sensitivity to reward changes (as predicted by the SR) but insensitivity to transition changes (a unique signature of the SR vs. model-based). Their data show a hybrid SR–MB model.

**Russek, Momennejad et al. (2017, PLoS Comp Bio)** formalized how model-based computations can be built on a foundation of TD learning via the SR, with Dyna-SR extensions that use hippocampal replay to update the matrix M offline.

### 1.2 Beyond physical space — cognitive spaces

**Bellmund et al. (2018, Science)** showed that hippocampal spatial codes operate on abstract "cognitive spaces" — spaces whose dimensions can be weight, social hierarchy, or semantic features.

**Theves, Fernandez & Doeller (2020, J. Neuroscience)** proved that the hippocampus maps conceptual space rather than raw feature space: the hippocampal distance signal selectively reflects conceptually relevant dimensions.

**Stoewer et al. (2023, Scientific Reports)** demonstrated that artificial neural networks learning SRs on semantic spaces (32 animal species) successfully build cognitive maps capturing similarities between concepts.

**Ekman et al. (2023, eLife)** showed that primary visual cortex V1 and the hippocampus represent a predictive map akin to the SR — predictive representations permeate perceptual processing itself.

### 1.3 Hippocampus and metacognition — the meta-map thesis

The most direct link to PRISM comes from the **hippocampal meta-map** thesis proposed by **Ambrogioni & Ólafsdóttir (2023, Trends in Cognitive Sciences)** — "Rethinking the hippocampal cognitive map as a meta-learning computational module": the hippocampus encodes not only maps of familiar environments, but also informational states and sources of information. Cognitive maps would be part of a broader meta-representation that supports exploration and provides a foundation for learning under uncertainty.

**Allen et al. (2017, NeuroImage)** showed through quantitative MRI that metacognitive capacity correlates with the microstructure of the hippocampus and the anterior prefrontal cortex — neuroanatomical confirmation that metacognition and spatial cognition share substrates.

**Qiu et al. (2024, Communications Biology)** confirmed via fMRI that the hippocampus, entorhinal cortex, and orbitofrontal cortex collaborate to learn the structure of abstract multidimensional spaces.

### 1.4 SR and uncertainty — existing work

**Janz et al. (2019, NeurIPS) — Successor Uncertainties.** Combination of successor features with Bayesian linear regression to propagate uncertainty through the temporal structure of the MDP. Uncertainty guides exploration via posterior sampling (PSRL). Surpasses human performance on 38/49 Atari games. This is the closest work to PRISM on the SR + uncertainty axis.

**Machado, Bellemare & Bowling (2020, AAAI) — Count-based exploration with SR.** Use the SR norm as a proxy for state visits, deriving count-based exploration bonuses from the SR structure.

**Flennerhag et al. (2020, DeepMind) — Temporal Difference Uncertainties as Signal for Exploration.** Propose using temporal difference uncertainties as an exploration signal, conceptually close to PRISM's TD error monitoring.

### 1.5 Metacognition in AI — existing frameworks

**Valiente & Pilly (2024, arXiv; 2025, Neural Networks) — MUSE Framework.** Integrates self-assessment and self-regulation into autonomous agents. Two implementations: world model and LLM. Tested in Meta-World and ALFWorld. The most comprehensive framework for computational metacognition, but does not use the SR as substrate.

**Kawato et al. (2021, Biological Cybernetics) — From Internal Models toward Metacognitive AI.** Proposes a computational model of metacognition based on pairs of generative-inverse models with a "responsibility signal" that gates selection and learning. The responsibility signal is conceptually close to PRISM's prediction error monitoring.

**Meta-Cognitive RL (VPES).** A recent framework where a meta-controller monitors the stability of Value Prediction Error Stability (VPES) to regulate the learning rate. Architecturally close to PRISM's meta-SR.

**Steyvers & Peters (2025, Perspectives on Psychological Science).** Survey on metacognition and uncertainty communication in humans and LLMs, identifying confidence calibration as a key metric.

### 1.6 Encompassing framework — the Global Neuronal Workspace (GNW)

The Global Neuronal Workspace of **Dehaene & Changeux (1998, 2011)** is the dominant theory of conscious access: pyramidal neurons with long-range axons (prefrontal, parietal) form a global workspace where information undergoes a non-linear, all-or-none "ignition," making it accessible to all specialized processors. The GNW is a theory of **broadcast** — what enters the workspace becomes conscious.

Two results make the GNW relevant for PRISM:

**The hippocampus is part of the workspace core.** Deco, Vidaurre & Kringelbach (2021, *Nature Human Behaviour*) empirically quantified the "functional rich club" constituting the global workspace across seven tasks + rest. The hippocampus is in the central core, alongside the precuneus, posterior cingulate, and nucleus accumbens. The predictive SR map is therefore not an isolated peripheral process — it directly feeds the global broadcast hub.

**The "predictive global workspace".** Whyte & Smith (2020, *Progress in Neurobiology*) integrate the GNW with Friston's active inference, showing that the workspace can be understood as the site where prediction errors are selected and broadcast. PRISM operates precisely in this prediction error space — upstream of broadcast.

The GNW and PRISM operate at different scales and are not in competition. The GNW describes how metacognitive information becomes **globally accessible**. PRISM describes **where it comes from** — the monitoring of the predictive SR structure within the hippocampal module. The precise positioning is developed in §3.4.

---

## 2. Review of Existing Results and Implementations

### 2.1 What has been demonstrated experimentally

| Result | Authors | Status |
|--------|---------|--------|
| Tabular SR converges in FourRooms | Juliani (2019, tutorial) | ✅ Reproduced, code available |
| Eigenvectors of M → grid-like patterns | Stachenfeld et al. (2017); Chelu (repo) | ✅ Reproduced, code available |
| SR transfer when R changes (M reused) | Juliani (2019); Barreto et al. (2017) | ✅ Reproduced, code available |
| Humans use SR + SR/MB arbitration | Momennejad et al. (2017) | ✅ Data + model available |
| SR + Bayesian uncertainty → exploration | Janz et al. (2019) | ✅ Atari-scale results |
| Count-based exploration via SR norm | Machado et al. (2020) | ✅ AAAI results |
| SF learned from pixels in MiniGrid | Chua et al. (2024) | ✅ Code available |
| Metacognition as RL self-assessment | Valiente & Pilly (2024) | ✅ Meta-World + ALFWorld |

### 2.2 What has NOT been done

| Gap | Why it is a gap | Does PRISM fill it? |
|-----|-----------------|---------------------|
| Uncertainty map iso-structural to the SR | Successor Uncertainties propagates uncertainty but does not build a parallel spatial map | ✅ Main contribution |
| Psychophysical calibration of an SR agent | Nobody has measured the ECE of an SR agent nor produced a reliability diagram | ✅ Exp A protocol |
| Calibrated and continuous "I don't know" signal | MUSE does self-assessment but without formal calibration metrics | ✅ Exp A protocol |
| Computational test of the hippocampal meta-map | The TiCS 2023 thesis is theoretical, never implemented | ✅ Project framing |
| Exploration driven by structural SR uncertainty | Machado (2020) uses the SR norm; Janz (2019) uses the posterior — neither uses a parallel U(s) map | ✅ Exp B protocol |
| Comparison of structural SR uncertainty vs. Bayesian vs. count-based | Each approach has been evaluated in isolation | ✅ Exp B protocol |

### 2.3 Reusable assets

| Asset | Source | Usage in PRISM |
|-------|--------|----------------|
| **MiniGrid** FourRooms | Farama Foundation (NeurIPS 2023) | Base environment — no custom gridworld |
| Tabular SR + visualizations | Juliani (2019) | Starting point for the SR agent |
| Spectral SR decomposition | Chelu (github/temporal_abstraction) | Eigenvector, eigenvalue visualization |
| Hybrid SR/MB model | Russek et al. (2017, github) | Reference for arbitration |
| Simple Successor Features | Chua et al. (2024, github) | Deep SF if future extension |
| RL baselines | Stable-Baselines3 | Q-learning, DQN baselines |

---

## 3. PRISM Positioning

### 3.1 Positioning map

```
    Y-axis: Metacognitive rigor (psychophysical metrics)
    ▲
    │
    │   ┌─────────┐
    │   │  PRISM  │  SR as a natural substrate for metacognition
    │   │         │  ECE calibration, reliability diagrams
    │   │         │  Iso-structural uncertainty map
    │   └─────────┘
    │        ▲
    │        │ brings metacognitive         brings the SR substrate
    │        │ metrics                           │
    │   ┌────┴────┐                    ┌────────┴─────────┐
    │   │  MUSE   │                    │ Succ. Uncertain. │
    │   │         │                    │                  │
    │   │ Self-assessment              │ SR + Bayesian    │
    │   │ Self-regulation              │ for exploration  │
    │   │ (world model / LLM)          │ (posterior samp.)│
    │   └─────────┘                    └──────────────────┘
    │
    ├──────────────────────────────────────────────────────► X-axis: SR grounding
    │
    │   ┌───────────┐          ┌──────────────┐
    │   │ VPES /    │          │ Machado 2020 │
    │   │ Meta-Cog  │          │ Count + SR   │
    │   │ RL        │          │              │
    │   └───────────┘          └──────────────┘
```

### 3.2 Unique contribution

**PRISM is the first project to:**

1. Build an **uncertainty map iso-structural** to the SR — the same formalism for first-order cognition (M: "where am I going?") and metacognition (U: "do I know where I'm going?")

2. Measure the **metacognitive calibration** of an SR agent with the tools of psychophysics (ECE, reliability diagrams, Metacognitive Index) — treating an RL agent as a cognitive psychology subject

3. **Computationally test** the hippocampal meta-map thesis (TiCS, 2023), by showing that the predictive structure of the SR is sufficient to give rise to metacognitive behaviors without an external metacognitive module

### 3.3 What PRISM does NOT claim to do

- Outperform Successor Uncertainties in exploration performance (they operate at Atari scale, PRISM is tabular)
- Replace MUSE as a general metacognition framework (PRISM is specific to the SR substrate)
- Prove that the brain uses the meta-SR (PRISM is a computational test, not a neurobiological validation)
- Model consciousness or conscious access (that is the territory of the GNW, see below)

### 3.4 Positioning relative to the Global Neuronal Workspace (GNW)

The Global Neuronal Workspace of Dehaene-Changeux (1998, 2011) is the dominant theory of conscious access. PRISM and the GNW are not in competition — they operate at different scales.

**PRISM models a specialized processor that feeds the workspace.** The hippocampus is part of the central core of the global workspace (Deco et al., 2021). The predictive SR map and the meta-map U(s) produce signals — prediction errors, uncertainty — that can be broadcast to the workspace. PRISM models the **local computation** that generates these signals. The GNW models how they become **globally accessible**.

| | GNW (Dehaene-Changeux) | PRISM |
|---|---|---|
| Scale | Whole brain | Hippocampal module |
| Key mechanism | Ignition + broadcast | SR prediction error + meta-map |
| Central question | How does information become conscious? | Where does the uncertainty signal come from? |
| Metacognition | Requires access to the workspace | Emerges from the local predictive structure |
| Dynamics | All-or-none (ignition threshold) | Continuous (U(s)) + threshold (change detection) |

**Key point of contact — the detection threshold.** PRISM's **change detection** — when `change_score > θ_change` — has the structure of a GNW ignition threshold: a discrete transition that reorients the agent's strategy. The `θ_change` could be the functional analog of the ignition threshold, local to the hippocampus. Testing whether this threshold exhibits ignition properties (non-linearity, hysteresis) is a future extension outside the scope of v1.

---

## 4. Focused Thesis

### Main hypothesis

> The successor representation provides a **natural** substrate for metacognition:
> an uncertainty map built from SR prediction errors
> (iso-structural to the predictive map itself) produces confidence signals
> that are **better calibrated** than unstructured uncertainty approaches,
> and this supports the neuroscientific thesis of the hippocampal meta-map.

### Testable predictions

**P1 — Calibration.** The confidence signal C(s) derived from the meta-SR is calibrated: high-confidence decisions are correct more often than low-confidence decisions. ECE < 0.15.

**P2 — Iso-structurality.** The uncertainty map U(s) has a spatial structure consistent with the predictive map M: uncertainty boundaries correspond to topological boundaries of the world (doors, unexplored zones, recently modified zones).

**P3 — Structural advantage.** Exploration guided by U(s) (spatially structured) is more efficient than exploration guided by unstructured uncertainty signals (count-based, epsilon-greedy, global variance).

---

## 5. Architecture

### 5.1 Overview

```
┌──────────────────────────────────────────────────────────┐
│              WORLD — MiniGrid FourRooms                  │
│  (Farama Foundation, existing asset)                     │
│  + DynamicsWrapper (to code)                             │
│    - Reward shift                                        │
│    - Door block/open                                     │
│    - Perturbation schedule                               │
└────────────────────────┬─────────────────────────────────┘
                         │ (s, a, r, s')
                         ▼
┌──────────────────────────────────────────────────────────┐
│                  PRISM AGENT                              │
│                                                          │
│  ┌────────────────────────────────────────────────┐      │
│  │  SR Layer — first order                        │      │
│  │  (adapted from Juliani 2019 / Chua et al. 2024)│      │
│  │  M(s,s'): predicted transitions (TD learning)  │      │
│  │  R(s): learned rewards                         │      │
│  │  V(s) = M · R                                  │      │
│  └─────────────────────┬──────────────────────────┘      │
│                        │ δ(s) = || TD error on M ||       │
│                        ▼                                  │
│  ┌────────────────────────────────────────────────┐      │
│  │  Meta-SR Layer — PRISM CONTRIBUTION ★          │      │
│  │  U(s): uncertainty map (sliding δ buffer)      │      │
│  │  C(s): calibrated confidence signal            │      │
│  │  Structural change detection                   │      │
│  │  Iso-structural to M by construction           │      │
│  └─────────────────────┬──────────────────────────┘      │
│                        │ C(s), U(s)                       │
│                        ▼                                  │
│  ┌────────────────────────────────────────────────┐      │
│  │  Controller                                    │      │
│  │  ε_adaptive(s) = f(U(s))                       │      │
│  │  V_explore(s) = V(s) + λ · U(s)               │      │
│  │  "I don't know" signal when C(s) < θ           │      │
│  └────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────┘
```

### 5.2 The world — MiniGrid + DynamicsWrapper

**Base:** `MiniGrid-FourRooms-v0` (Farama Foundation). Modular grid with 4 rooms connected by doors. Standard Gymnasium interface.

**Custom extension — `DynamicsWrapper`:** A Gymnasium wrapper that adds dynamic perturbations on top of any MiniGrid env. This is the only "world" component to code.

```python
class DynamicsWrapper(gymnasium.Wrapper):
    """Adds controlled perturbations to a MiniGrid env."""

    def apply_perturbation(self, ptype: str, **kwargs):
        """Types: 'reward_shift', 'door_block', 'door_open', 'combined'"""

    def set_schedule(self, schedule: PerturbationSchedule):
        """Configurable schedule: periodic, random, triggered."""

    def get_state_index(self, pos: tuple) -> int:
        """Mapping position → state index for the SR matrix."""

    def get_true_transition_matrix(self) -> np.ndarray:
        """Ground truth for validation."""
```

### 5.3 The SR layer — first order

Adapted from existing implementations (Juliani 2019). No contribution here — this is a standard component.

**SR matrix — M ∈ ℝ^(N×N):**

```
M(s, s') = E[ Σ_t γ^t 𝟙(s_t = s') | s_0 = s, π ]
```

**TD(0) update:**

```
δ_M(s) = e(s') + γ · M(s',:) - M(s,:)
M(s,:) ← M(s,:) + α_M · δ_M(s)
```

**Value function:** V(s) = M(s,:) · R

**Parameters:**

| Parameter | Symbol | Default | Role |
|-----------|--------|---------|------|
| Discount factor | γ | 0.95 | SR temporal horizon |
| SR learning rate | α_M | 0.1 | M learning speed |
| R learning rate | α_R | 0.3 | R learning speed |
| Base exploration | ε | 0.1 | Default exploration rate |

### 5.4 The Meta-SR layer — MAIN CONTRIBUTION ★

The founding idea: the uncertainty map has **exactly the same structure** as the predictive map. Same state indexing, same spatial granularity. It is not an external module that observes the SR — it is a **reflection** of the SR.

**Scalar SR prediction error per visit:**

```
δ(s) = || e(s') + γ · M(s',:) - M(s,:) ||₂
```

**Justification for scalar compression.** The full TD error vector δ_vec(s) ∈ ℝ^N contains directional information (toward which states the prediction is poor), but the L2 norm suffices for our main objective: measuring whether the agent knows its map is reliable *at a given state*. The scalar version preserves iso-structurality (one scalar per state, just as M has one row per state) while remaining computationally lightweight. A vectorial extension U(s, s') — which would preserve the full structure — is conceivable but outside the scope of v1. The scalar compression is tested empirically: if the MI (correlation between U(s) and the actual error) is high, the compression does not lose critical information for calibration.

**Sliding error buffer — ΔM_history(s):**

For each state s, a circular buffer of size K (default: 20) of the δ values observed during visits to s.

**Uncertainty map — U(s) ∈ [0, 1]:**

```
U(s) = {
    mean(ΔM_history(s))           if visits(s) ≥ K
    U_max                          if visits(s) = 0
    U_prior · decay^(visits(s))    if 0 < visits(s) < K
}
```

**Confidence signal — C(s) ∈ [0, 1]:**

```
C(s) = 1 - sigmoid(β · (U(s) - θ_C))
```

**Change detection:**

```
change_score = mean(U(s) for s in recently_visited)
change_detected = change_score > θ_change
```

**Adaptive exploration:**

```
ε_adaptive(s) = ε_min + (ε_max - ε_min) · U(s) / U_max
V_explore(s) = V(s) + λ · U(s)
```

**Meta-SR parameters — default values and justification:**

| Parameter | Symbol | Default | Justification |
|-----------|--------|---------|---------------|
| Buffer size | K | 20 | ~5 complete traversals of a FourRooms room. Enough to estimate variance, small enough to detect changes. |
| Uncertainty prior | U_prior | 0.8 | Conservative: an unvisited state is assumed to be highly uncertain. |
| Prior decay | decay | 0.95 | Optimized by sweep A.2 (81 configs). Each visit reduces U by 5%. decay is the dominant parameter: only configs with decay=0.95 achieve ECE < 0.15. |
| Confidence sigmoid slope | β | 5 | Moderate transition around θ_C. Optimized by sweep A.2: β=5 yields better separation of confidence bins than β=10 (less saturation). |
| Confidence threshold | θ_C | 0.3 | Center of the sigmoid C(s). U < 0.3 → high confidence, U > 0.3 → low. Confirmed by sweep A.2. |
| Change threshold | θ_change | 0.5 | Change detection. Validated by ROC analysis in Exp C. |
| Exploration bonus | λ | 0.5 | Relative exploration/exploitation weight in V_explore. |
| Epsilon min | ε_min | 0.01 | Exploration floor even under high confidence. |
| Epsilon max | ε_max | 0.5 | Exploration ceiling under high uncertainty. |

**Sensitivity analysis (Exp A.2 — sweep, A.3 — 100-run validation):** Factorial sweep over {U_prior, decay, β, θ_C} = 81 configs × 10 runs × 3 phases = 2430 measurements. Key results:
- **decay** is the dominant parameter: only configs with decay=0.95 achieve ECE < 0.15 AND MI > 0.4
- Retained config (rank 5/81): U_prior=0.8, **decay=0.95**, **beta=5**, θ_C=0.3 (ECE_med=0.098, MI_med=0.39, CV=0.149)
- Default config (decay=0.85, beta=10): rank 23/81, ECE=0.222 — suboptimal
- 5/81 configs pass both thresholds (ECE < 0.15 AND MI > 0.4), all with decay=0.95
- CP2: GO with reservation (4/5 criteria validated, median MI 0.39 < target 0.4)
- **A.3 validation (100 runs × 5 conditions)**: ECE(PRISM)=0.133 ✅ (< 0.15), MI(PRISM)=0.499 ⚠️ (borderline, CI [0.475, 0.506]). All pairwise ECE comparisons significant (p < 10⁻³³). SR-Count paradox: MI=0.695 > PRISM but ECE=0.34 >> PRISM (correlation ≠ calibration). CP3: GO.

**Key property — iso-structurality:** U is indexed by the same states as M. The predictive map and the uncertainty map can be visually superimposed. High-uncertainty boundaries should correspond to topological boundaries (doors, unexplored zones, perturbed zones). This property is tested in Exp A.

---

## 6. Experimental Protocol

Three deep experiments instead of five shallow ones. Each tests a specific prediction.

### 6.1 Experiment A — Metacognitive calibration (tests P1 + P2)

**Question:** Is the confidence signal C(s) calibrated? Is the map U(s) iso-structural to M?

**Protocol (revised — actual implementation):**

1. **Phase 1 — learning** (200 episodes): stable world, 4 rooms, fixed goal. The agent learns M and builds U.
2. **Phase 2 — perturbation R₁** (200 episodes): reward_shift — the goal is moved to another room. M remains valid, only R changes.
3. **Phase 3 — perturbation R₂** (200 episodes): second reward_shift — the goal is moved again. Tests the stability of calibration against successive perturbations.
4. At each step, the agent emits C(s) — its confidence.

> **Note:** The initial protocol planned 300+100+100 episodes with opening of a 5th room (door_open). The final implementation uses 3×200 episodes with reward_shift only, because FourRooms has no doors (open passages) and reward_shift cleanly isolates the effect on R without modifying M.

**Metrics:**

**Calibration — Expected Calibration Error (ECE):**

```
ECE = Σ_b (|B_b| / N) · |accuracy(B_b) - confidence(B_b)|
```

Predictions are divided into 10 confidence bins. For each bin, mean confidence C(s) is compared to the rate of "reliable predictions." **Operational definition of accuracy:** a prediction is considered reliable when the actual SR error is low, i.e. ||M(s,:) - M*(s,:)||₂ < τ_accuracy, where M* is the true transition matrix. This choice is consistent with what C(s) is intended to predict: not the stochasticity of transitions (which is zero in MiniGrid — the environment is deterministic), but the *reliability of the map M itself*. The threshold τ_accuracy is set at the 50th percentile of ||M - M*|| across all states, so that the baseline accuracy is ~50%. This ensures informative dynamics in the reliability diagram.

**Iso-structurality — Spatial correlation:**

```
ρ = corr(U(s), d(s, frontier))
```

The uncertainty map should correlate with the distance to topological boundaries (doors, unexplored zones). The correlation between U(s) and the actual SR error (ground truth) is also measured:

```
MI = corr(U(s), ||M(s,:) - M*(s,:)||)  where M* is the true transition matrix
```

MI = Metacognitive Index. This is the key metric: does the agent know what it doesn't know?

**Reliability diagram:** a graph of declared confidence vs. observed accuracy, per bin. A curve on the diagonal = perfect calibration.

**Conditions:**

| Condition | Confidence signal | Description |
|-----------|-------------------|-------------|
| **PRISM** | C(s) = f(U(s)), U spatially structured | Our approach |
| SR-Global | Confidence = f(global mean TD error) | Unstructured uncertainty |
| SR-Count | Confidence = f(1/√visits(s)) | Count-based (Machado-like) |
| SR-Bayesian | Posterior on V via linear regression | Successor Uncertainties-like |
| Random-Conf | Random confidence | Floor baseline |

**Success criteria:**
- ECE(PRISM) < 0.15
- MI(PRISM) > 0.5 (moderate to strong correlation)
- ECE(PRISM) < ECE(SR-Global) and ECE(SR-Count) — spatial structure helps calibration
- The reliability diagram shows a clear positive correlation

**Visualizations:**
- Heatmap of M for a few source states (standard SR validation)
- Heatmap of U superimposed on the world — the uncertainty map
- Reliability diagram per condition
- Temporal evolution of U after perturbation (animation or sequence)
- Top-6 eigenvectors of M (standard spectral validation)

### 6.2 Experiment B — Exploration driven by structural uncertainty (tests P3)

**Question:** Is exploration guided by U(s) (spatially structured) more efficient than the alternatives?

**Protocol:**

1. Large MiniGrid world (19×19) with 4+ rooms
2. 4 hidden goals, one per room (the agent does not know them initially)
3. The agent must find the 4 goals as quickly as possible
4. Compare exploration efficiency across strategies

**Conditions:**

| Condition | Exploration strategy | Guiding signal |
|-----------|---------------------|----------------|
| **PRISM** | V_explore = V + λ·U(s) | Structured U map |
| SR-Oracle | V + λ·||M(s,:) - M*(s,:)|| | True error (theoretical ceiling) |
| SR-ε-greedy | Fixed ε = 0.1 | None |
| SR-ε-decay | Decaying ε | None |
| SR-Count-Bonus | V + λ/√visits(s) | Count-based (Machado-like) |
| SR-Norm-Bonus | V + λ/||M(s,:)|| | SR norm (Machado 2020) |
| SR-Posterior | Posterior sampling on V | Bayesian (Janz-like) |
| Random | Uniformly random | Floor baseline |

**SR-Oracle** knows the true errors of M and uses them as a bonus. This is a performance ceiling — no realistic agent can do better. The ratio (PRISM performance - Random) / (Oracle performance - Random) quantifies what fraction of the theoretical gain PRISM captures ("efficiency ratio").

**Metrics:**
- Steps to find the 4 goals (mean over 100 runs)
- Coverage (% of states visited) vs. steps
- Redundancy: revisit / new visit ratio
- Correlation between region visit order and their U(s)
- Efficiency ratio: (steps_Random - steps_PRISM) / (steps_Random - steps_Oracle) — fraction of theoretical gain captured

**Success criterion:** PRISM finds the 4 goals in significantly fewer steps than SR-ε-greedy and SR-Count-Bonus.

**Key differential test:** PRISM vs. SR-Count-Bonus isolates the contribution of structure. Both provide an exploration bonus, but PRISM uses SR prediction error (structured) while Count-Bonus uses visits (unstructured). If PRISM wins, then the predictive structure of the SR contributes something beyond simple counting.

### 6.3 Experiment C — Adaptation to change (tests P1 + P2 in dynamic settings)

**Question:** Does the agent detect changes and adapt its behavior, while maintaining calibrated confidence?

**Protocol:**

1. **Stable phase** (200 episodes): fixed world, the agent masters the environment.
2. **R-type perturbation** (100 episodes): goal moved. M remains valid, R changes.
3. **Re-stabilization** (100 episodes): the agent re-adapts.
4. **M-type perturbation** (100 episodes): door blocked. M becomes invalid, R does not change.
5. **Final re-stabilization** (100 episodes).

This design tests the classic SR prediction (Momennejad 2017): adaptation to reward change should be fast (only R is updated), adaptation to transition change should be slow (the entire matrix must be relearned).

**Quantitative prediction of R/M asymmetry.** For a reward change (goal moved), adaptation requires ~O(1/α_R) episodes to converge — with α_R = 0.3, this gives ~3-5 episodes. For a transition change (door blocked), the rows of M corresponding to the N_affected states whose transitions change must be relearned — this takes ~O(N_affected / α_M) episodes. In FourRooms, blocking a door affects ~8-12 states adjacent to the door; with α_M = 0.1, this gives ~80-120 episodes. The predicted ratio is therefore latency_M / latency_R ≈ 15-40×. If the observed ratio falls significantly outside this range, it would point to a non-SR mechanism (too low → model-based; too high → no M relearning).

**Metrics:**
- **Detection latency**: episodes before `change_detected = true`
- **Adaptation latency**: episodes to recover 80% of pre-perturbation performance
- **Dynamic calibration**: ECE measured in a sliding window of 20 episodes — does calibration hold during and after transitions?
- **R vs. M asymmetry**: latency_M / latency_R ratio — should be >> 1 if the SR is indeed the underlying mechanism

**Conditions:**

| Condition | Description |
|-----------|-------------|
| **PRISM** | Complete agent with meta-SR and detection |
| SR-Blind | SR agent without monitoring (fixed ε) |
| Q-Learning | Classic model-free (Stable-Baselines3) |

**Success criteria:**
- PRISM detects changes in < 10 episodes
- Adaptation latency: PRISM ≤ 0.5 × SR-Blind
- R/M asymmetry observable (confirmation of the SR signature)
- ECE remains < 0.20 even during transitions

### 6.4 Statistical analysis plan

**Number of runs and power.** Each condition is run 100 times with different random seeds (Exp A and C: 100 runs × ~500 episodes; Exp B: 100 runs × variable duration). This number guarantees sufficient statistical power to detect mean effect differences (Cohen's d ≥ 0.5) with α = 0.05.

**Comparison tests (Exp A, B).** The metric distributions (ECE, steps, MI) between conditions are not assumed normal. Pairwise comparisons use the Mann-Whitney U test (one-tailed when the direction is predicted, two-tailed otherwise). Holm-Bonferroni correction is applied for multiple comparisons — only the comparisons pre-specified in the success criteria are tested, no fishing.

**Confidence intervals.** 95% confidence intervals on ECE and MI are computed by non-parametric bootstrap (10,000 resamples). Error bars in figures represent these intervals.

**Calibration tests (Exp A, C).** In addition to ECE, the Hosmer-Lemeshow test is applied to formally assess calibration quality in each condition. A p > 0.05 indicates acceptable calibration.

**Correlations (Exp A — iso-structurality).** Correlations ρ and MI are reported with bootstrap confidence intervals. Significance is assessed by a permutation test (1,000 permutations).

**Effect size.** All comparisons report Cohen's d (or rank r for Mann-Whitney) in addition to the p-value. A statistically significant result with a small effect size (d < 0.3) will be discussed as such.

---

## 7. Technical Stack

### 7.1 Dependencies

```
Python 3.11+
minigrid >= 2.3, < 3.0  # FourRooms environment (Farama)
gymnasium >= 0.29, < 1.0 # standard RL interface
numpy >= 1.24, < 2.0     # pinned < 2.0 for MiniGrid compatibility
scipy >= 1.11            # spectral decomposition
matplotlib >= 3.7
seaborn >= 0.12          # reliability diagrams, heatmaps
pandas >= 2.0            # results logging
tqdm >= 4.65             # progress bars
pytest >= 7.0            # tests
```

> **Note:** `stable-baselines3` is deferred to Phase 3 (Q-learning baseline Exp C).
> A tabular Q-learning fallback is planned if SB3 causes compatibility issues.

### 7.2 Project structure

```
PRISM/
├── master.md                          # ← this document
├── README.md
├── requirements.txt
├── checkpoints.md                     # Human validation protocols CP1-CP5
│
├── prism/                             # Main Python package
│   ├── __init__.py
│   ├── config.py                      # [DONE] Centralized hyperparameters (PRISMConfig)
│   │
│   ├── env/
│   │   ├── __init__.py
│   │   ├── state_mapper.py            # [DONE] MiniGrid position → SR index mapping (260 states)
│   │   ├── dynamics_wrapper.py        # [DONE] Perturbation wrapper on MiniGrid
│   │   ├── exploration_task.py        # [DONE] ExplorationTaskWrapper, place_goals, get_room_cells
│   │   └── perturbation_schedule.py   # [DONE] PerturbationSchedule.exp_a() (reward_shift 3 phases)
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── sr_layer.py                # [DONE] Tabular SR (from Juliani 2019)
│   │   ├── meta_sr.py                 # [DONE] ★ U(s) map, C(s) signal, detection
│   │   ├── controller.py              # [DONE] ★ Adaptive ε, greedy V_explore, "I don't know"
│   │   └── prism_agent.py             # [DONE] ★ Complete agent assembling the layers
│   │
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── base_agent.py              # [DONE] BaseSRAgent + RandomAgent (MiniGrid navigation)
│   │   ├── sr_blind.py                # [DONE] SREpsilonGreedy + SREpsilonDecay
│   │   ├── sr_count.py                # [DONE] SRCountBonus + SRNormBonus
│   │   ├── sr_bayesian.py             # [DONE] SRPosterior (Thompson sampling)
│   │   ├── sr_oracle.py               # [DONE] SROracle (bonus = ||M - M*||)
│   │   ├── calibration_baselines.py   # [DONE] SRGlobalConf, SRCountConf, SRBayesianConf, RandomConfAgent (Exp A)
│   │   └── sb3_baselines.py           # [STUB Phase 3] Q-learning via Stable-Baselines3
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── calibration.py             # [DONE] ★ ECE, reliability diagrams, MI, hosmer_lemeshow_test
│   │   ├── spectral.py                # [DONE] Eigenvectors of M (from Chelu)
│   │   ├── visualization.py           # [DONE] U/M superimposed heatmaps (animation Phase 3)
│   │   ├── metrics.py                 # [DONE] bootstrap_ci, mann_whitney, holm_bonferroni, compare_conditions
│   │   └── results.py                 # [DONE] Run catalog: get_latest_run(), list_runs(), load_run()
│   │
│   └── pedagogy/
│       └── toy_grid.py                # [DONE] Lightweight grid for notebooks (no MiniGrid dep)
│
├── experiments/
│   ├── __init__.py
│   ├── exp_b_exploration.py           # [DONE] Exp B — directed exploration (9 conditions)
│   ├── exp_a_calibration.py           # [DONE] Exp A — calibration runner (5 conditions, CLI config override, fine-grained parallelization)
│   ├── exp_a_sweep.py                 # [DONE] Exp A.2 — sweep 81 configs × n_runs × 3 phases
│   ├── _test_config.py                # [DONE] Config validation: default vs sweep_best (10 runs)
│   ├── exp_c_adaptation.py            # [STUB Phase 3] Exp C — adaptation to change
│   └── run_all.py                     # [STUB Phase 3] Batch script
│
├── notebooks/
│   ├── 00_prism_concepts.ipynb        # [DONE] Pedagogical introduction to PRISM concepts
│   ├── 00a_spectral_deep_dive.ipynb   # [DONE] Spectral analysis of the SR
│   ├── 00b_calibration_methods.ipynb  # [DONE] Calibration methods
│   ├── 01_sr_validation.ipynb         # [DONE] SR validation + CP1 go/no-go (9 sections)
│   ├── 02_experiment_tracking.ipynb   # [DONE] Exp B tracking and analysis
│   └── 03_calibration_tracking.ipynb  # [DONE] Exp A — sweep A.2 + calibration tracking
│
├── tests/                             # 205 tests, 11 files
│   ├── __init__.py
│   ├── conftest.py                    # Shared fixtures (env, mapper)
│   ├── test_imports.py                # Package import verification
│   ├── test_env_smoke.py              # MiniGrid smoke tests
│   ├── test_state_mapper.py           # StateMapper tests (260 states, bijection)
│   ├── test_sr_layer.py               # SR Layer tests (convergence, update)
│   ├── test_dynamics_wrapper.py       # DynamicsWrapper tests (perturbations)
│   ├── test_meta_sr.py                # MetaSR tests (uncertainty, confidence, change)
│   ├── test_calibration.py            # Calibration tests (ECE, MI, reliability, Hosmer-Lemeshow)
│   ├── test_baselines.py              # Baselines + metrics + goal placement tests (44 tests)
│   ├── test_perturbation_schedule.py  # PerturbationSchedule.exp_a() tests (10 tests)
│   └── test_calibration_baselines.py  # 4 calibration baselines Exp A tests (34 tests)
│
├── scripts/
│   └── verify_env.py                  # Python environment verification
│
└── results/                           # Automatically generated (.gitignored)
    ├── exp_a/
    ├── exp_b/
    └── exp_c/
```

├── monitor_p2.py                      # [DONE] Live monitoring: sweep, config test
│
**Legend:**
- ★ = PRISM contribution (original code)
- [DONE] = implemented and tested
- [STUB Phase 3] = file exists with interface, implementation in Phase 3

**Counter:** 205 tests pass, 24 Python modules, 6 notebooks.

> **Note:** Baselines (Exp B.1) and `metrics.py` were implemented in Phase 3.
> The 9th condition `SR-Count-Matched` was added in Exp B.2.
> The 4 calibration baselines (Exp A.1) and `hosmer_lemeshow_test()` were added in Exp A.1.
> The runner `exp_a_calibration.py` was rewritten with fine-grained parallelization and CLI config override.

---

## 8. Implementation Plan

### Phase 1 — Assembly — ✅ DONE

Objective: functional SR agent in MiniGrid, zero original contribution.

- [x] Install MiniGrid, verify FourRooms works
- [x] `state_mapper.py` — MiniGrid position → SR matrix index mapping
- [x] `sr_layer.py` — adapt Juliani's tabular SR implementation
- [x] `dynamics_wrapper.py` — perturbation wrapper (reward shift, door block)
- [x] `spectral.py` — adapt eigenvector visualization code (Chelu)
- [x] Notebook `01_sr_validation.ipynb` — sanity check: SR converges, eigenvectors ok
- [x] Unit tests: wrapper, SR layer, state mapper

> **Implementation notes:**
> - FourRooms = 19×19, **260 accessible states** (not ~100 as anticipated)
> - `agent_pos` is a tuple `(x, y)`, not an int — requires `to_grid()` in StateMapper
> - `max_steps=500` mandatory to avoid silent truncation
> - No doors in MiniGrid v2.5 (open passages between rooms)

**Milestone:** ✅ The SR agent navigates toward the goal in FourRooms. The M heatmaps and eigenvectors are consistent with Stachenfeld 2017.
→ **CP1 PASSED** (6 automated checks in notebook 01)

### Phase 2 — Meta-SR and calibration ★ — ✅ DONE

Objective: implement the main contribution and run Exp A.

- [x] `meta_sr.py` — error buffer, U(s) map, C(s) signal, detection
- [x] `controller.py` — adaptive ε, V_explore, "I don't know" signal
- [x] `prism_agent.py` — complete agent assembly
- [x] `calibration.py` — ECE, reliability diagrams, Metacognitive Index, hosmer_lemeshow_test
- [x] `visualization.py` — U/M superposition, animations
- [x] Baselines: `sr_blind.py`, `sr_count.py`, `sr_bayesian.py` — implemented (Exp B.1 + A.1)
- [x] **Meta-SR hyperparameter sweep** — DONE (Exp A.2, 81 configs × 10 runs)
- [ ] **Run complete Exp A** — deferred to Phase 3 → CP3
- [ ] Notebook `02_meta_sr_demo.ipynb` — deferred to Phase 3

> **Implementation notes:**
> - MetaSR uses adaptive p99 normalization (not naive min-max)
> - MetaSR has an incremental `_U_cache` (updated in `observe()`, O(1) lookup)
> - SRLayer pre-allocates `_e_next` buffer (avoids ~10M `np.zeros()` allocations per full run)
> - The controller's exploit action is random (not V_explore neighbors as planned)
> - `prism_agent.py` integrates `config.py` (PRISMConfig) to centralize hyperparameters
> - `hosmer_lemeshow_test()` added in calibration.py (Exp A.1)

**Milestone:** ✅ Software components implemented and tested (205 tests). Sweep A.2 completed, Exp A.3 to launch.

### Phase 3 — Exploration and adaptation ★ — IN PROGRESS

Objective: run Exp A, B, and C, comparisons with baselines.

**Exp B — Exploration (P3): ✅ DONE (B.1 + B.2)**
- [x] Common infrastructure (runner, results I/O, metrics)
- [x] 8 baselines (Oracle, ε-greedy, ε-decay, Count-Bonus, Norm-Bonus, Posterior, Count-Matched, Random)
- [x] Fix controller exploit branch (greedy on V_explore)
- [x] Exp B.1: 800 runs (8 conds × 100), CP4 diagnostic, notebook 02
- [x] Exp B.2: SR-Count-Matched (9th cond), discovery AUC, guidance index, parallelization, cataloging, 900 runs
- Result: PRISM efficiency 0.60, clusters around Count-Bonus. CP4 GO 5/6.

**Exp A — Calibration (P1+P2): ✅ DONE (A.1+A.2+A.3, CP3 GO)**
- [x] Exp A.1: PerturbationSchedule.exp_a(), hosmer_lemeshow_test(), 4 calibration baselines (SRGlobalConf, SRCountConf, SRBayesianConf, RandomConfAgent), runner, notebook 03 — 205 tests
- [x] Exp A.2: Hyperparameter sweep (81 configs × 10 runs × 3 phases), result: optimal config decay=0.95, beta=5, defaults rank 23/81 → CP2 GO with reservation 4/5
- [x] Exp A.3: Full run (100 runs × 5 conditions, sweep-optimized config, ~18h38). ECE(PRISM)=0.133 ✅, MI(PRISM)=0.499 ⚠️ borderline → CP3 GO

**Exp C — Adaptation (P4+P5): TODO**
- [ ] Exp C.1: Q-Learning baseline + exp_c runner
- [ ] Exp C.2: Run Exp C + analysis → CP5
- [ ] Exp C.3: Cross-analysis A×B×C

**Milestone:** PRISM beats baselines on exploration. The R/M asymmetry confirms the SR signature. Calibration holds in dynamic settings.

### 8.5 Checkpoint system

Human validation at each key stage. Detailed protocols in `checkpoints.md`.

| CP | Name | Phase | Key criteria | Status |
|----|------|-------|--------------|--------|
| CP1 | Basic SR validation | End Phase 1 | ‖ΔM‖ < 0.1, rank > 50%, ECE < 0.30, MI > 0 | ✅ PASSED |
| CP2 | Meta-SR hyperparameters | Phase 3 (sweep) | ECE < 0.15, inter-run stability | ✅ GO with reservation 4/5 (MI 0.39 < target 0.4) |
| CP3 | Metacognitive calibration | Phase 3 (Exp A) | ECE < 0.15, MI > 0.5, PRISM < baselines | ✅ GO (ECE=0.133, MI=0.499 borderline) |
| CP4 | Directed exploration | Phase 3 (Exp B) | PRISM −30% vs ε-greedy, < Count-Bonus | ✅ GO 5/6 (efficiency 0.60, PRISM ≈ Count-Bonus) |
| CP5 | Adaptation to change | Phase 3 (Exp C) | Detection < 10 episodes, asymmetry 15-40× | ⏳ UPCOMING |

---

## 9. Global Metrics

> **Status:** Exp A and B evaluated. Exp C remains to be done.

### Dashboard

| Exp | Metric | Baseline | PRISM target | Result | Tests |
|-----|--------|----------|--------------|--------|-------|
| A | ECE | — | < 0.15 | **0.133 ✅** | P1 |
| A | Metacognitive Index (MI) | — | > 0.5 | **0.499 ⚠️** (CI [0.475, 0.506]) | P2 |
| A | ECE vs. SR-Global | ECE(SR-Global) | ECE(PRISM) < ECE(SR-Global) | **✅** (p < 10⁻³³) | P1 |
| A | ECE vs. SR-Count | ECE(SR-Count) | ECE(PRISM) < ECE(SR-Count) | **✅** (p < 10⁻³³) | P1 |
| B | Steps to 4 goals | SR-ε-greedy | −30% | **−37% ✅** | P3 |
| B | Steps PRISM vs. SR-Count-Bonus | SR-Count-Bonus | PRISM < Count-Bonus | **≈ (p > 0.05)** | P3 (structure) |
| B | Efficiency ratio (PRISM vs. Oracle) | SR-Oracle | > 0.5 (captures >50% of theoretical gain) | **0.60 ✅** | P3 (ceiling) |
| C | Detection latency | SR-Blind | < 10 episodes | ⏳ | P1 |
| C | Adaptation latency PRISM / SR-Blind | SR-Blind | ≤ 0.5× | ⏳ | P2 |
| C | Asymmetry latency_M / latency_R | — | 15–40× (analytically derived) | ⏳ | SR signature |
| C | ECE during transitions | — | < 0.20 | ⏳ | P1 dynamic |

### Cross-cutting metrics

- **Metacognitive Index (MI)** = corr(U(s), actual SR error). Key metric: does the agent know what it doesn't know?
- **Calibration Maintenance** = ECE measured in a sliding window. Does calibration degrade?
- **Structure Advantage** = PRISM gain vs. SR-Count-Bonus. Isolates the contribution of SR structure.

---

## 10. Future Extensions

### Short term (if results are solid)

- **Multi-scale SR**: maintain multiple M with different γ, inspired by the longitudinal axis of the hippocampus. Test whether U maps at different scales capture different types of uncertainty.
- **Replay**: experience replay in offline phases to consolidate M, inspired by hippocampal replay. Test the impact on U stability.
- **SR/MB arbitration**: add a model-based planner and use U(s) for arbitration (Russek et al. 2017). Deferred from v1 but architecturally ready.

### Medium term

- **Deep SR**: replace the tabular matrix with a network (Chua et al. 2024 as starting point). Can the meta-SR work on learned representations?
- **Non-spatial spaces**: apply PRISM to a semantic space (Stoewer et al. 2023) — does SR metacognition work beyond navigation?

### Research

- Compare the spectral structure of the artificial SR + meta-SR with electrophysiological data
- Formalize the meta-SR ↔ variational free energy link (active inference)
- Explore whether the meta-SR is an approximation of Bayesian uncertainty (Successor Uncertainties) and under what conditions

---

## 11. References

### SR foundations

| Ref | Contribution |
|-----|--------------|
| Dayan (1993) — *Neural Computation* | Original SR formalism |
| Stachenfeld et al. (2017) — *Nature Neuroscience* | Hippocampus as predictive map |
| Gershman (2018) — *J. Neuroscience* | SR survey: computational logic and neural substrates |
| Momennejad et al. (2017) — *Nature Human Behaviour* | Behavioral evidence for SR in humans |
| Russek et al. (2017) — *PLoS Comp Bio* | Hybrid SR–MB, replay, Dyna-SR |
| Barreto et al. (2017) — *NeurIPS* | Successor features for transfer |

### Cognitive spaces

| Ref | Contribution |
|-----|--------------|
| Bellmund et al. (2018) — *Science* | Spatial codes for human thought |
| Theves et al. (2020) — *J. Neuroscience* | Hippocampus maps conceptual space |
| Stoewer et al. (2023) — *Scientific Reports* | SR on semantic spaces (artificial NNs) |
| Ekman et al. (2023) — *eLife* | SR in visual cortex |

### Metacognition and hippocampus

| Ref | Contribution |
|-----|--------------|
| Ambrogioni, L. & Ólafsdóttir, H. F. (2023) — *Trends in Cognitive Sciences*, 27(8), 702-712 | PRISM's founding thesis: hippocampal meta-map as a meta-learning module |
| Allen et al. (2017) — *NeuroImage* | Microstructural correlates of metacognition–hippocampus |
| Qiu et al. (2024) — *Communications Biology* | Hippocampus + OFC for abstract spaces |

### SR and uncertainty — direct positioning

| Ref | Contribution | Relation to PRISM |
|-----|--------------|-------------------|
| Janz et al. (2019) — *NeurIPS* | Successor Uncertainties | Bayesian approach — PRISM compares |
| Machado et al. (2020) — *AAAI* | Count-based exploration + SR | SR norm — PRISM uses as baseline |
| Flennerhag et al. (2020) — *DeepMind* | TD uncertainties for exploration | TD signal — PRISM extends into structured map |
| Chua et al. (2024) — *arXiv* | Simple Successor Features | Deep SF from pixels — future extension |

### Metacognition in AI — direct positioning

| Ref | Contribution | Relation to PRISM |
|-----|--------------|-------------------|
| Valiente & Pilly (2024/2025) — MUSE | Self-assessment + self-regulation | General framework — PRISM specific to SR |
| Kawato et al. (2021) — *Biol. Cybernetics* | Internal models → metacognitive AI | Responsibility signal ≈ meta-SR |
| Steyvers & Peters (2025) — *Perspectives Psych. Science* | LLM metacognition + calibration | ECE metrics — PRISM borrows |

### Global Neuronal Workspace — encompassing framework

| Ref | Contribution | Relation to PRISM |
|-----|--------------|-------------------|
| Dehaene, Kerszberg & Changeux (1998) — *PNAS* | Neuronal model of the GNW | Encompassing framework — PRISM = specialized processor |
| Dehaene & Changeux (2011) — *Neuron* | GNW: experimental and theoretical approaches | Ignition, broadcast, thresholds |
| Deco, Vidaurre & Kringelbach (2021) — *Nature Human Behaviour* | Functional rich club = empirical workspace | Hippocampus in the workspace core |
| Whyte & Smith (2020) — *Progress in Neurobiology* | Predictive Global Workspace (GNW + active inference) | Direct bridge: prediction errors in the workspace |

### Technical assets

| Asset | Source | Usage |
|-------|--------|-------|
| MiniGrid | github.com/Farama-Foundation/Minigrid | Environment |
| Tabular SR tutorial | Juliani (2019) | SR agent base |
| Temporal abstraction (spectral) | github.com/veronicachelu/temporal_abstraction | Eigenvector visualization |
| Hybrid SR/MB code | github.com/evanrussek | Arbitration reference |
| Stable-Baselines3 | github.com/DLR-RM/stable-baselines3 | RL baselines |

---

*Last updated: 2026-02-22 — Phases 0-2 DONE, Phase 3 IN PROGRESS (Exp B DONE, Exp A DONE (CP3 GO), Exp C TODO)*
