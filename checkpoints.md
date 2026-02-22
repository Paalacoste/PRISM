# PRISM — Human Checkpoints

> When to look, what to look at, what to run, when to worry.
> Everything else, let Claude run.

---

## Overview

```
Week     1 ──── 2 ──── 3 ──── 4 ──── 5 ──── 6 ──── 7 ──── 8
         │      │      │      │      │      │      │      │
Phase 1  ████████      │      │      │      │      │      │
         │    CP1      │      │      │      │      │      │
Phase 2  │      ████████████████      │      │      │      │
         │      │      │  CP2 │ CP3  │      │      │      │
Phase 3  │      │      │      │      ████████████████      │
         │      │      │      │      │    CP4 │  CP5│      │
                                                    └─ Writing

CP = Human checkpoint
Estimated total human time: ~3 days over 8 weeks
```

---

## Checkpoint 1 — SR Sanity Check (end of week 2)

> **Status:** EXECUTABLE — the script below works with the current code. Alternative: the notebook `notebooks/01_sr_validation.ipynb` covers the same checks + calibration.

**Duration:** 30 min – 1h
**Prerequisites:** Phase 1 complete, `pytest tests/ -v` passes green.

### What Claude has done

- MiniGrid FourRooms runs
- `state_mapper.py` converts positions <-> indices
- `sr_layer.py` learns M via TD(0)
- `dynamics_wrapper.py` can perturb the env
- `spectral.py` computes the eigenvectors of M

### What you verify

**1. Has matrix M converged?**

```python
# checkpoints/cp1_sr_convergence.py
"""
Lance 300 épisodes d'apprentissage SR dans FourRooms.
Affiche : convergence de M, heatmaps, eigenvectors.
Temps d'exécution : ~2 min.
"""
import numpy as np
import matplotlib.pyplot as plt
from prism.env.dynamics_wrapper import DynamicsWrapper
from prism.env.state_mapper import StateMapper
from prism.agent.sr_layer import SRLayer
from prism.analysis.spectral import sr_eigenvectors
import minigrid  # noqa: F401
import gymnasium as gym
import os

os.makedirs("results", exist_ok=True)

# --- Setup ---
env = gym.make("MiniGrid-FourRooms-v0", max_steps=500)
mapper = StateMapper(env)
n_states = mapper.n_states
agent = SRLayer(n_states, gamma=0.95, alpha_M=0.1, alpha_R=0.3)

# --- Train ---
M_snapshots = []
for ep in range(300):
    obs, _ = env.reset()
    s = mapper.get_index(env.agent_pos)
    done = False
    while not done:
        action = env.action_space.sample()  # random policy pour apprentissage latent
        obs, reward, terminated, truncated, _ = env.step(action)
        s_next = mapper.get_index(env.agent_pos)
        agent.update(s, s_next, reward)
        s = s_next
        done = terminated or truncated
    if ep % 50 == 0:
        M_snapshots.append((ep, agent.M.copy()))

# --- Figure 1 : Convergence ---
fig, axes = plt.subplots(1, len(M_snapshots), figsize=(4*len(M_snapshots), 4))
for i, (ep, M) in enumerate(M_snapshots):
    axes[i].imshow(M, cmap="viridis", aspect="auto")
    axes[i].set_title(f"M @ épisode {ep}")
    axes[i].set_xlabel("état s'")
    if i == 0:
        axes[i].set_ylabel("état s")
plt.suptitle("CP1 — Convergence de la matrice SR", fontsize=14)
plt.tight_layout()
plt.savefig("results/cp1_convergence.png", dpi=150)
plt.show()

# --- Figure 2 : Heatmap M pour quelques états sources ---
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
source_states = [0, n_states//4, n_states//2, 3*n_states//4, n_states-1, n_states//3]
grid_shape = mapper.get_grid_shape()  # (rows, cols)

for ax, src in zip(axes.flat, source_states[:6]):
    row_vec = agent.M[src, :]
    heatmap = mapper.to_grid(row_vec)
    im = ax.imshow(heatmap, cmap="hot", interpolation="nearest")
    src_pos = mapper.get_pos(src)
    ax.plot(src_pos[1], src_pos[0], 'c*', markersize=15)
    ax.set_title(f"M({src}, :)")
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle("CP1 — Successor representation depuis différents états", fontsize=14)
plt.tight_layout()
plt.savefig("results/cp1_heatmaps.png", dpi=150)
plt.show()

# --- Figure 3 : Top-6 eigenvectors ---
eigenvalues, eigenvectors = sr_eigenvectors(agent.M, k=6)

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for i, ax in enumerate(axes.flat):
    grid = mapper.to_grid(eigenvectors[:, i])
    im = ax.imshow(grid, cmap="RdBu_r", interpolation="nearest")
    ax.set_title(f"Eigenvector {i+1} (λ={eigenvalues[i]:.3f})")
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle("CP1 — Décomposition spectrale de M (≈ grid cells)", fontsize=14)
plt.tight_layout()
plt.savefig("results/cp1_eigenvectors.png", dpi=150)
plt.show()

# --- Métriques numériques ---
print("\n" + "="*60)
print("CP1 — MÉTRIQUES DE CONVERGENCE")
print("="*60)
M_diff = np.linalg.norm(M_snapshots[-1][1] - M_snapshots[-2][1])
print(f"  ||M_300 - M_250|| = {M_diff:.6f}  (< 0.1 = convergé)")
print(f"  Rang effectif de M : {np.linalg.matrix_rank(agent.M, tol=0.01)}")
print(f"  Nombre d'états : {n_states}")
print(f"  Top-3 eigenvalues : {eigenvalues[:3]}")
diag = np.diag(agent.M)
print(f"  Diagonale M — min: {diag.min():.3f}, max: {diag.max():.3f}, mean: {diag.mean():.3f}")
print(f"  (diag devrait être > 0 partout — c'est P(rester/revenir))")
```

### What you are looking for visually

**Heatmaps of M:** from a source state, the heat should diffuse toward accessible states and decay with distance. Walls should block the diffusion. If heat "passes through" walls -> bug in the state mapper or the wrapper.

**Eigenvectors:** the first 6 should show coherent spatial patterns — low spatial frequencies, separation between rooms. Compare mentally with Figure 3 of Stachenfeld 2017. No need for a perfect match, but the structure should be there. If it looks like noise -> M has not converged or the discount gamma is too low.

### Green light / Red light

| Signal | Green: Go | Red: Stop |
|--------|------|--------|
| `||M_300 - M_250||` | < 0.1 | > 1.0 |
| Heatmaps | Diffusion blocked by walls | Heat passes through walls |
| Eigenvectors | Smooth spatial patterns | Noise or uniform |
| Diagonal of M | > 0 everywhere | Zeros or negatives |

### If it does not pass

- Heatmaps pass through walls -> check `state_mapper.py`, walls may not be encoded as inaccessible states
- M does not converge -> increase the number of episodes to 500, or lower alpha_M to 0.05
- Noisy eigenvectors -> increase gamma toward 0.98 (horizon too short)

---

## Checkpoint 2 — Meta-SR Hyperparameter Sweep (mid-week 4)

> **Status:** TEMPLATE — the sweep will be run in Phase 3. The script below will be executable once `experiments/exp_a_calibration.py` is implemented.

**Duration:** 1-2h
**Prerequisites:** `meta_sr.py` and `calibration.py` functional.

### What Claude has done

- 81 configurations tested (4 params x 3 values x 10 runs)
- Table sorted by ECE on the stable learning phase
- Complete CSV file in `results/sweep/`

### What you verify

```python
# checkpoints/cp2_sweep_review.py
"""
Charge les résultats du sweep et affiche les diagnostics.
Temps d'exécution : ~10 sec (analyse uniquement, pas de simulation).
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Charger les résultats ---
df = pd.read_csv("results/sweep/sweep_results.csv")
# Colonnes attendues : U_prior, decay, beta, theta_C, run, ECE, MI

print("="*60)
print("CP2 — RÉSULTATS DU SWEEP HYPERPARAMÈTRES")
print("="*60)

# --- Top 10 configs par ECE médian ---
summary = df.groupby(["U_prior", "decay", "beta", "theta_C"]).agg(
    ECE_median=("ECE", "median"),
    ECE_std=("ECE", "std"),
    MI_median=("MI", "median"),
    MI_std=("MI", "std"),
).reset_index().sort_values("ECE_median")

print("\nTop 10 configs (ECE médian le plus bas) :")
print(summary.head(10).to_string(index=False))

best = summary.iloc[0]
print(f"\n★ Meilleure config :")
print(f"  U_prior = {best.U_prior}")
print(f"  decay   = {best.decay}")
print(f"  β       = {best.beta}")
print(f"  θ_C     = {best.theta_C}")
print(f"  ECE     = {best.ECE_median:.4f} ± {best.ECE_std:.4f}")
print(f"  MI      = {best.MI_median:.4f} ± {best.MI_std:.4f}")

# --- ALERTES DE SENS PHYSIQUE ---
print("\n" + "-"*60)
print("ALERTES DE SENS PHYSIQUE :")
alerts = []
if best.beta > 30:
    alerts.append(f"⚠️  β = {best.beta} → sigmoïde quasi-binaire, pas de zone de transition graduelle")
if best.beta < 3:
    alerts.append(f"⚠️  β = {best.beta} → sigmoïde trop plate, confiance presque uniforme")
if best.decay > 0.97:
    alerts.append(f"⚠️  decay = {best.decay} → le prior met ~100 visites à disparaître (trop lent)")
if best.decay < 0.5:
    alerts.append(f"⚠️  decay = {best.decay} → le prior disparaît en ~2 visites (trop rapide)")
if best.U_prior < 0.3:
    alerts.append(f"⚠️  U_prior = {best.U_prior} → états non visités considérés peu incertains (dangereux)")
if best.theta_C > 0.7:
    alerts.append(f"⚠️  θ_C = {best.theta_C} → presque tout est classé 'haute confiance'")

if alerts:
    for a in alerts:
        print(f"  {a}")
    print("  → Considère la 2e ou 3e meilleure config si les alertes sont sérieuses")
else:
    print("  ✅ Aucune alerte — les paramètres sont dans des plages raisonnables")

# --- Figure 1 : Sensibilité par paramètre ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
params = ["U_prior", "decay", "beta", "theta_C"]
for ax, param in zip(axes.flat, params):
    param_summary = df.groupby(param)["ECE"].agg(["median", "std"]).reset_index()
    ax.errorbar(param_summary[param], param_summary["median"],
                yerr=param_summary["std"], marker="o", capsize=4, color="#2E75B6")
    ax.set_xlabel(param, fontsize=12)
    ax.set_ylabel("ECE", fontsize=12)
    ax.set_title(f"Sensibilité à {param}")
    ax.axhline(y=0.15, color="red", linestyle="--", alpha=0.5, label="seuil 0.15")
    ax.legend()

plt.suptitle("CP2 — Sensibilité de l'ECE aux hyperparamètres méta-SR", fontsize=14)
plt.tight_layout()
plt.savefig("results/cp2_sensitivity.png", dpi=150)
plt.show()

# --- Figure 2 : ECE vs MI pour toutes les configs ---
plt.figure(figsize=(8, 6))
scatter = plt.scatter(summary.ECE_median, summary.MI_median,
                      c=summary.beta, cmap="viridis", s=60, alpha=0.7)
plt.colorbar(scatter, label="β")
plt.axvline(x=0.15, color="red", linestyle="--", alpha=0.5, label="ECE cible < 0.15")
plt.axhline(y=0.5, color="green", linestyle="--", alpha=0.5, label="MI cible > 0.5")
plt.xlabel("ECE médian (↓ meilleur)")
plt.ylabel("MI médian (↑ meilleur)")
plt.title("CP2 — Espace ECE × MI — chaque point = une config")
plt.legend()
plt.tight_layout()
plt.savefig("results/cp2_ece_vs_mi.png", dpi=150)
plt.show()

# --- Stabilité : la meilleure config est-elle robuste ? ---
best_runs = df[
    (df.U_prior == best.U_prior) & (df.decay == best.decay) &
    (df.beta == best.beta) & (df.theta_C == best.theta_C)
]
print(f"\nStabilité de la meilleure config (10 runs) :")
print(f"  ECE : {best_runs.ECE.values}")
print(f"  Coefficient de variation ECE : {best_runs.ECE.std()/best_runs.ECE.mean():.2f}")
print(f"  (< 0.3 = stable, > 0.5 = inquiétant)")
```

### What you are looking for

**Sensitivity:** if the ECE changes drastically for a small variation of a parameter, it signals that the result will be fragile. Ideally, there should be a "plateau" around the best configuration.

**ECE <-> MI consistency:** the best configurations in ECE should also be good in MI. If they diverge (good ECE, bad MI), the confidence is calibrated but does not reflect the true error — the mechanism is suspect.

**Stability:** the coefficient of variation over 10 runs must be < 0.3. If the same configuration yields ECE = 0.08 on one run and ECE = 0.35 on another, it is unusable.

### Green light / Red light

| Signal | Green: Go | Red: Stop |
|--------|------|--------|
| Best ECE | < 0.15 | > 0.25 |
| Associated MI | > 0.4 | < 0.2 |
| Physical alerts | 0 | >= 2 serious alerts |
| Stability CV | < 0.3 | > 0.5 |
| ECE<->MI correlation | Concordant | Divergent |

### If it does not pass

- ECE systematically > 0.20 -> the scalar compression mechanism may be losing too much information. Test a U(s) based on buffer variance rather than the mean.
- MI weak everywhere -> the TD errors do not reflect the true error of M. Check that `get_true_transition_matrix()` is correctly implemented.
- Instability -> increase K (buffer size) to 30 or 40 to smooth the signal.

---

## Checkpoint 3 — Exp A Results (end of week 5)

> **Status:** TEMPLATE — will be executable after Exp A implementation (Phase 3). The functions `hosmer_lemeshow_test` and `bootstrap_ci` are to be created in `prism/analysis/calibration.py`.

**Duration:** half a day
**Prerequisites:** Exp A complete, 100 runs x 5 conditions.
**This is the most important checkpoint. If Exp A does not hold, the project has a fundamental problem.**

### What Claude has done

- 500 runs total (100 runs x 5 conditions)
- ECE, MI, reliability diagrams, U heatmaps, statistical tests
- Results in `results/exp_a/`

### What you verify

```python
# checkpoints/cp3_exp_a_review.py
"""
Revue complète de l'Exp A — calibration et iso-structuralité.
Temps d'exécution : ~30 sec (analyse uniquement).

⚠️ ATTENTION : Ce script est un TEMPLATE écrit avant l'implémentation d'Exp A.
   Incompatibilités connues à corriger avant exécution :
   - hosmer_lemeshow_test() n'existe pas encore dans calibration.py
   - bootstrap_ci() est dans metrics.py, pas calibration.py
   - expected_calibration_error() et reliability_diagram_data() acceptent
     (confidences, accuracies), pas un DataFrame
   - plot_uncertainty_map() accepte (U, state_mapper, grid_shape), pas des paths
   Ces issues seront corrigées lors de l'implémentation d'Exp A (Phase 3).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy import stats
from prism.analysis.calibration import (
    expected_calibration_error, reliability_diagram_data,
    sr_errors, sr_accuracies, metacognitive_index,
)
# À créer en Phase 3 :
# from prism.analysis.calibration import hosmer_lemeshow_test, bootstrap_ci
from prism.analysis.visualization import plot_uncertainty_map

# --- Charger les résultats ---
results = {}
conditions = ["PRISM", "SR-Global", "SR-Count", "SR-Bayesian", "Random-Conf"]
for cond in conditions:
    results[cond] = pd.read_csv(f"results/exp_a/{cond}_results.csv")

# ═══════════════════════════════════════════
# FIGURE MAÎTRESSE — 6 panneaux
# ═══════════════════════════════════════════
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

# --- Panel A : Reliability diagram (toutes conditions) ---
ax_rel = fig.add_subplot(gs[0, 0])
colors = {"PRISM": "#E87722", "SR-Global": "#2E75B6", "SR-Count": "#888888",
          "SR-Bayesian": "#5B9E3A", "Random-Conf": "#CCCCCC"}

for cond in conditions:
    bins_conf, bins_acc, bins_count = reliability_diagram_data(results[cond])
    mask = bins_count > 5  # ne tracer que les bins peuplés
    ax_rel.plot(bins_conf[mask], bins_acc[mask], "o-", color=colors[cond],
                label=cond, linewidth=2, markersize=6)

ax_rel.plot([0, 1], [0, 1], "k--", alpha=0.3, label="calibration parfaite")
ax_rel.set_xlabel("Confiance déclarée C(s)")
ax_rel.set_ylabel("Accuracy observée")
ax_rel.set_title("A — Reliability diagram")
ax_rel.legend(fontsize=8)
ax_rel.set_xlim(0, 1)
ax_rel.set_ylim(0, 1)

# --- Panel B : Barplot ECE par condition ---
ax_ece = fig.add_subplot(gs[0, 1])
ece_values = {}
ece_cis = {}
for cond in conditions:
    eces = [expected_calibration_error(results[cond][results[cond].run == r])
            for r in results[cond].run.unique()]
    ece_values[cond] = np.median(eces)
    ece_cis[cond] = bootstrap_ci(eces)

x_pos = range(len(conditions))
bars = ax_ece.bar(x_pos, [ece_values[c] for c in conditions],
                  color=[colors[c] for c in conditions], alpha=0.8)
for i, cond in enumerate(conditions):
    lo, hi = ece_cis[cond]
    ax_ece.errorbar(i, ece_values[cond], yerr=[[ece_values[cond]-lo], [hi-ece_values[cond]]],
                    color="black", capsize=4)
ax_ece.axhline(y=0.15, color="red", linestyle="--", alpha=0.5, label="seuil 0.15")
ax_ece.set_xticks(x_pos)
ax_ece.set_xticklabels([c.replace("SR-", "") for c in conditions], rotation=30, ha="right")
ax_ece.set_ylabel("ECE (↓ meilleur)")
ax_ece.set_title("B — Expected Calibration Error")
ax_ece.legend()

# --- Panel C : Barplot MI par condition ---
ax_mi = fig.add_subplot(gs[0, 2])
mi_values = {}
for cond in conditions:
    mis = [results[cond][results[cond].run == r][["U_s", "true_error"]].corr().iloc[0,1]
           for r in results[cond].run.unique()]
    mi_values[cond] = np.median(mis)

ax_mi.bar(x_pos, [mi_values[c] for c in conditions],
          color=[colors[c] for c in conditions], alpha=0.8)
ax_mi.axhline(y=0.5, color="green", linestyle="--", alpha=0.5, label="seuil 0.5")
ax_mi.set_xticks(x_pos)
ax_mi.set_xticklabels([c.replace("SR-", "") for c in conditions], rotation=30, ha="right")
ax_mi.set_ylabel("Metacognitive Index (↑ meilleur)")
ax_mi.set_title("C — MI = corr(U(s), true error)")
ax_mi.legend()

# --- Panel D : Heatmap U(s) superposée au monde (phase apprentissage) ---
ax_u1 = fig.add_subplot(gs[1, 0])
plot_uncertainty_map(ax_u1, "results/exp_a/PRISM_U_phase1.npy",
                     "results/exp_a/world_layout.npy",
                     title="D — U map after learning (300 eps)")

# --- Panel E : Heatmap U(s) après ouverture 5e pièce ---
ax_u2 = fig.add_subplot(gs[1, 1])
plot_uncertainty_map(ax_u2, "results/exp_a/PRISM_U_phase2.npy",
                     "results/exp_a/world_layout_expanded.npy",
                     title="E — U map after opening new zone")

# --- Panel F : Évolution temporelle U(s) dans la nouvelle zone ---
ax_time = fig.add_subplot(gs[1, 2])
u_timeline = np.load("results/exp_a/PRISM_U_timeline_newzone.npy")
# shape : (n_episodes, n_states_new_zone)
mean_u = u_timeline.mean(axis=1)
std_u = u_timeline.std(axis=1)
episodes = np.arange(len(mean_u))
ax_time.fill_between(episodes, mean_u - std_u, mean_u + std_u, alpha=0.2, color="#E87722")
ax_time.plot(episodes, mean_u, color="#E87722", linewidth=2)
ax_time.axvline(x=0, color="red", linestyle="--", alpha=0.5, label="ouverture zone")
ax_time.set_xlabel("Épisodes après ouverture")
ax_time.set_ylabel("U(s) moyen dans nouvelle zone")
ax_time.set_title("F — Décroissance de l'incertitude")
ax_time.legend()

plt.suptitle("CHECKPOINT 3 — Exp A : Calibration métacognitive", fontsize=16, y=1.01)
plt.savefig("results/cp3_exp_a_master.png", dpi=150, bbox_inches="tight")
plt.show()

# ═══════════════════════════════════════════
# TESTS STATISTIQUES
# ═══════════════════════════════════════════
print("\n" + "="*60)
print("CP3 — TESTS STATISTIQUES")
print("="*60)

# ECE : PRISM vs chaque baseline
prism_eces = [expected_calibration_error(results["PRISM"][results["PRISM"].run == r])
              for r in results["PRISM"].run.unique()]

for cond in ["SR-Global", "SR-Count", "SR-Bayesian"]:
    other_eces = [expected_calibration_error(results[cond][results[cond].run == r])
                  for r in results[cond].run.unique()]
    stat, p = stats.mannwhitneyu(prism_eces, other_eces, alternative="less")
    d = (np.mean(other_eces) - np.mean(prism_eces)) / np.sqrt(
        (np.std(prism_eces)**2 + np.std(other_eces)**2) / 2)
    print(f"\n  PRISM vs {cond} (ECE) :")
    print(f"    Mann-Whitney U : p = {p:.6f} {'✅' if p < 0.05 else '❌'}")
    print(f"    Taille d'effet (Cohen's d) : {d:.3f}")
    print(f"    ECE médian : PRISM={np.median(prism_eces):.4f} vs {cond}={np.median(other_eces):.4f}")

# Hosmer-Lemeshow
for cond in ["PRISM", "SR-Global", "SR-Count"]:
    hl_stat, hl_p = hosmer_lemeshow_test(results[cond])
    print(f"\n  Hosmer-Lemeshow ({cond}) : χ²={hl_stat:.2f}, p={hl_p:.4f} "
          f"{'✅ calibré' if hl_p > 0.05 else '❌ mal calibré'}")

# ═══════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════
print("\n" + "="*60)
print("CP3 — VERDICT")
print("="*60)
ece_ok = np.median(prism_eces) < 0.15
mi_ok = mi_values["PRISM"] > 0.5
beats_global = ece_values["PRISM"] < ece_values["SR-Global"]
beats_count = ece_values["PRISM"] < ece_values["SR-Count"]

checks = [
    ("ECE(PRISM) < 0.15", ece_ok),
    ("MI(PRISM) > 0.5", mi_ok),
    ("ECE(PRISM) < ECE(SR-Global)", beats_global),
    ("ECE(PRISM) < ECE(SR-Count)", beats_count),
]
for label, passed in checks:
    print(f"  {'✅' if passed else '❌'} {label}")

all_pass = all(p for _, p in checks)
print(f"\n  {'🟢 GO — passer à Phase 3' if all_pass else '🔴 STOP — diagnostic nécessaire'}")
```

### What you are looking for visually

**Reliability diagram (Panel A):** the PRISM curve (orange) should be the closest to the dashed diagonal. If SR-Bayesian is as good or better, that is an interesting result but it weakens the thesis — note it. Check that the bins are populated: markers without points = artifact.

**U map (Panels D-E):** after opening the 5th room, the new zone should be bright red (high uncertainty). Already-learned rooms should be green/blue. Doors should show slight residual uncertainty (they are transition points). If everything is uniformly yellow, U has no spatial structure.

**Decay (Panel F):** the uncertainty in the new zone should decrease progressively over 20-50 episodes. If it drops to zero in 2 episodes, the buffer is too short. If it does not decrease after 100 episodes, learning is not happening.

### If it does not pass

- ECE > 0.15 but MI > 0.5 -> the confidence contains information but the transformation C(s) = sigmoid(U(s)) is poorly calibrated. Adjust beta and theta_C.
- MI < 0.3 -> fundamental problem: U(s) does not correlate with the true error of M. Check that TD errors are computed correctly. Possibly the L2 norm loses too much information -> test buffer variance instead of the mean.
- SR-Bayesian beats PRISM on everything -> negative result for the thesis, but honest. Document and discuss.

---

## Checkpoint 4 — Exp B Results (mid-week 7)

> **Status:** GO 5/6 — Exp B.1 + B.2 complete (9 conditions x 100 runs).
> Results in `results/exp_b/run_YYYYMMDD_HHMMSS/` (runs cataloged with `run_info.json`).

**Duration:** 1-2h
**Prerequisites:** Exp B complete, 100 runs x 9 conditions.

### What Claude has done

- Large 19x19 world with 4 hidden goals
- 8 conditions compared
- Results in `results/exp_b/`

### What you verify

```python
# checkpoints/cp4_exp_b_review.py
"""
Revue Exp B — exploration dirigée par incertitude.
Temps d'exécution : ~20 sec.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# --- Charger ---
from prism.analysis.results import get_latest_run
run = get_latest_run("exp_b")
df = pd.read_csv(run["path"] / "exploration_results.csv")
# Colonnes : condition, run, steps, goals_found, all_found, coverage, redundancy,
#             discovery_1, discovery_2, discovery_3, discovery_4

conditions_order = ["SR-Oracle", "PRISM", "SR-Posterior", "SR-Count-Bonus",
                    "SR-Norm-Bonus", "SR-e-greedy", "SR-e-decay", "Random"]
colors = {"SR-Oracle": "#2E75B6", "PRISM": "#E87722", "SR-Posterior": "#5B9E3A",
          "SR-Count-Bonus": "#888888", "SR-Norm-Bonus": "#AAAAAA",
          "SR-e-greedy": "#CCCCCC", "SR-e-decay": "#DDDDDD", "Random": "#EEEEEE"}

# --- Figure 1 : Boxplot steps ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A : Steps pour trouver les 4 goals
data_steps = [df[df.condition == c].steps.values for c in conditions_order]
bp = axes[0].boxplot(data_steps, labels=[c.replace("SR-", "") for c in conditions_order],
                     patch_artist=True, showfliers=False)
for patch, cond in zip(bp["boxes"], conditions_order):
    patch.set_facecolor(colors[cond])
axes[0].set_ylabel("Steps pour 4 goals (↓ meilleur)")
axes[0].set_title("A — Efficacité d'exploration")
axes[0].tick_params(axis="x", rotation=35)

# Panel B : Couverture finale
data_cov = [df[df.condition == c].coverage.astype(float).values for c in conditions_order]
bp2 = axes[1].boxplot(data_cov, labels=[c.replace("SR-", "") for c in conditions_order],
                      patch_artist=True, showfliers=False)
for patch, cond in zip(bp2["boxes"], conditions_order):
    patch.set_facecolor(colors[cond])
axes[1].set_ylabel("Couverture finale (↑ meilleur)")
axes[1].set_title("B — Couverture")
axes[1].tick_params(axis="x", rotation=35)

# Panel C : Efficiency ratio
prism_steps = df[df.condition == "PRISM"].steps.median()
oracle_steps = df[df.condition == "SR-Oracle"].steps.median()
random_steps = df[df.condition == "Random"].steps.median()
eff_ratio = (random_steps - prism_steps) / (random_steps - oracle_steps)

for cond in conditions_order:
    med = df[df.condition == cond].steps.median()
    r = (random_steps - med) / (random_steps - oracle_steps)
    axes[2].barh(cond.replace("SR-", ""), r, color=colors[cond], alpha=0.8)
axes[2].axvline(x=0.5, color="green", linestyle="--", alpha=0.5)
axes[2].axvline(x=1.0, color="blue", linestyle="--", alpha=0.3)
axes[2].set_xlabel("Efficiency ratio (1.0 = Oracle)")
axes[2].set_title("C — Fraction du gain théorique capturé")

plt.suptitle("CHECKPOINT 4 — Exp B : Exploration dirigée", fontsize=14)
plt.tight_layout()
plt.savefig("results/cp4_exp_b.png", dpi=150)
plt.show()

# --- Tests statistiques ---
print("\n" + "="*60)
print("CP4 — TESTS STATISTIQUES")
print("="*60)

prism_data = df[df.condition == "PRISM"].steps.values
for cond in ["SR-e-greedy", "SR-Count-Bonus", "SR-Norm-Bonus"]:
    other = df[df.condition == cond].steps.values
    stat, p = stats.mannwhitneyu(prism_data, other, alternative="less")
    improvement = 1 - np.median(prism_data) / np.median(other)
    print(f"\n  PRISM vs {cond} :")
    print(f"    Mann-Whitney p = {p:.6f} {'✅' if p < 0.05 else '❌'}")
    print(f"    Amélioration médiane : {improvement*100:.1f}%")

print(f"\n  Efficiency ratio PRISM : {eff_ratio:.3f} "
      f"{'✅' if eff_ratio > 0.5 else '⚠️'} (cible > 0.5)")

# --- Trajectoires individuelles à inspecter ---
print("\n" + "-"*60)
print("TRAJECTOIRES À INSPECTER VISUELLEMENT :")
print("(vérifie que les trajectoires ont du sens spatial)")
prism_runs = df[df.condition == "PRISM"].sort_values("steps")
print(f"  Meilleur run  : run #{prism_runs.iloc[0].run:.0f} ({prism_runs.iloc[0].steps:.0f} steps)")
print(f"  Run médian    : run #{prism_runs.iloc[len(prism_runs)//2].run:.0f}")
print(f"  Pire run      : run #{prism_runs.iloc[-1].run:.0f} ({prism_runs.iloc[-1].steps:.0f} steps)")
print(f"\n  Lance : python checkpoints/cp4_show_trajectory.py --run <N>")
```

```python
# checkpoints/cp4_show_trajectory.py
"""
Visualise la trajectoire d'un run spécifique.
Usage : python checkpoints/cp4_show_trajectory.py --run 42
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from prism.env.state_mapper import StateMapper

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=int, required=True)
parser.add_argument("--condition", default="PRISM")
args = parser.parse_args()

traj = np.load(f"results/exp_b/trajectories/{args.condition}_run{args.run}.npy")
world = np.load("results/exp_b/world_layout_large.npy")
goals = np.load("results/exp_b/goal_positions.npy")

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(world, cmap="gray_r", alpha=0.3)

# Trajectoire colorée par temps
n = len(traj)
for i in range(n - 1):
    color = plt.cm.plasma(i / n)
    ax.plot([traj[i, 1], traj[i+1, 1]], [traj[i, 0], traj[i+1, 0]],
            color=color, linewidth=0.5, alpha=0.6)

# Goals
for g in goals:
    ax.plot(g[1], g[0], "g*", markersize=20)

# Départ
ax.plot(traj[0, 1], traj[0, 0], "rs", markersize=12, label="départ")

ax.set_title(f"{args.condition} — Run #{args.run} ({n} steps)\n"
             f"Couleur : violet=début → jaune=fin")
ax.legend()
plt.tight_layout()
plt.savefig(f"results/cp4_trajectory_{args.condition}_{args.run}.png", dpi=150)
plt.show()
```

### What you are looking for

**Boxplots:** PRISM should be clearly to the left (fewer steps) than epsilon-greedy and Count-Bonus. If PRISM is in the same cluster, the structure does not help.

**Trajectories:** the best PRISM run should show a systematic exploration pattern — visit one room, then head to the next. The worst run may be chaotic, but the median run should be coherent. If even the best run zigzags randomly -> the U map does not actually guide behavior.

**Efficiency ratio:** > 0.5 means that PRISM captures more than half of the gain that the oracle (which knows the true errors) achieves. This is ambitious but attainable.

---

## Checkpoint 5 — Exp C Results + Synthesis (end of week 7)

> **Status:** TEMPLATE — will be executable after Exp C implementation (Phase 3).

**Duration:** 2-3h
**Prerequisites:** Exp C complete.

### What you verify

```python
# checkpoints/cp5_exp_c_review.py
"""
Revue Exp C — adaptation au changement + asymétrie R/M.
Temps d'exécution : ~20 sec.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/exp_c/adaptation_results.csv")
# Colonnes : condition, run, phase, episode, reward, U_mean,
#            change_detected, ECE_window

# --- Phase timeline ---
phase_boundaries = {
    "stable": (0, 200),
    "perturb_R": (200, 300),
    "restab_1": (300, 400),
    "perturb_M": (400, 500),
    "restab_2": (500, 600),
}

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

for cond, color in [("PRISM", "#E87722"), ("SR-Blind", "#2E75B6"), ("Q-Learning", "#888888")]:
    cond_df = df[df.condition == cond].groupby("episode").agg(
        reward_mean=("reward", "mean"),
        reward_std=("reward", "std"),
        U_mean=("U_mean", "mean"),
        ECE_mean=("ECE_window", "mean"),
    ).reset_index()

    ep = cond_df.episode

    # Panel A : Performance (reward)
    axes[0].plot(ep, cond_df.reward_mean, color=color, label=cond, linewidth=1.5)
    axes[0].fill_between(ep, cond_df.reward_mean - cond_df.reward_std,
                         cond_df.reward_mean + cond_df.reward_std, alpha=0.1, color=color)

    # Panel B : Incertitude moyenne
    if "U_mean" in cond_df.columns and cond != "Q-Learning":
        axes[1].plot(ep, cond_df.U_mean, color=color, label=cond, linewidth=1.5)

    # Panel C : ECE glissant
    axes[2].plot(ep, cond_df.ECE_mean, color=color, label=cond, linewidth=1.5)

# Phases
for ax in axes:
    for phase, (start, end) in phase_boundaries.items():
        if "perturb" in phase:
            ax.axvspan(start, end, alpha=0.1, color="red")
    ax.axvline(200, color="red", linestyle="--", alpha=0.3)
    ax.axvline(400, color="red", linestyle="--", alpha=0.3)

axes[0].set_ylabel("Reward moyen")
axes[0].set_title("A — Performance")
axes[0].legend()
axes[1].set_ylabel("U(s) moyen")
axes[1].set_title("B — Incertitude")
axes[1].legend()
axes[2].set_ylabel("ECE (fenêtre 20 éps)")
axes[2].set_title("C — Calibration dynamique")
axes[2].axhline(y=0.20, color="red", linestyle=":", alpha=0.5, label="seuil 0.20")
axes[2].legend()
axes[2].set_xlabel("Épisode")

plt.suptitle("CHECKPOINT 5 — Exp C : Adaptation au changement", fontsize=14)
plt.tight_layout()
plt.savefig("results/cp5_exp_c.png", dpi=150)
plt.show()

# --- Asymétrie R/M ---
print("\n" + "="*60)
print("CP5 — ASYMÉTRIE R/M")
print("="*60)

prism_df = df[df.condition == "PRISM"]

# Latence adaptation R (épisodes 200-300)
pre_perf = prism_df[(prism_df.episode >= 180) & (prism_df.episode < 200)].reward.mean()
threshold_R = 0.8 * pre_perf
post_R = prism_df[(prism_df.episode >= 200) & (prism_df.episode < 300)]
for ep in post_R.episode.unique():
    ep_perf = post_R[post_R.episode == ep].reward.mean()
    if ep_perf >= threshold_R:
        latence_R = ep - 200
        break
else:
    latence_R = 100  # pas convergé

# Latence adaptation M (épisodes 400-500)
pre_perf_M = prism_df[(prism_df.episode >= 380) & (prism_df.episode < 400)].reward.mean()
threshold_M = 0.8 * pre_perf_M
post_M = prism_df[(prism_df.episode >= 400) & (prism_df.episode < 600)]
for ep in post_M.episode.unique():
    ep_perf = post_M[post_M.episode == ep].reward.mean()
    if ep_perf >= threshold_M:
        latence_M = ep - 400
        break
else:
    latence_M = 200

ratio = latence_M / max(latence_R, 1)
print(f"  Latence adaptation R : {latence_R} épisodes")
print(f"  Latence adaptation M : {latence_M} épisodes")
print(f"  Ratio M/R : {ratio:.1f}×")
print(f"  Plage prédite : 15–40×")
if 10 < ratio < 60:
    print(f"  ✅ Compatible avec la signature SR")
elif ratio < 5:
    print(f"  ⚠️  Ratio trop bas — possiblement model-based plutôt que SR")
else:
    print(f"  ⚠️  Ratio trop haut — M ne se réapprend peut-être pas correctement")

# --- Latence de détection ---
detect_R = prism_df[(prism_df.episode >= 200) & (prism_df.change_detected == True)].episode.min() - 200
detect_M = prism_df[(prism_df.episode >= 400) & (prism_df.change_detected == True)].episode.min() - 400
print(f"\n  Latence détection R : {detect_R} épisodes {'✅' if detect_R < 10 else '❌'}")
print(f"  Latence détection M : {detect_M} épisodes {'✅' if detect_M < 10 else '❌'}")

# --- ECE pendant transitions ---
ece_trans_R = prism_df[(prism_df.episode >= 200) & (prism_df.episode < 220)].ECE_window.mean()
ece_trans_M = prism_df[(prism_df.episode >= 400) & (prism_df.episode < 420)].ECE_window.mean()
print(f"\n  ECE pendant transition R : {ece_trans_R:.3f} {'✅' if ece_trans_R < 0.20 else '❌'}")
print(f"  ECE pendant transition M : {ece_trans_M:.3f} {'✅' if ece_trans_M < 0.20 else '❌'}")

# --- Verdict global ---
print("\n" + "="*60)
print("CP5 — VERDICT GLOBAL DU PROJET")
print("="*60)
exp_a = pd.read_csv("results/exp_a/summary.csv")  # résumé généré par Exp A
exp_b = pd.read_csv("results/exp_b/summary.csv")

verdicts = [
    ("Exp A — ECE < 0.15", exp_a[exp_a.condition == "PRISM"].ECE_median.iloc[0] < 0.15),
    ("Exp A — MI > 0.5", exp_a[exp_a.condition == "PRISM"].MI_median.iloc[0] > 0.5),
    ("Exp A — PRISM bat SR-Global",
     exp_a[exp_a.condition == "PRISM"].ECE_median.iloc[0] < exp_a[exp_a.condition == "SR-Global"].ECE_median.iloc[0]),
    ("Exp B — PRISM bat e-greedy",
     exp_b[exp_b.condition == "PRISM"].steps_median.iloc[0] < exp_b[exp_b.condition == "SR-e-greedy"].steps_median.iloc[0]),
    ("Exp B — PRISM bat Count-Bonus",
     exp_b[exp_b.condition == "PRISM"].steps_median.iloc[0] < exp_b[exp_b.condition == "SR-Count-Bonus"].steps_median.iloc[0]),
    ("Exp C — Détection < 10 éps", max(detect_R, detect_M) < 10),
    ("Exp C — Asymétrie R/M dans [10, 60]", 10 < ratio < 60),
    ("Exp C — ECE transitions < 0.20", max(ece_trans_R, ece_trans_M) < 0.20),
]

n_pass = sum(1 for _, v in verdicts if v)
for label, passed in verdicts:
    print(f"  {'✅' if passed else '❌'} {label}")

print(f"\n  Score : {n_pass}/{len(verdicts)}")
if n_pass == len(verdicts):
    print("  🟢 Résultats solides — rédiger le rapport avec confiance")
elif n_pass >= 6:
    print("  🟡 Résultats partiels — rédiger en nuançant les points faibles")
elif n_pass >= 4:
    print("  🟠 Résultats mitigés — discuter honnêtement les limites")
else:
    print("  🔴 Résultats négatifs — pivoter vers l'analyse de pourquoi ça ne marche pas")
```

### What you are looking for

**Panel A (performance):** the reward should drop at the time of perturbations (episodes 200 and 400) then recover. PRISM should recover faster than SR-Blind. Q-Learning will serve as a contrast.

**Panel B (uncertainty):** U(s) should show a sharp peak at each perturbation then come back down. If U does not react to the R-type change (episode 200), that is normal — the SR is not affected. If U does not react to the M-type change (episode 400), that is a bug.

**Asymmetry:** this is the most theoretically informative test. The 15-40x ratio is derived from the architecture. A ratio in this range confirms that the agent truly operates as an SR agent.

---

## Memo — What you do NOT verify

For clarity, here is what does not require your attention:

- Unit tests (automated, `pytest`)
- SB3 baselines (standard wrapper)
- Statistical computations (Mann-Whitney, bootstrap, Holm-Bonferroni — automated in the scripts above)
- Logging and saving results
- Formatting of non-critical figures
- Spectral decomposition (adapted code, validated at CP1)
- Generation of result CSVs

---

*Last updated: 2026-02-14*
