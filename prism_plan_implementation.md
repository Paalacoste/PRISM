# PRISM — Plan d'implémentation détaillé

> Document de travail dérivé du master.md
> Organisation : tâches atomiques, dépendances explicites, critères de validation, pièges à éviter

---

## Vue d'ensemble des phases

| Phase | Semaines | Objectif | Livrable clé |
|-------|----------|----------|--------------|
| **0 — Setup** | Jour 1-2 | Environnement de dev prêt | `pytest` passe, MiniGrid tourne |
| **1 — Assemblage** | S1-S2 | Agent SR fonctionnel dans MiniGrid | Heatmaps M + eigenvectors cohérents |
| **2 — Méta-SR** | S3-S5 | Contribution principale + Exp A | ECE < 0.15, MI > 0.5 |
| **3 — Exploration & Adaptation** | S6-S8 | Exp B + C, comparaisons complètes | Résultats finaux, figures, rapport |

---

## Phase 0 — Setup environnement (Jours 1-2)

> **Status : DONE** (commit d5512bf)
> Python 3.11 + venv + pip. Package `prism/` (pas `src/`). `pip install -e .` via pyproject.toml.
> stable-baselines3 différé à Phase 3. numpy pinné `<2.0`.

### 0.1 Structure du projet

```bash
# Plan initial (src/ n'a pas été utilisé) :
mkdir -p prism/{src/{env,agent,baselines,analysis},experiments,notebooks,tests,results/{exp_a,exp_b,exp_c}}
touch prism/src/__init__.py prism/src/env/__init__.py prism/src/agent/__init__.py
touch prism/src/baselines/__init__.py prism/src/analysis/__init__.py
```

> **Réalisation :** Le package est `prism/` directement (pas `prism/src/`). Plus simple pour un projet de recherche solo. Imports : `from prism.agent.sr_layer import SRLayer`.

### 0.2 Dépendances

Créer `requirements.txt` :
```
# Plan initial :
Python 3.11+
minigrid>=2.3
gymnasium>=0.29
numpy>=1.24
scipy>=1.11
matplotlib>=3.7
seaborn
pandas
tqdm
pytest
stable-baselines3
```

> **Réalisation :** `stable-baselines3` retiré (PyTorch ~2 Go inutile avant Phase 3).
> `numpy>=1.24,<2.0` pinné pour éviter les breaking changes.
> Versions installées : minigrid 2.5, gymnasium 0.29.1, numpy 1.26.4, scipy 1.17.0.

**Action :**
```bash
# Plan initial :
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> **Réalisation :** Sous Windows : `py -3.11 -m venv .venv` puis `.venv\Scripts\activate`.

### 0.3 Vérification MiniGrid

```python
# Script de smoke test
import gymnasium as gym
import minigrid
env = gym.make("MiniGrid-FourRooms-v0")
obs, info = env.reset()
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
# Vérifier : 3 actions de mouvement (left, right, forward) + autres
# Vérifier : observation partielle par défaut (7x7)
```

**Piège MiniGrid :** L'observation par défaut est *partielle* (vue ego-centrique 7×7). Pour la SR tabulaire, on a besoin de la position absolue de l'agent. Deux options :
- Utiliser `env.unwrapped.agent_pos` pour extraire la position
- Ou wrapper avec `FullyObsWrapper` si besoin de la grille complète

> **Résolu :** `env.unwrapped.agent_pos` utilisé. C'est un **tuple** dans MiniGrid 2.5 (pas un ndarray).
> FourRooms 19×19 a **260 cellules accessibles** (pas 80-150 comme estimé initialement).
> Pas de portes — les pièces sont connectées par des passages ouverts.
> `max_steps=500` nécessaire (défaut 100 trop court pour convergence SR).

**Critère de validation :** L'agent random se déplace dans les 4 pièces. On peut extraire `agent_pos` à chaque step.

---

## Phase 1 — Assemblage SR (Semaines 1-2)

> **Status : DONE** — 97 tests passent (7 fichiers). Tous les composants implémentés dans `prism/` (pas `src/`).
> → CP1 PASSED (vérifié via `notebooks/01_sr_validation.ipynb`)

### 1.1 `state_mapper.py` — Mapping position → index SR

> **Status : DONE** — `prism/env/state_mapper.py` (97 lignes), tests : `tests/test_state_mapper.py`
> Ajout de `to_grid(values)` pour la visualisation et `get_grid_shape()`.
> n_states = 260 dans FourRooms (pas 80-150). Les `goal` sont aussi inclus comme cellules accessibles.

**But :** Convertir les positions (x, y) de MiniGrid en indices entiers pour la matrice SR N×N.

**Implémentation :**

```python
class StateMapper:
    def __init__(self, env):
        """Parcourt la grille, identifie toutes les cellules accessibles,
        crée un mapping bidirectionnel pos <-> index."""
        self.pos_to_idx = {}  # (x, y) -> int
        self.idx_to_pos = {}  # int -> (x, y)
        self.n_states = 0
        self._build_map(env)
    
    def _build_map(self, env):
        """Itérer sur la grille, exclure les murs."""
        grid = env.unwrapped.grid
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is None or cell.type == 'door':  # None = cellule vide = accessible
                    self.pos_to_idx[(x, y)] = self.n_states
                    self.idx_to_pos[self.n_states] = (x, y)
                    self.n_states += 1
    
    def get_index(self, pos: tuple) -> int:
        return self.pos_to_idx[pos]
    
    def get_pos(self, idx: int) -> tuple:
        return self.idx_to_pos[idx]
```

**Pièges :**
- Les portes dans FourRooms sont des cellules traversables — les inclure dans le mapping
- La grille a des murs en bordure — les exclure
- Reconstruire le mapping après chaque perturbation structurelle (porte bloquée/ouverte)
- Gérer l'orientation de l'agent : dans MiniGrid, l'agent a une direction (0-3). Pour la SR tabulaire, on peut soit ignorer la direction (state = position seule), soit l'inclure (state = position × direction). **Recommandation pour v1 : ignorer la direction**, ce qui donne ~100-120 states dans FourRooms standard. Argument : la SR de Stachenfeld opère sur des positions, pas des poses.

**Tests unitaires :**
- `test_all_walkable_cells_mapped()` — aucune cellule accessible n'est oubliée
- `test_no_wall_mapped()` — aucun mur dans le mapping
- `test_bijection()` — pos_to_idx et idx_to_pos sont inverses
- `test_n_states_reasonable()` — entre 80 et 150 pour FourRooms standard

> **Réalisation :** Borne ajustée à 80-300 (FourRooms a 260 cellules). Tous les tests `[x]`.

**Estimation :** 2-3 heures

---

### 1.2 `sr_layer.py` — SR tabulaire

> **Status : DONE** — `prism/agent/sr_layer.py` (81 lignes), tests : `tests/test_sr_layer.py` (147 lignes, 17 tests)
> Fidèle au plan. `update()` retourne `delta_M` comme spécifié. Tous les tests `[x]`.

**But :** Matrice M(s,s') apprise par TD(0), vecteur R(s), calcul V(s) = M·R.

**Base :** Adapter depuis le tutorial de Juliani (2019). L'adaptation principale est l'interface avec MiniGrid via le StateMapper.

**Structure :**

```python
class SRLayer:
    def __init__(self, n_states: int, gamma=0.95, alpha_M=0.1, alpha_R=0.3):
        self.M = np.eye(n_states)  # Init identité (chaque état se prédit lui-même)
        self.R = np.zeros(n_states)
        self.gamma = gamma
        self.alpha_M = alpha_M
        self.alpha_R = alpha_R
    
    def update(self, s: int, s_next: int, reward: float) -> np.ndarray:
        """Retourne le vecteur d'erreur TD sur M (nécessaire pour la méta-SR)."""
        # One-hot de s_next
        e_s_next = np.zeros(self.n_states)
        e_s_next[s_next] = 1.0
        
        # Erreur TD sur M
        delta_M = e_s_next + self.gamma * self.M[s_next] - self.M[s]
        
        # Mise à jour M
        self.M[s] += self.alpha_M * delta_M
        
        # Mise à jour R
        self.R[s_next] += self.alpha_R * (reward - self.R[s_next])
        
        return delta_M  # ← crucial : la méta-SR en a besoin
    
    def value(self, s: int) -> float:
        return np.dot(self.M[s], self.R)
    
    def all_values(self) -> np.ndarray:
        return self.M @ self.R
```

**Détail critique :** La méthode `update()` doit **retourner le vecteur d'erreur TD complet** `delta_M`, pas seulement mettre à jour M. C'est l'interface avec la couche méta-SR (phase 2).

**Tests unitaires :**
- `test_M_identity_init()` — M commence comme identité
- `test_update_changes_M()` — après un update, M[s] a changé
- `test_value_computation()` — V(s) = M[s] · R
- `test_convergence_simple_chain()` — sur un MDP en chaîne de 5 états avec reward terminal, M converge vers la solution analytique (à 10% près après 1000 updates)
- `test_reward_transfer()` — après convergence, déplacer R et vérifier que V change instantanément (sans réapprendre M)

**Estimation :** 3-4 heures (dont tests)

---

### 1.3 `dynamics_wrapper.py` — Perturbations contrôlées

> **Status : DONE** — `prism/env/dynamics_wrapper.py` (152 lignes), tests : `tests/test_dynamics_wrapper.py` (154 lignes, 12 tests)
> `get_true_transition_matrix()` itère sur 4 directions × 3 actions de mouvement, poids 1/(4×3).
> Piège "reset re-applique perturbations" géré comme prévu.
> Note : `door_block` cible des passages ouverts (pas de portes dans FourRooms).

**But :** Wrapper Gymnasium qui ajoute des perturbations dynamiques sur MiniGrid.

**Types de perturbations :**

| Type | Effet | Impact SR |
|------|-------|-----------|
| `reward_shift` | Déplace le goal à une nouvelle position | R change, M reste valide |
| `door_block` | Bloque une porte (mur temporaire) | M invalide localement, R inchangé |
| `door_open` | Ouvre un passage (nouvelle zone) | M doit être étendu/appris |
| `combined` | reward_shift + door_block simultanés | M et R changent |

**Implémentation :**

```python
class DynamicsWrapper(gymnasium.Wrapper):
    def __init__(self, env, seed=None):
        super().__init__(env)
        self.perturbations_log = []
        self.current_step = 0
        self.schedule = None
    
    def apply_perturbation(self, ptype: str, **kwargs):
        """Applique une perturbation.
        
        Pour reward_shift : kwargs = {'new_goal_pos': (x, y)}
        Pour door_block : kwargs = {'door_pos': (x, y)}
        Pour door_open : kwargs = {'wall_pos': (x, y)}
        """
        # Modifier la grille interne de MiniGrid
        # Logger la perturbation avec timestamp
    
    def set_schedule(self, schedule):
        """Schedule : liste de (episode, perturbation_type, kwargs)"""
        self.schedule = schedule
    
    def step(self, action):
        self.current_step += 1
        # Vérifier si une perturbation schedulée doit être appliquée
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
    
    def get_true_transition_matrix(self, state_mapper) -> np.ndarray:
        """Ground truth M* — pour calculer l'erreur réelle de la SR.
        Parcourt toutes les positions accessibles et simule chaque action."""
        # BFS/flood fill depuis chaque état
```

**Piège majeur — modification de la grille MiniGrid :** MiniGrid reconstruit la grille à chaque `reset()`. Les perturbations doivent être ré-appliquées après chaque reset, ou il faut surcharger `reset()` pour maintenir l'état perturbé.

```python
def reset(self, **kwargs):
    obs, info = self.env.reset(**kwargs)
    # Ré-appliquer toutes les perturbations actives
    for p in self.active_perturbations:
        self._apply_to_grid(p)
    return obs, info
```

**`get_true_transition_matrix()` — détail important :**
Cette méthode est nécessaire pour calculer M* (ground truth) et donc le Metacognitive Index. Elle doit simuler, pour chaque état accessible et chaque action, quel est l'état suivant. Dans MiniGrid déterministe, c'est un simple flood fill.

**Attention à l'action space de MiniGrid :** Les actions sont `{left, right, forward, pickup, drop, toggle, done}`. Pour la SR, on ne s'intéresse qu'aux actions de mouvement. Le wrapper peut filtrer ou la politique peut être restreinte.

**Tests unitaires :**
- `test_reward_shift_moves_goal()`
- `test_door_block_prevents_passage()`
- `test_perturbation_survives_reset()`
- `test_true_transition_matrix_symmetric()` — dans un env sans murs internes, T doit être symétrique
- `test_schedule_applies_at_correct_episode()`

**Estimation :** 6-8 heures (le plus complexe de Phase 1)

---

### 1.4 `perturbation_schedule.py`

> **Status : STUB** — `PerturbationEvent` dataclass existe, `PerturbationSchedule` lève `NotImplementedError`.
> Sera complété en Phase 3 quand les expériences seront exécutées.

**But :** Configs réutilisables de schedules de perturbation pour chaque expérience.

```python
@dataclass
class PerturbationEvent:
    episode: int
    ptype: str  # 'reward_shift', 'door_block', 'door_open', 'combined'
    kwargs: dict

class PerturbationSchedule:
    def __init__(self, events: list[PerturbationEvent]):
        self.events = sorted(events, key=lambda e: e.episode)
    
    @classmethod
    def exp_a(cls):
        """Phase apprentissage (0-300), exploration (300-400), perturbation (400-500)"""
        return cls([
            PerturbationEvent(300, 'door_open', {'wall_pos': (9, 5)}),
            PerturbationEvent(400, 'reward_shift', {'new_goal_pos': (12, 8)}),
        ])
    
    @classmethod
    def exp_c(cls):
        """Phase stable, perturbation R, re-stab, perturbation M, re-stab finale"""
        return cls([
            PerturbationEvent(200, 'reward_shift', {'new_goal_pos': (5, 12)}),
            PerturbationEvent(400, 'door_block', {'door_pos': (9, 9)}),
        ])
```

**Estimation :** 1-2 heures

---

### 1.5 `spectral.py` — Décomposition spectrale de M

> **Status : DONE** — `prism/analysis/spectral.py` (79 lignes)
> Utilise `scipy.linalg.eigh` avec `subset_by_index`. Symétrise M avec (M+M.T)/2.
> `plot_eigenvectors()` crée la grille 2×3 de heatmaps RdBu_r.

**But :** Calculer et visualiser les eigenvectors de M (validation que les grid cells émergent).

**Base :** Adapter depuis le repo de Chelu (temporal_abstraction).

```python
def compute_eigenvectors(M: np.ndarray, k: int = 6) -> tuple:
    """Retourne les k premiers eigenvectors et eigenvalues de M."""
    eigenvalues, eigenvectors = scipy.linalg.eigh(M, subset_by_index=[M.shape[0]-k, M.shape[0]-1])
    # Trier par eigenvalue décroissante
    idx = np.argsort(-eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]

def plot_eigenvectors_on_grid(eigenvectors, state_mapper, grid_shape, k=6):
    """Affiche les k premiers eigenvectors comme heatmaps sur la grille."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, ax in enumerate(axes.flat):
        grid = np.full(grid_shape, np.nan)
        for idx in range(state_mapper.n_states):
            x, y = state_mapper.get_pos(idx)
            grid[y, x] = eigenvectors[idx, i]
        ax.imshow(grid, cmap='RdBu_r')
        ax.set_title(f'Eigenvector {i+1}')
```

**Critère de validation :** Les premiers eigenvectors doivent montrer des patterns à grande échelle (séparation entre pièces), les suivants des patterns plus fins. Comparer visuellement avec la Figure 3 de Stachenfeld 2017.

**Estimation :** 2-3 heures

---

### 1.6 Notebook `01_sr_validation.ipynb` — Sanity check

> **Status : DONE** — 9 sections, inclut les checks CP1 (convergence, go/no-go automatique).
> Couvre : entraînement 300 épisodes, courbes d'apprentissage, heatmaps SR, eigenvecteurs,
> triptyque V/U/C, calibration (ECE, MI, reliability diagram), et diagnostic CP1.
> → **CP1 PASSED**

**Objectif :** Vérifier que tout fonctionne avant de passer à la contribution originale.

**Contenu :**

1. Initialiser FourRooms + StateMapper
2. Entraîner l'agent SR pendant 500 épisodes (random policy ou ε-greedy simple)
3. Visualiser M comme heatmap pour quelques états sources (coins de chaque pièce)
4. Visualiser les 6 premiers eigenvectors de M
5. Vérifier le transfert de reward : déplacer R, recalculer V sans réapprendre M
6. Comparer M appris vs. M* (ground truth) — erreur moyenne par état

**Critères de validation Phase 1 (milestone) :**
- [x] M converge : ||M - M*||_F décroît monotoniquement
- [x] Les heatmaps de M montrent des patterns de diffusion depuis l'état source
- [x] Les eigenvectors ressemblent à ceux de Stachenfeld 2017
- [x] Le transfert de reward fonctionne : nouveau V correct sans réapprentissage de M
- [x] Tous les tests passent (141 tests, 9 fichiers — mis à jour Palier 1 Exp B)
- [ ] Tous les tests unitaires passent

---

## Phase 2 — Méta-SR et calibration (Semaines 3-5) ⭐

> **Status : PARTIELLEMENT DONE** — Composants core (meta_sr, controller, agent, calibration, visualization) implémentés et testés.
> Baselines = **implémentés** (Palier 1 Exp B). `metrics.py` = **implémenté** (bootstrap_ci, mann_whitney, holm_bonferroni, compare_conditions).
> Controller exploit branch = **corrigé** (greedy sur V_explore).
> Sweep, Exp A, notebook 02 = reportés à Phase 3.
> → CP2, CP3 PENDING (seront validés après exécution d'Exp A)

C'est le cœur du projet — la contribution originale.

### 2.1 `meta_sr.py` — Carte d'incertitude U(s) et signal C(s)

> **Status : DONE** — `prism/agent/meta_sr.py` (144 lignes), tests : `tests/test_meta_sr.py` (179 lignes, 18 tests)
> Décisions d'implémentation vs. plan :
> - Normalisation **p99 adaptative** (deque des 5000 derniers δ) au lieu de running min/max
> - Régime cold-start simplifié : pur decay exponentiel (pas de blending data/prior)
> - `_recent_visits` deque(maxlen=50) pour la détection de changement
> - `all_uncertainties()` et `all_confidences()` vectorisés via numpy

**C'est le fichier le plus important du projet.**

**Structure détaillée :**

```python
class MetaSR:
    def __init__(self, n_states: int, buffer_size=20, U_prior=0.8, 
                 decay=0.85, beta=10, theta_C=0.3, theta_change=0.5):
        self.n_states = n_states
        self.buffer_size = buffer_size
        self.U_prior = U_prior
        self.decay = decay
        self.beta = beta
        self.theta_C = theta_C
        self.theta_change = theta_change
        
        # Buffer circulaire d'erreurs par état
        self.error_buffers = {s: deque(maxlen=buffer_size) for s in range(n_states)}
        self.visit_counts = np.zeros(n_states, dtype=int)
        self.U = np.full(n_states, U_prior)  # Carte d'incertitude
        self.recently_visited = deque(maxlen=50)  # Pour détection de changement
    
    def observe(self, s: int, delta_M: np.ndarray):
        """Appelé après chaque update SR. delta_M = vecteur d'erreur TD."""
        delta_scalar = np.linalg.norm(delta_M)  # Compression L2
        
        self.error_buffers[s].append(delta_scalar)
        self.visit_counts[s] += 1
        self.recently_visited.append(s)
        
        # Mettre à jour U(s)
        self._update_uncertainty(s)
    
    def _update_uncertainty(self, s: int):
        K = self.buffer_size
        visits = self.visit_counts[s]
        
        if visits == 0:
            self.U[s] = self.U_prior  # Maximum d'incertitude
        elif visits < K:
            # Transition progressive du prior vers les données
            data_weight = len(self.error_buffers[s]) / K
            data_estimate = np.mean(self.error_buffers[s])
            prior_estimate = self.U_prior * (self.decay ** visits)
            self.U[s] = (1 - data_weight) * prior_estimate + data_weight * data_estimate
        else:
            # Assez de données : utiliser la moyenne empirique
            self.U[s] = np.mean(self.error_buffers[s])
    
    def confidence(self, s: int) -> float:
        """Signal de confiance C(s) ∈ [0, 1]. 1 = haute confiance."""
        return 1.0 / (1.0 + np.exp(self.beta * (self.U[s] - self.theta_C)))
    
    def all_confidences(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(self.beta * (self.U - self.theta_C)))
    
    def detect_change(self) -> bool:
        """Détecte un changement structurel en regardant U des états récemment visités."""
        if len(self.recently_visited) < 10:
            return False
        recent_states = list(set(self.recently_visited))
        mean_U = np.mean([self.U[s] for s in recent_states])
        return mean_U > self.theta_change
    
    def change_score(self) -> float:
        """Score continu de changement (pour monitoring)."""
        if len(self.recently_visited) < 10:
            return 0.0
        recent_states = list(set(self.recently_visited))
        return np.mean([self.U[s] for s in recent_states])
```

**Points d'attention critiques :**

1. **Normalisation de U :** Les erreurs δ brutes ne sont pas dans [0, 1]. Options :
   - Normalisation running min/max sur les δ observés
   - Clipping à un percentile (99e) puis division
   - **Recommandation :** normalisation adaptative avec running statistics, recalculée tous les N steps

2. **Buffer à froid :** Au début de l'apprentissage, les δ sont grands partout (M est loin de M*). Il faut une période de "warm-up" avant de considérer U comme informative. Suggestion : ne pas calculer les métriques de calibration avant 100 épisodes.

3. **Détection de changement — faux positifs :** En début d'épisode après un reset, l'agent peut visiter des états qu'il n'a pas vus depuis longtemps → pic de U. Distinguer "je ne connais pas bien cet état" de "cet état a changé". La fenêtre `recently_visited` aide, mais à monitorer.

**Tests unitaires :**
- `test_unvisited_states_max_uncertainty()` — U(s) = U_prior si jamais visité
- `test_uncertainty_decreases_with_visits()` — U(s) décroît après des visites répétées avec δ faibles
- `test_uncertainty_increases_after_perturbation()` — si δ augmente soudainement, U(s) remonte
- `test_confidence_inverse_uncertainty()` — C(s) est décroissant en U(s)
- `test_change_detection_after_perturbation()`
- `test_buffer_circular()` — le buffer ne dépasse pas K

**Estimation :** 8-10 heures (le composant le plus critique)

---

### 2.2 `controller.py` — Politique adaptative

> **Status : DONE** — `prism/agent/controller.py` (101 lignes)
> Simplification vs. plan : l'action exploit ne fait PAS d'évaluation V_explore des voisins
> (nécessiterait un modèle de transition T[s][a] → s'). Utilise sélection aléatoire parmi
> les actions disponibles. L'évaluation voisins est une amélioration Phase 3.

**But :** Utiliser U(s) et C(s) pour adapter le comportement de l'agent.

```python
class PRISMController:
    def __init__(self, sr_layer, meta_sr, epsilon_min=0.01, epsilon_max=0.5, 
                 lambda_explore=0.5, theta_idk=0.3):
        self.sr = sr_layer
        self.meta = meta_sr
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.lambda_explore = lambda_explore
        self.theta_idk = theta_idk  # Seuil "je ne sais pas"
    
    def adaptive_epsilon(self, s: int) -> float:
        """ε adaptatif basé sur l'incertitude locale."""
        U_s = self.meta.U[s]
        U_max = max(self.meta.U.max(), 1e-8)
        return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * (U_s / U_max)
    
    def exploration_value(self, s: int) -> float:
        """V augmentée par le bonus d'exploration."""
        return self.sr.value(s) + self.lambda_explore * self.meta.U[s]
    
    def select_action(self, s: int, available_actions: list) -> tuple:
        """Retourne (action, confidence, idk_flag)."""
        eps = self.adaptive_epsilon(s)
        confidence = self.meta.confidence(s)
        idk = confidence < self.theta_idk
        
        if np.random.random() < eps:
            # Exploration : choisir l'action menant vers l'état de plus haute incertitude
            action = self._explore_action(s, available_actions)
        else:
            # Exploitation : action maximisant V_explore
            action = self._exploit_action(s, available_actions)
        
        return action, confidence, idk
    
    def _explore_action(self, s, actions):
        """Exploration dirigée : aller vers les états incertains."""
        # Pour chaque action, estimer l'état suivant et son U
        # Choisir l'action menant vers le plus haut U
        # Fallback : random si on ne peut pas estimer les transitions
        ...
    
    def _exploit_action(self, s, actions):
        """Exploitation : maximiser V_explore."""
        # Pour chaque action, estimer V_explore de l'état suivant
        ...
```

**Piège — estimation de l'état suivant :** Pour `_explore_action` et `_exploit_action`, il faut savoir quel état résulte de chaque action. En tabulaire, on peut maintenir une table T(s, a) → s' apprise par expérience. Alternative plus simple : utiliser la géométrie de la grille (forward = avancer d'une case dans la direction courante). **Attention :** si on ignore la direction de l'agent (voir 1.1), il faut un mécanisme alternatif. Solution pragmatique : maintenir un compteur T[s][a] → s' basé sur les transitions observées.

**Estimation :** 4-6 heures

---

### 2.3 `prism_agent.py` — Agent complet

> **Status : DONE** — `prism/agent/prism_agent.py` (195 lignes)
> Ajouts vs. plan :
> - Intégration `config.py` (PRISMConfig dataclass centralise tous les hyperparamètres)
> - Méthodes helper : `get_uncertainty_map()`, `get_confidence_map()`, `get_value_map()`
> - `_get_state()` gère `tuple(env.unwrapped.agent_pos)` pour MiniGrid 2.5
> - `MOVEMENT_ACTIONS = [0, 1, 2]` constante (turn_left, turn_right, forward)

**But :** Assembler SR + Méta-SR + Controller dans une boucle agent-environnement.

```python
class PRISMAgent:
    def __init__(self, env, state_mapper, **kwargs):
        self.env = env
        self.mapper = state_mapper
        self.sr = SRLayer(state_mapper.n_states, **sr_kwargs)
        self.meta = MetaSR(state_mapper.n_states, **meta_kwargs)
        self.controller = PRISMController(self.sr, self.meta, **ctrl_kwargs)
        
        # Logging
        self.history = []  # (episode, step, s, a, r, s', C(s), U(s), delta)
    
    def train_episode(self) -> dict:
        """Un épisode complet. Retourne les métriques."""
        obs, info = self.env.reset()
        s = self.mapper.get_index(self.env.unwrapped.agent_pos)
        
        episode_reward = 0
        episode_steps = 0
        confidences = []
        uncertainties = []
        
        done = False
        while not done:
            action, confidence, idk = self.controller.select_action(s, ...)
            obs, reward, terminated, truncated, info = self.env.step(action)
            s_next = self.mapper.get_index(self.env.unwrapped.agent_pos)
            
            # Update SR → récupérer delta_M
            delta_M = self.sr.update(s, s_next, reward)
            
            # Update Méta-SR
            self.meta.observe(s, delta_M)
            
            # Logging
            self.history.append({...})
            confidences.append(confidence)
            uncertainties.append(self.meta.U[s])
            
            episode_reward += reward
            episode_steps += 1
            s = s_next
            done = terminated or truncated
        
        return {
            'reward': episode_reward,
            'steps': episode_steps,
            'mean_confidence': np.mean(confidences),
            'mean_uncertainty': np.mean(uncertainties),
            'change_detected': self.meta.detect_change(),
        }
    
    def train(self, n_episodes: int, log_every: int = 10):
        """Boucle d'entraînement complète."""
        for ep in tqdm(range(n_episodes)):
            metrics = self.train_episode()
            if ep % log_every == 0:
                self._log_metrics(ep, metrics)
```

**Estimation :** 4-5 heures

---

### 2.4 `calibration.py` — Métriques psychophysiques ⭐

> **Status : DONE** — `prism/analysis/calibration.py` (207 lignes), tests : `tests/test_calibration.py` (149 lignes, 17 tests)
> Implémente : `sr_errors()`, `sr_accuracies()`, `expected_calibration_error()`,
> `reliability_diagram_data()`, `plot_reliability_diagram()`, `metacognitive_index()`.
> **Phase 3 :** Ajouter `hosmer_lemeshow_test()` (référencé dans checkpoints.md CP3).

**C'est la deuxième contribution clé — traiter l'agent comme un sujet de psychologie cognitive.**

```python
def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins=10) -> float:
    """Expected Calibration Error.
    
    confidences : C(s) pour chaque prédiction
    accuracies : 1 si ||M(s,:) - M*(s,:)|| < tau, 0 sinon
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_conf = confidences[mask].mean()
            bin_acc = accuracies[mask].mean()
            ece += mask.sum() / len(confidences) * abs(bin_acc - bin_conf)
    return ece

def reliability_diagram(confidences, accuracies, n_bins=10, ax=None):
    """Reliability diagram — confiance déclarée vs. accuracy observée."""
    # Produire le plot classique avec barres + diagonale parfaite
    ...

def metacognitive_index(U: np.ndarray, M: np.ndarray, M_star: np.ndarray) -> float:
    """MI = corr(U(s), ||M(s,:) - M*(s,:)||₂)
    L'agent sait-il ce qu'il ne sait pas ?"""
    real_errors = np.array([np.linalg.norm(M[s] - M_star[s]) for s in range(len(U))])
    return scipy.stats.spearmanr(U, real_errors).correlation

def accuracy_from_sr_error(M, M_star, tau_percentile=50):
    """Calcule le vecteur d'accuracy binaire pour l'ECE.
    tau = 50e percentile de ||M - M*|| → baseline accuracy ~50%."""
    errors = np.array([np.linalg.norm(M[s] - M_star[s]) for s in range(M.shape[0])])
    tau = np.percentile(errors, tau_percentile)
    return (errors < tau).astype(float), tau
```

**Point méthodologique crucial — définition de l'accuracy :** L'ECE standard en ML classification compare la confiance du modèle avec "la réponse est correcte ou non". Ici, il n'y a pas de classification binaire. L'accuracy est définie comme "la carte SR est fiable en cet état", seuillée par τ (50e percentile de l'erreur). Ce choix est justifié dans le master.md §6.1 et doit être discuté dans le rapport.

**Tests unitaires :**
- `test_ece_perfect_calibration()` — si C(s) = accuracy(s) exactement, ECE = 0
- `test_ece_worst_calibration()` — si C(s) = 1 - accuracy(s), ECE est maximal
- `test_mi_perfect_correlation()` — si U(s) = ||M(s) - M*(s)||, MI = 1.0
- `test_mi_random()` — si U est random, MI ≈ 0

**Estimation :** 5-6 heures

---

### 2.5 `visualization.py` — Superposition U/M

> **Status : DONE** — `prism/analysis/visualization.py` (98 lignes)
> Implémente : `plot_sr_heatmap()`, `plot_value_map()`, `plot_uncertainty_map()`.
> Animation `animate_U_after_perturbation` pas encore faite (Phase 3).

```python
def plot_uncertainty_map(U, state_mapper, grid_shape, ax=None):
    """Heatmap de U(s) sur la grille."""
    grid = np.full(grid_shape, np.nan)
    for idx in range(state_mapper.n_states):
        x, y = state_mapper.get_pos(idx)
        grid[y, x] = U[idx]
    ax.imshow(grid, cmap='YlOrRd', vmin=0, vmax=1)

def plot_M_and_U_overlay(M, U, source_state, state_mapper, grid_shape):
    """Deux heatmaps côte à côte : M[source,:] et U."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_sr_heatmap(M, source_state, state_mapper, grid_shape, ax=ax1)
    plot_uncertainty_map(U, state_mapper, grid_shape, ax=ax2)
    ax1.set_title(f'M(s={source_state}, ·) — Carte prédictive')
    ax2.set_title('U(s) — Carte d\'incertitude')

def animate_U_after_perturbation(U_snapshots, state_mapper, grid_shape, filename):
    """Animation de l'évolution de U après une perturbation."""
    # Utiliser matplotlib.animation.FuncAnimation
    ...
```

**Estimation :** 4-5 heures

---

### 2.6 Baselines pour Exp A

> **Status : STUBS** — `sr_blind.py`, `sr_count.py` lèvent `NotImplementedError("Phase 2")`.
> `sr_bayesian.py`, `sb3_baselines.py` lèvent `NotImplementedError("Phase 3")`.
> Implémentation complète reportée à Phase 3.

**`sr_blind.py`** — Agent SR sans méta-monitoring. Même SR, ε fixe, confiance = constante (ou random).

**`sr_count.py`** — SR + confiance count-based : C(s) = f(1/√visits(s)). Teste si le simple comptage de visites suffit.

**`sr_bayesian.py`** — SR + régression linéaire bayésienne sur V. Le posterior donne une variance par état → confiance. Implémentation simplifiée de Janz et al. (2019) en tabulaire.

```python
class SRBayesian:
    """SR avec incertitude bayésienne sur V = M · R.
    Utilise une régression linéaire bayésienne : V(s) ~ N(M[s]·μ_R, M[s]·Σ_R·M[s]ᵀ)"""
    def __init__(self, n_states, prior_variance=1.0):
        self.mu_R = np.zeros(n_states)
        self.Sigma_R = prior_variance * np.eye(n_states)
    
    def confidence(self, s, M):
        """Confiance = 1 / (1 + variance prédictive)"""
        var = M[s] @ self.Sigma_R @ M[s]
        return 1.0 / (1.0 + var)
```

**Estimation baselines :** 4-6 heures total

---

### 2.7 Sweep hyperparamètres méta-SR

> **Status : PAS COMMENCÉ** — Reporté à Phase 3.
> → CP2 (voir checkpoints.md) validera les résultats du sweep.

**Avant l'Exp A formelle**, sweep factoriel :

| Paramètre | Valeurs testées |
|-----------|-----------------|
| U_prior | {0.5, 0.8, 1.0} |
| decay | {0.75, 0.85, 0.95} |
| β | {5, 10, 20} |
| θ_C | {0.2, 0.3, 0.4} |

Total : 3⁴ = 81 configs × 10 runs = 810 runs.

**Critère de sélection :** ECE minimal sur la phase d'apprentissage stable (épisodes 200-300 de l'Exp A).

**Script :** `experiments/sweep_hyperparams.py` avec sauvegarde des résultats en CSV.

**Estimation :** 3 heures de code, temps de calcul variable (parallélisable)

---

### 2.8 Exécuter Exp A — Calibration métacognitive

> **Status : STUB** — `experiments/exp_a_calibration.py` lève `NotImplementedError("Phase 2")`.
> → CP3 (voir checkpoints.md) validera les résultats d'Exp A.

**`experiments/exp_a_calibration.py`**

```python
def run_exp_a(condition: str, n_runs=100, seed_base=42):
    """
    Phases :
    1. Apprentissage (300 épisodes) — monde stable
    2. Exploration (100 épisodes) — ouverture 5e pièce
    3. Perturbation (100 épisodes) — goal déplacé dans nouvelle zone
    
    Conditions : 'prism', 'sr_global', 'sr_count', 'sr_bayesian', 'random_conf'
    """
    results = []
    for run in range(n_runs):
        env = DynamicsWrapper(gym.make("MiniGrid-FourRooms-v0"))
        env.set_schedule(PerturbationSchedule.exp_a())
        
        agent = create_agent(condition, env)
        
        # Entraîner
        for ep in range(500):
            metrics = agent.train_episode()
            
            # Collecter confiance et accuracy à chaque step
            # (nécessite M* du wrapper)
            M_star = env.get_true_transition_matrix(agent.mapper)
            ...
        
        # Calculer métriques finales
        ece = compute_ece(all_confidences, all_accuracies)
        mi = metacognitive_index(agent.meta.U, agent.sr.M, M_star)
        results.append({'run': run, 'condition': condition, 'ece': ece, 'mi': mi})
    
    return pd.DataFrame(results)
```

**Livrables Exp A :**
- ECE par condition (tableau + boxplot)
- Reliability diagrams par condition
- MI par condition
- Heatmaps M et U à 3 moments : fin phase 1, fin phase 2, fin phase 3
- Animation U après perturbation
- Eigenvectors de M (validation spectrale)

**Critères de succès :**
- [ ] ECE(PRISM) < 0.15
- [ ] MI(PRISM) > 0.5
- [ ] ECE(PRISM) < ECE(SR-Global) — p < 0.05, Mann-Whitney
- [ ] ECE(PRISM) < ECE(SR-Count) — p < 0.05
- [ ] Reliability diagram : corrélation positive claire

**Estimation :** 6-8 heures (code + premières runs + debug)

---

### Milestone Phase 2

- [x] `meta_sr.py` passe tous les tests (18 tests)
- [x] `prism_agent.py` tourne dans FourRooms sans crash (train_episode loop OK)
- [ ] Le reliability diagram de PRISM montre une tendance positive → Phase 3 (Exp A)
- [ ] ECE < 0.15 atteint de manière reproductible → Phase 3 (Exp A)
- [ ] MI > 0.5 atteint de manière reproductible → Phase 3 (Exp A)
- [ ] Notebook `02_meta_sr_demo.ipynb` avec visualisations interactives → Phase 3

> **Note :** Les items non cochés dépendent du sweep (§2.7) et d'Exp A (§2.8) qui sont reportés en Phase 3.
> Les composants logiciels de Phase 2 sont tous implémentés et testés.

---

## Phase 3 — Exploration et Adaptation (Semaines 6-8) ⭐

> **Status : EN COURS** — Palier 1 d'Exp B complet (infrastructure, baselines, metrics, runner). Palier 1.6 pending (lancer les 800 runs).
>
> **Prérequis restants :**
>
> 1. **Stubs à compléter :**
>    - ~~`prism/baselines/` — **DONE** (7 agents : RandomAgent, SREpsilonGreedy, SREpsilonDecay, SRCountBonus, SRNormBonus, SRPosterior, SROracle)~~
>    - ~~`prism/analysis/metrics.py` — **DONE** (bootstrap_ci, mann_whitney_test, holm_bonferroni, compare_conditions, compare_all_pairs)~~
>    - `prism/env/perturbation_schedule.py` — implémenter les méthodes `exp_a()`, `exp_c()`
>    - `experiments/exp_a_calibration.py` — implémenter `run_exp_a()`
>
> 2. **Fonctions manquantes à créer :**
>    - `hosmer_lemeshow_test()` dans `prism/analysis/calibration.py`
>    - ~~`get_true_transition_matrix()` dans `DynamicsWrapper` — **DONE**~~
>
> 3. **Dépendance externe :**
>    - Installer `stable-baselines3` pour Q-learning/DQN baseline (Exp C)
>    - Fallback prévu : Q-learning tabulaire custom si incompatibilité SB3
>
> 4. **Leçons Phases 1-2 à appliquer :**
>    - FourRooms = 19×19, **260 états** accessibles (pas ~100)
>    - Pas de portes dans MiniGrid v2.5 (passages ouverts)
>    - `max_steps=500` obligatoire dans `gym.make()`
>    - Normalisation p99 dans MetaSR (pas min-max naïf)
>    - `agent_pos` est un tuple `(x, y)`, pas un int
>
> **Checkpoints Phase 3 :**
> - CP2 après sweep (§2.7) — valider hyperparamètres
> - CP3 après Exp A (§2.8) — valider calibration
> - CP4 après Exp B (§3.3) — valider exploration
> - CP5 après Exp C (§3.4) — valider adaptation

### 3.1 Config monde Exp B

**Grand monde 19×19 avec 4+ pièces et 4 goals cachés.**

> **Note Phase 1-2 :** FourRooms est déjà 19×19 avec 260 cellules. Pour un monde "plus grand",
> il faudra un env custom (héritant de `MiniGridEnv`). Le paramètre `size` n'existe pas
> dans FourRooms standard.

Options :
- ~~Utiliser `MiniGrid-FourRooms-v0` en augmentant la taille (paramètre `size`)~~ → pas supporté
- Créer un env custom avec `MiniGridEnv` comme base (seule option viable)

```python
# Vérifier si FourRooms supporte un paramètre de taille
env = gym.make("MiniGrid-FourRooms-v0", agent_pos=None, goal_pos=None)
# Si taille fixe → créer un custom env :
class LargeFourRooms(MiniGridEnv):
    def __init__(self, size=19, n_goals=4):
        super().__init__(grid_size=size, max_steps=500)
    
    def _gen_grid(self, width, height):
        # Créer 4+ pièces avec portes
        # Placer 4 goals (objets "goal") dans des pièces différentes
```

**Piège :** MiniGrid standard n'a qu'un seul goal. Pour 4 goals, il faudra soit utiliser des `Ball` objects comme goals custom, soit modifier la logique de terminaison.

**Estimation :** 3-4 heures

---

### 3.2 Baselines Exp B

En plus de PRISM, 7 baselines à implémenter :

| Baseline | Code | Difficulté |
|----------|------|------------|
| SR-Oracle | Réutilise PRISM + M* comme signal | Facile (wrapper) |
| SR-ε-greedy | Réutilise SR + ε fixe | Trivial |
| SR-ε-decay | SR + ε décroissant | Trivial |
| SR-Count-Bonus | SR + λ/√visits(s) | Facile |
| SR-Norm-Bonus | SR + λ/||M(s,:)|| | Facile (Machado 2020) |
| SR-Posterior | SR + posterior sampling (bayésien) | Moyen |
| Random | Uniform random | Trivial |

**Test différentiel clé :** PRISM vs. SR-Count-Bonus. Si PRISM gagne, c'est que la structure prédictive SR apporte quelque chose au-delà du comptage.

**Estimation baselines Exp B :** 4-5 heures

---

### 3.3 Exécuter Exp B — Exploration dirigée

> → CP4 (voir checkpoints.md) validera les résultats d'Exp B.

**`experiments/exp_b_exploration.py`**

100 runs par condition, 8 conditions = 800 runs total.

**Métriques à collecter :**
- Steps pour trouver chaque goal (1er, 2e, 3e, 4e)
- Steps total pour trouver les 4
- Couverture cumulée (% d'états visités) vs. steps
- Redondance (revisites / nouvelles visites)
- Efficiency ratio : (steps_Random - steps_PRISM) / (steps_Random - steps_Oracle)

**Critères de succès :**
- [ ] PRISM < SR-ε-greedy sur steps (−30%, p < 0.05)
- [ ] PRISM < SR-Count-Bonus (p < 0.05) — c'est le test de la structure
- [ ] Efficiency ratio > 0.5

**Estimation :** 5-6 heures (code + runs)

---

### 3.4 Exécuter Exp C — Adaptation au changement

> → CP5 (voir checkpoints.md) validera les résultats d'Exp C.

**`experiments/exp_c_adaptation.py`**

**Phases :**
1. Stable (200 épisodes)
2. Perturbation R : goal déplacé (100 épisodes)
3. Re-stabilisation (100 épisodes)
4. Perturbation M : porte bloquée (100 épisodes)
5. Re-stabilisation finale (100 épisodes)

**Conditions :** PRISM, SR-Blind, Q-Learning (via SB3)

**`sb3_baselines.py` :** Wrapper Stable-Baselines3 pour Q-learning/DQN. Attention à l'interface — SB3 attend un env Gymnasium standard. Le wrapper DynamicsWrapper doit être compatible.

**Métriques :**
- Latence de détection : épisodes avant `change_detected = true`
- Latence d'adaptation : épisodes pour retrouver 80% performance pré-perturbation
- ECE en fenêtre glissante de 20 épisodes
- Asymétrie R/M : latence_M / latence_R (prédiction : 15-40×)

**Critères de succès :**
- [ ] PRISM détecte changements en < 10 épisodes
- [ ] Latence adaptation PRISM ≤ 0.5 × SR-Blind
- [ ] Asymétrie R/M entre 15× et 40×
- [ ] ECE < 0.20 pendant transitions

**Estimation :** 6-8 heures

---

### 3.5 Analyse croisée et figures finales

**Notebook `03_results_analysis.ipynb` :**

**Figures essentielles pour le rapport :**

1. **Figure 1 — Architecture PRISM** (schéma, pas de code)
2. **Figure 2 — Validation SR** : heatmaps M + eigenvectors
3. **Figure 3 — Carte d'incertitude** : U superposée au monde, à 3 moments (stable, post-perturbation, re-stabilisé)
4. **Figure 4 — Reliability diagrams** : toutes conditions Exp A
5. **Figure 5 — ECE comparison** : boxplot par condition
6. **Figure 6 — MI comparison** : boxplot par condition
7. **Figure 7 — Exploration** : courbes de couverture Exp B (steps vs % couvert)
8. **Figure 8 — Adaptation** : performance vs. épisodes Exp C, avec marqueurs de perturbation
9. **Figure 9 — Asymétrie R/M** : latences comparées
10. **Figure 10 — ECE dynamique** : ECE en fenêtre glissante Exp C

**Tests statistiques :**
- Mann-Whitney U pour comparaisons deux-à-deux
- Correction Holm-Bonferroni
- Bootstrap 95% CI (10000 resamples)
- Cohen's d pour taille d'effet
- Hosmer-Lemeshow pour calibration
- Test de permutation pour corrélations

**Estimation :** 8-10 heures

---

## Récapitulatif temporel

```
Semaine 1 ──────────────────────────────── ✅ DONE
  Jour 1-2  : Phase 0 (setup) + StateMapper + SR Layer
  Jour 3-4  : DynamicsWrapper + PerturbationSchedule (stub)
  Jour 5    : Spectral + début notebook validation

Semaine 2 ──────────────────────────────── ✅ DONE
  Jour 6-7  : Finaliser notebook 01, tous tests Phase 1
  Jour 8-10 : ★ MetaSR — composant principal
              → CP1 PASSED (notebook 01 + 6 checks automatisés)

Semaine 3 ──────────────────────────────── ✅ DONE
  Jour 11-12 : Controller + PRISMAgent
  Jour 13-14 : Calibration.py + visualisation.py
  Jour 15    : Baselines Exp A (stubs seulement)

Semaine 4 ──────────────────────────────── ⏳ PARTIEL
  Jour 16    : Sweep hyperparamètres        → reporté Phase 3
  Jour 17-19 : ★ Exécuter Exp A             → reporté Phase 3
  Jour 20    : Analyser résultats Exp A     → reporté Phase 3
  ✅ FAIT : calibration.py, checkpoints.md, mise à jour docs

Semaine 5 ──────────────────────────────── À VENIR
  Compléter stubs (baselines, perturbation_schedule, metrics)
  Installer stable-baselines3
  Sweep hyperparamètres (§2.7)
  → CP2 : valider hyperparamètres choisis

Semaine 6 ──────────────────────────────── À VENIR
  ★ Exécuter Exp A (100 runs × 5 conditions)
  Analyser résultats, notebook 02
  → CP3 : valider calibration métacognitive

Semaine 7 ──────────────────────────────── À VENIR
  Config grand monde Exp B + baselines Exp B
  ★ Exécuter Exp B (100 runs × 8 conditions)
  → CP4 : valider exploration dirigée

Semaine 8 ──────────────────────────────── À VENIR
  SB3 wrapper + baselines Exp C
  ★ Exécuter Exp C (100 runs × 3 conditions)
  → CP5 : valider adaptation au changement

Semaine 9 (buffer) ────────────────────── À VENIR
  Analyse croisée, figures finales, notebook 03
  Rédaction rapport de résultats
```

---

## Risques et plans de contingence

| Risque | Impact | Probabilité | Mitigation | Statut |
|--------|--------|-------------|------------|--------|
| ECE > 0.15 même après sweep | Bloquant pour P1 | Moyenne | Ajuster τ_accuracy, tester d'autres fonctions de normalisation de U, envisager température scaling post-hoc | OUVERT |
| MI < 0.5 | Affaiblit la thèse iso-structurale | Moyenne | Analyser si le problème est la compression scalaire ; tester U vectoriel sur un sous-ensemble | OUVERT |
| MiniGrid FourRooms trop petit (< 100 états) | Statistiques pauvres sur les bins ECE | Faible | Utiliser une grille plus grande (11×11 au lieu de default) | **RÉSOLU** — FourRooms = 19×19 = 260 états accessibles |
| DynamicsWrapper incompatible avec SB3 | Bloque baseline Q-learning Exp C | Faible | Implémenter un Q-learning tabulaire custom (simple) | MITIGÉ — fallback tabulaire prévu |
| Temps de calcul 810 runs sweep | Retard Phase 2 | Moyenne | Réduire à 27 configs (3³) en fixant U_prior=0.8 d'abord | OUVERT |
| Asymétrie R/M hors plage 15-40× | Signature SR non confirmée | Faible | Analyser les causes, ajuster les learning rates, discuter | OUVERT |
| PRISM ≈ SR-Count-Bonus (pas de gain structure) | Affaiblit P3 | Moyenne | Analyser les conditions où la structure aide vs. pas (topologies simples vs. complexes) | OUVERT |

---

## Checklist de livraison finale

- [x] Tous les tests unitaires passent (`pytest tests/`) — 141 tests ✅
- [ ] 3 expériences exécutées avec 100 runs chacune → Phase 3
- [ ] Analyses statistiques complètes (p-values, CI, effect sizes) → Phase 3
- [ ] 10 figures principales générées → Phase 3
- [x] Notebook 01 : validation SR ✅ (+ CP1 go/no-go intégré)
- [ ] Notebook 02 : démo méta-SR → Phase 3
- [ ] Notebook 03 : analyse finale → Phase 3
- [ ] Code reproductible (seeds fixés, `run_all.py` fonctionne) → Phase 3
- [x] `README.md` avec instructions d'installation et reproduction ✅
- [ ] Rapport de résultats rédigé → Phase 3
- [ ] 5 checkpoints humains validés (CP1 ✅, CP2-CP5 → Phase 3)

---

## Checkpoints humains

Système de validation humaine à chaque étape clé. Détails complets dans `checkpoints.md`.

| CP | Nom | Quand | Critères clés | Statut |
|----|-----|-------|---------------|--------|
| CP1 | Validation SR de base | Fin Phase 1 | ‖ΔM‖ < 0.1, rang > 50%, eigenvalue > 1, ECE < 0.30, MI > 0 | ✅ PASSED |
| CP2 | Hyperparamètres méta-SR | Après sweep (§2.7) | ECE < 0.15 pour top config, stabilité inter-runs | ⏳ À VENIR |
| CP3 | Calibration métacognitive | Après Exp A (§2.8) | ECE < 0.15, MI > 0.5, PRISM < baselines (p < 0.05) | ⏳ À VENIR |
| CP4 | Exploration dirigée | Après Exp B (§3.3) | PRISM < ε-greedy (−30%), PRISM < Count-Bonus (p < 0.05) | ⏳ À VENIR |
| CP5 | Adaptation au changement | Après Exp C (§3.4) | Détection < 10 épisodes, asymétrie R/M 15-40× | ⏳ À VENIR |

**Ordre des tâches Phase 3 avec checkpoints intercalés :**

1. Compléter stubs (baselines, perturbation_schedule, metrics)
2. Sweep hyperparamètres → **CP2**
3. Exécuter Exp A → **CP3**
4. Config grand monde + baselines Exp B
5. Exécuter Exp B → **CP4**
6. SB3/tabulaire wrapper + Exp C → **CP5**
7. Analyse croisée, figures, notebook 03, rapport
