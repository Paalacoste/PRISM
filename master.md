# PRISM — Predictive Representation for Introspective Spatial Metacognition

> Premier test computationnel de la thèse neuroscientifique de la méta-carte hippocampique :
> la successor representation comme substrat unifié pour la cognition et la métacognition,
> évalué avec les outils de la psychophysique.

---

## 1. Revue de littérature

### 1.1 Successor representations — fondements

Le formalisme des successor representations a été introduit par **Dayan (1993)** comme compromis entre apprentissage model-free (efficace mais rigide) et model-based (flexible mais coûteux). L'idée centrale est la décomposition de la fonction de valeur V(s) = M · R, où M encode les transitions prédites et R les récompenses, permettant une adaptation rapide quand l'un change indépendamment de l'autre.

**Stachenfeld, Botvinick & Gershman (2017, Nature Neuroscience)** ont reformulé l'hippocampe comme "carte prédictive" : les cellules de lieu CA1 n'encodent pas la position géodésique mais la probabilité de transition vers les positions futures. Les grid cells du cortex entorhinal émergent comme eigenvectors de la matrice SR — une compression spectrale multi-échelle. Cette théorie prédit et explique l'expansion asymétrique des champs de lieu, le clustering, la sensibilité à la récompense et les cellules de temps.

**Gershman (2018, J. Neuroscience)** a fourni une synthèse de la logique computationnelle et des substrats neuronaux de la SR, établissant qu'elle ne fonctionne pas en isolation mais interagit avec des computations model-based et model-free.

**Momennejad, Russek et al. (2017, Nature Human Behaviour)** ont fourni les premières preuves comportementales chez l'humain : les sujets montrent une sensibilité aux changements de récompense (comme prédit par la SR) mais une insensibilité aux changements de transition (signature unique de la SR vs. model-based). Leurs données montrent un modèle hybride SR–MB.

**Russek, Momennejad et al. (2017, PLoS Comp Bio)** ont formalisé comment les computations model-based peuvent être construites sur un socle de TD learning via la SR, avec des extensions Dyna-SR qui utilisent le replay hippocampique pour mettre à jour la matrice M offline.

### 1.2 Au-delà de l'espace physique — espaces cognitifs

**Bellmund et al. (2018, Science)** ont montré que les codes spatiaux hippocampiques opèrent sur des "espaces cognitifs" abstraits — des espaces dont les dimensions peuvent être le poids, la hiérarchie sociale, ou les features sémantiques.

**Theves, Fernandez & Doeller (2020, J. Neuroscience)** ont prouvé que l'hippocampe cartographie l'espace conceptuel plutôt que l'espace des features bruts : le signal de distance hippocampique reflète sélectivement les dimensions conceptuellement pertinentes.

**Stoewer et al. (2023, Scientific Reports)** ont démontré que des réseaux de neurones artificiels apprenant des SR sur des espaces sémantiques (32 espèces animales) construisent avec succès des cartes cognitives capturant les similarités entre concepts.

**Ekman et al. (2023, eLife)** ont montré que le cortex visuel primaire V1 et l'hippocampe représentent une carte prédictive apparentée à la SR — les représentations prédictives imprègnent le traitement perceptif lui-même.

### 1.3 Hippocampe et métacognition — la thèse de la méta-carte

Le lien le plus direct avec PRISM provient de la thèse de la **méta-carte hippocampique** proposée par **Ambrogioni & Ólafsdóttir (2023, Trends in Cognitive Sciences)** — « Rethinking the hippocampal cognitive map as a meta-learning computational module » : l'hippocampe n'encode pas seulement des cartes d'environnements familiers, mais aussi des états informationnels et des sources d'information. Les cartes cognitives feraient partie d'une méta-représentation plus large qui soutient l'exploration et fournit un fondement pour l'apprentissage en contexte d'incertitude.

**Allen et al. (2017, NeuroImage)** ont montré par IRM quantitative que la capacité métacognitive corrèle avec la microstructure de l'hippocampe et du cortex préfrontal antérieur — confirmation neuroanatomique que métacognition et cognition spatiale partagent des substrats.

**Qiu et al. (2024, Communications Biology)** ont confirmé en IRMf que l'hippocampe, le cortex entorhinal et le cortex orbitofrontal collaborent pour apprendre la structure d'espaces abstraits multidimensionnels.

### 1.4 SR et incertitude — travaux existants

**Janz et al. (2019, NeurIPS) — Successor Uncertainties.** Combinaison de successor features avec la régression linéaire bayésienne pour propager l'incertitude à travers la structure temporelle du MDP. L'incertitude guide l'exploration via posterior sampling (PSRL). Surpasse la performance humaine sur 38/49 jeux Atari. C'est le travail le plus proche de PRISM sur l'axe SR + incertitude.

**Machado, Bellemare & Bowling (2020, AAAI) — Count-based exploration with SR.** Utilisent la norme de la SR comme proxy pour les visites d'états, dérivant des bonus d'exploration count-based à partir de la structure SR.

**Flennerhag et al. (2020, DeepMind) — Temporal Difference Uncertainties as Signal for Exploration.** Proposent d'utiliser les incertitudes des différences temporelles comme signal d'exploration, conceptuellement proche du monitoring d'erreurs TD de PRISM.

### 1.5 Métacognition en IA — frameworks existants

**Valiente & Pilly (2024, arXiv; 2025, Neural Networks) — MUSE Framework.** Intègre self-assessment et self-regulation dans des agents autonomes. Deux implémentations : world model et LLM. Testé dans Meta-World et ALFWorld. Le framework le plus complet pour la métacognition computationnelle, mais n'utilise pas la SR comme substrat.

**Kawato et al. (2021, Biological Cybernetics) — From Internal Models toward Metacognitive AI.** Propose un modèle computationnel de la métacognition basé sur des paires de modèles génératifs-inverses avec un "responsibility signal" qui gate la sélection et l'apprentissage. Le signal de responsabilité est conceptuellement proche du monitoring d'erreurs de prédiction de PRISM.

**Meta-Cognitive RL (VPES).** Framework récent où un méta-contrôleur monitore la stabilité des erreurs de prédiction de valeur (Value Prediction Error Stability) pour réguler le taux d'apprentissage. Architecturalement proche de la méta-SR de PRISM.

**Steyvers & Peters (2025, Perspectives on Psychological Science).** Survey sur la métacognition et la communication d'incertitude chez les humains et les LLMs, identifiant la calibration de confiance comme métrique clé.

### 1.6 Cadre englobant — l'Espace de Travail Neuronal Global (GNW)

La Global Neuronal Workspace de **Dehaene & Changeux (1998, 2011)** est la théorie dominante de l'accès conscient : des neurones pyramidaux à axones longs (préfrontaux, pariétaux) forment un workspace global où l'information subit une "ignition" non-linéaire, tout-ou-rien, la rendant accessible à l'ensemble des processeurs spécialisés. La GNW est une théorie du **broadcast** — ce qui entre dans le workspace devient conscient.

Deux résultats rendent la GNW pertinente pour PRISM :

**L'hippocampe fait partie du core du workspace.** Deco, Vidaurre & Kringelbach (2021, *Nature Human Behaviour*) ont quantifié empiriquement le "functional rich club" constituant le workspace global à travers sept tâches + repos. L'hippocampe figure dans le noyau central, aux côtés du precuneus, du cingulaire postérieur et du noyau accumbens. La carte prédictive SR n'est donc pas un processus périphérique isolé — elle alimente directement le hub de diffusion global.

**Le "predictive global workspace".** Whyte & Smith (2020, *Progress in Neurobiology*) intègrent la GNW avec l'active inference de Friston, montrant que le workspace peut être compris comme le lieu où les erreurs de prédiction sont sélectionnées et diffusées. PRISM opère exactement dans cet espace d'erreurs de prédiction — en amont du broadcast.

La GNW et PRISM opèrent à des échelles différentes et ne sont pas en compétition. La GNW décrit comment l'information métacognitive devient **globalement accessible**. PRISM décrit **d'où elle vient** — le monitoring de la structure prédictive SR au sein du module hippocampique. Le positionnement précis est développé en §3.4.

---

## 2. Revue des résultats et implémentations existants

### 2.1 Ce qui a été démontré expérimentalement

| Résultat | Auteurs | Statut |
|----------|---------|--------|
| SR tabulaire converge dans FourRooms | Juliani (2019, tutorial) | ✅ Reproduit, code dispo |
| Eigenvectors de M → patterns grid-like | Stachenfeld et al. (2017); Chelu (repo) | ✅ Reproduit, code dispo |
| Transfert SR quand R change (M réutilisé) | Juliani (2019); Barreto et al. (2017) | ✅ Reproduit, code dispo |
| Humains utilisent SR + arbitrage SR/MB | Momennejad et al. (2017) | ✅ Données + modèle dispo |
| SR + incertitude bayésienne → exploration | Janz et al. (2019) | ✅ Résultats Atari-scale |
| Count-based exploration via norme SR | Machado et al. (2020) | ✅ Résultats AAAI |
| SF apprises depuis pixels dans MiniGrid | Chua et al. (2024) | ✅ Code dispo |
| Métacognition comme self-assessment RL | Valiente & Pilly (2024) | ✅ Meta-World + ALFWorld |

### 2.2 Ce qui n'a PAS été fait

| Gap | Pourquoi c'est un gap | PRISM le comble ? |
|-----|----------------------|-------------------|
| Carte d'incertitude iso-structurale à la SR | Successor Uncertainties propage l'incertitude mais ne construit pas une carte spatiale parallèle | ✅ Contribution principale |
| Calibration psychophysique d'un agent SR | Personne n'a mesuré l'ECE d'un agent SR ni produit de reliability diagram | ✅ Protocole Exp A |
| Signal "je ne sais pas" calibré et continu | MUSE fait du self-assessment mais sans métriques de calibration formelles | ✅ Protocole Exp A |
| Test computationnel de la méta-carte hippocampique | La thèse TiCS 2023 est théorique, jamais implémentée | ✅ Cadrage du projet |
| Exploration dirigée par incertitude SR structurale | Machado (2020) utilise la norme SR ; Janz (2019) utilise le posterior — ni l'un ni l'autre n'utilise une carte U(s) parallèle | ✅ Protocole Exp B |
| Comparaison incertitude SR structurale vs. bayésienne vs. count-based | Chaque approche a été évaluée isolément | ✅ Protocole Exp B |

### 2.3 Assets réutilisables

| Asset | Source | Usage dans PRISM |
|-------|--------|-----------------|
| **MiniGrid** FourRooms | Farama Foundation (NeurIPS 2023) | Environnement de base — pas de gridworld custom |
| SR tabulaire + visualisations | Juliani (2019) | Point de départ pour l'agent SR |
| Décomposition spectrale SR | Chelu (github/temporal_abstraction) | Visualisation eigenvectors, eigenvalues |
| Modèle SR/MB hybride | Russek et al. (2017, github) | Référence pour l'arbitrage |
| Simple Successor Features | Chua et al. (2024, github) | Deep SF si extension future |
| Baselines RL | Stable-Baselines3 | Q-learning, DQN baselines |

---

## 3. Positionnement de PRISM

### 3.1 Carte de positionnement

```
    Axe Y : Rigueur métacognitive (métriques psychophysiques)
    â–²
    │
    │   ┌─────────┐
    │   │  PRISM  │  SR comme substrat naturel pour la métacognition
    │   │         │  Calibration ECE, reliability diagrams
    │   │         │  Carte d'incertitude iso-structurale
    │   └─────────┘
    │        ▲
    │        │ apporte les métriques        apporte le substrat SR
    │        │ métacognitives                    │
    │   ┌────┴────┐                    ┌────────┴─────────┐
    │   │  MUSE   │                    │ Succ. Uncertain. │
    │   │         │                    │                  │
    │   │ Self-assessment              │ SR + bayésien    │
    │   │ Self-regulation              │ pour exploration │
    │   │ (world model / LLM)          │ (posterior samp.)│
    │   └─────────┘                    └──────────────────┘
    │
    ├──────────────────────────────────────────────────────► Axe X : Ancrage SR
    │
    │   ┌───────────┐          ┌──────────────┐
    │   │ VPES /    │          │ Machado 2020 │
    │   │ Meta-Cog  │          │ Count + SR   │
    │   │ RL        │          │              │
    │   └───────────┘          └──────────────┘
```

### 3.2 Contribution unique

**PRISM est le premier projet à :**

1. Construire une **carte d'incertitude iso-structurale** à la SR — même formalisme pour cognition de premier ordre (M : "où vais-je ?") et métacognition (U : "est-ce que je sais où je vais ?")

2. Mesurer la **calibration métacognitive** d'un agent SR avec les outils de la psychophysique (ECE, reliability diagrams, Metacognitive Index) — traiter un agent RL comme un sujet de psychologie cognitive

3. **Tester computationnellement** la thèse de la méta-carte hippocampique (TiCS, 2023), en montrant que la structure prédictive de la SR suffit à faire émerger des comportements métacognitifs sans module métacognitif externe

### 3.3 Ce que PRISM ne prétend PAS faire

- Surpasser Successor Uncertainties en performance d'exploration (ils opèrent à l'échelle Atari, PRISM est tabulaire)
- Remplacer MUSE comme framework général de métacognition (PRISM est spécifique au substrat SR)
- Prouver que le cerveau utilise la méta-SR (PRISM est un test computationnel, pas une validation neurobiologique)
- Modéliser la conscience ou l'accès conscient (c'est le territoire de la GNW, voir ci-dessous)

### 3.4 Positionnement par rapport à l'Espace de Travail Neuronal Global (GNW)

La Global Neuronal Workspace de Dehaene-Changeux (1998, 2011) est la théorie dominante de l'accès conscient. PRISM et la GNW ne sont pas en compétition — ils opèrent à des échelles différentes.

**PRISM modélise un processeur spécialisé qui alimente le workspace.** L'hippocampe fait partie du noyau central du workspace global (Deco et al., 2021). La carte prédictive SR et la méta-carte U(s) produisent des signaux — erreurs de prédiction, incertitude — qui peuvent être diffusés vers le workspace. PRISM modélise la **computation locale** qui génère ces signaux. La GNW modélise comment ils deviennent **globalement accessibles**.

| | GNW (Dehaene-Changeux) | PRISM |
|---|---|---|
| Échelle | Cerveau entier | Module hippocampique |
| Mécanisme clé | Ignition + broadcast | Erreur de prédiction SR + méta-carte |
| Question centrale | Comment l'information devient consciente ? | D'où vient le signal d'incertitude ? |
| Métacognition | Requiert l'accès au workspace | Émerge de la structure prédictive locale |
| Dynamique | Tout-ou-rien (seuil d'ignition) | Continue (U(s)) + seuil (détection changement) |

**Point de contact clé — le seuil de détection.** La **détection de changement** de PRISM — quand `change_score > θ_change` — a la structure d'un seuil d'ignition GNW : une transition discrète qui réoriente la stratégie de l'agent. Le `θ_change` pourrait être l'analogue fonctionnel du seuil d'ignition, local à l'hippocampe. Tester si ce seuil exhibe les propriétés de l'ignition (non-linéarité, hystérésis) est une extension future hors-scope de la v1.

---

## 4. Thèse resserrée

### Hypothèse principale

> La successor representation fournit un substrat **naturel** pour la métacognition :
> une carte d'incertitude construite à partir des erreurs de prédiction SR
> (iso-structurale à la carte prédictive elle-même) produit des signaux de confiance
> **mieux calibrés** que les approches d'incertitude non-structurées,
> et cela soutient la thèse neuroscientifique de la méta-carte hippocampique.

### Prédictions testables

**P1 — Calibration.** Le signal de confiance C(s) dérivé de la méta-SR est calibré : les décisions à haute confiance sont correctes plus souvent que les décisions à basse confiance. ECE < 0.15.

**P2 — Iso-structuralité.** La carte d'incertitude U(s) a une structure spatiale cohérente avec la carte prédictive M : les frontières d'incertitude correspondent aux frontières topologiques du monde (portes, zones inexplorées, zones récemment modifiées).

**P3 — Avantage de la structure.** L'exploration guidée par U(s) (structurée spatialement) est plus efficace que l'exploration guidée par des signaux d'incertitude non-structurés (count-based, ε-greedy, variance globale).

---

## 5. Architecture

### 5.1 Vue d'ensemble

```
┌──────────────────────────────────────────────────────────┐
│              MONDE — MiniGrid FourRooms                  │
│  (Farama Foundation, asset existant)                     │
│  + DynamicsWrapper (à coder)                             │
│    - Déplacement de récompense                           │
│    - Blocage/ouverture de porte                          │
│    - Schedule de perturbations                           │
└────────────────────────┬─────────────────────────────────┘
                         │ (s, a, r, s')
                         â–¼
┌──────────────────────────────────────────────────────────┐
│                  AGENT PRISM                              │
│                                                          │
│  ┌────────────────────────────────────────────────┐      │
│  │  Couche SR — premier ordre                     │      │
│  │  (adapté de Juliani 2019 / Chua et al. 2024)   │      │
│  │  M(s,s') : transitions prédites (TD learning)  │      │
│  │  R(s) : récompenses apprises                   │      │
│  │  V(s) = M · R                                  │      │
│  └─────────────────────┬──────────────────────────┘      │
│                        │ δ(s) = || TD error on M ||       │
│                        ▼                                  │
│  ┌────────────────────────────────────────────────┐      │
│  │  Couche Méta-SR — CONTRIBUTION PRISM ★         │      │
│  │  U(s) : carte d'incertitude (buffer δ glissant)│      │
│  │  C(s) : signal de confiance calibré            │      │
│  │  Détection de changement structurel            │      │
│  │  Iso-structurale à M par construction          │      │
│  └─────────────────────┬──────────────────────────┘      │
│                        │ C(s), U(s)                       │
│                        ▼                                  │
│  ┌────────────────────────────────────────────────┐      │
│  │  Contrôleur                                    │      │
│  │  ε_adaptive(s) = f(U(s))                       │      │
│  │  V_explore(s) = V(s) + λ · U(s)               │      │
│  │  Signal "je ne sais pas" quand C(s) < θ        │      │
│  └────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────┘
```

### 5.2 Le monde — MiniGrid + DynamicsWrapper

**Base :** `MiniGrid-FourRooms-v0` (Farama Foundation). Grille modulaire avec 4 pièces connectées par des portes. Interface Gymnasium standard.

**Extension custom — `DynamicsWrapper` :** Wrapper Gymnasium qui ajoute les perturbations dynamiques au-dessus de n'importe quel env MiniGrid. C'est le seul composant "monde" à coder.

```python
class DynamicsWrapper(gymnasium.Wrapper):
    """Ajoute des perturbations contrôlées à un env MiniGrid."""
    
    def apply_perturbation(self, ptype: str, **kwargs):
        """Types : 'reward_shift', 'door_block', 'door_open', 'combined'"""
    
    def set_schedule(self, schedule: PerturbationSchedule):
        """Schedule configurable : périodique, aléatoire, triggered."""
    
    def get_state_index(self, pos: tuple) -> int:
        """Mapping position → index d'état pour la matrice SR."""
    
    def get_true_transition_matrix(self) -> np.ndarray:
        """Ground truth pour validation."""
```

### 5.3 La couche SR — premier ordre

Adapté depuis les implémentations existantes (Juliani 2019). Pas de contribution ici — c'est un composant standard.

**Matrice SR — M ∈ ℝ^(N×N) :**

```
M(s, s') = E[ Σ_t γ^t 𝟙(s_t = s') | s_0 = s, π ]
```

**Mise à jour TD(0) :**

```
δ_M(s) = e(s') + γ · M(s',:) - M(s,:)
M(s,:) ← M(s,:) + α_M · δ_M(s)
```

**Fonction de valeur :** V(s) = M(s,:) · R

**Paramètres :**

| Paramètre | Symbole | Défaut | Rôle |
|-----------|---------|--------|------|
| Discount factor | γ | 0.95 | Horizon temporel SR |
| Learning rate SR | α_M | 0.1 | Vitesse d'apprentissage M |
| Learning rate R | α_R | 0.3 | Vitesse d'apprentissage R |
| Exploration base | ε | 0.1 | Taux exploration par défaut |

### 5.4 La couche Méta-SR — CONTRIBUTION PRINCIPALE ★

L'idée fondatrice : la carte d'incertitude a **exactement la même structure** que la carte prédictive. Même indexation par état, même granularité spatiale. Ce n'est pas un module externe qui observe la SR — c'est un **reflet** de la SR.

**Erreur de prédiction SR scalaire par visite :**

```
δ(s) = || e(s') + γ · M(s',:) - M(s,:) ||₂
```

**Justification de la compression scalaire.** Le vecteur d'erreur TD complet δ_vec(s) ∈ ℝ^N contient de l'information directionnelle (vers quels états la prédiction est mauvaise), mais la norme L2 suffit pour notre objectif principal : mesurer si l'agent sait que sa carte est fiable *en un état donné*. La version scalaire permet de maintenir l'iso-structuralité (un scalaire par état, comme M a une ligne par état) tout en restant computationnellement légère. Une extension vectorielle U(s, s') — qui conserverait la structure complète — est envisageable mais sort du scope de la v1. La compression scalaire est testée empiriquement : si le MI (corrélation entre U(s) et l'erreur réelle) est élevé, la compression ne perd pas d'information critique pour la calibration.

**Buffer d'erreurs glissant — ΔM_history(s) :**

Pour chaque état s, buffer circulaire de taille K (défaut : 20) des δ observés lors des visites à s.

**Carte d'incertitude — U(s) ∈ [0, 1] :**

```
U(s) = {
    mean(ΔM_history(s))           si visits(s) ≥ K
    U_max                          si visits(s) = 0
    U_prior · decay^(visits(s))    si 0 < visits(s) < K
}
```

**Signal de confiance — C(s) ∈ [0, 1] :**

```
C(s) = 1 - sigmoid(β · (U(s) - θ_C))
```

**Détection de changement :**

```
change_score = mean(U(s) for s in recently_visited)
change_detected = change_score > θ_change
```

**Exploration adaptative :**

```
ε_adaptive(s) = ε_min + (ε_max - ε_min) · U(s) / U_max
V_explore(s) = V(s) + λ · U(s)
```

**Paramètres méta-SR — valeurs par défaut et justification :**

| Paramètre | Symbole | Défaut | Justification |
|-----------|---------|--------|---------------|
| Taille buffer | K | 20 | ~5 traversées complètes d'une pièce de FourRooms. Assez pour estimer la variance, assez petit pour détecter les changements. |
| Prior d'incertitude | U_prior | 0.8 | Conservateur : un état non visité est supposé hautement incertain. |
| Decay du prior | decay | 0.85 | Chaque visite réduit l'incertitude prior de 15%. Après 10 visites, U ≈ 0.16 (basse). |
| Pente sigmoïde confiance | β | 10 | Transition nette autour de θ_C. Validé par sweep [5, 10, 20] en Exp A. |
| Seuil de confiance | θ_C | 0.3 | Centre de la sigmoïde C(s). U < 0.3 → haute confiance, U > 0.3 → basse. |
| Seuil de changement | θ_change | 0.5 | Détection de changement. Validé par analyse ROC en Exp C. |
| Bonus exploration | λ | 0.5 | Poids relatif exploration/exploitation dans V_explore. |
| Epsilon min | ε_min | 0.01 | Plancher d'exploration même en haute confiance. |
| Epsilon max | ε_max | 0.5 | Plafond d'exploration en haute incertitude. |

**Analyse de sensibilité (Exp A, phase préliminaire) :** Avant les comparaisons formelles, un sweep factoriel sur {U_prior, decay, β, θ_C} sera réalisé (4 paramètres × 3 valeurs = 81 configs, 10 runs chacune). Le critère de sélection est l'ECE minimal sur la phase d'apprentissage stable. Les paramètres sélectionnés sont ensuite fixés pour toutes les expériences. Ce sweep est reporté en annexe pour éviter le p-hacking.

**Propriété clé — iso-structuralité :** U est indexé par les mêmes états que M. On peut superposer visuellement la carte prédictive et la carte d'incertitude. Les frontières de haute incertitude devraient correspondre aux frontières topologiques (portes, zones inexplorées, zones perturbées). C'est cette propriété qui est testée dans l'Exp A.

---

## 6. Protocole expérimental

Trois expériences profondes au lieu de cinq superficielles. Chacune teste une prédiction spécifique.

### 6.1 Expérience A — Calibration métacognitive (teste P1 + P2)

**Question :** Le signal de confiance C(s) est-il calibré ? La carte U(s) est-elle iso-structurale à M ?

**Protocole :**

1. **Phase apprentissage** (300 épisodes) : monde stable, 4 pièces, goal fixe. L'agent apprend M et construit U.
2. **Phase exploration** (100 épisodes) : on ouvre une nouvelle zone (5e pièce) jamais vue.
3. **Phase perturbation** (100 épisodes) : on déplace le goal dans la nouvelle zone.
4. À chaque step, l'agent émet C(s) — sa confiance.

**Métriques :**

**Calibration — Expected Calibration Error (ECE) :**

```
ECE = Σ_b (|B_b| / N) · |accuracy(B_b) - confidence(B_b)|
```

On découpe les prédictions en 10 bins de confiance. Pour chaque bin, on compare la confiance moyenne C(s) et le taux de « prédictions fiables ». **Définition opérationnelle de l'accuracy :** une prédiction est considérée comme fiable quand l'erreur réelle de la SR est faible, i.e. ||M(s,:) - M*(s,:)||₂ < τ_accuracy, où M* est la vraie matrice de transition. Ce choix est cohérent avec ce que C(s) est censé prédire : non pas la stochasticité des transitions (nulle dans MiniGrid — l'environnement est déterministe), mais la *fiabilité de la carte M elle-même*. Le seuil τ_accuracy est fixé au 50e percentile de ||M - M*|| sur l'ensemble des états, de sorte que la baseline d'accuracy est ~50%. Cela garantit une dynamique informative dans le reliability diagram.

**Iso-structuralité — Corrélation spatiale :**

```
ρ = corr(U(s), d(s, frontier))
```

La carte d'incertitude devrait corréler avec la distance aux frontières topologiques (portes, zones inexplorées). On mesure aussi la corrélation entre U(s) et l'erreur réelle de la SR (ground truth) :

```
MI = corr(U(s), ||M(s,:) - M*(s,:)||)  où M* est la vraie matrice de transition
```

MI = Metacognitive Index. C'est la métrique reine : l'agent sait-il ce qu'il ne sait pas ?

**Reliability diagram :** graphique confiance déclarée vs. accuracy observée, par bin. Une courbe sur la diagonale = calibration parfaite.

**Conditions :**

| Condition | Signal de confiance | Description |
|-----------|--------------------|-------------|
| **PRISM** | C(s) = f(U(s)), U structuré spatialement | Notre approche |
| SR-Global | Confiance = f(erreur TD moyenne globale) | Incertitude non-structurée |
| SR-Count | Confiance = f(1/√visits(s)) | Count-based (Machado-like) |
| SR-Bayesian | Posterior sur V via régression linéaire | Successor Uncertainties-like |
| Random-Conf | Confiance aléatoire | Baseline plancher |

**Critères de succès :**
- ECE(PRISM) < 0.15
- MI(PRISM) > 0.5 (corrélation modérée à forte)
- ECE(PRISM) < ECE(SR-Global) et ECE(SR-Count) — la structure spatiale aide la calibration
- Le reliability diagram montre une corrélation positive claire

**Visualisations :**
- Heatmap de M pour quelques états sources (validation SR standard)
- Heatmap de U superposée au monde — la carte d'incertitude
- Reliability diagram par condition
- Évolution temporelle de U après perturbation (animation ou séquence)
- Top-6 eigenvectors de M (validation spectrale standard)

### 6.2 Expérience B — Exploration dirigée par incertitude structurelle (teste P3)

**Question :** L'exploration guidée par U(s) (structuré spatialement) est-elle plus efficace que les alternatives ?

**Protocole :**

1. Grand monde MiniGrid (19×19) avec 4+ pièces
2. 4 goals cachés, un par pièce (l'agent ne les connaît pas au départ)
3. L'agent doit trouver les 4 goals le plus vite possible
4. Comparer l'efficacité d'exploration selon la stratégie

**Conditions :**

| Condition | Stratégie d'exploration | Signal directeur |
|-----------|------------------------|------------------|
| **PRISM** | V_explore = V + λ·U(s) | Carte U structurée |
| SR-Oracle | V + λ·||M(s,:) - M*(s,:)|| | Erreur réelle (plafond théorique) |
| SR-ε-greedy | ε fixe = 0.1 | Aucun |
| SR-ε-decay | ε décroissant | Aucun |
| SR-Count-Bonus | V + λ/√visits(s) | Comptage (Machado-like) |
| SR-Norm-Bonus | V + λ/||M(s,:)|| | Norme SR (Machado 2020) |
| SR-Posterior | Posterior sampling sur V | Bayésien (Janz-like) |
| Random | Uniformément aléatoire | Baseline plancher |

**SR-Oracle** connaît les vraies erreurs de M et les utilise comme bonus. C'est un plafond de performance — aucun agent réaliste ne peut faire mieux. Le ratio (performance PRISM - Random) / (performance Oracle - Random) quantifie quelle fraction du gain théorique PRISM capture (« efficiency ratio »).

**Métriques :**
- Steps pour trouver les 4 goals (moyenne sur 100 runs)
- Couverture (% d'états visités) vs. steps
- Redondance : ratio revisites / nouvelles visites
- Corrélation entre l'ordre de visite des régions et leur U(s)
- Efficiency ratio : (steps_Random - steps_PRISM) / (steps_Random - steps_Oracle) — fraction du gain théorique capturée

**Critère de succès :** PRISM trouve les 4 goals en significativement moins de steps que SR-ε-greedy et SR-Count-Bonus.

**Test différentiel clé :** PRISM vs. SR-Count-Bonus isole l'apport de la structure. Les deux donnent un bonus d'exploration, mais PRISM utilise l'erreur de prédiction SR (structurée) tandis que Count-Bonus utilise les visites (non-structurée). Si PRISM gagne, c'est que la structure prédictive de la SR apporte quelque chose au-delà du simple comptage.

### 6.3 Expérience C — Adaptation au changement (teste P1 + P2 en dynamique)

**Question :** L'agent détecte-t-il les changements et adapte-t-il son comportement, tout en maintenant une confiance calibrée ?

**Protocole :**

1. **Phase stable** (200 épisodes) : monde fixe, l'agent maîtrise l'environnement.
2. **Perturbation de type R** (100 épisodes) : goal déplacé. M reste valide, R change.
3. **Re-stabilisation** (100 épisodes) : l'agent se réadapte.
4. **Perturbation de type M** (100 épisodes) : porte bloquée. M devient invalide, R ne change pas.
5. **Re-stabilisation finale** (100 épisodes).

Ce design teste la prédiction SR classique (Momennejad 2017) : l'adaptation au changement de R devrait être rapide (seul R est mis à jour), l'adaptation au changement de M devrait être lente (toute la matrice doit être réapprise).

**Prédiction quantitative de l'asymétrie R/M.** Pour un changement de R (goal déplacé), l'adaptation nécessite ~O(1/α_R) épisodes pour converger — avec α_R = 0.3, cela donne ~3-5 épisodes. Pour un changement de M (porte bloquée), les lignes de M correspondant aux N_affected états dont les transitions changent doivent être réapprises — cela prend ~O(N_affected / α_M) épisodes. Dans FourRooms, bloquer une porte affecte ~8-12 états adjacents à la porte ; avec α_M = 0.1, cela donne ~80-120 épisodes. Le ratio prédit est donc latence_M / latence_R ≈ 15-40×. Si le ratio observé tombe significativement en dehors de cette plage, cela pointerait vers un mécanisme non-SR (trop bas → model-based ; trop haut → pas de réapprentissage M).

**Métriques :**
- **Latence de détection** : épisodes avant `change_detected = true`
- **Latence d'adaptation** : épisodes pour retrouver 80% de la performance pré-perturbation
- **Calibration dynamique** : ECE mesuré dans une fenêtre glissante de 20 épisodes — la calibration se maintient-elle pendant et après les transitions ?
- **Asymétrie R vs. M** : ratio latence_M / latence_R — devrait être >> 1 si la SR est bien le mécanisme sous-jacent

**Conditions :**

| Condition | Description |
|-----------|-------------|
| **PRISM** | Agent complet avec méta-SR et détection |
| SR-Blind | Agent SR sans monitoring (ε fixe) |
| Q-Learning | Model-free classique (Stable-Baselines3) |

**Critères de succès :**
- PRISM détecte les changements en < 10 épisodes
- Latence d'adaptation : PRISM ≤ 0.5 × SR-Blind
- Asymétrie R/M observable (confirmation de la signature SR)
- ECE reste < 0.20 même pendant les transitions

### 6.4 Plan d'analyse statistique

**Nombre de runs et puissance.** Chaque condition est exécutée 100 fois avec des seeds aléatoires différentes (Exp A et C : 100 runs × ~500 épisodes ; Exp B : 100 runs × durée variable). Ce nombre garantit une puissance statistique suffisante pour détecter des différences d'effet moyen (Cohen's d ≥ 0.5) avec α = 0.05.

**Tests de comparaison (Exp A, B).** Les distributions de métriques (ECE, steps, MI) entre conditions ne sont pas supposées normales. Les comparaisons deux-à-deux utilisent le test de Mann-Whitney U (unilatéral quand la direction est prédite, bilatéral sinon). La correction de Holm-Bonferroni est appliquée pour les comparaisons multiples — seules les comparaisons pré-spécifiées dans les critères de succès sont testées, pas de fishing.

**Intervalles de confiance.** Les intervalles de confiance à 95% sur l'ECE et le MI sont calculés par bootstrap non-paramétrique (10 000 re-échantillonnages). Les barres d'erreur dans les figures représentent ces intervalles.

**Tests de calibration (Exp A, C).** En plus de l'ECE, le test de Hosmer-Lemeshow est appliqué pour évaluer formellement la qualité de la calibration dans chaque condition. Un p > 0.05 indique une calibration acceptable.

**Corrélations (Exp A — iso-structuralité).** Les corrélations ρ et MI sont reportées avec des intervalles de confiance bootstrap. La significativité est évaluée par un test de permutation (1000 permutations).

**Taille d'effet.** Toutes les comparaisons reportent le Cohen's d (ou r de rang pour Mann-Whitney) en plus du p-value. Un résultat statistiquement significatif mais avec une taille d'effet faible (d < 0.3) sera discuté comme tel.

---

## 7. Stack technique

### 7.1 Dépendances

```
Python 3.11+
minigrid >= 2.3, < 3.0  # environnement FourRooms (Farama)
gymnasium >= 0.29, < 1.0 # interface standard RL
numpy >= 1.24, < 2.0     # pinné < 2.0 pour compatibilité MiniGrid
scipy >= 1.11            # décomposition spectrale
matplotlib >= 3.7
seaborn >= 0.12          # reliability diagrams, heatmaps
pandas >= 2.0            # logging des résultats
tqdm >= 4.65             # progress bars
pytest >= 7.0            # tests
```

> **Note :** `stable-baselines3` est différé à Phase 3 (baseline Q-learning Exp C).
> Un fallback Q-learning tabulaire est prévu si SB3 pose des problèmes de compatibilité.

### 7.2 Structure du projet

```
PRISM/
├── master.md                          # ← ce document
├── README.md
├── requirements.txt
├── checkpoints.md                     # Protocoles de validation humaine CP1-CP5
│
├── prism/                             # Package Python principal
│   ├── __init__.py
│   ├── config.py                      # [DONE] Hyperparamètres centralisés (PRISMConfig)
│   │
│   ├── env/
│   │   ├── __init__.py
│   │   ├── state_mapper.py            # [DONE] Mapping position MiniGrid → index SR (260 états)
│   │   ├── dynamics_wrapper.py        # [DONE] Wrapper perturbations sur MiniGrid
│   │   ├── exploration_task.py        # [DONE] ExplorationTaskWrapper, place_goals, get_room_cells
│   │   └── perturbation_schedule.py   # [STUB Phase 3] Configs de schedules (exp_a/c)
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── sr_layer.py                # [DONE] SR tabulaire (depuis Juliani 2019)
│   │   ├── meta_sr.py                 # [DONE] ★ Carte U(s), signal C(s), détection
│   │   ├── controller.py              # [DONE] ★ ε adaptatif, V_explore greedy, "je ne sais pas"
│   │   └── prism_agent.py             # [DONE] ★ Agent complet assemblant les couches
│   │
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── base_agent.py              # [DONE] BaseSRAgent + RandomAgent (navigation MiniGrid)
│   │   ├── sr_blind.py                # [DONE] SREpsilonGreedy + SREpsilonDecay
│   │   ├── sr_count.py                # [DONE] SRCountBonus + SRNormBonus
│   │   ├── sr_bayesian.py             # [DONE] SRPosterior (Thompson sampling)
│   │   ├── sr_oracle.py               # [DONE] SROracle (bonus = ||M - M*||)
│   │   └── sb3_baselines.py           # [STUB Phase 3] Q-learning via Stable-Baselines3
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── calibration.py             # [DONE] ★ ECE, reliability diagrams, MI
│   │   ├── spectral.py                # [DONE] Eigenvectors M (depuis Chelu)
│   │   ├── visualization.py           # [DONE] Heatmaps U/M superposées (animation Phase 3)
│   │   └── metrics.py                 # [DONE] bootstrap_ci, mann_whitney, holm_bonferroni, compare_conditions
│   │
│   └── pedagogy/
│       └── toy_grid.py                # [DONE] Grille légère pour notebooks (sans MiniGrid)
│
├── experiments/
│   ├── __init__.py
│   ├── exp_b_exploration.py           # [DONE] Exp B — exploration dirigée (8 conditions)
│   ├── exp_a_calibration.py           # [STUB Phase 3] Exp A — calibration métacognitive
│   ├── exp_c_adaptation.py            # [STUB Phase 3] Exp C — adaptation au changement
│   └── run_all.py                     # [STUB Phase 3] Script batch
│
├── notebooks/
│   ├── 00_prism_concepts.ipynb        # [DONE] Introduction pédagogique aux concepts PRISM
│   ├── 00a_spectral_deep_dive.ipynb   # [DONE] Analyse spectrale de la SR
│   ├── 00b_calibration_methods.ipynb  # [DONE] Méthodes de calibration
│   ├── 01_sr_validation.ipynb         # [DONE] Validation SR + CP1 go/no-go (9 sections)
│   └── 02_experiment_tracking.ipynb   # [TODO] Suivi et analyse Exp B
│
├── tests/                             # 141 tests, 9 fichiers
│   ├── __init__.py
│   ├── conftest.py                    # Fixtures partagées (env, mapper)
│   ├── test_imports.py                # Vérification imports package
│   ├── test_env_smoke.py              # Smoke tests MiniGrid
│   ├── test_state_mapper.py           # Tests StateMapper (260 états, bijection)
│   ├── test_sr_layer.py               # Tests SR Layer (convergence, update)
│   ├── test_dynamics_wrapper.py       # Tests DynamicsWrapper (perturbations)
│   ├── test_meta_sr.py                # Tests MetaSR (uncertainty, confidence, change)
│   ├── test_calibration.py            # Tests calibration (ECE, MI, reliability)
│   └── test_baselines.py              # Tests baselines + metrics + goal placement (44 tests)
│
├── scripts/
│   └── verify_env.py                  # Vérification environnement Python
│
└── results/                           # Généré automatiquement (.gitignored)
    ├── exp_a/
    ├── exp_b/
    └── exp_c/
```

**Légende :**
- ★ = contribution PRISM (code original)
- [DONE] = implémenté et testé (Phases 0-2)
- [STUB Phase 3] = fichier existe avec interface, implémentation en Phase 3

**Compteur :** 141 tests passent, 20 modules Python, 4 notebooks (+1 en cours).

> **Note :** Les baselines (Palier 1 Exp B) et `metrics.py` ont été implémentés en Phase 3.
> Une 9e condition `SR-Count-Matched` est prévue pour le Palier 2 d'Exp B.

---

## 8. Plan d'implémentation

### Phase 1 — Assemblage (semaines 1-2) — ✅ DONE

Objectif : agent SR fonctionnel dans MiniGrid, zéro contribution originale.

- [x] Installer MiniGrid, vérifier FourRooms fonctionne
- [x] `state_mapper.py` — mapping position MiniGrid → index pour matrice SR
- [x] `sr_layer.py` — adapter l'implémentation SR tabulaire de Juliani
- [x] `dynamics_wrapper.py` — wrapper perturbations (reward shift, door block)
- [x] `spectral.py` — adapter le code de visualisation eigenvectors (Chelu)
- [x] Notebook `01_sr_validation.ipynb` — sanity check : SR converge, eigenvectors ok
- [x] Tests unitaires : wrapper, SR layer, state mapper

> **Notes d'implémentation :**
> - FourRooms = 19×19, **260 états** accessibles (pas ~100 comme anticipé)
> - `agent_pos` est un tuple `(x, y)`, pas un int — nécessite `to_grid()` dans StateMapper
> - `max_steps=500` obligatoire pour éviter la troncation silencieuse
> - Pas de portes dans MiniGrid v2.5 (passages ouverts entre les pièces)

**Milestone :** ✅ L'agent SR navigue vers le goal dans FourRooms. Les heatmaps de M et les eigenvectors sont cohérents avec Stachenfeld 2017.
→ **CP1 PASSED** (6 checks automatisés dans notebook 01)

### Phase 2 — Méta-SR et calibration (semaines 3-5) ★ — ⏳ PARTIELLEMENT DONE

Objectif : implémenter la contribution principale et exécuter l'Exp A.

- [x] `meta_sr.py` — buffer d'erreurs, carte U(s), signal C(s), détection
- [x] `controller.py` — ε adaptatif, V_explore, signal "je ne sais pas"
- [x] `prism_agent.py` — assemblage agent complet
- [x] `calibration.py` — ECE, reliability diagrams, Metacognitive Index
- [x] `visualization.py` — superposition U/M, animations
- [ ] Baselines : `sr_blind.py`, `sr_count.py`, `sr_bayesian.py` — stubs seulement
- [ ] **Sweep hyperparamètres méta-SR** — reporté Phase 3 → CP2
- [ ] **Exécuter Exp A** — reporté Phase 3 → CP3
- [ ] Notebook `02_meta_sr_demo.ipynb` — reporté Phase 3

> **Notes d'implémentation :**
> - MetaSR utilise normalisation p99 adaptative (pas min-max naïf)
> - L'action exploit du controller est random (pas V_explore voisins comme prévu)
> - `prism_agent.py` intègre `config.py` (PRISMConfig) pour centraliser les hyperparamètres
> - `hosmer_lemeshow_test()` manquant dans calibration.py — à créer en Phase 3

**Milestone :** Composants logiciels implémentés et testés (97 tests). Exp A et sweep reportés Phase 3.

### Phase 3 — Exploration et adaptation (semaines 6-8) ★ — À VENIR

Objectif : exécuter les Exp B et C, comparaisons avec baselines.

- [ ] Compléter stubs Phase 2 (baselines, perturbation_schedule, metrics)
- [ ] **Sweep hyperparamètres** (depuis Phase 2) → CP2
- [ ] **Exécuter Exp A** (depuis Phase 2) → CP3
- [ ] Config grand monde pour Exp B (19×19, 4+ pièces, 4 goals cachés)
- [ ] **Exécuter Exp B** — exploration dirigée, toutes conditions → CP4
- [ ] **Exécuter Exp C** — adaptation au changement (perturbations R puis M) → CP5
- [ ] `sb3_baselines.py` — wrapper Stable-Baselines3 pour Q-learning baseline
- [ ] SR-Oracle baseline (utilise M* comme signal — plafond théorique Exp B)
- [ ] Analyse croisée des 3 expériences
- [ ] Notebook `03_results_analysis.ipynb` — figures finales
- [ ] Rédaction du rapport de résultats

**Milestone :** PRISM bat les baselines sur l'exploration. L'asymétrie R/M confirme la signature SR. La calibration se maintient en dynamique.

### 8.5 Système de checkpoints

Validation humaine à chaque étape clé. Protocoles détaillés dans `checkpoints.md`.

| CP | Nom | Phase | Critères clés | Statut |
|----|-----|-------|---------------|--------|
| CP1 | Validation SR de base | Fin Phase 1 | ‖ΔM‖ < 0.1, rang > 50%, ECE < 0.30, MI > 0 | ✅ PASSED |
| CP2 | Hyperparamètres méta-SR | Phase 3 (sweep) | ECE < 0.15, stabilité inter-runs | ⏳ À VENIR |
| CP3 | Calibration métacognitive | Phase 3 (Exp A) | ECE < 0.15, MI > 0.5, PRISM < baselines | ⏳ À VENIR |
| CP4 | Exploration dirigée | Phase 3 (Exp B) | PRISM −30% vs ε-greedy, < Count-Bonus | ⏳ À VENIR |
| CP5 | Adaptation au changement | Phase 3 (Exp C) | Détection < 10 épisodes, asymétrie 15-40× | ⏳ À VENIR |

---

## 9. Métriques globales

> **Status :** Cibles définies, pas encore testées expérimentalement.
> Seul CP1 a validé les métriques de base (ECE < 0.30, MI > 0).
> Les cibles ci-dessous seront évaluées en Phase 3 (Exp A/B/C).

### Tableau de bord

| Exp | Métrique | Baseline | Cible PRISM | Teste |
|-----|----------|----------|-------------|-------|
| A | ECE | — | < 0.15 | P1 |
| A | Metacognitive Index (MI) | — | > 0.5 | P2 |
| A | ECE vs. SR-Global | ECE(SR-Global) | ECE(PRISM) < ECE(SR-Global) | P1 |
| B | Steps pour 4 goals | SR-ε-greedy | −30% | P3 |
| B | Steps PRISM vs. SR-Count-Bonus | SR-Count-Bonus | PRISM < Count-Bonus | P3 (structure) |
| B | Efficiency ratio (PRISM vs. Oracle) | SR-Oracle | > 0.5 (capture >50% du gain théorique) | P3 (plafond) |
| C | Latence de détection | SR-Blind | < 10 épisodes | P1 |
| C | Latence adaptation PRISM / SR-Blind | SR-Blind | ≤ 0.5× | P2 |
| C | Asymétrie latence_M / latence_R | — | 15–40× (dérivé analytiquement) | Signature SR |
| C | ECE pendant transitions | — | < 0.20 | P1 dynamique |

### Métriques transversales

- **Metacognitive Index (MI)** = corr(U(s), erreur réelle SR). Métrique reine : l'agent sait-il ce qu'il ne sait pas ?
- **Calibration Maintenance** = ECE mesuré en fenêtre glissante. La calibration se dégrade-t-elle ?
- **Structure Advantage** = gain PRISM vs. SR-Count-Bonus. Isole l'apport de la structure SR.

---

## 10. Extensions futures

### Court terme (si les résultats sont solides)

- **SR multi-échelle** : maintenir plusieurs M avec différents γ, inspiré de l'axe longitudinal de l'hippocampe. Tester si les cartes U à différentes échelles capturent différents types d'incertitude.
- **Replay** : rejeu d'expériences en phases offline pour consolider M, inspiré du replay hippocampique. Tester l'impact sur la stabilité de U.
- **Arbitrage SR/MB** : ajouter un planificateur model-based et utiliser U(s) pour l'arbitrage (Russek et al. 2017). Reporté de la v1 mais prêt architecturalement.

### Moyen terme

- **Deep SR** : remplacer la matrice tabulaire par un réseau (Chua et al. 2024 comme point de départ). La méta-SR peut-elle fonctionner sur des représentations apprises ?
- **Espaces non-spatiaux** : appliquer PRISM à un espace sémantique (Stoewer et al. 2023) — la métacognition SR fonctionne-t-elle au-delà de la navigation ?

### Recherche

- Comparer la structure spectrale de la SR + méta-SR artificielles avec les données électrophysiologiques
- Formaliser le lien méta-SR ↔ énergie libre variationnelle (active inference)
- Explorer si la méta-SR est une approximation de l'incertitude bayésienne (Successor Uncertainties) et sous quelles conditions

---

## 11. Références

### Fondations SR

| Réf | Apport |
|-----|--------|
| Dayan (1993) — *Neural Computation* | Formalisme SR original |
| Stachenfeld et al. (2017) — *Nature Neuroscience* | Hippocampe comme carte prédictive |
| Gershman (2018) — *J. Neuroscience* | Survey SR : logique computationnelle et substrats neuronaux |
| Momennejad et al. (2017) — *Nature Human Behaviour* | Preuves comportementales SR chez l'humain |
| Russek et al. (2017) — *PLoS Comp Bio* | SR–MB hybride, replay, Dyna-SR |
| Barreto et al. (2017) — *NeurIPS* | Successor features pour le transfert |

### Espaces cognitifs

| Réf | Apport |
|-----|--------|
| Bellmund et al. (2018) — *Science* | Codes spatiaux pour la pensée humaine |
| Theves et al. (2020) — *J. Neuroscience* | Hippocampe cartographie l'espace conceptuel |
| Stoewer et al. (2023) — *Scientific Reports* | SR sur espaces sémantiques (NN artificiels) |
| Ekman et al. (2023) — *eLife* | SR dans le cortex visuel |

### Métacognition et hippocampe

| Réf | Apport |
|-----|--------|
| Ambrogioni, L. & Ólafsdóttir, H. F. (2023) — *Trends in Cognitive Sciences*, 27(8), 702-712 | Thèse fondatrice de PRISM : méta-carte hippocampique comme module de méta-apprentissage |
| Allen et al. (2017) — *NeuroImage* | Corrélats microstructuraux métacognition–hippocampe |
| Qiu et al. (2024) — *Communications Biology* | Hippocampe + OFC pour espaces abstraits |

### SR et incertitude — positionnement direct

| Réf | Apport | Relation à PRISM |
|-----|--------|------------------|
| Janz et al. (2019) — *NeurIPS* | Successor Uncertainties | Approche bayésienne — PRISM compare |
| Machado et al. (2020) — *AAAI* | Count-based exploration + SR | Norme SR — PRISM utilise comme baseline |
| Flennerhag et al. (2020) — *DeepMind* | TD uncertainties pour exploration | Signal TD — PRISM étend en carte structurée |
| Chua et al. (2024) — *arXiv* | Simple Successor Features | Deep SF depuis pixels — extension future |

### Métacognition en IA — positionnement direct

| Réf | Apport | Relation à PRISM |
|-----|--------|------------------|
| Valiente & Pilly (2024/2025) — MUSE | Self-assessment + self-regulation | Framework général — PRISM spécifique SR |
| Kawato et al. (2021) — *Biol. Cybernetics* | Internal models → metacognitive AI | Responsibility signal ≈ méta-SR |
| Steyvers & Peters (2025) — *Perspectives Psych. Science* | Métacognition LLMs + calibration | Métriques ECE — PRISM emprunte |

### Global Neuronal Workspace — cadre englobant

| Réf | Apport | Relation à PRISM |
|-----|--------|------------------|
| Dehaene, Kerszberg & Changeux (1998) — *PNAS* | Modèle neuronal du GNW | Cadre englobant — PRISM = processeur spécialisé |
| Dehaene & Changeux (2011) — *Neuron* | GNW : approches expérimentales et théoriques | Ignition, broadcast, seuils |
| Deco, Vidaurre & Kringelbach (2021) — *Nature Human Behaviour* | Functional rich club = workspace empirique | Hippocampe dans le core du workspace |
| Whyte & Smith (2020) — *Progress in Neurobiology* | Predictive Global Workspace (GNW + active inference) | Pont direct : erreurs de prédiction dans le workspace |

### Assets techniques

| Asset | Source | Usage |
|-------|--------|-------|
| MiniGrid | github.com/Farama-Foundation/Minigrid | Environnement |
| SR tabulaire tutorial | Juliani (2019) | Base agent SR |
| Temporal abstraction (spectral) | github.com/veronicachelu/temporal_abstraction | Visualisation eigenvectors |
| SR/MB hybride code | github.com/evanrussek | Référence arbitrage |
| Stable-Baselines3 | github.com/DLR-RM/stable-baselines3 | Baselines RL |

---

*Dernière mise à jour : 2026-02-14 — Phases 0-2 DONE, Phase 3 À VENIR*
