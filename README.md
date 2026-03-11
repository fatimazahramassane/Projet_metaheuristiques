# Projet Métaheuristiques — ENSET Master
## Conception, implémentation et évaluation comparative de métaheuristiques
### Fatima-zahra Massane 

---

## Structure du projet

```
Projet_metaheuristiques/
│
├── config.py                          # Tous les paramètres centralisés
├── README.md                          # Ce fichier
│
├── volet_A/
│   ├
│   ├── fonction_cout.py               # f(s) + énumération exacte + figures A0
│   ├── A1_descente_locale.py          # Descente selon la plus grande pente
│   ├── A2_recherche_taboue.py         # Recherche taboue (2 stratégies, 4 tailles)
│   └── A3_recuit_simule.py            # Recuit simulé (9 combinaisons T0/lambda)
│
├── volet_B/
│   ├
│   ├── fonction_objectif.py           # f(x) 
│   ├── B1_codage.py                   # Codage binaire, décodage, précision
│   ├── B2_operateurs_genetiques.py    # AG complet : sélection, croisement, mutation
│   └── B3_schemas.py                  # Théorème des schèmes, analyse dynamique
│
├── comparaison/
│   ├── __init__.py
│   ├── tableaux_resultats.py          # Tableau comparatif global + CSV + figures
│   └── courbes_evolution.py           # Courbes d'évolution comparatives
│
└── resultats/
    ├── figures/                       # Tous les PNG générés automatiquement
    └── logs/                          # Logs JSON + tableau CSV
```

---

## Prérequis

```bash
pip install numpy matplotlib
```

Python 3.8+ requis. Aucune autre dépendance.

---

## Exécution

**Important :** tous les scripts doivent être lancés **depuis la racine du projet**
(le dossier `Projet_metaheuristiques/`), pas depuis les sous-dossiers.

### Étape 0 — Vérifier la configuration
```bash
python config.py
```

### Volet A — Recherche locale et métaheuristiques

```bash
# Modélisation + énumération exacte (surface de coût, minima locaux)
python volet_A/fonction_cout.py

# A1 — Descente locale (30 départs, 5 figures)
python volet_A/A1_descente_locale.py

# A2 — Recherche taboue (2 stratégies × 4 tailles × 30 runs)
python volet_A/A2_recherche_taboue.py

# A3 — Recuit simulé (9 configs × 30 runs)
python volet_A/A3_recuit_simule.py
```

### Volet B — Algorithme génétique

```bash
# Fonction objectif + codage binaire
python volet_B/fonction_objectif.py

# B1 — Codage, décodage, précision
python volet_B/B1_codage.py

# B2 — AG complet (roulette vs tournoi, effet pm/pc/taille)
python volet_B/B2_operateurs_genetiques.py

# B3 — Schèmes : calcul o(H), u(H), suivi dynamique
python volet_B/B3_schemas.py
```

### Comparaison globale (Section 3 du sujet)

```bash
# Tableau comparatif toutes méthodes + export CSV
python comparaison/tableaux_resultats.py

# Courbes d'évolution comparatives + analyse de sensibilité AG
python comparaison/courbes_evolution.py
```

---

## Reproductibilité

Toutes les expériences utilisent `RANDOM_SEED = 42` (défini dans `config.py`).
Pour changer la graine ou le nombre de runs :

```python
# config.py
RANDOM_SEED = 42     
NB_RUNS     = 30     
```

---

## Figures générées

| Fichier | Description |
|---------|-------------|
| `fig_A0_1_surface_cout.png` | Surface de coût complète (1024 solutions) |
| `fig_A0_2_histogramme.png` | Distribution des valeurs de coût |
| `fig_A0_3_coefficients.png` | Coefficients α et β |
| `fig_A0_4_minima_locaux.png` | Profil des minima locaux |
| `fig_A0_5_voisinage_optimum.png` | Voisinage de la solution optimale |
| `fig_A1_1_couts_finaux.png` | Coûts finaux par run — descente locale |
| `fig_A1_2_initial_vs_final.png` | Évolution coût initial → final |
| `fig_A1_3_nb_iterations.png` | Nombre d'itérations avant convergence |
| `fig_A1_4_courbes_convergence.png` | Courbes de convergence DL |
| `fig_A1_5_stats.png` | Statistiques descente locale |
| `fig_A2_*.png` | Recherche taboue (5 figures) |
| `fig_A3_*.png` | Recuit simulé (5 figures) |
| `fig_B1_*.png` | Codage binaire (4 figures) |
| `fig_B2_*.png` | AG — opérateurs et expériences |
| `fig_B3_*.png` | Schèmes — analyse statique et dynamique |
| `fig_COMP_1_boxplot_final.png` | Boxplot qualité finale toutes méthodes |
| `fig_COMP_2_taux_gap.png` | Taux de succès et gap moyen |
| `fig_COMP_3_radar.png` | Radar chart des profils de méthodes |
| `fig_COMP_4_tableau.png` | Tableau synthétique visuel |
| `fig_COMP_5_evolution_volet_a.png` | Courbes d'évolution DL/RT/RS |
| `fig_COMP_6_vitesse_convergence.png` | Vitesse de convergence par seuil |
| `fig_COMP_7_ag_evolution.png` | Fitness et diversité AG par génération |
| `fig_COMP_8_ag_sensibilite.png` | Sensibilité AG (pm, pc, taille pop) |
| `fig_COMP_9_convergence_normalisee.png` | Convergence normalisée toutes méthodes |

---

## Paramètres clés (config.py)

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `N_BITS` | 10 | Taille du vecteur binaire (Volet A) |
| `NB_RUNS` | 30 | Runs indépendants pour les statistiques |
| `TABU_LIST_SIZES` | [1,3,5,10] | Tailles testées pour la liste taboue |
| `SA_T0_VALUES` | [10,50,100] | Températures initiales testées |
| `SA_LAMBDA_VALUES` | [0.85,0.92,0.99] | Taux de refroidissement testés |
| `GA_NB_BITS` | 10 | Bits par individu (Volet B) |
| `GA_POP_SIZES` | [20,50,100] | Tailles de population testées |

## Résultats clés obtenus

| Méthode | Taux de succès | Meilleur résultat |
|---------|---------------|-------------------|
| Descente locale | 80 % | f = -16.0 |
| Recherche taboue | 100 % | f = -16.0 |
| Recuit simulé (T0=10, λ=0.92) | 100 % | f = -16.0 |
| Algorithme génétique | 100 % | f(x*) ≈ 2.7183 |
```

