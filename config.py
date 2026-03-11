

import numpy as np


# VOLET A — Espace binaire


# Taille du vecteur binaire (n >= 8 imposé par le sujet)
N_BITS = 10

# Coefficients alpha_i pour la fonction de coût f(s)
# Choix justifié : valeurs mixtes (positives et négatives) pour créer
# plusieurs minima locaux et un minimum global non trivial
ALPHA = np.array([3.0, -5.0, 2.0, -4.0, 6.0, -2.0, 1.0, -3.0, 4.0, -1.0])

# Coefficients beta_ij (matrice symétrique, triangulaire supérieure utilisée)
# Valeurs choisies pour créer des interactions entre bits → minima locaux
np.random.seed(42)  # reproductibilité
BETA = np.zeros((N_BITS, N_BITS))
for i in range(N_BITS):
    for j in range(i+1, N_BITS):
        BETA[i][j] = np.random.choice([-3, -2, -1, 1, 2, 3])

# Nombre de solutions initiales pour les tests (A1, A2, A3)
NB_INIT_SOLUTIONS = 30

# Graine aléatoire pour la reproductibilité
RANDOM_SEED = 42


# A1 — Descente locale

DL_NB_STARTS = 30          # Nombre de départs aléatoires


# A2 — Recherche Taboue

TABU_MAX_ITER    = 200      # Nombre maximum d'itérations
TABU_LIST_SIZES  = [1, 3, 5, 10]   # Tailles de la liste taboue à tester


# A3 — Recuit Simulé

SA_T0_VALUES     = [10.0, 50.0, 100.0]     # Températures initiales à tester
SA_LAMBDA_VALUES = [0.85, 0.92, 0.99]      # Taux de refroidissement à tester
SA_MAX_ITER      = 500                      # Itérations maximales
SA_T_MIN         = 1e-3                     # Température minimale d'arrêt


# VOLET B — Algorithme Génétique (fonction continue)


# Domaine de la fonction f(x) = sin(x) * e^sin(x)
GA_X_MIN = -5.0
GA_X_MAX =  5.0

# Codage binaire
GA_NB_BITS = 10            # Nombre de bits par individu

# Paramètres de l'AG à tester
GA_POP_SIZES        = [20, 50, 100]    # Tailles de population
GA_NB_GENERATIONS   = 100             # Nombre de générations
GA_PC_VALUES        = [0.6, 0.8, 0.9] # Taux de croisement
GA_PM_VALUES        = [0.01, 0.05, 0.1] # Taux de mutation

# Paramètres par défaut (pour les runs principaux)
GA_POP_SIZE_DEFAULT = 50
GA_PC_DEFAULT       = 0.8
GA_PM_DEFAULT       = 0.05


# COMPARAISON — Statistiques

NB_RUNS = 30       # Nombre d'exécutions indépendantes pour les stats



# RÉSULTATS — Chemins de sortie

RESULTS_DIR  = "resultats"
FIGURES_DIR  = "resultats/figures"
LOGS_DIR     = "resultats/logs"


# Vérification au lancement

if __name__ == "__main__":
    print("=" * 60)
    print("   CONFIG.PY — Vérification des paramètres")
    print("=" * 60)
    print(f"\n Volet A — Espace binaire")
    print(f"   N_BITS          = {N_BITS}")
    print(f"   ALPHA           = {ALPHA}")
    print(f"   NB_INIT         = {NB_INIT_SOLUTIONS}")
    print(f"\n Recherche Taboue")
    print(f"   MAX_ITER        = {TABU_MAX_ITER}")
    print(f"   TAILLES TABOUE  = {TABU_LIST_SIZES}")
    print(f"\n Recuit Simulé")
    print(f"   T0 testés       = {SA_T0_VALUES}")
    print(f"   Lambda testés   = {SA_LAMBDA_VALUES}")
    print(f"   MAX_ITER        = {SA_MAX_ITER}")
    print(f"\n Volet B — Algorithme Génétique")
    print(f"   Domaine x       = [{GA_X_MIN}, {GA_X_MAX}]")
    print(f"   NB_BITS/individu= {GA_NB_BITS}")
    print(f"   Tailles pop.    = {GA_POP_SIZES}")
    print(f"   Taux mutation   = {GA_PM_VALUES}")
    print(f"   Taux croisement = {GA_PC_VALUES}")
    print(f"\n Statistiques")
    print(f"   NB_RUNS         = {NB_RUNS}")
    print(f"\n Tous les paramètres sont chargés correctement.")
    print("=" * 60)
