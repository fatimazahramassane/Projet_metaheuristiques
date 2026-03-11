import numpy as np
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (GA_X_MIN, GA_X_MAX, GA_NB_BITS, GA_NB_GENERATIONS,
                    GA_POP_SIZES, GA_PC_VALUES, GA_PM_VALUES,
                    GA_POP_SIZE_DEFAULT, GA_PC_DEFAULT, GA_PM_DEFAULT,
                    NB_RUNS, RANDOM_SEED)
from volet_B.fonction_objectif import (decoder, fitness,
                                        chromosome_aleatoire, analyser_fonction)

np.random.seed(RANDOM_SEED)


# =============================================================
# 1.  INITIALISATION DE LA POPULATION
# =============================================================

def initialiser_population(taille, nb_bits=GA_NB_BITS):
    #Cree une population de taille individus binaires aleatoires.
    return [chromosome_aleatoire(nb_bits) for _ in range(taille)]


# =============================================================
# 2.  EVALUATION DE LA POPULATION
# =============================================================

def evaluer_population(population):
    
    #Calcule la fitness de chaque individu.
    #Retourne un array de fitness (peut contenir des valeurs negatives).
    
    return np.array([fitness(ind) for ind in population])


# =============================================================
# 3.  SELECTION PAR ROULETTE
# =============================================================

def selection_roulette(population, fitnesses, nb_selectionnes):
    
    # Decalage pour avoir des fitness positives
    f_min    = np.min(fitnesses)
    f_shift  = fitnesses - f_min + 1e-6   # toutes > 0
    total    = np.sum(f_shift)
    probs    = f_shift / total

    indices  = np.random.choice(len(population), size=nb_selectionnes,
                                 replace=True, p=probs)
    return [population[i].copy() for i in indices]


# =============================================================
# 4.  SELECTION PAR TOURNOI
# =============================================================

def selection_tournoi(population, fitnesses, nb_selectionnes, taille_tournoi=3):
    selectionnes = []
    n = len(population)
    for _ in range(nb_selectionnes):
        candidats = np.random.choice(n, size=taille_tournoi, replace=False)
        meilleur  = candidats[np.argmax(fitnesses[candidats])]
        selectionnes.append(population[meilleur].copy())
    return selectionnes


# =============================================================
# 5.  CROISEMENT BIPOINTS
# =============================================================

def croisement_bipoints(parent1, parent2, pc=GA_PC_DEFAULT):
 
    if np.random.random() > pc:
        return parent1.copy(), parent2.copy()

    n      = len(parent1)
    pts    = sorted(np.random.choice(range(1, n), size=2, replace=False))
    p1, p2 = pts[0], pts[1]

    enfant1 = np.concatenate([parent1[:p1], parent2[p1:p2], parent1[p2:]])
    enfant2 = np.concatenate([parent2[:p1], parent1[p1:p2], parent2[p2:]])

    return enfant1, enfant2


# =============================================================
# 6.  MUTATION BIT A BIT
# =============================================================

def mutation(individu, pm=GA_PM_DEFAULT):
    
    mute = individu.copy()
    for i in range(len(mute)):
        if np.random.random() < pm:
            mute[i] = 1 - mute[i]
    return mute


# =============================================================
# 7.  UNE GENERATION COMPLETE DE L AG
# =============================================================

def une_generation(population, fitnesses, pc, pm, methode_selection='roulette'):
    
    taille       = len(population)
    selectionnes = (selection_roulette(population, fitnesses, taille)
                    if methode_selection == 'roulette'
                    else selection_tournoi(population, fitnesses, taille))

    # Croisement par paires
    nouvelle_pop = []
    for i in range(0, taille - 1, 2):
        e1, e2 = croisement_bipoints(selectionnes[i], selectionnes[i+1], pc)
        nouvelle_pop.extend([e1, e2])
    if len(nouvelle_pop) < taille:
        nouvelle_pop.append(selectionnes[-1].copy())

    # Mutation
    nouvelle_pop = [mutation(ind, pm) for ind in nouvelle_pop]

    # Elitisme : conserver le meilleur de la generation courante
    idx_meilleur = np.argmax(fitnesses)
    nouvelle_pop[0] = population[idx_meilleur].copy()

    return nouvelle_pop[:taille]


# =============================================================
# 8.  ALGORITHME GENETIQUE COMPLET
# =============================================================

def algorithme_genetique(taille_pop=GA_POP_SIZE_DEFAULT,
                          nb_gen=GA_NB_GENERATIONS,
                          pc=GA_PC_DEFAULT,
                          pm=GA_PM_DEFAULT,
                          methode='roulette'):
   
    population        = initialiser_population(taille_pop)
    meilleur_global   = None
    meilleure_f       = -np.inf

    historique_best   = []
    historique_moy    = []
    historique_div    = []

    for gen in range(nb_gen):
        fitnesses = evaluer_population(population)

        # Statistiques de la generation
        f_best = np.max(fitnesses)
        f_moy  = np.mean(fitnesses)
        x_vals = np.array([decoder(ind) for ind in population])
        div    = np.std(x_vals)

        historique_best.append(f_best)
        historique_moy.append(f_moy)
        historique_div.append(div)

        # Meilleur global
        idx_best = np.argmax(fitnesses)
        if fitnesses[idx_best] > meilleure_f:
            meilleure_f     = fitnesses[idx_best]
            meilleur_global = population[idx_best].copy()

        # Nouvelle generation
        population = une_generation(population, fitnesses, pc, pm, methode)

    return {
        "meilleur_individu" : meilleur_global,
        "meilleur_x"        : decoder(meilleur_global),
        "meilleure_fitness" : meilleure_f,
        "historique_best"   : historique_best,
        "historique_moy"    : historique_moy,
        "historique_div"    : historique_div
    }


# =============================================================
# 9.  EXPERIENCES COMPLETES
# =============================================================

def lancer_experiences(nb_runs=NB_RUNS):
    
    resultats = {}
    f_max_ref = analyser_fonction()["f_max"]

    # A) Comparaison methodes de selection
    print("  [A] Comparaison roulette vs tournoi...")
    resultats["selection"] = {}
    for methode in ["roulette", "tournoi"]:
        runs = []
        np.random.seed(RANDOM_SEED)
        for _ in range(nb_runs):
            r = algorithme_genetique(methode=methode)
            runs.append(r)
        resultats["selection"][methode] = {
            "runs"     : runs,
            "moy_best" : np.mean([r["meilleure_fitness"] for r in runs]),
            "std_best" : np.std([r["meilleure_fitness"] for r in runs]),
            "taux"     : 100 * sum(1 for r in runs
                                    if abs(r["meilleure_fitness"] - f_max_ref) < 0.05)
                         / nb_runs
        }

    # B) Effet de pm
    print("  [B] Effet du taux de mutation pm...")
    resultats["pm"] = {}
    for pm in GA_PM_VALUES:
        runs = []
        np.random.seed(RANDOM_SEED)
        for _ in range(nb_runs):
            r = algorithme_genetique(pm=pm)
            runs.append(r)
        resultats["pm"][pm] = {
            "runs"     : runs,
            "moy_best" : np.mean([r["meilleure_fitness"] for r in runs]),
            "std_best" : np.std([r["meilleure_fitness"] for r in runs]),
            "taux"     : 100 * sum(1 for r in runs
                                    if abs(r["meilleure_fitness"] - f_max_ref) < 0.05)
                         / nb_runs
        }

    # C) Effet de pc
    print("  [C] Effet du taux de croisement pc...")
    resultats["pc"] = {}
    for pc in GA_PC_VALUES:
        runs = []
        np.random.seed(RANDOM_SEED)
        for _ in range(nb_runs):
            r = algorithme_genetique(pc=pc)
            runs.append(r)
        resultats["pc"][pc] = {
            "runs"     : runs,
            "moy_best" : np.mean([r["meilleure_fitness"] for r in runs]),
            "std_best" : np.std([r["meilleure_fitness"] for r in runs]),
            "taux"     : 100 * sum(1 for r in runs
                                    if abs(r["meilleure_fitness"] - f_max_ref) < 0.05)
                         / nb_runs
        }

    # D) Effet de la taille de population
    print("  [D] Effet de la taille de population...")
    resultats["pop"] = {}
    for taille in GA_POP_SIZES:
        runs = []
        np.random.seed(RANDOM_SEED)
        for _ in range(nb_runs):
            r = algorithme_genetique(taille_pop=taille)
            runs.append(r)
        resultats["pop"][taille] = {
            "runs"     : runs,
            "moy_best" : np.mean([r["meilleure_fitness"] for r in runs]),
            "std_best" : np.std([r["meilleure_fitness"] for r in runs]),
            "taux"     : 100 * sum(1 for r in runs
                                    if abs(r["meilleure_fitness"] - f_max_ref) < 0.05)
                         / nb_runs
        }

    return resultats


# =============================================================
# 10.  GENERATION DES FIGURES
# =============================================================

def generer_figures(resultats, f_max_ref, figures_dir="resultats/figures"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(figures_dir, exist_ok=True)
    plt.rcParams.update({
        'font.family'   : 'DejaVu Sans',
        'font.size'     : 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'figure.dpi'    : 150,
        'axes.grid'     : True,
        'grid.alpha'    : 0.3
    })

    generations = range(GA_NB_GENERATIONS)

    def moyenne_historique(runs, cle):
        mat = np.array([r[cle] for r in runs])
        return np.mean(mat, axis=0), np.std(mat, axis=0)

    # ----------------------------------------------------------
    # FIGURE 1 — Roulette vs Tournoi
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    couleurs  = {"roulette": "#3498db", "tournoi": "#e74c3c"}

    for methode, color in couleurs.items():
        runs      = resultats["selection"][methode]["runs"]
        moy, std  = moyenne_historique(runs, "historique_best")
        moy_m, _  = moyenne_historique(runs, "historique_moy")
        moy_d, _  = moyenne_historique(runs, "historique_div")

        axes[0].plot(generations, moy, color=color, lw=2,
                     label=f"{methode} (moy={resultats['selection'][methode]['moy_best']:.4f})")
        axes[0].fill_between(generations, moy - std, moy + std,
                              alpha=0.15, color=color)

        axes[1].plot(generations, moy_m, color=color, lw=2, label=methode)
        axes[2].plot(generations, moy_d, color=color, lw=2, label=methode)

    for ax in axes:
        ax.legend(fontsize=9)
    axes[0].axhline(f_max_ref, color='black', ls='--', lw=1.5,
                    label=f'f_max={f_max_ref:.4f}')
    axes[0].legend(fontsize=9)
    axes[0].set_title("Meilleure fitness / generation")
    axes[0].set_ylabel("Meilleure fitness")
    axes[1].set_title("Fitness moyenne / generation")
    axes[1].set_ylabel("Fitness moyenne")
    axes[2].set_title("Diversite (std de x) / generation")
    axes[2].set_ylabel("Diversite")
    for ax in axes:
        ax.set_xlabel("Generation")

    fig.suptitle(f"Comparaison Selection Roulette vs Tournoi — {NB_RUNS} runs",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    p1 = f"{figures_dir}/fig_B2_1_selection_roulette_vs_tournoi.png"
    fig.savefig(p1, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p1}")

    # ----------------------------------------------------------
    # FIGURE 2 — Effet du taux de mutation pm
    # ----------------------------------------------------------
    couleurs_pm = {0.01: '#2ecc71', 0.05: '#f39c12', 0.1: '#e74c3c'}
    fig, axes   = plt.subplots(1, 3, figsize=(16, 5))

    for pm, color in couleurs_pm.items():
        runs     = resultats["pm"][pm]["runs"]
        moy, std = moyenne_historique(runs, "historique_best")
        moy_m, _ = moyenne_historique(runs, "historique_moy")
        moy_d, _ = moyenne_historique(runs, "historique_div")

        axes[0].plot(generations, moy, color=color, lw=2,
                     label=f"pm={pm} (moy={resultats['pm'][pm]['moy_best']:.4f})")
        axes[0].fill_between(generations, moy - std, moy + std,
                              alpha=0.12, color=color)
        axes[1].plot(generations, moy_m, color=color, lw=2, label=f"pm={pm}")
        axes[2].plot(generations, moy_d, color=color, lw=2, label=f"pm={pm}")

    axes[0].axhline(f_max_ref, color='black', ls='--', lw=1.5,
                    label=f'f_max={f_max_ref:.4f}')
    axes[0].set_title("Meilleure fitness / generation")
    axes[0].set_ylabel("Meilleure fitness")
    axes[1].set_title("Fitness moyenne / generation")
    axes[1].set_ylabel("Fitness moyenne")
    axes[2].set_title("Diversite / generation")
    axes[2].set_ylabel("Diversite (std de x)")
    for ax in axes:
        ax.set_xlabel("Generation")
        ax.legend(fontsize=9)

    fig.suptitle(f"Effet du taux de mutation pm — {NB_RUNS} runs par valeur",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    p2 = f"{figures_dir}/fig_B2_2_effet_mutation.png"
    fig.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p2}")

    # ----------------------------------------------------------
    # FIGURE 3 — Effet du taux de croisement pc
    # ----------------------------------------------------------
    couleurs_pc = {0.6: '#9b59b6', 0.8: '#3498db', 0.9: '#e74c3c'}
    fig, axes   = plt.subplots(1, 2, figsize=(13, 5))

    for pc, color in couleurs_pc.items():
        runs     = resultats["pc"][pc]["runs"]
        moy, std = moyenne_historique(runs, "historique_best")
        moy_m, _ = moyenne_historique(runs, "historique_moy")

        axes[0].plot(generations, moy, color=color, lw=2,
                     label=f"pc={pc} (moy={resultats['pc'][pc]['moy_best']:.4f})")
        axes[0].fill_between(generations, moy - std, moy + std,
                              alpha=0.12, color=color)
        axes[1].plot(generations, moy_m, color=color, lw=2, label=f"pc={pc}")

    axes[0].axhline(f_max_ref, color='black', ls='--', lw=1.5,
                    label=f'f_max={f_max_ref:.4f}')
    axes[0].set_title("Meilleure fitness / generation")
    axes[0].set_ylabel("Meilleure fitness")
    axes[1].set_title("Fitness moyenne / generation")
    axes[1].set_ylabel("Fitness moyenne")
    for ax in axes:
        ax.set_xlabel("Generation")
        ax.legend(fontsize=9)

    fig.suptitle(f"Effet du taux de croisement pc — {NB_RUNS} runs par valeur",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    p3 = f"{figures_dir}/fig_B2_3_effet_croisement.png"
    fig.savefig(p3, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p3}")

    # ----------------------------------------------------------
    # FIGURE 4 — Effet de la taille de population
    # ----------------------------------------------------------
    couleurs_pop = {20: '#e74c3c', 50: '#f39c12', 100: '#2ecc71'}
    fig, axes    = plt.subplots(1, 3, figsize=(16, 5))

    for taille, color in couleurs_pop.items():
        runs     = resultats["pop"][taille]["runs"]
        moy, std = moyenne_historique(runs, "historique_best")
        moy_m, _ = moyenne_historique(runs, "historique_moy")
        moy_d, _ = moyenne_historique(runs, "historique_div")

        axes[0].plot(generations, moy, color=color, lw=2,
                     label=f"N={taille} (moy={resultats['pop'][taille]['moy_best']:.4f})")
        axes[0].fill_between(generations, moy - std, moy + std,
                              alpha=0.12, color=color)
        axes[1].plot(generations, moy_m, color=color, lw=2, label=f"N={taille}")
        axes[2].plot(generations, moy_d, color=color, lw=2, label=f"N={taille}")

    axes[0].axhline(f_max_ref, color='black', ls='--', lw=1.5,
                    label=f'f_max={f_max_ref:.4f}')
    axes[0].set_title("Meilleure fitness / generation")
    axes[0].set_ylabel("Meilleure fitness")
    axes[1].set_title("Fitness moyenne / generation")
    axes[1].set_ylabel("Fitness moyenne")
    axes[2].set_title("Diversite / generation")
    axes[2].set_ylabel("Diversite (std de x)")
    for ax in axes:
        ax.set_xlabel("Generation")
        ax.legend(fontsize=9)

    fig.suptitle(f"Effet de la taille de population — {NB_RUNS} runs par valeur",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    p4 = f"{figures_dir}/fig_B2_4_effet_population.png"
    fig.savefig(p4, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p4}")

    # ----------------------------------------------------------
    # FIGURE 5 — Boxplot comparaison globale
    # ----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # A) Selection
    data_sel  = [
        [r["meilleure_fitness"] for r in resultats["selection"][m]["runs"]]
        for m in ["roulette", "tournoi"]
    ]
    bp = axes[0,0].boxplot(data_sel, tick_labels=["Roulette", "Tournoi"],
                            patch_artist=True,
                            medianprops=dict(color='black', lw=2))
    for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    axes[0,0].axhline(f_max_ref, color='red', ls='--', lw=1.5,
                       label=f'f_max={f_max_ref:.4f}')
    axes[0,0].set_title("Selection : Roulette vs Tournoi")
    axes[0,0].set_ylabel("Meilleure fitness")
    axes[0,0].legend(fontsize=9)

    # B) pm
    data_pm   = [
        [r["meilleure_fitness"] for r in resultats["pm"][pm]["runs"]]
        for pm in GA_PM_VALUES
    ]
    bp = axes[0,1].boxplot(data_pm, tick_labels=[f"pm={pm}" for pm in GA_PM_VALUES],
                            patch_artist=True,
                            medianprops=dict(color='black', lw=2))
    for patch, color in zip(bp['boxes'], ['#2ecc71', '#f39c12', '#e74c3c']):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    axes[0,1].axhline(f_max_ref, color='red', ls='--', lw=1.5)
    axes[0,1].set_title("Effet du taux de mutation pm")
    axes[0,1].set_ylabel("Meilleure fitness")

    # C) pc
    data_pc   = [
        [r["meilleure_fitness"] for r in resultats["pc"][pc]["runs"]]
        for pc in GA_PC_VALUES
    ]
    bp = axes[1,0].boxplot(data_pc, tick_labels=[f"pc={pc}" for pc in GA_PC_VALUES],
                            patch_artist=True,
                            medianprops=dict(color='black', lw=2))
    for patch, color in zip(bp['boxes'], ['#9b59b6', '#3498db', '#e74c3c']):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    axes[1,0].axhline(f_max_ref, color='red', ls='--', lw=1.5)
    axes[1,0].set_title("Effet du taux de croisement pc")
    axes[1,0].set_ylabel("Meilleure fitness")

    # D) taille pop
    data_pop  = [
        [r["meilleure_fitness"] for r in resultats["pop"][t]["runs"]]
        for t in GA_POP_SIZES
    ]
    bp = axes[1,1].boxplot(data_pop, tick_labels=[f"N={t}" for t in GA_POP_SIZES],
                            patch_artist=True,
                            medianprops=dict(color='black', lw=2))
    for patch, color in zip(bp['boxes'], ['#e74c3c', '#f39c12', '#2ecc71']):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    axes[1,1].axhline(f_max_ref, color='red', ls='--', lw=1.5)
    axes[1,1].set_title("Effet de la taille de population")
    axes[1,1].set_ylabel("Meilleure fitness")

    for ax in axes.flatten():
        ax.set_xlabel("")

    fig.suptitle(f"Comparaison globale des parametres AG — {NB_RUNS} runs",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    p5 = f"{figures_dir}/fig_B2_5_boxplot_global.png"
    fig.savefig(p5, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p5}")

    # ----------------------------------------------------------
    # FIGURE 6 — Meilleur run : positions de la population
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Courbe reference
    x_ref  = np.linspace(GA_X_MIN, GA_X_MAX, 1000)
    f_ref  = np.sin(x_ref) * np.exp(np.sin(x_ref))
    analyse = analyser_fonction()

    for ax, methode, color in zip(axes, ["roulette", "tournoi"],
                                   ["#3498db", "#e74c3c"]):
        runs    = resultats["selection"][methode]["runs"]
        best_run = max(runs, key=lambda r: r["meilleure_fitness"])

        ax.plot(x_ref, f_ref, color='#bdc3c7', lw=2, alpha=0.8, zorder=1)
        ax.axvline(best_run["meilleur_x"], color=color, ls='--', lw=2,
                   label=f'x*={best_run["meilleur_x"]:.5f}\nf={best_run["meilleure_fitness"]:.5f}')
        ax.scatter([best_run["meilleur_x"]], [best_run["meilleure_fitness"]],
                   color=color, s=200, marker='*', zorder=5)
        ax.scatter([analyse["x_max"]], [analyse["f_max"]], color='green',
                   s=120, marker='D', zorder=4,
                   label=f'Vrai max ({analyse["x_max"]:.5f})')
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title(f"Meilleur resultat — Selection {methode}")
        ax.legend(fontsize=9)

    fig.suptitle("Meilleur resultat trouve par l AG sur f(x) = sin(x)·exp(sin(x))",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    p6 = f"{figures_dir}/fig_B2_6_meilleur_resultat.png"
    fig.savefig(p6, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p6}")

    return [p1, p2, p3, p4, p5, p6]


# =============================================================
# 11.  SAUVEGARDE LOGS
# =============================================================

def sauvegarder_logs(resultats, logs_dir="resultats/logs"):
    os.makedirs(logs_dir, exist_ok=True)
    path = f"{logs_dir}/B2_operateurs_genetiques.json"
    data = {}

    for experience, contenu in resultats.items():
        data[experience] = {}
        for cle, val in contenu.items():
            data[experience][str(cle)] = {
                "moy_best" : float(val["moy_best"]),
                "std_best" : float(val["std_best"]),
                "taux"     : float(val["taux"]),
                "best_final": [float(r["meilleure_fitness"]) for r in val["runs"]],
                "best_x"   : [float(r["meilleur_x"]) for r in val["runs"]]
            }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  OK  {path}")


# =============================================================
# PROGRAMME PRINCIPAL
# =============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  VOLET B — B2_operateurs_genetiques.py")
    print(f"  AG : {GA_NB_GENERATIONS} generations, {NB_RUNS} runs par config")
    print("=" * 65)

    analyse    = analyser_fonction()
    f_max_ref  = analyse["f_max"]
    x_max_ref  = analyse["x_max"]
    print(f"\n  Reference : f_max = {f_max_ref:.6f}  a  x* = {x_max_ref:.6f}")

    print(f"\n  Lancement des experiences...")
    t0         = time.time()
    resultats  = lancer_experiences(nb_runs=NB_RUNS)
    duree      = time.time() - t0
    print(f"  Termine en {duree:.2f}s")

    # Tableau de synthese
    print("\n" + "-" * 65)
    print(f"  {'Experience':<25} {'Config':<12} {'Moy':>8} {'Std':>7} {'Taux(%)':>9}")
    print("-" * 65)

    for methode in ["roulette", "tournoi"]:
        r = resultats["selection"][methode]
        print(f"  {'Selection':<25} {methode:<12} {r['moy_best']:>8.4f} "
              f"{r['std_best']:>7.4f} {r['taux']:>8.1f}%")

    for pm in GA_PM_VALUES:
        r = resultats["pm"][pm]
        print(f"  {'Mutation':<25} {'pm='+str(pm):<12} {r['moy_best']:>8.4f} "
              f"{r['std_best']:>7.4f} {r['taux']:>8.1f}%")

    for pc in GA_PC_VALUES:
        r = resultats["pc"][pc]
        print(f"  {'Croisement':<25} {'pc='+str(pc):<12} {r['moy_best']:>8.4f} "
              f"{r['std_best']:>7.4f} {r['taux']:>8.1f}%")

    for taille in GA_POP_SIZES:
        r = resultats["pop"][taille]
        print(f"  {'Taille pop':<25} {'N='+str(taille):<12} {r['moy_best']:>8.4f} "
              f"{r['std_best']:>7.4f} {r['taux']:>8.1f}%")

    print("-" * 65)

    # Figures
    print("\n  Generation des figures...")
    base        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(base, "resultats", "figures")
    logs_dir    = os.path.join(base, "resultats", "logs")
    generer_figures(resultats, f_max_ref, figures_dir=figures_dir)

    # Logs
    print("\n  Sauvegarde des logs...")
    sauvegarder_logs(resultats, logs_dir=logs_dir)

    print("\n" + "=" * 65)
    print("  B2_operateurs_genetiques.py termine avec succes")
    print(f"  Figures dans : {figures_dir}")
    print(f"  Logs dans    : {logs_dir}")
    print("=" * 65)