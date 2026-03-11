import numpy as np
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (N_BITS, RANDOM_SEED, TABU_MAX_ITER,
                    TABU_LIST_SIZES, NB_RUNS)
from volet_A.fonction_cout import fonction_cout, get_voisins, solution_aleatoire, enumeration_exacte

np.random.seed(RANDOM_SEED)


# =============================================================
# 1.  RECHERCHE TABOUE — STRATEGIE 1 : solutions taboues
# =============================================================

def recherche_taboue_solutions(s_init, taille_taboue=5, max_iter=TABU_MAX_ITER):
    
    s_courant      = np.array(s_init, dtype=int).copy()
    s_meilleur     = s_courant.copy()
    f_courant      = fonction_cout(s_courant)
    f_meilleur     = f_courant

    liste_taboue   = []          # liste des solutions taboue
    evolution_cout = [f_courant] # historique du cout meilleur
    nb_depla       = 0

    for iteration in range(max_iter):
        voisins        = get_voisins(s_courant)
        meilleur_voisin = None
        meilleur_f_voisin = float('inf')

        for v in voisins:
            cle = tuple(v)
            # Accepter si non tabou OU si ameliore le meilleur global
            if cle not in [tuple(t) for t in liste_taboue]:
                f_v = fonction_cout(v)
                if f_v < meilleur_f_voisin:
                    meilleur_f_voisin = f_v
                    meilleur_voisin   = v.copy()
            else:
                # Critere d'aspiration : accepter quand meme si meilleur global
                f_v = fonction_cout(v)
                if f_v < f_meilleur:
                    meilleur_f_voisin = f_v
                    meilleur_voisin   = v.copy()

        if meilleur_voisin is None:
            break  # tous les voisins sont tabous

        # Mise a jour
        liste_taboue.append(s_courant.copy())
        if len(liste_taboue) > taille_taboue:
            liste_taboue.pop(0)

        s_courant = meilleur_voisin.copy()
        f_courant = meilleur_f_voisin
        nb_depla += 1

        if f_courant < f_meilleur:
            f_meilleur = f_courant
            s_meilleur = s_courant.copy()

        evolution_cout.append(f_meilleur)

    return {
        "solution"      : s_meilleur,
        "cout"          : f_meilleur,
        "evolution"     : evolution_cout,
        "nb_depla"      : nb_depla,
        "strategie"     : "solutions"
    }


# =============================================================
# 2.  RECHERCHE TABOUE — STRATEGIE 2 : mouvements inverses
# =============================================================

def recherche_taboue_mouvements(s_init, taille_taboue=5, max_iter=TABU_MAX_ITER):
    
    s_courant      = np.array(s_init, dtype=int).copy()
    s_meilleur     = s_courant.copy()
    f_courant      = fonction_cout(s_courant)
    f_meilleur     = f_courant

    liste_taboue   = []          # liste des indices de bits interdits
    evolution_cout = [f_courant]
    nb_depla       = 0

    for iteration in range(max_iter):
        voisins           = get_voisins(s_courant)
        meilleur_voisin   = None
        meilleur_f_voisin = float('inf')
        meilleur_idx      = -1

        for idx, v in enumerate(voisins):
            # idx = bit qui a ete flippe pour obtenir v
            mouvement_inverse = idx   # re-flipper idx serait l'inverse
            if mouvement_inverse not in liste_taboue:
                f_v = fonction_cout(v)
                if f_v < meilleur_f_voisin:
                    meilleur_f_voisin = f_v
                    meilleur_voisin   = v.copy()
                    meilleur_idx      = idx
            else:
                # Critere d'aspiration
                f_v = fonction_cout(v)
                if f_v < f_meilleur:
                    meilleur_f_voisin = f_v
                    meilleur_voisin   = v.copy()
                    meilleur_idx      = idx

        if meilleur_voisin is None:
            break

        # Ajouter le mouvement inverse dans la liste taboue
        liste_taboue.append(meilleur_idx)
        if len(liste_taboue) > taille_taboue:
            liste_taboue.pop(0)

        s_courant = meilleur_voisin.copy()
        f_courant = meilleur_f_voisin
        nb_depla += 1

        if f_courant < f_meilleur:
            f_meilleur = f_courant
            s_meilleur = s_courant.copy()

        evolution_cout.append(f_meilleur)

    return {
        "solution"  : s_meilleur,
        "cout"      : f_meilleur,
        "evolution" : evolution_cout,
        "nb_depla"  : nb_depla,
        "strategie" : "mouvements"
    }


# =============================================================
# 3.  EXPERIENCE COMPLETE
#     NB_RUNS runs x 4 tailles x 2 strategies
# =============================================================

def lancer_experiences(nb_runs=NB_RUNS):
    
    res_global = {}

    for strategie in ["solutions", "mouvements"]:
        res_global[strategie] = {}

        for k in TABU_LIST_SIZES:
            couts_finaux  = []
            nb_deplas     = []
            evolutions    = []
            convergences  = []   # iteration de convergence au meilleur

            np.random.seed(RANDOM_SEED)
            for run in range(nb_runs):
                s0 = solution_aleatoire()

                if strategie == "solutions":
                    r = recherche_taboue_solutions(s0, taille_taboue=k)
                else:
                    r = recherche_taboue_mouvements(s0, taille_taboue=k)

                couts_finaux.append(r["cout"])
                nb_deplas.append(r["nb_depla"])
                evolutions.append(r["evolution"])

                # Iteration ou le meilleur a ete atteint
                best = r["cout"]
                conv = next((i for i, v in enumerate(r["evolution"]) if v == best), len(r["evolution"]))
                convergences.append(conv)

            res_global[strategie][k] = {
                "couts_finaux"  : couts_finaux,
                "nb_deplas"     : nb_deplas,
                "evolutions"    : evolutions,
                "convergences"  : convergences,
                "moyenne"       : np.mean(couts_finaux),
                "std"           : np.std(couts_finaux),
                "meilleur"      : np.min(couts_finaux),
                "pire"          : np.max(couts_finaux),
                "conv_moyenne"  : np.mean(convergences)
            }

    return res_global


# =============================================================
# 4.  GENERATION DES FIGURES
# =============================================================

def generer_figures(res_global, min_global, figures_dir="resultats/figures"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    os.makedirs(figures_dir, exist_ok=True)
    plt.rcParams.update({
        'font.family'   : 'DejaVu Sans',
        'font.size'     : 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'figure.dpi'    : 150,
        'axes.grid'     : True,
        'grid.alpha'    : 0.3
    })

    couleurs_k = {1: '#e74c3c', 3: '#e67e22', 5: '#2ecc71', 10: '#3498db'}
    strategies = ["solutions", "mouvements"]

    # ----------------------------------------------------------
    # FIGURE 1 — Evolution du cout meilleur (moyenne sur NB_RUNS)
    #            pour chaque taille k, strategie solutions
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, strat in zip(axes, strategies):
        for k in TABU_LIST_SIZES:
            evols = res_global[strat][k]["evolutions"]
            max_len = max(len(e) for e in evols)
            mat = np.full((len(evols), max_len), np.nan)
            for i, e in enumerate(evols):
                mat[i, :len(e)] = e
            moy = np.nanmean(mat, axis=0)
            ax.plot(moy, color=couleurs_k[k], lw=2,
                    label=f"k = {k}  (moy finale={res_global[strat][k]['moyenne']:.2f})")

        ax.axhline(min_global, color='black', ls='--', lw=1.5,
                   label=f'Min. global = {min_global:.2f}')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Meilleur cout trouve")
        ax.set_title(f"Strategie : {strat}\nEvolution du meilleur cout (moy. {NB_RUNS} runs)")
        ax.legend(fontsize=9)

    fig.suptitle("Recherche Taboue — Evolution du cout par taille de liste",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    p1 = f"{figures_dir}/fig_A2_1_evolution_cout.png"
    fig.savefig(p1, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p1}")

    # ----------------------------------------------------------
    # FIGURE 2 — Boxplot couts finaux : solutions vs mouvements
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, strat in zip(axes, strategies):
        data   = [res_global[strat][k]["couts_finaux"] for k in TABU_LIST_SIZES]
        labels = [f"k={k}" for k in TABU_LIST_SIZES]
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        medianprops=dict(color='black', lw=2))
        for patch, k in zip(bp['boxes'], TABU_LIST_SIZES):
            patch.set_facecolor(couleurs_k[k])
            patch.set_alpha(0.75)
        ax.axhline(min_global, color='red', ls='--', lw=1.5,
                   label=f'Min. global = {min_global:.2f}')
        ax.set_xlabel("Taille liste taboue k")
        ax.set_ylabel("Cout final")
        ax.set_title(f"Strategie : {strat}\nDistribution des couts finaux ({NB_RUNS} runs)")
        ax.legend(fontsize=9)

    fig.suptitle("Recherche Taboue — Qualite finale selon la taille de liste",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    p2 = f"{figures_dir}/fig_A2_2_boxplot_couts.png"
    fig.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p2}")

    # ----------------------------------------------------------
    # FIGURE 3 — Comparaison des deux strategies (barres groupees)
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x     = np.arange(len(TABU_LIST_SIZES))
    width = 0.35

    # Moyenne cout final
    moy_sol = [res_global["solutions"][k]["moyenne"]   for k in TABU_LIST_SIZES]
    moy_mov = [res_global["mouvements"][k]["moyenne"]  for k in TABU_LIST_SIZES]
    std_sol = [res_global["solutions"][k]["std"]       for k in TABU_LIST_SIZES]
    std_mov = [res_global["mouvements"][k]["std"]      for k in TABU_LIST_SIZES]

    axes[0].bar(x - width/2, moy_sol, width, label="Solutions taboues",
                color='#3498db', alpha=0.85, yerr=std_sol, capsize=4, ecolor='black')
    axes[0].bar(x + width/2, moy_mov, width, label="Mouvements inverses",
                color='#e74c3c', alpha=0.85, yerr=std_mov, capsize=4, ecolor='black')
    axes[0].axhline(min_global, color='black', ls='--', lw=1.5,
                    label=f'Min. global = {min_global:.2f}')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"k={k}" for k in TABU_LIST_SIZES])
    axes[0].set_xlabel("Taille liste taboue k")
    axes[0].set_ylabel("Cout moyen final (± std)")
    axes[0].set_title("Qualite finale : Solutions vs Mouvements")
    axes[0].legend(fontsize=9)

    # Nombre moyen de deplacements
    dep_sol = [np.mean(res_global["solutions"][k]["nb_deplas"])  for k in TABU_LIST_SIZES]
    dep_mov = [np.mean(res_global["mouvements"][k]["nb_deplas"]) for k in TABU_LIST_SIZES]

    axes[1].bar(x - width/2, dep_sol, width, label="Solutions taboues",
                color='#3498db', alpha=0.85)
    axes[1].bar(x + width/2, dep_mov, width, label="Mouvements inverses",
                color='#e74c3c', alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"k={k}" for k in TABU_LIST_SIZES])
    axes[1].set_xlabel("Taille liste taboue k")
    axes[1].set_ylabel("Nombre moyen de deplacements")
    axes[1].set_title("Mobilite : Solutions vs Mouvements")
    axes[1].legend(fontsize=9)

    fig.suptitle("Comparaison des deux strategies de memoire taboue",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    p3 = f"{figures_dir}/fig_A2_3_comparaison_strategies.png"
    fig.savefig(p3, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p3}")

    # ----------------------------------------------------------
    # FIGURE 4 — Taux d'atteinte du minimum global
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    x     = np.arange(len(TABU_LIST_SIZES))
    width = 0.35

    taux_sol = [
        100 * sum(1 for c in res_global["solutions"][k]["couts_finaux"]
                  if abs(c - min_global) < 1e-6) / NB_RUNS
        for k in TABU_LIST_SIZES
    ]
    taux_mov = [
        100 * sum(1 for c in res_global["mouvements"][k]["couts_finaux"]
                  if abs(c - min_global) < 1e-6) / NB_RUNS
        for k in TABU_LIST_SIZES
    ]

    bars1 = ax.bar(x - width/2, taux_sol, width,
                   label="Solutions taboues",   color='#3498db', alpha=0.85)
    bars2 = ax.bar(x + width/2, taux_mov, width,
                   label="Mouvements inverses", color='#e74c3c', alpha=0.85)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                f'{h:.0f}%', ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in TABU_LIST_SIZES])
    ax.set_xlabel("Taille liste taboue k")
    ax.set_ylabel("Taux d'atteinte du minimum global (%)")
    ax.set_title(f"Capacite a trouver le minimum global — {NB_RUNS} runs")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.axhline(100, color='black', ls=':', lw=1)

    fig.tight_layout()
    p4 = f"{figures_dir}/fig_A2_4_taux_global.png"
    fig.savefig(p4, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p4}")

    # ----------------------------------------------------------
    # FIGURE 5 — Iterations de convergence
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    conv_sol = [res_global["solutions"][k]["conv_moyenne"]  for k in TABU_LIST_SIZES]
    conv_mov = [res_global["mouvements"][k]["conv_moyenne"] for k in TABU_LIST_SIZES]

    ax.plot([f"k={k}" for k in TABU_LIST_SIZES], conv_sol,
            'o-', color='#3498db', lw=2, ms=8, label="Solutions taboues")
    ax.plot([f"k={k}" for k in TABU_LIST_SIZES], conv_mov,
            's--', color='#e74c3c', lw=2, ms=8, label="Mouvements inverses")

    ax.set_xlabel("Taille liste taboue k")
    ax.set_ylabel("Iteration moyenne de convergence")
    ax.set_title("Vitesse de convergence vers le meilleur trouve")
    ax.legend(fontsize=10)
    fig.tight_layout()
    p5 = f"{figures_dir}/fig_A2_5_convergence.png"
    fig.savefig(p5, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p5}")

    return [p1, p2, p3, p4, p5]


# =============================================================
# 5.  SAUVEGARDE DES LOGS
# =============================================================

def sauvegarder_logs(res_global, logs_dir="resultats/logs"):
    os.makedirs(logs_dir, exist_ok=True)
    path = f"{logs_dir}/A2_recherche_taboue.json"

    data = {}
    for strat in res_global:
        data[strat] = {}
        for k in res_global[strat]:
            r = res_global[strat][k]
            data[strat][str(k)] = {
                "moyenne"      : float(r["moyenne"]),
                "std"          : float(r["std"]),
                "meilleur"     : float(r["meilleur"]),
                "pire"         : float(r["pire"]),
                "conv_moyenne" : float(r["conv_moyenne"]),
                "couts_finaux" : [float(c) for c in r["couts_finaux"]]
            }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  OK  {path}")


# =============================================================
# PROGRAMME PRINCIPAL
# =============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  VOLET A — A2_recherche_taboue.py")
    print(f"  {NB_RUNS} runs x {len(TABU_LIST_SIZES)} tailles x 2 strategies")
    print("=" * 65)

    # Minimum global de reference
    res_enum    = enumeration_exacte()
    min_global  = res_enum["minimum_global_exact"]
    sol_opt     = res_enum["solution_optimale"]
    print(f"\n  Reference : minimum global = {min_global:.4f}  s* = {sol_opt}")

    # Lancement des experiences
    print(f"\n  Lancement de {NB_RUNS * len(TABU_LIST_SIZES) * 2} runs...")
    t0          = time.time()
    res_global  = lancer_experiences(nb_runs=NB_RUNS)
    duree       = time.time() - t0
    print(f"  Termine en {duree:.2f}s")

    # Tableau de resultats
    print("\n" + "-" * 65)
    print(f"  {'Strategie':<15} {'k':<5} {'Moy':>8} {'Std':>8} "
          f"{'Meilleur':>10} {'Taux(%)':>9} {'Conv':>7}")
    print("-" * 65)

    for strat in ["solutions", "mouvements"]:
        for k in TABU_LIST_SIZES:
            r     = res_global[strat][k]
            taux  = 100 * sum(1 for c in r["couts_finaux"]
                              if abs(c - min_global) < 1e-6) / NB_RUNS
            print(f"  {strat:<15} {k:<5} {r['moyenne']:>8.3f} {r['std']:>8.3f} "
                  f"{r['meilleur']:>10.3f} {taux:>8.1f}% {r['conv_moyenne']:>6.1f}")

    print("-" * 65)

    # Figures
    print("\n  Generation des figures...")
    base        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(base, "resultats", "figures")
    logs_dir    = os.path.join(base, "resultats", "logs")
    generer_figures(res_global, min_global, figures_dir=figures_dir)

    # Logs
    print("\n  Sauvegarde des logs...")
    sauvegarder_logs(res_global, logs_dir=logs_dir)

    print("\n" + "=" * 65)
    print("  A2_recherche_taboue.py termine avec succes")
    print(f"  Figures dans : {figures_dir}")
    print(f"  Logs dans    : {logs_dir}")
    print("=" * 65)