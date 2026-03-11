
import numpy as np
import sys, os, csv, json, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    N_BITS, RANDOM_SEED, NB_RUNS,
    DL_NB_STARTS,
    TABU_MAX_ITER, TABU_LIST_SIZES,
    SA_T0_VALUES, SA_LAMBDA_VALUES, SA_MAX_ITER,
    GA_NB_GENERATIONS, GA_POP_SIZE_DEFAULT,
    GA_PC_DEFAULT, GA_PM_DEFAULT
)

np.random.seed(RANDOM_SEED)


# =============================================================
# 1.  COLLECTE DES RESULTATS — Descente locale
# =============================================================

def collecter_descente_locale(nb_runs=NB_RUNS):
    """
    Lance nb_runs descentes locales independantes (1 depart/run).
    Retourne un dict de statistiques standardise.
    """
    from volet_A.fonction_cout   import solution_aleatoire, enumeration_exacte
    from volet_A.A1_descente_locale import descente_locale

    ref        = enumeration_exacte()
    min_global = ref["minimum_global_exact"]

    np.random.seed(RANDOM_SEED)
    couts, iters = [], []

    for _ in range(nb_runs):
        s0  = solution_aleatoire()
        res = descente_locale(s0)
        couts.append(res["cout_final"])
        iters.append(res["nb_iterations"])

    couts = np.array(couts)
    succes = np.sum(np.isclose(couts, min_global, atol=1e-6))

    return {
        "methode"      : "Descente locale",
        "min_global"   : min_global,
        "couts"        : couts,
        "iters"        : np.array(iters),
        "moyenne"      : float(np.mean(couts)),
        "std"          : float(np.std(couts)),
        "meilleur"     : float(np.min(couts)),
        "pire"         : float(np.max(couts)),
        "taux_succes"  : float(succes / nb_runs * 100),
        "iter_moy"     : float(np.mean(iters)),
        "nb_runs"      : nb_runs,
        # Ecart au minimum global (0 = optimal)
        "gap_moy"      : float(np.mean(couts) - min_global),
    }


# =============================================================
# 2.  COLLECTE DES RESULTATS — Recherche Taboue
# =============================================================

def collecter_recherche_taboue(nb_runs=NB_RUNS):
    """
    Lance nb_runs runs de recherche taboue pour chaque (strategie, k).
    Conserve la meilleure configuration pour le tableau de comparaison.
    Retourne AUSSI toutes les configurations pour l analyse fine.
    """
    from volet_A.fonction_cout        import solution_aleatoire, enumeration_exacte
    from volet_A.A2_recherche_taboue  import (recherche_taboue_solutions,
                                               recherche_taboue_mouvements)

    ref        = enumeration_exacte()
    min_global = ref["minimum_global_exact"]

    toutes_configs = {}   # (strategie, k) -> stats

    for strat, fn in [("solutions",  recherche_taboue_solutions),
                       ("mouvements", recherche_taboue_mouvements)]:
        toutes_configs[strat] = {}
        for k in TABU_LIST_SIZES:
            np.random.seed(RANDOM_SEED)
            couts, iters = [], []
            for _ in range(nb_runs):
                s0  = solution_aleatoire()
                res = fn(s0, taille_taboue=k, max_iter=TABU_MAX_ITER)
                couts.append(res["cout"])
                iters.append(res["nb_depla"])

            couts  = np.array(couts)
            succes = np.sum(np.isclose(couts, min_global, atol=1e-6))

            toutes_configs[strat][k] = {
                "couts"       : couts,
                "iters"       : np.array(iters),
                "moyenne"     : float(np.mean(couts)),
                "std"         : float(np.std(couts)),
                "meilleur"    : float(np.min(couts)),
                "pire"        : float(np.max(couts)),
                "taux_succes" : float(succes / nb_runs * 100),
                "iter_moy"    : float(np.mean(iters)),
                "gap_moy"     : float(np.mean(couts) - min_global),
            }

    # --- Meilleure configuration globale (taux de succes max)
    best_taux, best_strat, best_k = -1, None, None
    for strat in toutes_configs:
        for k in toutes_configs[strat]:
            t = toutes_configs[strat][k]["taux_succes"]
            if t > best_taux:
                best_taux, best_strat, best_k = t, strat, k

    best = toutes_configs[best_strat][best_k].copy()
    best["methode"]          = f"Recherche Taboue (strat={best_strat}, k={best_k})"
    best["min_global"]       = min_global
    best["nb_runs"]          = nb_runs
    best["toutes_configs"]   = toutes_configs
    best["meilleure_strat"]  = best_strat
    best["meilleur_k"]       = best_k

    return best


# =============================================================
# 3.  COLLECTE DES RESULTATS — Recuit Simule
# =============================================================

def collecter_recuit_simule(nb_runs=NB_RUNS):
    """
    Lance nb_runs runs de recuit simule pour chaque (T0, lambda).
    Conserve la meilleure configuration pour le tableau de comparaison.
    """
    from volet_A.fonction_cout    import solution_aleatoire, enumeration_exacte
    from volet_A.A3_recuit_simule import recuit_simule

    ref        = enumeration_exacte()
    min_global = ref["minimum_global_exact"]

    toutes_configs = {}  # (T0, lam) -> stats

    for T0 in SA_T0_VALUES:
        for lam in SA_LAMBDA_VALUES:
            cle = (T0, lam)
            np.random.seed(RANDOM_SEED)
            couts, iters = [], []
            for _ in range(nb_runs):
                s0  = solution_aleatoire()
                res = recuit_simule(s0, T0=T0, lam=lam, max_iter=SA_MAX_ITER)
                couts.append(res["cout"])
                iters.append(res["iterations"])

            couts  = np.array(couts)
            succes = np.sum(np.isclose(couts, min_global, atol=1e-6))

            toutes_configs[cle] = {
                "couts"       : couts,
                "iters"       : np.array(iters),
                "moyenne"     : float(np.mean(couts)),
                "std"         : float(np.std(couts)),
                "meilleur"    : float(np.min(couts)),
                "pire"        : float(np.max(couts)),
                "taux_succes" : float(succes / nb_runs * 100),
                "iter_moy"    : float(np.mean(iters)),
                "gap_moy"     : float(np.mean(couts) - min_global),
            }

    # --- Meilleure configuration
    best_taux, best_cle = -1, None
    for cle in toutes_configs:
        t = toutes_configs[cle]["taux_succes"]
        if t > best_taux:
            best_taux, best_cle = t, cle

    best = toutes_configs[best_cle].copy()
    best["methode"]        = f"Recuit Simule (T0={best_cle[0]}, lam={best_cle[1]})"
    best["min_global"]     = min_global
    best["nb_runs"]        = nb_runs
    best["toutes_configs"] = toutes_configs
    best["meilleur_T0"]    = best_cle[0]
    best["meilleur_lam"]   = best_cle[1]

    return best


# =============================================================
# 4.  COLLECTE DES RESULTATS — Algorithme Genetique
# =============================================================

def collecter_ag(nb_runs=NB_RUNS):
    """
    Lance nb_runs runs de l AG avec les parametres par defaut.
    Retourne stats standardisees (fitness maximisee -> on negat. pour comparaison).
    """
    from volet_B.fonction_objectif        import analyser_fonction
    from volet_B.B2_operateurs_genetiques import algorithme_genetique

    analyse  = analyser_fonction()
    f_max_ref = analyse["f_max"]

    np.random.seed(RANDOM_SEED)
    fitnesses, iters_conv = [], []

    for _ in range(nb_runs):
        res = algorithme_genetique(
            taille_pop = GA_POP_SIZE_DEFAULT,
            nb_gen     = GA_NB_GENERATIONS,
            pc         = GA_PC_DEFAULT,
            pm         = GA_PM_DEFAULT,
            methode    = "tournoi"
        )
        fitnesses.append(res["meilleure_fitness"])
        # Iteration de premiere convergence a 95% du max reference
        hist = res["historique_best"]
        conv = next((i for i, v in enumerate(hist) if v >= 0.95 * f_max_ref),
                    GA_NB_GENERATIONS)
        iters_conv.append(conv)

    fitnesses  = np.array(fitnesses)
    iters_conv = np.array(iters_conv)
    succes = np.sum(np.abs(fitnesses - f_max_ref) < 0.05)

    return {
        "methode"      : f"Algo Genetique (pop={GA_POP_SIZE_DEFAULT}, "
                         f"pc={GA_PC_DEFAULT}, pm={GA_PM_DEFAULT})",
        "f_max_ref"    : f_max_ref,
        # On stocke les fitnesses (maximisation) comme "couts" negatifs
        # pour uniformiser l affichage des tableaux
        "fitnesses"    : fitnesses,
        "iters_conv"   : iters_conv,
        "moyenne"      : float(np.mean(fitnesses)),
        "std"          : float(np.std(fitnesses)),
        "meilleur"     : float(np.max(fitnesses)),
        "pire"         : float(np.min(fitnesses)),
        "taux_succes"  : float(succes / nb_runs * 100),
        "iter_moy"     : float(np.mean(iters_conv)),
        "gap_moy"      : float(f_max_ref - np.mean(fitnesses)),
        "nb_runs"      : nb_runs,
    }


# =============================================================
# 5.  TABLEAU SYNTHÉTIQUE
# =============================================================

def afficher_tableau(dl, rt, rs, ag):
    """Affiche le tableau comparatif dans le terminal."""
    sep = "=" * 95

    print(f"\n{sep}")
    print("  TABLEAU COMPARATIF — Toutes les methodes  (sur {nb} runs independants)".format(
          nb=dl["nb_runs"]))
    print(sep)
    print(f"  {'Methode':<45} {'Moy':>8} {'Std':>7} {'Meilleur':>10} "
          f"{'Taux %':>8} {'Gap':>8} {'Iter moy':>9}")
    print("  " + "-" * 93)

    # Volet A — minimisation
    for res in [dl, rt, rs]:
        print(f"  {res['methode']:<45} "
              f"{res['moyenne']:>8.4f} "
              f"{res['std']:>7.4f} "
              f"{res['meilleur']:>10.4f} "
              f"{res['taux_succes']:>7.1f}% "
              f"{res['gap_moy']:>8.4f} "
              f"{res['iter_moy']:>9.1f}")

    print("  " + "-" * 93)

    # Volet B — maximisation (affichage separe)
    print(f"  {ag['methode']:<45} "
          f"{ag['moyenne']:>8.4f} "
          f"{ag['std']:>7.4f} "
          f"{ag['meilleur']:>10.4f} "
          f"{ag['taux_succes']:>7.1f}% "
          f"{ag['gap_moy']:>8.4f} "
          f"{ag['iter_moy']:>9.1f}")

    print(sep)
    print("  Notes :")
    print("  - Volet A (DL, RT, RS) : MINIMISATION — meilleur = valeur la plus basse")
    print("  - Volet B (AG)         : MAXIMISATION — meilleur = valeur la plus haute")
    print("  - Taux succes          : % runs proches de l optimum (tolerance 1e-6 / 0.05)")
    print("  - Gap                  : |moyenne - optimum|  (0 = parfait)")
    print(f"{sep}\n")


def exporter_csv(dl, rt, rs, ag, logs_dir="resultats/logs"):
    """Exporte le tableau comparatif en CSV."""
    os.makedirs(logs_dir, exist_ok=True)
    path = f"{logs_dir}/tableau_comparatif.csv"

    lignes = [
        ["Methode", "Moyenne", "Std", "Meilleur", "Pire",
         "Taux_succes_%", "Gap_moy", "Iter_moy", "NB_Runs"],
    ]
    for res in [dl, rt, rs]:
        lignes.append([
            res["methode"],
            f"{res['moyenne']:.6f}",
            f"{res['std']:.6f}",
            f"{res['meilleur']:.6f}",
            f"{res['pire']:.6f}",
            f"{res['taux_succes']:.2f}",
            f"{res['gap_moy']:.6f}",
            f"{res['iter_moy']:.2f}",
            res["nb_runs"],
        ])
    # AG — meilleur = max, donc pire = min
    lignes.append([
        ag["methode"],
        f"{ag['moyenne']:.6f}",
        f"{ag['std']:.6f}",
        f"{ag['meilleur']:.6f}",
        f"{ag['pire']:.6f}",
        f"{ag['taux_succes']:.2f}",
        f"{ag['gap_moy']:.6f}",
        f"{ag['iter_moy']:.2f}",
        ag["nb_runs"],
    ])

    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(lignes)

    print(f"  OK  {path}")
    return path


# =============================================================
# 6.  GENERATION DES FIGURES
# =============================================================

def generer_figures(dl, rt, rs, ag, figures_dir="resultats/figures"):
    """
    Genere les figures de comparaison globale.
    """
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

    couleurs = {
        "Descente locale" : '#e74c3c',
        "Recherche Taboue": '#e67e22',
        "Recuit Simule"   : '#2ecc71',
        "Algo Genetique"  : '#3498db',
    }

    labels_courts = [
        "Descente\nlocale",
        "Recherche\nTaboue",
        "Recuit\nSimule",
        "Algo\nGenetique",
    ]

    # ----------------------------------------------------------
    # FIGURE 1 — Boxplot des couts finaux (Volet A)
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gauche : Volet A — minimisation
    ax = axes[0]
    data_a = [dl["couts"], rt["couts"], rs["couts"]]
    bp = ax.boxplot(data_a,
                    labels=["Descente\nlocale", "Taboue\n(best)", "Recuit\n(best)"],
                    patch_artist=True,
                    medianprops=dict(color='black', lw=2))
    c_list = [couleurs["Descente locale"],
              couleurs["Recherche Taboue"],
              couleurs["Recuit Simule"]]
    for patch, c in zip(bp['boxes'], c_list):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.axhline(dl["min_global"], color='red', ls='--', lw=1.8,
               label=f"Optimum global = {dl['min_global']:.4f}")
    ax.set_ylabel("Valeur de f(s)  (minimisation)")
    ax.set_title(f"Volet A — Distribution des couts finaux\n({dl['nb_runs']} runs independants)")
    ax.legend(fontsize=10)

    # Droite : Volet B — maximisation
    ax2 = axes[1]
    bp2 = ax2.boxplot([ag["fitnesses"]],
                       labels=["AG\n(best config)"],
                       patch_artist=True,
                       medianprops=dict(color='black', lw=2))
    bp2['boxes'][0].set_facecolor(couleurs["Algo Genetique"])
    bp2['boxes'][0].set_alpha(0.75)
    ax2.axhline(ag["f_max_ref"], color='red', ls='--', lw=1.8,
                label=f"Max reel = {ag['f_max_ref']:.4f}")
    ax2.set_ylabel("Fitness f(x)  (maximisation)")
    ax2.set_title(f"Volet B — Distribution des fitness\n({ag['nb_runs']} runs independants)")
    ax2.legend(fontsize=10)

    fig.suptitle("Comparaison globale — Qualite finale par methode",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    p1 = f"{figures_dir}/fig_COMP_1_boxplot_final.png"
    fig.savefig(p1, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p1}")

    # ----------------------------------------------------------
    # FIGURE 2 — Barres : Taux de succes + Gap moyen
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    methodes_a = ["Descente\nlocale", "Taboue\n(best)", "Recuit\n(best)"]
    couleurs_a = [couleurs["Descente locale"],
                  couleurs["Recherche Taboue"],
                  couleurs["Recuit Simule"]]

    # Taux de succes
    ax = axes[0]
    taux_a = [dl["taux_succes"], rt["taux_succes"], rs["taux_succes"]]
    bars = ax.bar(methodes_a, taux_a, color=couleurs_a,
                  edgecolor='black', lw=0.7)
    # AG sur axe secondaire (meme figure)
    ax_r = ax.twinx()
    ax_r.bar(["AG\n(best)"], [ag["taux_succes"]],
             color=couleurs["Algo Genetique"],
             edgecolor='black', lw=0.7, alpha=0.85)
    ax_r.set_ylabel("Taux succes AG (%)", color=couleurs["Algo Genetique"])
    for bar, val in zip(bars, taux_a):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + 1, f"{val:.1f}%", ha='center', fontsize=10)
    ax.set_ylim(0, 120)
    ax.set_ylabel("Taux de succes (%)")
    ax.set_title("Taux de convergence vers l'optimum\n(% runs atteignant la solution optimale)")

    # Gap moyen
    ax2 = axes[1]
    all_labels = ["Descente\nlocale", "Taboue\n(best)",
                  "Recuit\n(best)", "AG\n(best)"]
    all_gaps   = [dl["gap_moy"], rt["gap_moy"], rs["gap_moy"], ag["gap_moy"]]
    all_cols   = [couleurs["Descente locale"], couleurs["Recherche Taboue"],
                  couleurs["Recuit Simule"],   couleurs["Algo Genetique"]]
    bars2 = ax2.bar(all_labels, all_gaps, color=all_cols,
                    edgecolor='black', lw=0.7)
    for bar, val in zip(bars2, all_gaps):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 val + 0.01, f"{val:.4f}", ha='center', fontsize=9)
    ax2.set_ylabel("Gap moyen  |moyenne - optimum|")
    ax2.set_title("Ecart moyen a l'optimum\n(plus petit = meilleur)")

    fig.suptitle("Comparaison globale — Qualite et robustesse",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    p2 = f"{figures_dir}/fig_COMP_2_taux_gap.png"
    fig.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p2}")

    # ----------------------------------------------------------
    # FIGURE 3 — Radar chart des criteres (Volet A seulement)
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    criteres   = ["Taux\nsucces", "Stabilite\n(1/std)", "Qualite\n(1/gap)",
                  "Rapidite\n(1/iter)", "Robustesse\n(1/pire-opt)"]
    n_crit     = len(criteres)
    angles     = [k * 2 * np.pi / n_crit for k in range(n_crit)]
    angles    += angles[:1]

    def normaliser(val, mini, maxi):
        if maxi == mini:
            return 0.5
        return (val - mini) / (maxi - mini)

    # Calcul des valeurs normalisees pour chaque methode
    eps = 1e-9
    stats_a = [
        {
            "label"  : "Descente locale",
            "taux"   : dl["taux_succes"],
            "stab"   : 1.0 / (dl["std"] + eps),
            "qualite": 1.0 / (dl["gap_moy"] + eps),
            "rapid"  : 1.0 / (dl["iter_moy"] + eps),
            "robust" : 1.0 / (dl["pire"] - dl["min_global"] + eps),
        },
        {
            "label"  : "Recherche Taboue",
            "taux"   : rt["taux_succes"],
            "stab"   : 1.0 / (rt["std"] + eps),
            "qualite": 1.0 / (rt["gap_moy"] + eps),
            "rapid"  : 1.0 / (rt["iter_moy"] + eps),
            "robust" : 1.0 / (rt["pire"] - rt["min_global"] + eps),
        },
        {
            "label"  : "Recuit Simule",
            "taux"   : rs["taux_succes"],
            "stab"   : 1.0 / (rs["std"] + eps),
            "qualite": 1.0 / (rs["gap_moy"] + eps),
            "rapid"  : 1.0 / (rs["iter_moy"] + eps),
            "robust" : 1.0 / (rs["pire"] - rs["min_global"] + eps),
        },
    ]

    # Normalisation globale par critere
    keys = ["taux", "stab", "qualite", "rapid", "robust"]
    for ki, key in enumerate(keys):
        vals = [s[key] for s in stats_a]
        vmin, vmax = min(vals), max(vals)
        for s in stats_a:
            s[key + "_norm"] = normaliser(s[key], vmin, vmax)

    coul_radar = [couleurs["Descente locale"],
                  couleurs["Recherche Taboue"],
                  couleurs["Recuit Simule"]]

    for s, col in zip(stats_a, coul_radar):
        vals_norm = [s[key + "_norm"] for key in keys]
        vals_norm += vals_norm[:1]
        ax.plot(angles, vals_norm, color=col, lw=2, label=s["label"])
        ax.fill(angles, vals_norm, color=col, alpha=0.15)

    ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], criteres)
    ax.set_ylim(0, 1)
    ax.set_title("Profil des methodes — Volet A\n(valeurs normalisees 0-1)",
                 fontsize=13, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.tight_layout()
    p3 = f"{figures_dir}/fig_COMP_3_radar.png"
    fig.savefig(p3, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p3}")

    # ----------------------------------------------------------
    # FIGURE 4 — Tableau visuel (heatmap des performances)
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    col_labels = ["Methode", "Moyenne", "Std", "Meilleur",
                  "Taux succes", "Gap moy", "Iter moy"]

    rows_a = []
    for res, nom in [(dl, "Descente locale"),
                     (rt, "Taboue (best config)"),
                     (rs, "Recuit (best config)")]:
        rows_a.append([
            nom,
            f"{res['moyenne']:.4f}",
            f"{res['std']:.4f}",
            f"{res['meilleur']:.4f}",
            f"{res['taux_succes']:.1f}%",
            f"{res['gap_moy']:.4f}",
            f"{res['iter_moy']:.1f}",
        ])

    rows_a.append([
        "AG (best config) [MAX]",
        f"{ag['moyenne']:.4f}",
        f"{ag['std']:.4f}",
        f"{ag['meilleur']:.4f}",
        f"{ag['taux_succes']:.1f}%",
        f"{ag['gap_moy']:.4f}",
        f"{ag['iter_moy']:.1f}",
    ])

    table = ax.table(cellText=rows_a, colLabels=col_labels,
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Coloration de l'entete
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Coloration des lignes par methode
    row_colors = [
        couleurs["Descente locale"],
        couleurs["Recherche Taboue"],
        couleurs["Recuit Simule"],
        couleurs["Algo Genetique"],
    ]
    for i, col in enumerate(row_colors):
        table[i+1, 0].set_facecolor(col)
        table[i+1, 0].set_text_props(color='white', fontweight='bold')
        for j in range(1, len(col_labels)):
            table[i+1, j].set_facecolor(col + '30')  # transparence

    ax.set_title("Tableau synthétique — Comparaison globale des methodes",
                 fontsize=13, fontweight='bold', pad=10)
    fig.tight_layout()
    p4 = f"{figures_dir}/fig_COMP_4_tableau.png"
    fig.savefig(p4, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p4}")

    return [p1, p2, p3, p4]


# =============================================================
# PROGRAMME PRINCIPAL
# =============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  COMPARAISON — tableaux_resultats.py")
    print(f"  {NB_RUNS} runs independants par methode")
    print("=" * 70)

    t_total = time.time()

    print("\n  [1/4] Descente locale...")
    t0 = time.time()
    dl = collecter_descente_locale(nb_runs=NB_RUNS)
    print(f"        Done en {time.time()-t0:.1f}s  "
          f"| taux succes = {dl['taux_succes']:.1f}%")

    print("\n  [2/4] Recherche Taboue (toutes configs)...")
    t0 = time.time()
    rt = collecter_recherche_taboue(nb_runs=NB_RUNS)
    print(f"        Done en {time.time()-t0:.1f}s  "
          f"| meilleure config : strat={rt['meilleure_strat']}, k={rt['meilleur_k']}"
          f"  | taux succes = {rt['taux_succes']:.1f}%")

    print("\n  [3/4] Recuit Simule (toutes configs)...")
    t0 = time.time()
    rs = collecter_recuit_simule(nb_runs=NB_RUNS)
    print(f"        Done en {time.time()-t0:.1f}s  "
          f"| meilleure config : T0={rs['meilleur_T0']}, lam={rs['meilleur_lam']}"
          f"  | taux succes = {rs['taux_succes']:.1f}%")

    print("\n  [4/4] Algorithme Genetique...")
    t0 = time.time()
    ag = collecter_ag(nb_runs=NB_RUNS)
    print(f"        Done en {time.time()-t0:.1f}s  "
          f"| taux succes = {ag['taux_succes']:.1f}%")

    # --- Affichage terminal
    afficher_tableau(dl, rt, rs, ag)

    # --- Export CSV
    print("  Export CSV...")
    base     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(base, "resultats", "logs")
    exporter_csv(dl, rt, rs, ag, logs_dir=logs_dir)

    # --- Figures
    print("\n  Generation des figures...")
    figures_dir = os.path.join(base, "resultats", "figures")
    generer_figures(dl, rt, rs, ag, figures_dir=figures_dir)

    print("\n" + "=" * 70)
    print(f"  tableaux_resultats.py termine en {time.time()-t_total:.1f}s")
    print(f"  Figures dans : {figures_dir}")
    print(f"  CSV dans     : {logs_dir}/tableau_comparatif.csv")
    print("=" * 70)