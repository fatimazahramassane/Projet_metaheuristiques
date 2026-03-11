import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_BITS, DL_NB_STARTS, RANDOM_SEED
from volet_A.fonction_cout import fonction_cout, get_voisins, solution_aleatoire, enumeration_exacte

np.random.seed(RANDOM_SEED)


# =============================================================
# 1.  ALGORITHME — DESCENTE LOCALE (plus grande pente)
# =============================================================

def descente_locale(s_init):
    
    s_courant      = np.array(s_init, dtype=int)
    cout_courant   = fonction_cout(s_courant)
    historique     = [cout_courant]
    nb_iterations  = 0

    while True:
        voisins       = get_voisins(s_courant)
        couts_voisins = [fonction_cout(v) for v in voisins]
        idx_meilleur  = np.argmin(couts_voisins)
        meilleur_cout = couts_voisins[idx_meilleur]

        # Si le meilleur voisin n'améliore pas donc minimum local atteint
        if meilleur_cout >= cout_courant:
            break

        # Déplacement vers le meilleur voisin
        s_courant    = voisins[idx_meilleur]
        cout_courant = meilleur_cout
        historique.append(cout_courant)
        nb_iterations += 1

    return {
        "solution_finale"  : s_courant,
        "cout_final"       : cout_courant,
        "nb_iterations"    : nb_iterations,
        "historique_couts" : historique
    }


# =============================================================
# 2.  MULTI-DEMARRAGE — 30 solutions initiales
# =============================================================

def multi_demarrage(nb_starts=DL_NB_STARTS, seed=RANDOM_SEED):
    """
    Lance la descente locale depuis nb_starts solutions initiales.

    Retourne
    --------
    dict :
      - resultats        : liste des résultats pour chaque départ
      - meilleur_global  : meilleure solution trouvée sur tous les runs
      - cout_meilleur    : coût de la meilleure solution
      - nb_convergences  : nombre de fois où le minimum global est atteint
      - proba_global     : probabilité empirique d'atteindre le minimum global
    """
    np.random.seed(seed)

    # Référence : minimum global exact
    ref          = enumeration_exacte()
    min_global   = ref["minimum_global_exact"]

    resultats          = []
    meilleur_global    = None
    cout_meilleur      = float('inf')
    nb_convergences    = 0

    for i in range(nb_starts):
        s_init  = solution_aleatoire()
        res     = descente_locale(s_init)

        resultats.append({
            "run"             : i + 1,
            "s_init"          : s_init.copy(),
            "cout_init"       : fonction_cout(s_init),
            "solution_finale" : res["solution_finale"],
            "cout_final"      : res["cout_final"],
            "nb_iterations"   : res["nb_iterations"],
            "historique_couts": res["historique_couts"],
            "atteint_global"  : np.isclose(res["cout_final"], min_global, atol=1e-6)
        })

        if res["cout_final"] < cout_meilleur:
            cout_meilleur   = res["cout_final"]
            meilleur_global = res["solution_finale"].copy()

        if np.isclose(res["cout_final"], min_global, atol=1e-6):
            nb_convergences += 1

    proba_global = nb_convergences / nb_starts

    return {
        "resultats"       : resultats,
        "meilleur_global" : meilleur_global,
        "cout_meilleur"   : cout_meilleur,
        "min_global_ref"  : min_global,
        "nb_convergences" : nb_convergences,
        "proba_global"    : proba_global,
        "nb_starts"       : nb_starts
    }


# =============================================================
# 3.  GENERATION DES FIGURES
# =============================================================

def generer_figures(bilan, figures_dir="resultats/figures"):
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

    resultats     = bilan["resultats"]
    min_global    = bilan["min_global_ref"]
    nb_starts     = bilan["nb_starts"]
    proba_global  = bilan["proba_global"]
    nb_conv       = bilan["nb_convergences"]

    couts_finaux  = [r["cout_final"]    for r in resultats]
    couts_initiaux= [r["cout_init"]     for r in resultats]
    nb_iters      = [r["nb_iterations"] for r in resultats]
    atteint       = [r["atteint_global"] for r in resultats]
    runs          = [r["run"]           for r in resultats]

    # ----------------------------------------------------------
    # FIGURE 1 — Couts finaux par run
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 5))

    colors = ['#2ecc71' if a else '#e74c3c' for a in atteint]
    bars   = ax.bar(runs, couts_finaux, color=colors,
                    edgecolor='black', linewidth=0.5)

    ax.axhline(min_global, color='red', ls='--', lw=2,
               label=f'Minimum global = {min_global:.2f}')

    # Annotations sur les barres
    for bar, val, ok in zip(bars, couts_finaux, atteint):
        sym = '★' if ok else ''
        ax.text(bar.get_x() + bar.get_width()/2,
                val - 0.8, f'{val:.1f}{sym}',
                ha='center', fontsize=7.5, color='white', fontweight='bold')

    green_p = mpatches.Patch(color='#2ecc71',
                              label=f'Atteint le global ({nb_conv}/{nb_starts})')
    red_p   = mpatches.Patch(color='#e74c3c',
                              label=f'Bloqué en minimum local ({nb_starts-nb_conv}/{nb_starts})')
    ax.legend(handles=[ax.get_lines()[0], green_p, red_p], fontsize=10)

    ax.set_xlabel("Numéro du run (départ aléatoire)")
    ax.set_ylabel("Coût final f(s)")
    ax.set_title(
        f"Descente locale — Coûts finaux sur {nb_starts} départs aléatoires\n"
        f"Probabilité d'atteindre le minimum global = {proba_global:.1%}  "
        f"({nb_conv}/{nb_starts})"
    )
    ax.set_xticks(runs)
    fig.tight_layout()
    p1 = f"{figures_dir}/fig_A1_1_couts_finaux.png"
    fig.savefig(p1, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p1}")

    # ----------------------------------------------------------
    # FIGURE 2 — Cout initial vs Cout final (scatter)
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, r in enumerate(resultats):
        color = '#2ecc71' if r["atteint_global"] else '#e74c3c'
        ax.annotate("",
                    xy  =(i+1, r["cout_final"]),
                    xytext=(i+1, r["cout_init"]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    ax.scatter(runs, couts_initiaux, c='#3498db', s=60, zorder=5,
               label='Coût initial', marker='o')
    ax.scatter(runs, couts_finaux,
               c=['#2ecc71' if a else '#e74c3c' for a in atteint],
               s=80, zorder=5, marker='D', label='Coût final')

    ax.axhline(min_global, color='red', ls='--', lw=2,
               label=f'Min. global = {min_global:.2f}')
    ax.set_xlabel("Numéro du run")
    ax.set_ylabel("Valeur de f(s)")
    ax.set_title("Evolution : coût initial → coût final\npour chaque départ")
    ax.legend(fontsize=10)
    ax.set_xticks(runs)
    fig.tight_layout()
    p2 = f"{figures_dir}/fig_A1_2_initial_vs_final.png"
    fig.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p2}")

    # ----------------------------------------------------------
    # FIGURE 3 — Nombre d'itérations par run
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 4))
    colors3 = ['#2ecc71' if a else '#f39c12' for a in atteint]
    ax.bar(runs, nb_iters, color=colors3, edgecolor='black', lw=0.5)
    ax.axhline(np.mean(nb_iters), color='blue', ls='--', lw=2,
               label=f'Moyenne = {np.mean(nb_iters):.1f} itérations')
    for i, (nb, ok) in enumerate(zip(nb_iters, atteint)):
        ax.text(i+1, nb + 0.1, str(nb), ha='center', fontsize=8)
    ax.set_xlabel("Numéro du run")
    ax.set_ylabel("Nombre d'itérations")
    ax.set_title("Nombre d'itérations avant convergence par run")
    ax.legend(fontsize=10)
    ax.set_xticks(runs)
    fig.tight_layout()
    p3 = f"{figures_dir}/fig_A1_3_nb_iterations.png"
    fig.savefig(p3, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p3}")

    # ----------------------------------------------------------
    # FIGURE 4 — Courbes d'évolution des meilleurs runs
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gauche : runs qui atteignent le global
    ax_ok  = axes[0]
    runs_ok = [r for r in resultats if r["atteint_global"]]
    for r in runs_ok[:8]:
        ax_ok.plot(r["historique_couts"], marker='o', ms=4,
                   label=f"Run {r['run']}")
    ax_ok.axhline(min_global, color='red', ls='--', lw=1.5,
                  label=f'Global = {min_global:.2f}')
    ax_ok.set_xlabel("Itération")
    ax_ok.set_ylabel("f(s)")
    ax_ok.set_title(f"Runs atteignant le minimum global\n({len(runs_ok)} runs)")
    ax_ok.legend(fontsize=8, ncol=2)

    # Droite : runs bloqués en minimum local
    ax_ko  = axes[1]
    runs_ko = [r for r in resultats if not r["atteint_global"]]
    for r in runs_ko[:8]:
        ax_ko.plot(r["historique_couts"], marker='s', ms=4,
                   ls='--', label=f"Run {r['run']} → {r['cout_final']:.1f}")
    ax_ko.axhline(min_global, color='red', ls='--', lw=1.5,
                  label=f'Global = {min_global:.2f}')
    ax_ko.set_xlabel("Itération")
    ax_ko.set_ylabel("f(s)")
    ax_ko.set_title(f"Runs bloqués en minimum local\n({len(runs_ko)} runs)")
    ax_ko.legend(fontsize=8, ncol=2)

    fig.suptitle("Courbes de convergence — Descente locale",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    p4 = f"{figures_dir}/fig_A1_4_courbes_convergence.png"
    fig.savefig(p4, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p4}")

    # ----------------------------------------------------------
    # FIGURE 5 — Histogramme des couts finaux
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogramme
    ax5a = axes[0]
    unique_couts, counts = np.unique(couts_finaux, return_counts=True)
    bar_colors5 = ['#2ecc71' if np.isclose(c, min_global) else '#e74c3c'
                   for c in unique_couts]
    ax5a.bar([str(round(c, 1)) for c in unique_couts],
             counts, color=bar_colors5, edgecolor='black', lw=0.7)
    for i, cnt in enumerate(counts):
        ax5a.text(i, cnt + 0.1, str(cnt), ha='center', fontsize=10)
    ax5a.set_xlabel("Valeur de coût final atteinte")
    ax5a.set_ylabel("Nombre de runs")
    ax5a.set_title("Distribution des minima locaux atteints")
    gp5 = mpatches.Patch(color='#2ecc71', label='Minimum global')
    rp5 = mpatches.Patch(color='#e74c3c', label='Minimum local')
    ax5a.legend(handles=[gp5, rp5], fontsize=10)

    # Pie chart
    ax5b = axes[1]
    sizes  = [nb_conv, nb_starts - nb_conv]
    labels = [f'Global atteint\n{nb_conv}/{nb_starts}',
              f'Minimum local\n{nb_starts-nb_conv}/{nb_starts}']
    explode = (0.05, 0)
    ax5b.pie(sizes, labels=labels, autopct='%1.1f%%',
             colors=['#2ecc71', '#e74c3c'],
             explode=explode, startangle=90,
             textprops={'fontsize': 11})
    ax5b.set_title(f"Probabilité d'atteindre\nle minimum global = {proba_global:.1%}")

    fig.suptitle("Analyse statistique — Descente locale",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    p5 = f"{figures_dir}/fig_A1_5_stats.png"
    fig.savefig(p5, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p5}")

    return [p1, p2, p3, p4, p5]


# =============================================================
# PROGRAMME PRINCIPAL
# =============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  VOLET A — A1_descente_locale.py")
    print(f"  {DL_NB_STARTS} départs aléatoires")
    print("=" * 65)

    # --- Exécution
    bilan = multi_demarrage(nb_starts=DL_NB_STARTS)

    # --- Affichage des résultats
    print(f"\n  Minimum global de référence : {bilan['min_global_ref']:.4f}")
    print(f"\n  {'Run':<5} {'Cout init':>10} {'Cout final':>10} "
          f"{'Iterations':>11} {'Global?':>8}")
    print("  " + "-" * 50)
    for r in bilan["resultats"]:
        ok  = "OUI ★" if r["atteint_global"] else "non"
        print(f"  {r['run']:<5} {r['cout_init']:>10.4f} "
              f"{r['cout_final']:>10.4f} {r['nb_iterations']:>11} {ok:>8}")

    print("\n" + "=" * 65)
    print(f"  BILAN")
    print(f"  Meilleure solution trouvée  : {bilan['meilleur_global']}")
    print(f"  Meilleur coût               : {bilan['cout_meilleur']:.4f}")
    print(f"  Convergences vers le global : {bilan['nb_convergences']}/{bilan['nb_starts']}")
    print(f"  Probabilité empirique       : {bilan['proba_global']:.1%}")
    print("=" * 65)

    # --- Génération des figures
    print("\n  Génération des figures...")
    base        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(base, "resultats", "figures")
    generer_figures(bilan, figures_dir=figures_dir)

    print("\n" + "=" * 65)
    print("  A1_descente_locale.py terminé avec succès")
    print(f"  Figures dans : {figures_dir}")
    print("=" * 65)# cela juste pour facilite la visualisation 