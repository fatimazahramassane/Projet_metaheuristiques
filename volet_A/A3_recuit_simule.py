import numpy as np
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (N_BITS, RANDOM_SEED, SA_T0_VALUES, SA_LAMBDA_VALUES,
                    SA_MAX_ITER, SA_T_MIN, NB_RUNS)
from volet_A.fonction_cout import (fonction_cout, get_voisins,
                                   solution_aleatoire, enumeration_exacte)

np.random.seed(RANDOM_SEED)


# =============================================================
# 1.  RECUIT SIMULE  — algorithme principal
# =============================================================

def recuit_simule(s_init, T0=50.0, lam=0.95, max_iter=SA_MAX_ITER, T_min=SA_T_MIN):
    
    s_courant  = np.array(s_init, dtype=int).copy()
    s_meilleur = s_courant.copy()
    f_courant  = fonction_cout(s_courant)
    f_meilleur = f_courant
    T          = T0

    evolution_best = [f_meilleur]
    evolution_curr = [f_courant]
    evolution_T    = [T]
    nb_acceptees   = 0   # solutions degradantes acceptees
    nb_refusees    = 0   # solutions degradantes refusees
    iteration      = 0

    for it in range(max_iter):
        if T < T_min:
            break

        iteration += 1

        # Choisir un voisin aleatoire
        idx     = np.random.randint(0, N_BITS)
        voisin  = s_courant.copy()
        voisin[idx] = 1 - voisin[idx]
        f_voisin = fonction_cout(voisin)

        delta_E = f_voisin - f_courant

        # Regle d'acceptation
        if delta_E <= 0:
            # Amelioration : toujours accepter
            s_courant = voisin.copy()
            f_courant = f_voisin
        else:
            # Degradation : accepter avec probabilite exp(-DE/T)
            prob = np.exp(-delta_E / T)
            if np.random.random() < prob:
                s_courant = voisin.copy()
                f_courant = f_voisin
                nb_acceptees += 1
            else:
                nb_refusees += 1

        # Mise a jour du meilleur global
        if f_courant < f_meilleur:
            f_meilleur = f_courant
            s_meilleur = s_courant.copy()

        # Refroidissement
        T *= lam

        evolution_best.append(f_meilleur)
        evolution_curr.append(f_courant)
        evolution_T.append(T)

    return {
        "solution"       : s_meilleur,
        "cout"           : f_meilleur,
        "evolution_best" : evolution_best,
        "evolution_curr" : evolution_curr,
        "evolution_T"    : evolution_T,
        "nb_acceptees"   : nb_acceptees,
        "nb_refusees"    : nb_refusees,
        "iterations"     : iteration
    }


# =============================================================
# 2.  EXPERIENCES COMPLETES
#     NB_RUNS x len(T0) x len(lambda) combinaisons
# =============================================================

def lancer_experiences(nb_runs=NB_RUNS):
    """
    Lance toutes les combinaisons (T0, lambda) sur nb_runs runs.
    Retourne un dictionnaire de resultats.
    """
    res_global = {}

    for T0 in SA_T0_VALUES:
        for lam in SA_LAMBDA_VALUES:
            cle = (T0, lam)
            couts_finaux  = []
            nb_iter_list  = []
            nb_acc_list   = []
            evols_best    = []
            evols_T       = []
            convergences  = []

            np.random.seed(RANDOM_SEED)
            for run in range(nb_runs):
                s0 = solution_aleatoire()
                r  = recuit_simule(s0, T0=T0, lam=lam)

                couts_finaux.append(r["cout"])
                nb_iter_list.append(r["iterations"])
                nb_acc_list.append(r["nb_acceptees"])
                evols_best.append(r["evolution_best"])
                evols_T.append(r["evolution_T"])

                # Iteration de convergence au meilleur
                best = r["cout"]
                conv = next((i for i, v in enumerate(r["evolution_best"])
                             if v == best), len(r["evolution_best"]))
                convergences.append(conv)

            res_global[cle] = {
                "couts_finaux" : couts_finaux,
                "nb_iter"      : nb_iter_list,
                "nb_acceptees" : nb_acc_list,
                "evols_best"   : evols_best,
                "evols_T"      : evols_T,
                "convergences" : convergences,
                "moyenne"      : np.mean(couts_finaux),
                "std"          : np.std(couts_finaux),
                "meilleur"     : np.min(couts_finaux),
                "pire"         : np.max(couts_finaux),
                "conv_moy"     : np.mean(convergences),
                "acc_moy"      : np.mean(nb_acc_list)
            }

    return res_global


# =============================================================
# 3.  GENERATION DES FIGURES
# =============================================================

def generer_figures(res_global, min_global, figures_dir="resultats/figures"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

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

    # Palette de couleurs pour les 9 combinaisons
    couleurs_T0  = {10.0: '#e74c3c', 50.0: '#2ecc71', 100.0: '#3498db'}
    styles_lam   = {0.85: '-', 0.92: '--', 0.99: ':'}
    marqueurs    = {0.85: 'o', 0.92: 's', 0.99: '^'}

    # ----------------------------------------------------------
    # FIGURE 1 — Evolution du meilleur cout pour chaque (T0, lam)
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, T0 in zip(axes, SA_T0_VALUES):
        for lam in SA_LAMBDA_VALUES:
            cle   = (T0, lam)
            evols = res_global[cle]["evols_best"]
            max_l = max(len(e) for e in evols)
            mat   = np.full((len(evols), max_l), np.nan)
            for i, e in enumerate(evols):
                mat[i, :len(e)] = e
            moy = np.nanmean(mat, axis=0)
            ax.plot(moy, color=couleurs_T0[T0],
                    ls=styles_lam[lam], lw=2,
                    label=f"lam={lam}  fin={res_global[cle]['moyenne']:.2f}")

        ax.axhline(min_global, color='black', ls='--', lw=1.5,
                   label=f'Min. global={min_global:.1f}')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Meilleur cout")
        ax.set_title(f"T0 = {T0}")
        ax.legend(fontsize=8)

    fig.suptitle("Recuit Simule — Evolution du meilleur cout (moy. 30 runs)",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    p1 = f"{figures_dir}/fig_A3_1_evolution_best.png"
    fig.savefig(p1, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p1}")

    # ----------------------------------------------------------
    # FIGURE 2 — Evolution de la temperature
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    for T0 in SA_T0_VALUES:
        for lam in SA_LAMBDA_VALUES:
            cle  = (T0, lam)
            evol = res_global[cle]["evols_T"][0]  # un seul run suffit
            ax.plot(evol, color=couleurs_T0[T0],
                    ls=styles_lam[lam], lw=2,
                    label=f"T0={T0}, lam={lam}")

    ax.axhline(SA_T_MIN, color='black', ls=':', lw=1.5,
               label=f'T_min = {SA_T_MIN}')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Temperature T")
    ax.set_title("Profils de refroidissement T(k) = lam^k * T0")
    ax.set_yscale('log')
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    p2 = f"{figures_dir}/fig_A3_2_temperature.png"
    fig.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p2}")

    # ----------------------------------------------------------
    # FIGURE 3 — Boxplot couts finaux pour les 9 combinaisons
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 5))

    labels_box = []
    data_box   = []
    colors_box = []

    for T0 in SA_T0_VALUES:
        for lam in SA_LAMBDA_VALUES:
            cle = (T0, lam)
            data_box.append(res_global[cle]["couts_finaux"])
            labels_box.append(f"T0={int(T0)}\nlam={lam}")
            colors_box.append(couleurs_T0[T0])

    bp = ax.boxplot(data_box, tick_labels=labels_box, patch_artist=True,
                    medianprops=dict(color='black', lw=2))
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.axhline(min_global, color='red', ls='--', lw=1.5,
               label=f'Min. global = {min_global:.2f}')

    patches_legend = [mpatches.Patch(color=couleurs_T0[T0], label=f'T0={T0}')
                      for T0 in SA_T0_VALUES]
    ax.legend(handles=patches_legend + [
        mpatches.Patch(color='red', label=f'Min. global={min_global:.1f}')
    ], fontsize=9)

    ax.set_xlabel("Combinaison (T0, lambda)")
    ax.set_ylabel("Cout final")
    ax.set_title(f"Distribution des couts finaux — 9 combinaisons x {NB_RUNS} runs")
    fig.tight_layout()
    p3 = f"{figures_dir}/fig_A3_3_boxplot.png"
    fig.savefig(p3, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p3}")

    # ----------------------------------------------------------
    # FIGURE 4 — Taux d'atteinte du minimum global
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))

    x     = np.arange(len(SA_LAMBDA_VALUES))
    width = 0.25

    for i, T0 in enumerate(SA_T0_VALUES):
        taux = [
            100 * sum(1 for c in res_global[(T0, lam)]["couts_finaux"]
                      if abs(c - min_global) < 1e-6) / NB_RUNS
            for lam in SA_LAMBDA_VALUES
        ]
        bars = ax.bar(x + i * width, taux, width,
                      label=f"T0={T0}", color=couleurs_T0[T0],
                      alpha=0.85, edgecolor='black', lw=0.5)
        for bar, v in zip(bars, taux):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                    f'{v:.0f}%', ha='center', fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"lam={l}" for l in SA_LAMBDA_VALUES])
    ax.set_xlabel("Taux de refroidissement lambda")
    ax.set_ylabel("Taux d'atteinte du minimum global (%)")
    ax.set_title(f"Capacite a trouver le minimum global — {NB_RUNS} runs")
    ax.set_ylim(0, 115)
    ax.axhline(100, color='black', ls=':', lw=1)
    ax.legend(fontsize=10)
    fig.tight_layout()
    p4 = f"{figures_dir}/fig_A3_4_taux_global.png"
    fig.savefig(p4, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p4}")

    # ----------------------------------------------------------
    # FIGURE 5 — Acceptation des solutions degradantes
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Nombre moyen de solutions degradantes acceptees
    acc_data   = []
    labels_acc = []
    colors_acc = []
    for T0 in SA_T0_VALUES:
        for lam in SA_LAMBDA_VALUES:
            cle = (T0, lam)
            acc_data.append(res_global[cle]["acc_moy"])
            labels_acc.append(f"T0={int(T0)}\nlam={lam}")
            colors_acc.append(couleurs_T0[T0])

    bars = axes[0].bar(range(len(acc_data)), acc_data,
                       color=colors_acc, edgecolor='black', lw=0.5, alpha=0.85)
    axes[0].set_xticks(range(len(labels_acc)))
    axes[0].set_xticklabels(labels_acc, fontsize=8)
    axes[0].set_ylabel("Nb moyen d'acceptations degradantes")
    axes[0].set_title("Nb de solutions degradantes acceptees\n(mesure de la diversification)")
    for bar, v in zip(bars, acc_data):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 1,
                     f'{v:.0f}', ha='center', fontsize=8)

    # Illustration probabilite p = exp(-DE/T) en fonction de T
    DE_values = [1.0, 3.0, 5.0, 10.0]
    T_range   = np.linspace(0.1, 100, 500)
    for de in DE_values:
        axes[1].plot(T_range, np.exp(-de / T_range), lw=2,
                     label=f"ΔE = {de}")
    for T0 in SA_T0_VALUES:
        axes[1].axvline(T0, ls='--', lw=1.2,
                        color=couleurs_T0[T0], label=f"T0={T0}")
    axes[1].set_xlabel("Temperature T")
    axes[1].set_ylabel("Probabilite d'acceptation p = exp(-DE/T)")
    axes[1].set_title("Effet de la temperature sur l'acceptation")
    axes[1].legend(fontsize=8, ncol=2)

    fig.suptitle("Recuit Simule — Analyse de l'acceptation des solutions degradantes",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    p5 = f"{figures_dir}/fig_A3_5_acceptation.png"
    fig.savefig(p5, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p5}")

    # ----------------------------------------------------------
    # FIGURE 6 — Heatmap qualite : T0 x lambda
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    mat_moy  = np.zeros((len(SA_T0_VALUES), len(SA_LAMBDA_VALUES)))
    mat_taux = np.zeros((len(SA_T0_VALUES), len(SA_LAMBDA_VALUES)))

    for i, T0 in enumerate(SA_T0_VALUES):
        for j, lam in enumerate(SA_LAMBDA_VALUES):
            cle = (T0, lam)
            mat_moy[i, j]  = res_global[cle]["moyenne"]
            mat_taux[i, j] = 100 * sum(
                1 for c in res_global[cle]["couts_finaux"]
                if abs(c - min_global) < 1e-6) / NB_RUNS

    im1 = axes[0].imshow(mat_moy, cmap='RdYlGn_r', aspect='auto')
    plt.colorbar(im1, ax=axes[0], label='Cout moyen final')
    axes[0].set_xticks(range(len(SA_LAMBDA_VALUES)))
    axes[0].set_xticklabels([f"lam={l}" for l in SA_LAMBDA_VALUES])
    axes[0].set_yticks(range(len(SA_T0_VALUES)))
    axes[0].set_yticklabels([f"T0={t}" for t in SA_T0_VALUES])
    axes[0].set_title("Cout moyen final")
    for i in range(len(SA_T0_VALUES)):
        for j in range(len(SA_LAMBDA_VALUES)):
            axes[0].text(j, i, f"{mat_moy[i,j]:.2f}",
                         ha='center', va='center', fontsize=10, fontweight='bold',
                         color='white' if mat_moy[i,j] < np.mean(mat_moy) else 'black')

    im2 = axes[1].imshow(mat_taux, cmap='RdYlGn', aspect='auto',
                          vmin=0, vmax=100)
    plt.colorbar(im2, ax=axes[1], label='Taux succes (%)')
    axes[1].set_xticks(range(len(SA_LAMBDA_VALUES)))
    axes[1].set_xticklabels([f"lam={l}" for l in SA_LAMBDA_VALUES])
    axes[1].set_yticks(range(len(SA_T0_VALUES)))
    axes[1].set_yticklabels([f"T0={t}" for t in SA_T0_VALUES])
    axes[1].set_title("Taux d'atteinte du minimum global (%)")
    for i in range(len(SA_T0_VALUES)):
        for j in range(len(SA_LAMBDA_VALUES)):
            axes[1].text(j, i, f"{mat_taux[i,j]:.0f}%",
                         ha='center', va='center', fontsize=11, fontweight='bold',
                         color='white' if mat_taux[i,j] < 50 else 'black')

    fig.suptitle("Heatmap : Qualite du recuit simule selon (T0, lambda)",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    p6 = f"{figures_dir}/fig_A3_6_heatmap_parametres.png"
    fig.savefig(p6, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p6}")

    return [p1, p2, p3, p4, p5, p6]


# =============================================================
# 4.  SAUVEGARDE LOGS
# =============================================================

def sauvegarder_logs(res_global, logs_dir="resultats/logs"):
    os.makedirs(logs_dir, exist_ok=True)
    path = f"{logs_dir}/A3_recuit_simule.json"
    data = {}
    for (T0, lam), r in res_global.items():
        cle_str = f"T0={T0}_lam={lam}"
        data[cle_str] = {
            "moyenne"      : float(r["moyenne"]),
            "std"          : float(r["std"]),
            "meilleur"     : float(r["meilleur"]),
            "pire"         : float(r["pire"]),
            "conv_moy"     : float(r["conv_moy"]),
            "acc_moy"      : float(r["acc_moy"]),
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
    print("  VOLET A — A3_recuit_simule.py")
    print(f"  {NB_RUNS} runs x {len(SA_T0_VALUES)} T0 x {len(SA_LAMBDA_VALUES)} lambda")
    print(f"  = {NB_RUNS * len(SA_T0_VALUES) * len(SA_LAMBDA_VALUES)} runs au total")
    print("=" * 65)

    # Reference
    res_enum   = enumeration_exacte()
    min_global = res_enum["minimum_global_exact"]
    sol_opt    = res_enum["solution_optimale"]
    print(f"\n  Reference : minimum global = {min_global:.4f}  s* = {sol_opt}")
    print(f"\n  Parametres :")
    print(f"    T0       = {SA_T0_VALUES}")
    print(f"    lambda   = {SA_LAMBDA_VALUES}")
    print(f"    max_iter = {SA_MAX_ITER}   T_min = {SA_T_MIN}")

    # Experiences
    print(f"\n  Lancement des experiences...")
    t0         = time.time()
    res_global = lancer_experiences(nb_runs=NB_RUNS)
    duree      = time.time() - t0
    print(f"  Termine en {duree:.2f}s")

    # Tableau de resultats
    print("\n" + "-" * 70)
    print(f"  {'T0':>6} {'lambda':>7} {'Moy':>8} {'Std':>7} "
          f"{'Meilleur':>10} {'Taux(%)':>9} {'Iter_moy':>9}")
    print("-" * 70)

    for T0 in SA_T0_VALUES:
        for lam in SA_LAMBDA_VALUES:
            cle  = (T0, lam)
            r    = res_global[cle]
            taux = 100 * sum(1 for c in r["couts_finaux"]
                             if abs(c - min_global) < 1e-6) / NB_RUNS
            iter_moy = np.mean(r["nb_iter"])
            print(f"  {T0:>6.1f} {lam:>7.2f} {r['moyenne']:>8.3f} "
                  f"{r['std']:>7.3f} {r['meilleur']:>10.3f} "
                  f"{taux:>8.1f}% {iter_moy:>8.1f}")

    print("-" * 70)

    # Meilleure combinaison
    meilleure_cle = min(res_global.keys(), key=lambda k: res_global[k]["moyenne"])
    print(f"\n  Meilleure combinaison : T0={meilleure_cle[0]}, lambda={meilleure_cle[1]}")
    print(f"    Cout moyen = {res_global[meilleure_cle]['moyenne']:.4f}")

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
    print("  A3_recuit_simule.py termine avec succes")
    print(f"  Figures dans : {figures_dir}")
    print(f"  Logs dans    : {logs_dir}")
    print("=" * 65)