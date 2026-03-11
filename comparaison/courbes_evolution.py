import numpy as np
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    N_BITS, RANDOM_SEED, NB_RUNS,
    TABU_MAX_ITER, TABU_LIST_SIZES,
    SA_T0_VALUES, SA_LAMBDA_VALUES, SA_MAX_ITER,
    GA_NB_GENERATIONS, GA_POP_SIZE_DEFAULT,
    GA_PC_DEFAULT, GA_PM_DEFAULT,
    GA_POP_SIZES, GA_PC_VALUES, GA_PM_VALUES
)

np.random.seed(RANDOM_SEED)


# =============================================================
# 1.  COLLECTE DES HISTORIQUES D EVOLUTION
# =============================================================

def collecter_historiques_volet_a(nb_runs=NB_RUNS):
    
    
    from volet_A.fonction_cout        import solution_aleatoire, enumeration_exacte
    from volet_A.A1_descente_locale   import descente_locale
    from volet_A.A2_recherche_taboue  import (recherche_taboue_solutions,
                                               recherche_taboue_mouvements)
    from volet_A.A3_recuit_simule     import recuit_simule
    from comparaison.tableaux_resultats import (collecter_recherche_taboue,
                                                  collecter_recuit_simule)

    ref        = enumeration_exacte()
    min_global = ref["minimum_global_exact"]

    # --- Recherche de la meilleure config RT et RS
    print("    Identification meilleure config RT...")
    rt_stats = collecter_recherche_taboue(nb_runs=10)   # rapide
    best_strat = rt_stats["meilleure_strat"]
    best_k     = rt_stats["meilleur_k"]
    fn_tabu    = (recherche_taboue_solutions
                  if best_strat == "solutions"
                  else recherche_taboue_mouvements)

    print("    Identification meilleure config RS...")
    rs_stats = collecter_recuit_simule(nb_runs=10)
    best_T0  = rs_stats["meilleur_T0"]
    best_lam = rs_stats["meilleur_lam"]

    # --- Collecte des historiques
    histo = {
        "DL" : {"label": "Descente locale",
                "evolutions": [], "min_global": min_global},
        "RT" : {"label": f"Taboue (strat={best_strat}, k={best_k})",
                "evolutions": [], "min_global": min_global},
        "RS" : {"label": f"Recuit (T0={best_T0}, lam={best_lam})",
                "evolutions": [], "min_global": min_global},
    }

    np.random.seed(RANDOM_SEED)
    for _ in range(nb_runs):
        s0 = solution_aleatoire()

        res_dl = descente_locale(s0)
        histo["DL"]["evolutions"].append(res_dl["historique_couts"])

        res_rt = fn_tabu(s0, taille_taboue=best_k, max_iter=TABU_MAX_ITER)
        histo["RT"]["evolutions"].append(res_rt["evolution"])

        res_rs = recuit_simule(s0, T0=best_T0, lam=best_lam, max_iter=SA_MAX_ITER)
        histo["RS"]["evolutions"].append(res_rs["evolution_best"])

    return histo


def collecter_historiques_ag(nb_runs=NB_RUNS):
    """
    Collecte les historiques best-fitness et moy-fitness de l AG
    pour nb_runs runs independants (config par defaut).
    """
    from volet_B.B2_operateurs_genetiques import algorithme_genetique

    np.random.seed(RANDOM_SEED)
    hist_best, hist_moy, hist_div = [], [], []

    for _ in range(nb_runs):
        res = algorithme_genetique(
            taille_pop = GA_POP_SIZE_DEFAULT,
            nb_gen     = GA_NB_GENERATIONS,
            pc         = GA_PC_DEFAULT,
            pm         = GA_PM_DEFAULT,
            methode    = "tournoi"
        )
        hist_best.append(res["historique_best"])
        hist_moy.append(res["historique_moy"])
        hist_div.append(res["historique_div"])

    return {
        "hist_best" : np.array(hist_best),   # (nb_runs, nb_gen)
        "hist_moy"  : np.array(hist_moy),
        "hist_div"  : np.array(hist_div),
    }


def collecter_historiques_ag_params():
    """
    Collecte les courbes de convergence pour differents parametres AG.
    Retourne {param_name -> {valeur -> hist_best moyen}}.
    """
    from volet_B.B2_operateurs_genetiques import algorithme_genetique

    resultats = {}

    # A) Effet de pm
    resultats["pm"] = {}
    for pm in GA_PM_VALUES:
        np.random.seed(RANDOM_SEED)
        hists = []
        for _ in range(15):   # 15 runs pour la courbe, assez rapide
            res = algorithme_genetique(pm=pm)
            hists.append(res["historique_best"])
        resultats["pm"][pm] = np.mean(np.array(hists), axis=0)

    # B) Effet de pc
    resultats["pc"] = {}
    for pc in GA_PC_VALUES:
        np.random.seed(RANDOM_SEED)
        hists = []
        for _ in range(15):
            res = algorithme_genetique(pc=pc)
            hists.append(res["historique_best"])
        resultats["pc"][pc] = np.mean(np.array(hists), axis=0)

    # C) Effet de la taille de population
    resultats["pop"] = {}
    for taille in GA_POP_SIZES:
        np.random.seed(RANDOM_SEED)
        hists = []
        for _ in range(15):
            res = algorithme_genetique(taille_pop=taille)
            hists.append(res["historique_best"])
        resultats["pop"][taille] = np.mean(np.array(hists), axis=0)

    return resultats


# =============================================================
# 2.  UTILITAIRE — Courbe moyenne ± std a partir de listes
# =============================================================

def moy_std_courbes(evolutions, max_len=None):
    """
    Prend une liste de listes de longueurs variables.
    Remplit avec la derniere valeur connue (plateau apres convergence).
    Retourne (moyenne, std, max_len_effective).
    """
    if max_len is None:
        max_len = max(len(e) for e in evolutions)
    mat = np.full((len(evolutions), max_len), np.nan)
    for i, e in enumerate(evolutions):
        mat[i, :len(e)] = e
        # Remplace NaN par la derniere valeur (plateau)
        if len(e) < max_len:
            mat[i, len(e):] = e[-1]
    return np.mean(mat, axis=0), np.std(mat, axis=0), max_len


# =============================================================
# 3.  GENERATION DES FIGURES
# =============================================================

def generer_figures(histo_a, histo_ag, histo_ag_params,
                    figures_dir="resultats/figures"):
    """
    Genere toutes les figures de comparaison d evolution.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

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
        "DL" : '#e74c3c',
        "RT" : '#e67e22',
        "RS" : '#2ecc71',
        "AG" : '#3498db',
    }

    min_global = histo_a["DL"]["min_global"]

    # ----------------------------------------------------------
    # FIGURE 1 — Evolution du meilleur cout : DL, RT, RS
    #            Moyenne ± std sur NB_RUNS runs
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Longueur max commune pour les 3 methodes
    max_len = max(
        max(len(e) for e in histo_a["DL"]["evolutions"]),
        max(len(e) for e in histo_a["RT"]["evolutions"]),
        max(len(e) for e in histo_a["RS"]["evolutions"]),
    )

    for cle, col in [("DL", couleurs["DL"]),
                      ("RT", couleurs["RT"]),
                      ("RS", couleurs["RS"])]:
        moy, std, _ = moy_std_courbes(histo_a[cle]["evolutions"], max_len)
        label = histo_a[cle]["label"]
        x = np.arange(len(moy))
        axes[0].plot(x, moy, color=col, lw=2, label=label)
        axes[0].fill_between(x, moy - std, moy + std, color=col, alpha=0.12)
        axes[1].plot(x, moy, color=col, lw=2, label=label)
        axes[1].fill_between(x, moy - std, moy + std, color=col, alpha=0.12)

    axes[0].axhline(min_global, color='black', ls='--', lw=1.5,
                    label=f"Optimum = {min_global:.4f}")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Meilleur cout f(s)")
    axes[0].set_title(f"Evolution complete (0 — {max_len} iterations)\nmoy ± std sur {NB_RUNS} runs")
    axes[0].legend(fontsize=9)

    # Zoom sur les 30 premieres iterations (convergence rapide)
    axes[1].axhline(min_global, color='black', ls='--', lw=1.5,
                    label=f"Optimum = {min_global:.4f}")
    axes[1].set_xlim(0, min(30, max_len))
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Meilleur cout f(s)")
    axes[1].set_title("Zoom : 30 premieres iterations\n(vitesse de convergence initiale)")
    axes[1].legend(fontsize=9)

    fig.suptitle("Volet A — Courbes d'evolution comparatives (moy ± std)",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    p1 = f"{figures_dir}/fig_COMP_5_evolution_volet_a.png"
    fig.savefig(p1, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p1}")

    # ----------------------------------------------------------
    # FIGURE 2 — Seuils de convergence : iterations necessaires
    #            pour atteindre 75%, 90%, 100% de l optimum
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    seuils     = [0.75, 0.90, 1.00]
    labels_s   = ["75% de\nl'optimum", "90% de\nl'optimum", "100%\n(optimum)"]
    x_pos      = np.arange(len(seuils))
    width      = 0.22

    # Pour chaque methode, calcul de l iteration mediane d atteinte du seuil
    offsets = [-width, 0, width]
    for (cle, col), offset in zip(
            [("DL", couleurs["DL"]), ("RT", couleurs["RT"]),
             ("RS", couleurs["RS"])],
            offsets):
        evols   = histo_a[cle]["evolutions"]
        iters_s = []
        for seuil in seuils:
            cible = min_global + seuil * abs(min_global) if min_global != 0 \
                    else seuil
            # Seuil : valeur <= cible (minimisation)
            cible = min_global / seuil if seuil > 0 and min_global < 0 \
                    else min_global * (2 - seuil)
            cible = min_global + (1 - seuil) * abs(min_global - max(e[0] for e in evols))

            med_iters = []
            for ev in evols:
                idx = next((i for i, v in enumerate(ev) if v <= min_global + (1-seuil)*abs(ev[0]-min_global)),
                           len(ev))
                med_iters.append(idx)
            iters_s.append(np.median(med_iters))

        ax.bar(x_pos + offset, iters_s, width=width * 0.9,
               color=col, edgecolor='black', lw=0.6,
               label=histo_a[cle]["label"])
        for xi, val in zip(x_pos + offset, iters_s):
            ax.text(xi, val + 0.3, f"{val:.0f}",
                    ha='center', fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_s)
    ax.set_ylabel("Iteration mediane d'atteinte du seuil")
    ax.set_title("Vitesse de convergence — Iterations medianes\npour atteindre chaque seuil de qualite")
    ax.legend(fontsize=10)
    fig.tight_layout()
    p2 = f"{figures_dir}/fig_COMP_6_vitesse_convergence.png"
    fig.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p2}")

    # ----------------------------------------------------------
    # FIGURE 3 — AG : best fitness et fitness moyenne par generation
    #            (moyenne sur NB_RUNS runs)
    # ----------------------------------------------------------
    from volet_B.fonction_objectif import analyser_fonction
    analyse   = analyser_fonction()
    f_max_ref = analyse["f_max"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    moy_best = np.mean(histo_ag["hist_best"], axis=0)
    std_best = np.std(histo_ag["hist_best"],  axis=0)
    moy_moy  = np.mean(histo_ag["hist_moy"],  axis=0)
    std_moy  = np.std(histo_ag["hist_moy"],   axis=0)
    moy_div  = np.mean(histo_ag["hist_div"],  axis=0)
    gen      = np.arange(GA_NB_GENERATIONS)

    # Gauche : fitness
    ax = axes[0]
    ax.plot(gen, moy_best, color=couleurs["AG"],  lw=2.5,
            label="Meilleure fitness (moy)")
    ax.fill_between(gen, moy_best - std_best, moy_best + std_best,
                    color=couleurs["AG"], alpha=0.15)
    ax.plot(gen, moy_moy, color='#8e44ad', lw=2, ls='--',
            label="Fitness moyenne pop. (moy)")
    ax.fill_between(gen, moy_moy - std_moy, moy_moy + std_moy,
                    color='#8e44ad', alpha=0.10)
    ax.axhline(f_max_ref, color='red', ls=':', lw=1.8,
               label=f"Max reel = {f_max_ref:.4f}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness f(x)")
    ax.set_title(f"Evolution de la fitness AG\n(moy ± std sur {NB_RUNS} runs)")
    ax.legend(fontsize=9)

    # Droite : diversite
    ax2 = axes[1]
    ax2.plot(gen, moy_div, color='#16a085', lw=2.5,
             label="Diversite σ(x) (moy)")
    ax2.fill_between(gen, moy_div - np.std(histo_ag["hist_div"], axis=0),
                     moy_div + np.std(histo_ag["hist_div"], axis=0),
                     color='#16a085', alpha=0.15)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Diversite — ecart-type des x")
    ax2.set_title("Evolution de la diversite\n(convergence de la population)")
    ax2.legend(fontsize=9)

    fig.suptitle("Algorithme Genetique — Convergence et diversite",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    p3 = f"{figures_dir}/fig_COMP_7_ag_evolution.png"
    fig.savefig(p3, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p3}")

    # ----------------------------------------------------------
    # FIGURE 4 — Sensibilite AG : effet pm, pc, taille pop
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    gen = np.arange(GA_NB_GENERATIONS)

    palette_pm  = ['#e74c3c', '#e67e22', '#3498db']
    palette_pc  = ['#8e44ad', '#2ecc71', '#1abc9c']
    palette_pop = ['#e74c3c', '#f39c12', '#27ae60']

    # Effet pm
    ax = axes[0]
    for pm, col in zip(GA_PM_VALUES, palette_pm):
        courbe = histo_ag_params["pm"][pm]
        ax.plot(gen, courbe, color=col, lw=2,
                label=f"pm = {pm}")
    ax.axhline(f_max_ref, color='black', ls='--', lw=1.3,
               label=f"Max reel = {f_max_ref:.3f}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Meilleure fitness (moy)")
    ax.set_title("Effet du taux de mutation pm")
    ax.legend(fontsize=9)

    # Effet pc
    ax = axes[1]
    for pc, col in zip(GA_PC_VALUES, palette_pc):
        courbe = histo_ag_params["pc"][pc]
        ax.plot(gen, courbe, color=col, lw=2,
                label=f"pc = {pc}")
    ax.axhline(f_max_ref, color='black', ls='--', lw=1.3)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Meilleure fitness (moy)")
    ax.set_title("Effet du taux de croisement pc")
    ax.legend(fontsize=9)

    # Effet taille population
    ax = axes[2]
    for taille, col in zip(GA_POP_SIZES, palette_pop):
        courbe = histo_ag_params["pop"][taille]
        ax.plot(gen, courbe, color=col, lw=2,
                label=f"N = {taille}")
    ax.axhline(f_max_ref, color='black', ls='--', lw=1.3)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Meilleure fitness (moy)")
    ax.set_title("Effet de la taille de population")
    ax.legend(fontsize=9)

    fig.suptitle("Analyse de sensibilite — Parametres de l'AG (15 runs/config)",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    p4 = f"{figures_dir}/fig_COMP_8_ag_sensibilite.png"
    fig.savefig(p4, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p4}")

    # ----------------------------------------------------------
    # FIGURE 5 — Convergence comparee : DL, RT, RS, AG (meme echelle)
    #            Normalise : (f(t) - optimum) / (f(0) - optimum)
    #            0 = optimal, 1 = depart
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))

    # Volet A — minimisation : gap(t) / gap(0)
    for cle, col in [("DL", couleurs["DL"]),
                      ("RT", couleurs["RT"]),
                      ("RS", couleurs["RS"])]:
        moy, _, ml = moy_std_courbes(histo_a[cle]["evolutions"])
        gap0 = moy[0] - min_global
        if gap0 < 1e-9:
            gap0 = 1.0
        gap_norm = (moy - min_global) / gap0
        ax.plot(np.linspace(0, 1, len(gap_norm)),
                gap_norm, color=col, lw=2.5,
                label=histo_a[cle]["label"])

    # Volet B — maximisation : (f_max - f(t)) / (f_max - f(0))
    moy_b = np.mean(histo_ag["hist_best"], axis=0)
    gap0_b = f_max_ref - moy_b[0]
    if gap0_b < 1e-9:
        gap0_b = 1.0
    gap_norm_b = (f_max_ref - moy_b) / gap0_b
    ax.plot(np.linspace(0, 1, len(gap_norm_b)),
            gap_norm_b, color=couleurs["AG"], lw=2.5,
            label=f"AG (maximisation)")

    ax.axhline(0, color='black', ls='--', lw=1.5, label="Optimum (0)")
    ax.set_xlabel("Progression (fraction du budget d'iterations)")
    ax.set_ylabel("Gap normalise  (f(t)-opt)/(f(0)-opt)")
    ax.set_title("Convergence normalisee — Comparaison toutes methodes\n"
                 "0 = optimum atteint | 1 = pas d'amelioration")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=10)
    fig.tight_layout()
    p5 = f"{figures_dir}/fig_COMP_9_convergence_normalisee.png"
    fig.savefig(p5, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p5}")

    return [p1, p2, p3, p4, p5]


# =============================================================
# PROGRAMME PRINCIPAL
# =============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  COMPARAISON — courbes_evolution.py")
    print(f"  {NB_RUNS} runs independants | generation des courbes comparatives")
    print("=" * 70)

    t_total = time.time()

    print("\n  [1/3] Collecte historiques Volet A (DL, RT, RS)...")
    histo_a = collecter_historiques_volet_a(nb_runs=NB_RUNS)

    print("\n  [2/3] Collecte historiques AG (config par defaut)...")
    histo_ag = collecter_historiques_ag(nb_runs=NB_RUNS)

    print("\n  [3/3] Collecte historiques AG — analyse de sensibilite...")
    histo_ag_params = collecter_historiques_ag_params()

    print("\n  Generation des figures...")
    base        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(base, "resultats", "figures")
    generer_figures(histo_a, histo_ag, histo_ag_params,
                    figures_dir=figures_dir)

    print("\n" + "=" * 70)
    print(f"  courbes_evolution.py termine en {time.time()-t_total:.1f}s")
    print(f"  Figures dans : {figures_dir}")
    print("=" * 70)