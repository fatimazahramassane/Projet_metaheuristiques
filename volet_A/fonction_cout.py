import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_BITS, ALPHA, BETA, RANDOM_SEED

np.random.seed(RANDOM_SEED)


# =============================================================
# 1.  FONCTION DE COÛT
# =============================================================

def fonction_cout(s):
    
    s = np.array(s, dtype=float)
    terme_lineaire    = np.dot(ALPHA, s)
    terme_quadratique = 0.0
    for i in range(N_BITS):
        for j in range(i + 1, N_BITS):
            terme_quadratique += BETA[i][j] * s[i] * s[j]
    return terme_lineaire + terme_quadratique


# =============================================================
# 2.  VOISINAGE — DISTANCE DE HAMMING 1
# =============================================================

def get_voisins(s):

    s = np.array(s, dtype=int)
    voisins = []
    for i in range(len(s)):
        v = s.copy()
        v[i] = 1 - v[i]
        voisins.append(v)
    return voisins


# =============================================================
# 3.  SOLUTION ALEATOIRE
# =============================================================

def solution_aleatoire(n=N_BITS):
    """Genere un vecteur binaire aleatoire de taille n."""
    return np.random.randint(0, 2, size=n)


# =============================================================
# 4.  ENUMERATION EXACTE  (2^10 = 1024 solutions)
# =============================================================

def enumeration_exacte():
    
    #on fait le Parcourt TOUTES les 2^N_BITS solutions.
    #Retourne le minimum global EXACT et tous les minima locaux.

    n = N_BITS
    toutes_solutions, tous_couts = [], []

    for entier in range(2**n):
        s = np.array(list(map(int, format(entier, f'0{n}b'))), dtype=int)
        toutes_solutions.append(s)
        tous_couts.append(fonction_cout(s))

    tous_couts = np.array(tous_couts)
    idx_min    = np.argmin(tous_couts)

    minima_locaux = []
    for i, s in enumerate(toutes_solutions):
        cout_s = tous_couts[i]
        if all(fonction_cout(v) >= cout_s for v in get_voisins(s)):
            minima_locaux.append((s.copy(), cout_s))

    minima_locaux = sorted(minima_locaux, key=lambda x: x[1])

    return {
        "minimum_global_exact" : tous_couts[idx_min],
        "solution_optimale"    : toutes_solutions[idx_min],
        "nb_total_minima"      : len(minima_locaux),
        "tous_minima_locaux"   : minima_locaux,
        "tous_couts"           : tous_couts,
        "toutes_solutions"     : toutes_solutions
    }


# =============================================================
# 5.  GENERATION DES FIGURES
# =============================================================

def generer_figures(res, figures_dir="resultats/figures"):
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

    tous_couts       = res["tous_couts"]
    toutes_solutions = res["toutes_solutions"]
    minima_locaux    = res["tous_minima_locaux"]
    min_global_val   = res["minimum_global_exact"]
    sol_opt          = res["solution_optimale"]
    indices          = np.arange(2**N_BITS)

    # ----------------------------------------------------------
    # FIGURE 1 — Surface de cout complete
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(indices, tous_couts, c='#3498db', s=8, alpha=0.5,
               linewidths=0, label='Toutes les solutions')
    for s, c in minima_locaux:
        entier = int("".join(map(str, s)), 2)
        ax.scatter([entier], [c], c='orange', s=55, zorder=4, marker='^')
    idx_glob = int("".join(map(str, sol_opt)), 2)
    ax.scatter([idx_glob], [min_global_val],
               c='red', s=200, zorder=5, marker='*')
    blue_p   = mpatches.Patch(color='#3498db', label='Toutes les solutions')
    orange_p = mpatches.Patch(color='orange',
                               label=f'Minima locaux ({len(minima_locaux)})')
    red_p    = mpatches.Patch(color='red',
                               label=f'Minimum global = {min_global_val:.2f}')
    ax.legend(handles=[blue_p, orange_p, red_p], fontsize=10)
    ax.set_xlabel("Indice de la solution (entier binaire)")
    ax.set_ylabel("Valeur de f(s)")
    ax.set_title(
        f"Surface de cout complete — {2**N_BITS} solutions enumerees\n"
        f"Minimum global = {min_global_val:.4f}  |  {len(minima_locaux)} minima locaux"
    )
    fig.tight_layout()
    p1 = f"{figures_dir}/fig_A0_1_surface_cout.png"
    fig.savefig(p1, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p1}")

    # ----------------------------------------------------------
    # FIGURE 2 — Histogramme des couts
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    n_h, bins, patches = ax.hist(tous_couts, bins=40,
                                  color='#3498db', edgecolor='white', alpha=0.85)
    bw = bins[1] - bins[0]
    for patch, left in zip(patches, bins[:-1]):
        if left <= min_global_val <= left + bw:
            patch.set_facecolor('#e74c3c')
    ax.axvline(min_global_val, color='red', ls='--', lw=2,
               label=f'Minimum global = {min_global_val:.2f}')
    ax.axvline(np.mean(tous_couts), color='green', ls='--', lw=2,
               label=f'Moyenne = {np.mean(tous_couts):.2f}')
    ax.axvline(np.median(tous_couts), color='purple', ls=':', lw=2,
               label=f'Mediane = {np.median(tous_couts):.2f}')
    ax.set_xlabel("Valeur de f(s)")
    ax.set_ylabel("Nombre de solutions")
    ax.set_title("Distribution des valeurs de cout — 1024 solutions")
    ax.legend(fontsize=10)
    fig.tight_layout()
    p2 = f"{figures_dir}/fig_A0_2_histogramme.png"
    fig.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p2}")

    # ----------------------------------------------------------
    # FIGURE 3 — Coefficients alpha et beta
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1 = axes[0]
    bar_colors = ['#e74c3c' if a < 0 else '#2ecc71' for a in ALPHA]
    bars = ax1.bar(range(N_BITS), ALPHA, color=bar_colors,
                   edgecolor='black', linewidth=0.7)
    ax1.axhline(0, color='black', linewidth=0.8)
    for bar, val in zip(bars, ALPHA):
        offset = 0.2 if val >= 0 else -0.5
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 val + offset, f'{val:.1f}', ha='center', fontsize=9)
    ax1.set_xlabel("Indice i")
    ax1.set_ylabel("Valeur alpha_i")
    ax1.set_title("Coefficients lineaires alpha_i")
    ax1.set_xticks(range(N_BITS))
    gp = mpatches.Patch(color='#2ecc71', label='alpha > 0  penalise b=1')
    rp = mpatches.Patch(color='#e74c3c', label='alpha < 0  recompense b=1')
    ax1.legend(handles=[gp, rp], fontsize=9)

    ax2      = axes[1]
    beta_sym = BETA + BETA.T
    np.fill_diagonal(beta_sym, 0)
    im = ax2.imshow(beta_sym, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)
    plt.colorbar(im, ax=ax2, label='beta_ij')
    ax2.set_xlabel("Indice j")
    ax2.set_ylabel("Indice i")
    ax2.set_title("Interactions quadratiques beta_ij")
    ax2.set_xticks(range(N_BITS))
    ax2.set_yticks(range(N_BITS))
    for i in range(N_BITS):
        for j in range(N_BITS):
            ax2.text(j, i, f'{beta_sym[i,j]:.0f}',
                     ha='center', va='center', fontsize=7,
                     color='white' if abs(beta_sym[i, j]) > 1.5 else 'black')

    fig.suptitle("Parametres de la fonction de cout f(s)",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    p3 = f"{figures_dir}/fig_A0_3_coefficients.png"
    fig.savefig(p3, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p3}")

    # ----------------------------------------------------------
    # FIGURE 4 — Profil des minima locaux
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    vals_ml   = [c for _, c in minima_locaux]
    labels_ml = [f"ML{i+1}" for i in range(len(minima_locaux))]
    bar_c     = ['#e74c3c'] + ['#f39c12'] * (len(minima_locaux) - 1)

    axes[0].bar(labels_ml, vals_ml, color=bar_c, edgecolor='black', lw=0.6)
    axes[0].axhline(min_global_val, color='red', ls='--', lw=1.5,
                    label=f'Min. global = {min_global_val:.2f}')
    for i, v in enumerate(vals_ml):
        axes[0].text(i, v + 0.3, f'{v:.1f}', ha='center', fontsize=8)
    axes[0].set_xlabel("Minima locaux (tries par cout croissant)")
    axes[0].set_ylabel("f(s)")
    axes[0].set_title(f"{len(minima_locaux)} minima locaux detectes")
    axes[0].legend(fontsize=9)
    axes[0].tick_params(axis='x', rotation=45)

    top = minima_locaux[:min(10, len(minima_locaux))]
    mat = np.array([s for s, _ in top])
    im2 = axes[1].imshow(mat, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    axes[1].set_xlabel("Bit b_i")
    axes[1].set_ylabel("Minima locaux")
    axes[1].set_title("Vecteurs binaires des 10 premiers minima locaux")
    axes[1].set_xticks(range(N_BITS))
    axes[1].set_xticklabels([f'b{i+1}' for i in range(N_BITS)], fontsize=9)
    axes[1].set_yticks(range(len(top)))
    axes[1].set_yticklabels(
        [f"ML{i+1}  f={top[i][1]:.1f}" for i in range(len(top))], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            axes[1].text(j, i, str(mat[i, j]),
                         ha='center', va='center', fontsize=8,
                         color='white' if mat[i, j] == 1 else 'black')

    fig.suptitle("Analyse des minima locaux", fontsize=14, fontweight='bold')
    fig.tight_layout()
    p4 = f"{figures_dir}/fig_A0_4_minima_locaux.png"
    fig.savefig(p4, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p4}")

    # ----------------------------------------------------------
    # FIGURE 5 — Voisinage de la solution optimale
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    voisins_opt   = get_voisins(sol_opt)
    couts_voisins = [fonction_cout(v) for v in voisins_opt]
    labels_v      = [f"flip b{i+1}" for i in range(N_BITS)]
    bar_cv        = ['#e74c3c' if c < min_global_val else '#3498db'
                     for c in couts_voisins]

    ax.bar(labels_v, couts_voisins, color=bar_cv, edgecolor='black', lw=0.6)
    ax.axhline(min_global_val, color='red', ls='--', lw=2,
               label=f'f(s*) = {min_global_val:.4f}')
    for i, val in enumerate(couts_voisins):
        ax.text(i, val + 0.3, f'{val:.1f}', ha='center', fontsize=8)
    ax.set_xlabel("Voisin (bit retourne)")
    ax.set_ylabel("f(voisin)")
    ax.set_title(
        f"Voisinage de s* = {list(sol_opt)}\n"
        f"Tous les voisins ont f > f(s*)  =>  s* est un minimum local (et global)"
    )
    ax.legend(fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    p5 = f"{figures_dir}/fig_A0_5_voisinage_optimum.png"
    fig.savefig(p5, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p5}")

    return [p1, p2, p3, p4, p5]


# =============================================================
# PROGRAMME PRINCIPAL
# =============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  VOLET A — fonction_cout.py")
    print("  Modelisation + Enumeration exacte + Figures")
    print("=" * 65)

    res = enumeration_exacte()

    print(f"\n  N = {N_BITS}  |  2^N = {2**N_BITS} solutions")
    print(f"\n  Minimum global EXACT  = {res['minimum_global_exact']:.4f}")
    print(f"  Solution optimale     = {res['solution_optimale']}")
    print(f"\n  Nombre de minima locaux : {res['nb_total_minima']}")
    print(f"\n  Top 5 minima locaux :")
    for i, (s, c) in enumerate(res['tous_minima_locaux'][:5]):
        tag = "  <- GLOBAL" if i == 0 else ""
        print(f"    #{i+1}  f = {c:.4f}   s = {s}{tag}")

    s0 = np.zeros(N_BITS, dtype=int)
    s1 = np.ones(N_BITS, dtype=int)
    print(f"\n  Exemples de calcul :")
    print(f"    f(0...0) = {fonction_cout(s0):.4f}")
    print(f"    f(1...1) = {fonction_cout(s1):.4f}")
    print(f"    f(s*)    = {fonction_cout(res['solution_optimale']):.4f}")

    print("\n  Generation des figures...")
    base        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(base, "resultats", "figures")
    generer_figures(res, figures_dir=figures_dir)

    print("\n" + "=" * 65)
    print("  fonction_cout.py termine avec succes")
    print(f"  Figures dans : {figures_dir}")
    print("=" * 65)