import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GA_X_MIN, GA_X_MAX, GA_NB_BITS, RANDOM_SEED

np.random.seed(RANDOM_SEED)


# =============================================================
# 1.  FONCTION OBJECTIVE
# =============================================================

def fonction_objectif(x):
    return np.sin(x) * np.exp(np.sin(x))


# =============================================================
# 2.  CODAGE BINAIRE  (B.1 du sujet)
# =============================================================

def decoder(chromosome):
    chromosome = np.array(chromosome, dtype=int)
    n          = len(chromosome)
    entier     = 0
    for i, bit in enumerate(chromosome):
        entier += bit * (2 ** (n - 1 - i))
    x = GA_X_MIN + entier * (GA_X_MAX - GA_X_MIN) / (2**n - 1)
    return x


def encoder(x):
    """
    Encode une valeur reelle x en chromosome binaire de GA_NB_BITS bits.
    (Inverse de decoder — utile pour les tests)

    Parametres
    ----------
    x : float dans [X_MIN, X_MAX]

    Retourne
    --------
    np.array de 0/1 de longueur GA_NB_BITS
    """
    x       = np.clip(x, GA_X_MIN, GA_X_MAX)
    entier  = round((x - GA_X_MIN) * (2**GA_NB_BITS - 1) / (GA_X_MAX - GA_X_MIN))
    entier  = int(np.clip(entier, 0, 2**GA_NB_BITS - 1))
    bits    = list(map(int, format(entier, f'0{GA_NB_BITS}b')))
    return np.array(bits, dtype=int)


def precision_codage():
    
    return (GA_X_MAX - GA_X_MIN) / (2**GA_NB_BITS - 1)


def fitness(chromosome):
    
    x = decoder(chromosome)
    return fonction_objectif(x)


def chromosome_aleatoire(n=GA_NB_BITS):
    #Genere un chromosome binaire aleatoire de longueur n
    return np.random.randint(0, 2, size=n)


# =============================================================
# 3.  ANALYSE DE LA FONCTION
# =============================================================

def analyser_fonction(nb_points=10000):
   
    x_vals = np.linspace(GA_X_MIN, GA_X_MAX, nb_points)
    f_vals = fonction_objectif(x_vals)

    idx_max = np.argmax(f_vals)
    idx_min = np.argmin(f_vals)

    return {
        "x_max"     : x_vals[idx_max],
        "f_max"     : f_vals[idx_max],
        "x_min"     : x_vals[idx_min],
        "f_min"     : f_vals[idx_min],
        "x_vals"    : x_vals,
        "f_vals"    : f_vals,
        "moyenne"   : np.mean(f_vals),
        "std"       : np.std(f_vals)
    }


# =============================================================
# 4.  GENERATION DES FIGURES
# =============================================================

def generer_figures(figures_dir="resultats/figures"):
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

    analyse = analyser_fonction()
    x_vals  = analyse["x_vals"]
    f_vals  = analyse["f_vals"]
    x_max   = analyse["x_max"]
    f_max   = analyse["f_max"]
    prec    = precision_codage()

    # ----------------------------------------------------------
    # FIGURE 1 — Courbe de f(x) avec maximum et zones cles
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(x_vals, f_vals, color='#3498db', lw=2.5, label='f(x) = sin(x)·exp(sin(x))')
    ax.fill_between(x_vals, f_vals, alpha=0.12, color='#3498db')
    ax.axhline(0, color='black', lw=0.8, ls='--')

    # Maximum global
    ax.scatter([x_max], [f_max], color='red', s=150, zorder=5,
               marker='*', label=f'Maximum global ≈ ({x_max:.4f}, {f_max:.4f})')

    # Minimum local
    ax.scatter([analyse["x_min"]], [analyse["f_min"]], color='orange', s=100,
               zorder=5, marker='v', label=f'Minimum ≈ ({analyse["x_min"]:.4f}, {analyse["f_min"]:.4f})')

    # Bornes du domaine
    ax.axvline(GA_X_MIN, color='gray', ls=':', lw=1.5, label=f'x_min = {GA_X_MIN}')
    ax.axvline(GA_X_MAX, color='gray', ls=':', lw=1.5, label=f'x_max = {GA_X_MAX}')

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Fonction objective f(x) = sin(x)·exp(sin(x))  sur [-5, 5]\n"
                 "Objectif de l'AG : MAXIMISER f(x)")
    ax.legend(fontsize=9)
    ax.set_xlim(GA_X_MIN - 0.3, GA_X_MAX + 0.3)
    fig.tight_layout()
    p1 = f"{figures_dir}/fig_B0_1_fonction_objectif.png"
    fig.savefig(p1, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p1}")

    # ----------------------------------------------------------
    # FIGURE 2 — Codage binaire : 4 chromosomes illustres
    # ----------------------------------------------------------
    # Chromosomes choisis par l etudiant (sujet B.1.d)
    chromosomes_ex = [
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),   # entier=0   -> x_min
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),   # entier=1023 -> x_max
        np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1]),   # chromosome intermediaire 1
        np.array([1, 0, 0, 1, 1, 1, 0, 1, 0, 0]),   # chromosome intermediaire 2
    ]
    labels_ex = ["C1 (tout 0)", "C2 (tout 1)", "C3 (mixte A)", "C4 (mixte B)"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    x_disc = np.array([decoder(encoder(xi)) for xi in
                       np.linspace(GA_X_MIN, GA_X_MAX, 2**GA_NB_BITS)])
    f_disc = fonction_objectif(x_disc)

    for ax, chrom, label in zip(axes, chromosomes_ex, labels_ex):
        x_val = decoder(chrom)
        f_val = fonction_objectif(x_val)

        # Fond : courbe continue
        ax.plot(x_vals, f_vals, color='#bdc3c7', lw=1.5, alpha=0.7)

        # Tous les points decodables (grille de precision)
        ax.scatter(x_disc, f_disc, c='#3498db', s=15, alpha=0.4, zorder=2)

        # Ce chromosome
        ax.scatter([x_val], [f_val], color='red', s=200, zorder=5,
                   marker='*', label=f'x = {x_val:.5f}\nf(x) = {f_val:.5f}')
        ax.axvline(x_val, color='red', ls='--', lw=1.2, alpha=0.6)

        # Bits affiches
        bits_str = ''.join(map(str, chrom))
        entier   = int(bits_str, 2)
        ax.set_title(f"{label}\nBits : {bits_str}  (entier={entier})\n"
                     f"x = {GA_X_MIN} + {entier} × {prec:.5f} = {x_val:.5f}")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend(fontsize=9)
        ax.set_xlim(GA_X_MIN - 0.3, GA_X_MAX + 0.3)

    fig.suptitle(f"Codage binaire sur {GA_NB_BITS} bits — Precision = {prec:.6f}\n"
                 f"Formule : x = {GA_X_MIN} + entier × (10 / {2**GA_NB_BITS-1})",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    p2 = f"{figures_dir}/fig_B0_2_codage_chromosomes.png"
    fig.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p2}")

    # ----------------------------------------------------------
    # FIGURE 3 — Precision du codage : erreur de representation
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Gauche : tous les x decodables vs courbe continue
    axes[0].plot(x_vals, f_vals, color='#3498db', lw=2,
                 label='f(x) continue', zorder=1)
    axes[0].scatter(x_disc, f_disc, color='red', s=20, zorder=2,
                    alpha=0.7, label=f'{2**GA_NB_BITS} points decodables')
    axes[0].scatter([x_max], [f_max], color='green', s=150,
                    marker='*', zorder=3, label=f'Vrai maximum ({x_max:.4f})')
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].set_title(f"Grille de {2**GA_NB_BITS} points ({GA_NB_BITS} bits)\n"
                      f"Precision = {prec:.6f}")
    axes[0].legend(fontsize=9)

    # Droite : erreur entre f continue et f discretisee
    x_disc_fine = np.linspace(GA_X_MIN, GA_X_MAX, 2**GA_NB_BITS)
    erreur      = np.abs(fonction_objectif(x_disc_fine) - fonction_objectif(x_disc))
    axes[1].plot(x_disc_fine, erreur, color='#e74c3c', lw=1.5)
    axes[1].fill_between(x_disc_fine, erreur, alpha=0.2, color='#e74c3c')
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("|f(x_continu) - f(x_decode)|")
    axes[1].set_title(f"Erreur de representation due au codage binaire\n"
                      f"Erreur max = {np.max(erreur):.6f}")
    axes[1].axhline(prec, color='blue', ls='--', lw=1.5,
                    label=f'Precision = {prec:.6f}')
    axes[1].legend(fontsize=9)

    fig.suptitle("Analyse de la precision du codage binaire", fontsize=13, fontweight='bold')
    fig.tight_layout()
    p3 = f"{figures_dir}/fig_B0_3_precision_codage.png"
    fig.savefig(p3, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p3}")

    # ----------------------------------------------------------
    # FIGURE 4 — Processus de decodage etape par etape
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis('off')

    # Tableau de decodage pour les 4 chromosomes exemples
    col_labels = ['Chromosome (bits)', 'Entier', 'x decode', 'f(x)', 'Rang fitness']
    cell_data  = []
    fitness_vals = [(fitness(c), i) for i, c in enumerate(chromosomes_ex)]
    fitness_vals_sorted = sorted(fitness_vals, reverse=True)
    rang = {idx: r+1 for r, (_, idx) in enumerate(fitness_vals_sorted)}

    for i, (chrom, label) in enumerate(zip(chromosomes_ex, labels_ex)):
        bits_str = ''.join(map(str, chrom))
        entier   = int(bits_str, 2)
        x_val    = decoder(chrom)
        f_val    = fitness(chrom)
        cell_data.append([
            bits_str,
            str(entier),
            f'{x_val:.5f}',
            f'{f_val:.5f}',
            f'#{rang[i]}'
        ])

    table = ax.table(
        cellText   = cell_data,
        colLabels  = col_labels,
        cellLoc    = 'center',
        loc        = 'center',
        bbox       = [0.0, 0.15, 1.0, 0.75]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    # Colorier l'entete
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Colorier la meilleure ligne en vert
    best_row = rang.copy()
    for i in range(len(chromosomes_ex)):
        if rang[i] == 1:
            for j in range(len(col_labels)):
                table[i+1, j].set_facecolor('#d5f5e3')

    ax.set_title(
        f"Illustration du decodage binaire → reel\n"
        f"Formule : x = {GA_X_MIN} + entier × ({GA_X_MAX}-{GA_X_MIN}) / ({2**GA_NB_BITS}-1)\n"
        f"Precision du codage : {prec:.6f}",
        fontsize=12, fontweight='bold', pad=20
    )

    fig.tight_layout()
    p4 = f"{figures_dir}/fig_B0_4_tableau_decodage.png"
    fig.savefig(p4, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p4}")

    # ----------------------------------------------------------
    # FIGURE 5 — Distribution de la fitness sur la population
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Generer une population aleatoire de 200 individus
    np.random.seed(RANDOM_SEED)
    pop       = [chromosome_aleatoire() for _ in range(200)]
    x_pop     = [decoder(c) for c in pop]
    f_pop     = [fitness(c) for c in pop]

    # Gauche : positions sur la courbe
    axes[0].plot(x_vals, f_vals, color='#bdc3c7', lw=2, alpha=0.8, zorder=1)
    sc = axes[0].scatter(x_pop, f_pop, c=f_pop, cmap='RdYlGn',
                          s=40, zorder=2, alpha=0.8)
    plt.colorbar(sc, ax=axes[0], label='f(x)')
    axes[0].scatter([x_max], [f_max], color='red', s=200, marker='*',
                    zorder=3, label=f'Max global ({x_max:.3f})')
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].set_title("Population aleatoire de 200 individus\n(positions sur la courbe)")
    axes[0].legend(fontsize=9)

    # Droite : histogramme de la fitness
    axes[1].hist(f_pop, bins=25, color='#3498db', edgecolor='white', alpha=0.85)
    axes[1].axvline(np.mean(f_pop), color='green', ls='--', lw=2,
                    label=f'Moyenne = {np.mean(f_pop):.3f}')
    axes[1].axvline(f_max, color='red', ls='--', lw=2,
                    label=f'Max global = {f_max:.3f}')
    axes[1].set_xlabel("Valeur de fitness f(x)")
    axes[1].set_ylabel("Nombre d individus")
    axes[1].set_title("Distribution de la fitness\n(population aleatoire de 200 individus)")
    axes[1].legend(fontsize=9)

    fig.suptitle("Analyse de la distribution de fitness — Population initiale aleatoire",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    p5 = f"{figures_dir}/fig_B0_5_distribution_fitness.png"
    fig.savefig(p5, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p5}")

    return [p1, p2, p3, p4, p5]


# =============================================================
# PROGRAMME PRINCIPAL
# =============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  VOLET B — fonction_objectif.py")
    print("  Codage binaire, decodage, precision, illustrations")
    print("=" * 65)

    # --- Analyse de la fonction
    analyse = analyser_fonction()
    print(f"\n  Fonction : f(x) = sin(x) * exp(sin(x))")
    print(f"  Domaine  : [{GA_X_MIN}, {GA_X_MAX}]")
    print(f"\n  Maximum global approche :")
    print(f"    x* = {analyse['x_max']:.6f}")
    print(f"    f(x*) = {analyse['f_max']:.6f}")
    print(f"\n  Minimum global :")
    print(f"    x  = {analyse['x_min']:.6f}")
    print(f"    f  = {analyse['f_min']:.6f}")

    # --- Precision du codage
    prec = precision_codage()
    print(f"\n  Codage binaire sur {GA_NB_BITS} bits :")
    print(f"    2^{GA_NB_BITS} = {2**GA_NB_BITS} valeurs possibles")
    print(f"    Precision = ({GA_X_MAX} - {GA_X_MIN}) / ({2**GA_NB_BITS} - 1)")
    print(f"    Precision = {prec:.8f}")

    # --- Illustrations B.1.d : 4 chromosomes
    print(f"\n  Illustration sur 4 chromosomes :")
    chromosomes_ex = [
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1]),
        np.array([1, 0, 0, 1, 1, 1, 0, 1, 0, 0]),
    ]
    labels_ex = ["C1 (tout 0)", "C2 (tout 1)", "C3 (mixte A)", "C4 (mixte B)"]
    print(f"\n  {'Chromosome':<30} {'Entier':>8} {'x decode':>12} {'f(x)':>10}")
    print("  " + "-" * 65)
    for chrom, label in zip(chromosomes_ex, labels_ex):
        bits_str = ''.join(map(str, chrom))
        entier   = int(bits_str, 2)
        x_val    = decoder(chrom)
        f_val    = fitness(chrom)
        print(f"  {bits_str}  ({label:<12}) {entier:>6}  {x_val:>12.6f}  {f_val:>10.6f}")

    # --- Test encodage / decodage
    print(f"\n  Test encode -> decode (doit etre proche de x) :")
    for x_test in [-5.0, -2.5, 0.0, 1.5708, 5.0]:
        chrom    = encoder(x_test)
        x_retour = decoder(chrom)
        erreur   = abs(x_test - x_retour)
        print(f"    x={x_test:6.4f}  ->  bits={''.join(map(str,chrom))}  "
              f"->  x_decode={x_retour:.6f}  erreur={erreur:.6f}")

    # --- Figures
    print("\n  Generation des figures...")
    base        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(base, "resultats", "figures")
    generer_figures(figures_dir=figures_dir)

    print("\n" + "=" * 65)
    print("  fonction_objectif.py termine avec succes")
    print(f"  Figures dans : {figures_dir}")
    print("=" * 65)