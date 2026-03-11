import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GA_X_MIN, GA_X_MAX, GA_NB_BITS, RANDOM_SEED
from volet_B.fonction_objectif import (decoder, encoder, fitness,
                                        fonction_objectif, chromosome_aleatoire)

np.random.seed(RANDOM_SEED)



def decodage_detail(chromosome):
    
    #Retourne toutes les etapes du decodage pour une illustration complete.
    
    n       = len(chromosome)
    bits    = list(chromosome)

    # Etape 1 : calcul de l entier
    entier  = 0
    contrib = []
    for i, b in enumerate(bits):
        puissance = 2 ** (n - 1 - i)
        val       = b * puissance
        entier   += val
        contrib.append((i+1, b, puissance, val))

    # Etape 2 : normalisation
    delta   = (GA_X_MAX - GA_X_MIN) / (2**n - 1)
    x       = GA_X_MIN + entier * delta
    f_val   = fonction_objectif(x)

    return {
        "bits"          : bits,
        "contributions" : contrib,
        "entier"        : entier,
        "entier_max"    : 2**n - 1,
        "delta"         : delta,
        "x"             : x,
        "f_x"           : f_val,
        "formule"       : f"{GA_X_MIN} + {entier} × {delta:.6f} = {x:.6f}"
    }


# =============================================================
# 2.  LES 6 CHROMOSOMES ILLUSTRES (sujet demande >= 2)
# =============================================================

CHROMOSOMES_ILLUSTRES = [
    {
        "nom"    : "C1 — Minimum du domaine",
        "bits"   : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "couleur": "#3498db"
    },
    {
        "nom"    : "C2 — Maximum du domaine",
        "bits"   : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "couleur": "#e74c3c"
    },
    {
        "nom"    : "C3 — Milieu du domaine",
        "bits"   : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "couleur": "#2ecc71"
    },
    {
        "nom"    : "C4 — Proche de x* (pi/2)",
        "bits"   : [1, 0, 1, 0, 0, 1, 1, 1, 1, 1],
        "couleur": "#f39c12"
    },
    {
        "nom"    : "C5 — Chromosome aleatoire A",
        "bits"   : [0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
        "couleur": "#9b59b6"
    },
    {
        "nom"    : "C6 — Chromosome aleatoire B",
        "bits"   : [1, 0, 0, 1, 1, 1, 0, 1, 0, 0],
        "couleur": "#1abc9c"
    },
]


# =============================================================
# 3.  GENERATION DES FIGURES
# =============================================================

def generer_figures(figures_dir="resultats/figures"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    os.makedirs(figures_dir, exist_ok=True)
    plt.rcParams.update({
        'font.family'   : 'DejaVu Sans',
        'font.size'     : 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'figure.dpi'    : 150,
        'axes.grid'     : True,
        'grid.alpha'    : 0.3
    })

    x_cont  = np.linspace(GA_X_MIN, GA_X_MAX, 1000)
    f_cont  = fonction_objectif(x_cont)
    x_opt   = x_cont[np.argmax(f_cont)]
    f_opt   = np.max(f_cont)
    delta   = (GA_X_MAX - GA_X_MIN) / (2**GA_NB_BITS - 1)

    # ----------------------------------------------------------
    # FIGURE 1 — Schema du processus de codage / decodage
    # ----------------------------------------------------------
    fig = plt.figure(figsize=(14, 7))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.4)

    ax_main = fig.add_subplot(gs[0, :])   # courbe complete en haut
    ax_zoom = fig.add_subplot(gs[1, 0])   # zoom autour de x*
    ax_prec = fig.add_subplot(gs[1, 1])   # precision vs nb_bits
    ax_err  = fig.add_subplot(gs[1, 2])   # erreur de representation

    # -- Courbe principale avec les 6 chromosomes
    ax_main.plot(x_cont, f_cont, color='#bdc3c7', lw=2, zorder=1)
    ax_main.axhline(0, color='black', lw=0.7, ls='--', alpha=0.4)

    for ch in CHROMOSOMES_ILLUSTRES:
        chrom = np.array(ch["bits"])
        x_v   = decoder(chrom)
        f_v   = fitness(chrom)
        ax_main.scatter([x_v], [f_v], color=ch["couleur"],
                        s=120, zorder=5, marker='o', edgecolors='black', lw=0.8,
                        label=f"{ch['nom'].split('—')[0].strip()}  x={x_v:.3f}")

    ax_main.scatter([x_opt], [f_opt], color='red', s=220, zorder=6,
                    marker='*', label=f"Optimum x*={x_opt:.4f}")
    ax_main.set_xlabel("x")
    ax_main.set_ylabel("f(x) = sin(x)·exp(sin(x))")
    ax_main.set_title("Positions des 6 chromosomes illustres sur f(x)")
    ax_main.legend(fontsize=8, ncol=4)

    # -- Zoom autour de x*
    x_zoom = np.linspace(1.0, 2.2, 500)
    f_zoom = fonction_objectif(x_zoom)
    ax_zoom.plot(x_zoom, f_zoom, color='#3498db', lw=2)
    ch4 = np.array(CHROMOSOMES_ILLUSTRES[3]["bits"])
    x4  = decoder(ch4)
    ax_zoom.scatter([x4], [fitness(ch4)], color='#f39c12', s=150,
                    marker='o', zorder=5, label=f"C4  x={x4:.4f}")
    ax_zoom.scatter([x_opt], [f_opt], color='red', s=180,
                    marker='*', zorder=6, label=f"x*={x_opt:.4f}")
    ax_zoom.set_xlabel("x")
    ax_zoom.set_ylabel("f(x)")
    ax_zoom.set_title("Zoom autour du maximum\n(x ≈ π/2 ≈ 1.5708)")
    ax_zoom.legend(fontsize=8)

    # -- Precision vs nb_bits
    nb_bits_range = range(4, 17)
    precisions    = [(GA_X_MAX - GA_X_MIN) / (2**n - 1) for n in nb_bits_range]
    ax_prec.plot(list(nb_bits_range), precisions, 'o-',
                 color='#2ecc71', lw=2, ms=6)
    ax_prec.axvline(GA_NB_BITS, color='red', ls='--', lw=1.5,
                    label=f'n={GA_NB_BITS} (notre choix)\ndelta={delta:.5f}')
    ax_prec.set_xlabel("Nombre de bits n")
    ax_prec.set_ylabel("Precision delta")
    ax_prec.set_title("Precision du codage\nselon le nombre de bits")
    ax_prec.set_yscale('log')
    ax_prec.legend(fontsize=8)

    # -- Erreur de representation (discretisation)
    x_disc = np.array([decoder(encoder(xi)) for xi in
                       np.linspace(GA_X_MIN, GA_X_MAX, 2**GA_NB_BITS)])
    x_line = np.linspace(GA_X_MIN, GA_X_MAX, 2**GA_NB_BITS)
    erreur = np.abs(x_line - x_disc)
    ax_err.plot(x_line, erreur, color='#e74c3c', lw=1.5, alpha=0.8)
    ax_err.axhline(delta, color='blue', ls='--', lw=1.5,
                   label=f'delta={delta:.5f}')
    ax_err.fill_between(x_line, erreur, alpha=0.2, color='#e74c3c')
    ax_err.set_xlabel("x")
    ax_err.set_ylabel("|x - x_decode|")
    ax_err.set_title("Erreur de representation\npar la grille binaire")
    ax_err.legend(fontsize=8)

    fig.suptitle(f"Codage binaire sur {GA_NB_BITS} bits — Domaine [{GA_X_MIN}, {GA_X_MAX}]  "
                 f"|  Precision = {delta:.6f}",
                 fontsize=13, fontweight='bold')
    p1 = f"{figures_dir}/fig_B1_1_schema_codage.png"
    fig.savefig(p1, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p1}")

    # ----------------------------------------------------------
    # FIGURE 2 — Decodage etape par etape (4 chromosomes)
    # ----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, ch in zip(axes, CHROMOSOMES_ILLUSTRES[:4]):
        chrom  = np.array(ch["bits"])
        detail = decodage_detail(chrom)
        n      = GA_NB_BITS
        color  = ch["couleur"]

        # Affichage des bits avec contributions
        ax.set_xlim(-0.5, n + 0.5)
        ax.set_ylim(-1.0, 5.5)
        ax.axis('off')

        # Titre du chromosome
        ax.text(n/2 - 0.5, 5.2, ch["nom"], ha='center', fontsize=11,
                fontweight='bold', color=color)

        # Cases des bits
        for i, b in enumerate(detail["bits"]):
            bg = color if b == 1 else '#ecf0f1'
            fc = 'white' if b == 1 else '#2c3e50'
            rect = plt.Rectangle((i - 0.4, 3.6), 0.8, 0.8,
                                  color=bg, alpha=0.9, zorder=2)
            ax.add_patch(rect)
            ax.text(i, 4.0, str(b), ha='center', va='center',
                    fontsize=13, fontweight='bold', color=fc, zorder=3)
            ax.text(i, 3.4, f'b{i+1}', ha='center', fontsize=7, color='gray')

        # Contributions de chaque bit
        ax.text(-0.5, 2.8, "Contribution :", fontsize=9, fontweight='bold', color='navy')
        for i, (pos, b, puiss, val) in enumerate(detail["contributions"]):
            col = color if val > 0 else '#bdc3c7'
            ax.text(i, 2.5,
                    f'{b}×2^{n-i}' if i < 4 else f'{b}×{puiss}',
                    ha='center', fontsize=7.5, color=col)
            ax.text(i, 2.1, f'={val}', ha='center', fontsize=7.5,
                    color=col, fontweight='bold')

        # Ligne de somme
        ax.axhline(1.8, xmin=0.02, xmax=0.98, color='gray', lw=0.8)
        ax.text(n/2 - 0.5, 1.5,
                f"Entier = {detail['entier']}  (sur {detail['entier_max']} max)",
                ha='center', fontsize=10, color='navy', fontweight='bold')

        # Formule de normalisation
        ax.text(n/2 - 0.5, 0.8,
                f"x = {GA_X_MIN} + {detail['entier']} × {detail['delta']:.5f}",
                ha='center', fontsize=9.5, color='#2c3e50')
        ax.text(n/2 - 0.5, 0.3,
                f"x = {detail['x']:.6f}",
                ha='center', fontsize=11, color=color, fontweight='bold')
        ax.text(n/2 - 0.5, -0.2,
                f"f(x) = {detail['f_x']:.6f}",
                ha='center', fontsize=10, color='black')

        # Bord decoratif
        rect_bord = plt.Rectangle((-0.5, -0.5), n + 0.5, 6.0,
                                   fill=False, edgecolor=color,
                                   linewidth=2, alpha=0.5)
        ax.add_patch(rect_bord)

    fig.suptitle(f"Decodage etape par etape — Genotype binaire → Phenotype reel\n"
                 f"Formule : x = {GA_X_MIN} + entier × ({GA_X_MAX}−{GA_X_MIN}) / "
                 f"(2^{GA_NB_BITS}−1)",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    p2 = f"{figures_dir}/fig_B1_2_decodage_detaille.png"
    fig.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p2}")

    # ----------------------------------------------------------
    # FIGURE 3 — Grille complete des 1024 points decodables
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    entiers = np.arange(2**GA_NB_BITS)
    x_grille = GA_X_MIN + entiers * delta
    f_grille = fonction_objectif(x_grille)

    # Gauche : tous les points sur la courbe
    axes[0].plot(x_cont, f_cont, color='#bdc3c7', lw=1.5,
                 alpha=0.8, zorder=1, label='f(x) continu')
    sc = axes[0].scatter(x_grille, f_grille, c=f_grille,
                          cmap='RdYlGn', s=12, zorder=2, alpha=0.85)
    plt.colorbar(sc, ax=axes[0], label='f(x)')
    axes[0].scatter([x_opt], [f_opt], color='red', s=200,
                    marker='*', zorder=5, label=f'Max ({x_opt:.4f})')
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].set_title(f"Les {2**GA_NB_BITS} points representables\nsur {GA_NB_BITS} bits")
    axes[0].legend(fontsize=9)

    # Droite : entier vs x decode
    axes[1].plot(entiers, x_grille, color='#3498db', lw=1.5, alpha=0.7)
    axes[1].fill_between(entiers, x_grille, alpha=0.1, color='#3498db')
    axes[1].axhline(GA_X_MIN, color='gray', ls='--', lw=1,
                    label=f'x_min = {GA_X_MIN}')
    axes[1].axhline(GA_X_MAX, color='gray', ls='--', lw=1,
                    label=f'x_max = {GA_X_MAX}')

    # Marquer les 6 chromosomes
    for ch in CHROMOSOMES_ILLUSTRES:
        chrom  = np.array(ch["bits"])
        entier = int("".join(map(str, ch["bits"])), 2)
        x_v    = decoder(chrom)
        axes[1].scatter([entier], [x_v], color=ch["couleur"], s=80,
                        zorder=5, marker='o', edgecolors='black', lw=0.7)

    axes[1].set_xlabel("Entier (valeur binaire decodee)")
    axes[1].set_ylabel("x correspondant")
    axes[1].set_title(f"Relation entier → x\n"
                      f"Pas = {delta:.5f}  (lineaire)")
    axes[1].legend(fontsize=9)

    fig.suptitle(f"Grille de discretisation — {GA_NB_BITS} bits → "
                 f"{2**GA_NB_BITS} valeurs dans [{GA_X_MIN}, {GA_X_MAX}]",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    p3 = f"{figures_dir}/fig_B1_3_grille_decodage.png"
    fig.savefig(p3, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p3}")

    # ----------------------------------------------------------
    # FIGURE 4 — Tableau recapitulatif des 6 chromosomes
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    col_labels = [
        'Chromosome', 'Description',
        'Bits (genotype)', 'Entier',
        'x (phenotype)', 'f(x)', 'Rang'
    ]

    # Calcul des rangs
    all_fitness = [fitness(np.array(ch["bits"])) for ch in CHROMOSOMES_ILLUSTRES]
    rangs       = np.argsort(np.argsort([-f for f in all_fitness])) + 1

    cell_data = []
    for i, ch in enumerate(CHROMOSOMES_ILLUSTRES):
        chrom    = np.array(ch["bits"])
        bits_str = ''.join(map(str, ch["bits"]))
        detail   = decodage_detail(chrom)
        cell_data.append([
            ch["nom"].split("—")[0].strip(),
            ch["nom"].split("—")[1].strip() if "—" in ch["nom"] else "",
            bits_str,
            str(detail["entier"]),
            f"{detail['x']:.6f}",
            f"{detail['f_x']:.6f}",
            f"#{rangs[i]}"
        ])

    table = ax.table(
        cellText  = cell_data,
        colLabels = col_labels,
        cellLoc   = 'center',
        loc       = 'center',
        bbox      = [0.0, 0.05, 1.0, 0.88]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(col_labels))))

    # Entete bleu fonce
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Colorier chaque ligne avec la couleur du chromosome
    for i, ch in enumerate(CHROMOSOMES_ILLUSTRES):
        for j in range(len(col_labels)):
            table[i+1, j].set_facecolor(ch["couleur"] + '30')
        # Rang en gras
        table[i+1, 6].set_text_props(fontweight='bold')
        # Meilleur en vert
        if rangs[i] == 1:
            for j in range(len(col_labels)):
                table[i+1, j].set_facecolor('#d5f5e3')
            table[i+1, 6].set_text_props(
                fontweight='bold', color='#1a8a4a'
            )

    ax.set_title(
        f"Tableau recapitulatif — Codage binaire sur {GA_NB_BITS} bits\n"
        f"Precision : delta = ({GA_X_MAX}-{GA_X_MIN}) / "
        f"(2^{GA_NB_BITS}-1) = {delta:.8f}\n"
        f"(Ligne verte = meilleur individu parmi les 6)",
        fontsize=11, fontweight='bold', pad=15
    )
    fig.tight_layout()
    p4 = f"{figures_dir}/fig_B1_4_tableau_recapitulatif.png"
    fig.savefig(p4, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p4}")

    # ----------------------------------------------------------
    # FIGURE 5 — Effet du nombre de bits sur la precision
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    nb_bits_range  = list(range(4, 17))
    precisions_all = [(GA_X_MAX - GA_X_MIN) / (2**n - 1) for n in nb_bits_range]
    nb_points_all  = [2**n for n in nb_bits_range]

    # Distance au vrai optimum
    distances_opt = []
    for n in nb_bits_range:
        delta_n  = (GA_X_MAX - GA_X_MIN) / (2**n - 1)
        ent_opt  = round((x_opt - GA_X_MIN) / delta_n)
        ent_opt  = int(np.clip(ent_opt, 0, 2**n - 1))
        x_best   = GA_X_MIN + ent_opt * delta_n
        distances_opt.append(abs(x_opt - x_best))

    ax1 = axes[0]
    ax1.semilogy(nb_bits_range, precisions_all, 'o-',
                 color='#3498db', lw=2, ms=7, label='Precision delta')
    ax1.semilogy(nb_bits_range, distances_opt, 's--',
                 color='#e74c3c', lw=2, ms=7, label='Dist. au vrai optimum')
    ax1.axvline(GA_NB_BITS, color='green', ls='--', lw=2,
                label=f'Choix : n={GA_NB_BITS} bits\ndelta={delta:.5f}')
    ax1.set_xlabel("Nombre de bits n")
    ax1.set_ylabel("Precision (echelle log)")
    ax1.set_title("Precision vs nombre de bits\n"
                  "(plus de bits = meilleure precision)")
    ax1.legend(fontsize=9)
    ax1.grid(True, which='both', alpha=0.3)

    ax2 = axes[1]
    ax2.bar(nb_bits_range, [np.log2(npts) for npts in nb_points_all],
            color=['#2ecc71' if n == GA_NB_BITS else '#3498db'
                   for n in nb_bits_range],
            alpha=0.85, edgecolor='black', lw=0.5)
    ax2.set_xlabel("Nombre de bits n")
    ax2.set_ylabel("log2(nombre de points)")
    ax2.set_title(f"Taille de l espace de recherche\n"
                  f"n={GA_NB_BITS} → {2**GA_NB_BITS} chromosomes possibles")
    for i, n in enumerate(nb_bits_range):
        ax2.text(n, np.log2(2**n) + 0.1, f'{2**n}',
                 ha='center', fontsize=7,
                 color='white' if n == GA_NB_BITS else '#2c3e50',
                 fontweight='bold' if n == GA_NB_BITS else 'normal')

    fig.suptitle("Analyse du choix du nombre de bits pour le codage",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    p5 = f"{figures_dir}/fig_B1_5_choix_nb_bits.png"
    fig.savefig(p5, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p5}")

    return [p1, p2, p3, p4, p5]


# =============================================================
# PROGRAMME PRINCIPAL
# =============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  VOLET B — B1_codage.py")
    print(f"  Codage sur {GA_NB_BITS} bits  |  "
          f"Domaine [{GA_X_MIN}, {GA_X_MAX}]")
    print("=" * 65)

    delta = (GA_X_MAX - GA_X_MIN) / (2**GA_NB_BITS - 1)

    # --- B.1.a et B.1.b : formule de decodage
    print(f"\n  B.1.a — Codage sur {GA_NB_BITS} bits")
    print(f"    Nombre de valeurs representables : 2^{GA_NB_BITS} = {2**GA_NB_BITS}")
    print(f"\n  B.1.b — Formule de decodage :")
    print(f"    entier = sum_i b_i * 2^(n-1-i)")
    print(f"    x = {GA_X_MIN} + entier * "
          f"({GA_X_MAX} - {GA_X_MIN}) / (2^{GA_NB_BITS} - 1)")

    # --- B.1.c : precision
    print(f"\n  B.1.c — Precision du codage :")
    print(f"    delta = ({GA_X_MAX} - {GA_X_MIN}) / (2^{GA_NB_BITS} - 1)")
    print(f"    delta = 10 / {2**GA_NB_BITS - 1}")
    print(f"    delta = {delta:.8f}")
    print(f"    => On peut representer une valeur tous les {delta:.6f}")

    # --- B.1.d : illustration sur les 6 chromosomes
    print(f"\n  B.1.d — Illustration sur {len(CHROMOSOMES_ILLUSTRES)} chromosomes :")
    print(f"\n  {'Nom':<30} {'Bits':<14} {'Entier':>8} "
          f"{'x decode':>12} {'f(x)':>10}")
    print("  " + "-" * 78)

    all_fitness = []
    for ch in CHROMOSOMES_ILLUSTRES:
        chrom  = np.array(ch["bits"])
        detail = decodage_detail(chrom)
        all_fitness.append(detail["f_x"])
        print(f"  {ch['nom']:<30} "
              f"{''.join(map(str, ch['bits'])):<14} "
              f"{detail['entier']:>8}  "
              f"{detail['x']:>12.6f}  "
              f"{detail['f_x']:>10.6f}")

    best_idx = np.argmax(all_fitness)
    print(f"\n  => Meilleur individu : {CHROMOSOMES_ILLUSTRES[best_idx]['nom']}")
    print(f"     f(x) = {all_fitness[best_idx]:.6f}")

    # --- Test encodage <-> decodage
    print(f"\n  Test de coherence encode → decode :")
    for x_test in [-5.0, -2.5, 0.0, 1.5708, 5.0]:
        chrom    = encoder(x_test)
        x_retour = decoder(chrom)
        erreur   = abs(x_test - x_retour)
        print(f"    x={x_test:+.4f}  bits={''.join(map(str,chrom))}  "
              f"x_decode={x_retour:+.6f}  erreur={erreur:.6f}")

    # --- Figures
    print("\n  Generation des figures...")
    base        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(base, "resultats", "figures")
    generer_figures(figures_dir=figures_dir)

    print("\n" + "=" * 65)
    print("  B1_codage.py termine avec succes")
    print(f"  Figures dans : {figures_dir}")
    print("=" * 65)