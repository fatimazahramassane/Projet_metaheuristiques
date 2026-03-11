import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GA_NB_BITS, GA_NB_GENERATIONS, GA_POP_SIZE_DEFAULT
from config import GA_PC_DEFAULT, GA_PM_DEFAULT, RANDOM_SEED
from volet_B.fonction_objectif import decoder, fitness, analyser_fonction
from volet_B.B2_operateurs_genetiques import algorithme_genetique, initialiser_population

np.random.seed(RANDOM_SEED)


# =============================================================
# 1.  DEFINITION DES SCHEMAS
# =============================================================
SCHEMAS = {
    "H1": {
        "patron"     : [ 1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        "description": "b1=1  (region droite x>0)",
        "notation"   : "1*********"
    },
    "H2": {
        "patron"     : [ 1,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        "description": "b1=1, b2=0  (region x proche de pi/2)",
        "notation"   : "10********"
    },
    "H3": {
        "patron"     : [ 1, -1, -1, -1, -1, -1, -1, -1, -1,  1],
        "description": "b1=1, b10=1  (sous-espace large)",
        "notation"   : "1********1"
    },
    "H4": {
        "patron"     : [ 1,  0,  1, -1, -1, -1, -1, -1, -1, -1],
        "description": "b1=1, b2=0, b3=1  (region etroite autour de x*)",
        "notation"   : "101*******"
    },
    "H5": {
        "patron"     : [-1, -1, -1, -1, -1, -1, -1, -1, -1,  0],
        "description": "b10=0  (valeurs paires de l entier)",
        "notation"   : "*********0"
    },
    "H6": {
        "patron"     : [ 1,  0,  1,  0, -1, -1, -1, -1, -1, -1],
        "description": "b1=1,b2=0,b3=1,b4=0  (schema court et specifique)",
        "notation"   : "1010******"
    },
}


# =============================================================
# 2.  CALCUL DES PROPRIETES D UN SCHEMA
# =============================================================

def ordre(schema):
    #o(H) = nombre de bits fixes (positions != -1).
    return sum(1 for b in schema["patron"] if b != -1)


def longueur_utile(schema):

    positions_fixes = [i for i, b in enumerate(schema["patron"]) if b != -1]
    if len(positions_fixes) <= 1:
        return 0
    return positions_fixes[-1] - positions_fixes[0]


def prob_destruction_croisement(schema, pc=GA_PC_DEFAULT, n=GA_NB_BITS):
    
    #Probabilite de destruction par croisement bipoints.
    #Approximation : Pd_c >= pc * u(H) / (n - 1)
    
    u = longueur_utile(schema)
    return pc * u / (n - 1)


def prob_destruction_mutation(schema, pm=GA_PM_DEFAULT):
    #Probabilite de destruction par mutation.
    #Pd_m = 1 - (1 - pm)^o(H)  ≈  o(H) * pm  pour pm petit
    
    o = ordre(schema)
    return 1 - (1 - pm) ** o


def prob_survie(schema, pc=GA_PC_DEFAULT, pm=GA_PM_DEFAULT, n=GA_NB_BITS):
    
    #Probabilite de survie totale :
    #Ps = (1 - Pd_c) * (1 - Pd_m)

    pd_c = prob_destruction_croisement(schema, pc, n)
    pd_m = prob_destruction_mutation(schema, pm)
    return (1 - pd_c) * (1 - pd_m)


# =============================================================
# 3.  INDIVIDUS CORRESPONDANT A UN SCHEMA
# =============================================================

def individu_correspond(individu, schema):
    #test
    for i, (bit_ind, bit_sch) in enumerate(zip(individu, schema["patron"])):
        if bit_sch != -1 and bit_ind != bit_sch:
            return False
    return True


def compter_dans_population(population, schema):
    #Compte le nombre d individus correspondant au schema.
    return sum(1 for ind in population if individu_correspond(ind, schema))


def fitness_moyenne_schema(population, fitnesses, schema):
    
    #Calcule la fitness moyenne des individus correspondant au schema.
    #Retourne nan si aucun individu ne correspond.
    
    vals = [fitnesses[i] for i, ind in enumerate(population)
            if individu_correspond(ind, schema)]
    return np.mean(vals) if vals else np.nan


# =============================================================
# 4.  SUIVI DE L EVOLUTION DES SCHEMAS DANS L AG
# =============================================================

def suivre_schemas_dans_ag(taille_pop=GA_POP_SIZE_DEFAULT,
                            nb_gen=GA_NB_GENERATIONS,
                            pc=GA_PC_DEFAULT,
                            pm=GA_PM_DEFAULT):
   
    from volet_B.B2_operateurs_genetiques import (
        initialiser_population, evaluer_population, une_generation
    )

    population   = initialiser_population(taille_pop)
    historiques  = {nom: {"m": [], "f_schema": [], "m_theorique": []}
                    for nom in SCHEMAS}
    f_moy_hist   = []

    for gen in range(nb_gen):
        fitnesses = evaluer_population(population)
        f_moy     = np.mean(fitnesses)
        f_moy_hist.append(f_moy)

        for nom, schema in SCHEMAS.items():
            m_t       = compter_dans_population(population, schema)
            f_schema  = fitness_moyenne_schema(population, fitnesses, schema)

            historiques[nom]["m"].append(m_t)
            historiques[nom]["f_schema"].append(f_schema if not np.isnan(f_schema) else 0)

            # Prediction theoreme des schemas pour la generation suivante
            if m_t > 0 and f_moy > 0 and not np.isnan(f_schema):
                Pc_dest = prob_destruction_croisement(schema, pc)
                Pm_dest = prob_destruction_mutation(schema, pm)
                m_pred  = m_t * (f_schema / f_moy) * (1 - Pc_dest) * (1 - Pm_dest)
            else:
                m_pred = 0

            historiques[nom]["m_theorique"].append(max(0, m_pred))

        # Nouvelle generation
        population = une_generation(population, fitnesses, pc, pm, 'roulette')

    return historiques, f_moy_hist


# =============================================================
# 5.  GENERATION DES FIGURES
# =============================================================

def generer_figures(figures_dir="resultats/figures"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

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

    generations = range(GA_NB_GENERATIONS)
    couleurs    = {
        "H1": '#3498db', "H2": '#2ecc71', "H3": '#e74c3c',
        "H4": '#f39c12', "H5": '#9b59b6', "H6": '#1abc9c'
    }

    # ----------------------------------------------------------
    # FIGURE 1 — Tableau des proprietes des schemas
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    col_labels = [
        'Schema', 'Notation', 'Description',
        'o(H)', 'u(H)', 'Pd_crois.', 'Pd_mut.', 'Ps (survie)'
    ]
    cell_data = []
    for nom, schema in SCHEMAS.items():
        o    = ordre(schema)
        u    = longueur_utile(schema)
        pd_c = prob_destruction_croisement(schema)
        pd_m = prob_destruction_mutation(schema)
        ps   = prob_survie(schema)
        cell_data.append([
            nom,
            schema["notation"],
            schema["description"],
            str(o),
            str(u),
            f"{pd_c:.4f}",
            f"{pd_m:.4f}",
            f"{ps:.4f}"
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

    # Entete
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Colorier lignes alternees
    for i in range(len(SCHEMAS)):
        color = '#eaf4fb' if i % 2 == 0 else '#ffffff'
        for j in range(len(col_labels)):
            table[i+1, j].set_facecolor(color)
        # Couleur de la colonne schema
        table[i+1, 0].set_facecolor(couleurs[list(SCHEMAS.keys())[i]])
        table[i+1, 0].set_text_props(color='white', fontweight='bold')

    ax.set_title(
        f"Proprietes des schemas — n={GA_NB_BITS} bits, "
        f"pc={GA_PC_DEFAULT}, pm={GA_PM_DEFAULT}\n"
        f"o(H) = ordre  |  u(H) = longueur utile  |  "
        f"Pd = prob. destruction  |  Ps = prob. survie",
        fontsize=11, fontweight='bold', pad=15
    )
    fig.tight_layout()
    p1 = f"{figures_dir}/fig_B3_1_tableau_schemas.png"
    fig.savefig(p1, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p1}")

    # ----------------------------------------------------------
    # FIGURE 2 — Visualisation des schemas sur 10 bits
    # ----------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()

    for ax, (nom, schema) in zip(axes, SCHEMAS.items()):
        patron = schema["patron"]
        n      = len(patron)

        # Fond blanc
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, 1.5)

        for i, b in enumerate(patron):
            if b == 1:
                rect = plt.Rectangle((i - 0.45, 0.05), 0.9, 0.9,
                                      color='#2ecc71', alpha=0.85)
                ax.add_patch(rect)
                ax.text(i, 0.5, '1', ha='center', va='center',
                        fontsize=14, fontweight='bold', color='white')
            elif b == 0:
                rect = plt.Rectangle((i - 0.45, 0.05), 0.9, 0.9,
                                      color='#e74c3c', alpha=0.85)
                ax.add_patch(rect)
                ax.text(i, 0.5, '0', ha='center', va='center',
                        fontsize=14, fontweight='bold', color='white')
            else:
                rect = plt.Rectangle((i - 0.45, 0.05), 0.9, 0.9,
                                      color='#bdc3c7', alpha=0.4)
                ax.add_patch(rect)
                ax.text(i, 0.5, '*', ha='center', va='center',
                        fontsize=14, color='#7f8c8d')

        # Marquer la longueur utile
        fixes = [i for i, b in enumerate(patron) if b != -1]
        if len(fixes) >= 2:
            ax.annotate('', xy=(fixes[-1], 1.25), xytext=(fixes[0], 1.25),
                        arrowprops=dict(arrowstyle='<->', color='navy', lw=2))
            ax.text((fixes[0] + fixes[-1]) / 2, 1.38,
                    f"u(H)={fixes[-1]-fixes[0]}",
                    ha='center', fontsize=9, color='navy', fontweight='bold')

        o = ordre(schema)
        u = longueur_utile(schema)
        ax.set_title(f"{nom} : {schema['notation']}\n"
                     f"o={o}  u={u}  Ps={prob_survie(schema):.3f}",
                     fontsize=10, color=couleurs[nom], fontweight='bold')
        ax.set_xticks(range(n))
        ax.set_xticklabels([f'b{i+1}' for i in range(n)], fontsize=8)
        ax.set_yticks([])
        ax.set_xlabel(schema["description"], fontsize=8)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    fig.suptitle("Representation visuelle des 6 schemas\n"
                 "(vert=1 fixe, rouge=0 fixe, gris=joker *)",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    p2 = f"{figures_dir}/fig_B3_2_visualisation_schemas.png"
    fig.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p2}")

    # ----------------------------------------------------------
    # FIGURE 3 — Evolution m(H,t) : nombre d individus par schema
    # ----------------------------------------------------------
    print("    Suivi des schemas dans l AG (peut prendre 10-15s)...")
    historiques, f_moy_hist = suivre_schemas_dans_ag()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, (nom, schema) in zip(axes, SCHEMAS.items()):
        histo  = historiques[nom]
        m_reel = histo["m"]
        m_pred = histo["m_theorique"]
        color  = couleurs[nom]

        ax.plot(generations, m_reel, color=color, lw=2.5,
                label="m(H,t) reel")
        ax.plot(generations, m_pred, color=color, lw=1.5,
                ls='--', alpha=0.7, label="Prediction theorique")
        ax.fill_between(generations, 0, m_reel, alpha=0.1, color=color)

        ax.set_title(f"{nom} : {schema['notation']}\n"
                     f"o={ordre(schema)}, u={longueur_utile(schema)}, "
                     f"Ps={prob_survie(schema):.3f}",
                     fontsize=10, color=color, fontweight='bold')
        ax.set_xlabel("Generation")
        ax.set_ylabel("Nombre d individus")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    fig.suptitle(f"Evolution de m(H,t) — Nombre d individus correspondant "
                 f"a chaque schema\npop={GA_POP_SIZE_DEFAULT}, "
                 f"pc={GA_PC_DEFAULT}, pm={GA_PM_DEFAULT}",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    p3 = f"{figures_dir}/fig_B3_3_evolution_m_schemas.png"
    fig.savefig(p3, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p3}")

    # ----------------------------------------------------------
    # FIGURE 4 — Evolution de la fitness moyenne de chaque schema
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(generations, f_moy_hist, color='black', lw=2,
            ls='--', label='f_moy (population entiere)', zorder=5)

    for nom, schema in SCHEMAS.items():
        f_sch = historiques[nom]["f_schema"]
        ax.plot(generations, f_sch, color=couleurs[nom], lw=2,
                label=f"{nom} : {schema['notation']}")

    analyse = analyser_fonction()
    ax.axhline(analyse["f_max"], color='red', ls=':', lw=1.5,
               label=f'f_max = {analyse["f_max"]:.4f}')

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness moyenne")
    ax.set_title("Evolution de la fitness moyenne de chaque schema\n"
                 "Les schemas a haute fitness croissent par selection")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    p4 = f"{figures_dir}/fig_B3_4_fitness_schemas.png"
    fig.savefig(p4, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p4}")

    # ----------------------------------------------------------
    # FIGURE 5 — Comparaison o(H) vs u(H) vs Ps (graphique bulle)
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    noms   = list(SCHEMAS.keys())
    o_vals = [ordre(SCHEMAS[n])          for n in noms]
    u_vals = [longueur_utile(SCHEMAS[n]) for n in noms]
    ps_vals= [prob_survie(SCHEMAS[n])    for n in noms]
    pdc_v  = [prob_destruction_croisement(SCHEMAS[n]) for n in noms]
    pdm_v  = [prob_destruction_mutation(SCHEMAS[n])   for n in noms]

    # Bulle : taille proportionnelle a Ps
    sc = axes[0].scatter(o_vals, u_vals,
                          s=[ps * 2000 for ps in ps_vals],
                          c=[couleurs[n] for n in noms],
                          alpha=0.75, edgecolors='black', lw=1.2)
    for i, nom in enumerate(noms):
        axes[0].annotate(f"{nom}\nPs={ps_vals[i]:.3f}",
                          (o_vals[i], u_vals[i]),
                          textcoords="offset points", xytext=(8, 4),
                          fontsize=9)
    axes[0].set_xlabel("Ordre o(H)")
    axes[0].set_ylabel("Longueur utile u(H)")
    axes[0].set_title("Schemas : o(H) vs u(H)\n"
                       "(taille bulle ∝ probabilite de survie Ps)")

    # Barres groupees Pd_crois vs Pd_mut
    x     = np.arange(len(noms))
    width = 0.35
    axes[1].bar(x - width/2, pdc_v, width, label='Pd croisement',
                color='#3498db', alpha=0.85, edgecolor='black', lw=0.5)
    axes[1].bar(x + width/2, pdm_v, width, label='Pd mutation',
                color='#e74c3c', alpha=0.85, edgecolor='black', lw=0.5)
    axes[1].plot(x, ps_vals, 'D--', color='#2ecc71', lw=2, ms=8,
                 label='Ps (survie)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(noms)
    axes[1].set_ylabel("Probabilite")
    axes[1].set_title("Probabilites de destruction vs survie")
    axes[1].legend(fontsize=9)

    fig.suptitle("Analyse comparative des proprietes des schemas",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    p5 = f"{figures_dir}/fig_B3_5_analyse_proprietes.png"
    fig.savefig(p5, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p5}")

    # ----------------------------------------------------------
    # FIGURE 6 — Theoreme des schemas : reel vs prediction
    # ----------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, (nom, schema) in zip(axes, SCHEMAS.items()):
        m_reel = np.array(historiques[nom]["m"])
        color  = couleurs[nom]

        # Calcul de la prediction theorique generee pas a pas
        m_pred_seq = [m_reel[0]]
        f_sch_seq  = historiques[nom]["f_schema"]
        for t in range(len(generations) - 1):
            f_s   = f_sch_seq[t] if f_sch_seq[t] != 0 else 0.001
            f_m   = f_moy_hist[t] if f_moy_hist[t] != 0 else 0.001
            Pd_c  = prob_destruction_croisement(schema)
            Pd_m  = prob_destruction_mutation(schema)
            m_t1  = max(0, m_pred_seq[-1] * (f_s / f_m) * (1 - Pd_c) * (1 - Pd_m))
            m_pred_seq.append(m_t1)

        ax.plot(generations, m_reel,
                color=color, lw=2.5, label="Observe m(H,t)")
        ax.plot(generations, m_pred_seq,
                color='black', lw=1.5, ls='--', alpha=0.8,
                label="Theoreme des schemas")
        ax.fill_between(generations, m_reel, m_pred_seq,
                        alpha=0.1, color='gray', label="Ecart")

        o = ordre(schema)
        u = longueur_utile(schema)
        ps = prob_survie(schema)
        ax.set_title(f"{nom} — o={o}, u={u}, Ps={ps:.3f}",
                     fontsize=10, color=color, fontweight='bold')
        ax.set_xlabel("Generation")
        ax.set_ylabel("m(H, t)")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    fig.suptitle("Theoreme des schemas : m(H,t) observe vs predit\n"
                 "m(H,t+1) >= m(H,t) · [f(H)/f_moy] · [1 - Pd_c - Pd_m]",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    p6 = f"{figures_dir}/fig_B3_6_theoreme_schemas.png"
    fig.savefig(p6, bbox_inches='tight')
    plt.close()
    print(f"  OK  {p6}")

    return [p1, p2, p3, p4, p5, p6]


# =============================================================
# PROGRAMME PRINCIPAL
# =============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  VOLET B — B3_schemas.py")
    print(f"  {len(SCHEMAS)} schemas analyses — n={GA_NB_BITS} bits")
    print("=" * 65)

    # --- Tableau des proprietes
    print(f"\n  {'Schema':<6} {'Notation':<14} {'o(H)':>5} {'u(H)':>5} "
          f"{'Pd_crois':>10} {'Pd_mut':>9} {'Ps':>8}")
    print("  " + "-" * 60)
    for nom, schema in SCHEMAS.items():
        o    = ordre(schema)
        u    = longueur_utile(schema)
        pd_c = prob_destruction_croisement(schema)
        pd_m = prob_destruction_mutation(schema)
        ps   = prob_survie(schema)
        print(f"  {nom:<6} {schema['notation']:<14} {o:>5} {u:>5} "
              f"{pd_c:>10.4f} {pd_m:>9.4f} {ps:>8.4f}")

    print(f"\n  Interpretation :")
    print(f"    -> Schemas courts (u petit) et simples (o petit) survivent mieux")
    print(f"    -> H1 (o=1, u=0) : schema le plus robuste, Ps la plus haute")
    print(f"    -> H6 (o=4, u=3) : schema le plus fragile, Ps la plus basse")

    # --- Illustration : individus correspondants
    print(f"\n  Exemples d individus correspondant a chaque schema :")
    np.random.seed(RANDOM_SEED)
    pop_test  = initialiser_population(50)
    for nom, schema in SCHEMAS.items():
        nb = compter_dans_population(pop_test, schema)
        print(f"    {nom} {schema['notation']} : {nb}/50 individus correspondent")

    # --- Figures
    print("\n  Generation des figures...")
    base        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(base, "resultats", "figures")
    generer_figures(figures_dir=figures_dir)

    print("\n" + "=" * 65)
    print("  B3_schemas.py termine avec succes")
    print(f"  Figures dans : {figures_dir}")
    print("=" * 65)