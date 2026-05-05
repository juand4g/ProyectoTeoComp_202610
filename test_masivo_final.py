# ── Parámetros a ajustar manualmente ──────────────────────────────────────────
P0_BINARY   = 0.05
P1_BINARY   = 150
P0_RRLYRAE  = 0.15
P1_RRLYRAE  = 1.0
P0_CEFEID   = 0.2
P1_CEFEID   = 150
P_NUM       = 10000
N_SAMPLE    = 5
RANDOM_SEED = 7
N_CANDIDATES = 6
PEAK_PROMINENCE = 0.02
PEAK_DISTANCE   = 30

ACTIVE_TYPES    = ["cefeid", "rrlyrae", "binary"]  # Subconjunto a procesar
PLOT_ENTROPIES  = True   # Graficar periodograma de entropía (recomendado con N_SAMPLE pequeños)
PLOT_LIGHTCURVE = True   # Graficar curvas de luz en fase

# ── Divisiones de grilla por tipo (L=fases, K=magnitudes) ─────────────────────
GRID_PARTS = {
    "binary":  {"L": 7,  "K": 7},
    "rrlyrae": {"L": 7,  "K": 7},
    "cefeid":  {"L": 10, "K": 10},
}
# ──────────────────────────────────────────────────────────────────────────────

# ── CAMBIO: se usa la versión vectorizada del algoritmo ──────────────────────
import least_entropy_vectorized as le
# ── FIN CAMBIO ────────────────────────────────────────────────────────────────

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import testing_candidates as tst
import random
from datetime import datetime

COLLECTIONS = {
    "binary": {
        "dat_dir":  "ogle_collection/binaries_phot",
        "ref_file": "ogle_collection/binaries.txt",
        "p_col":    "P",
        "p0": P0_BINARY,
        "p1": P1_BINARY,
    },
    "rrlyrae": {
        "dat_dir":  "ogle_collection/rrlyrae_phot",
        "ref_file": "ogle_collection/rrlyrae.txt",
        "p_col":    "P_1",
        "p0": P0_RRLYRAE,
        "p1": P1_RRLYRAE,
    },
    "cefeid": {
        "dat_dir":  "ogle_collection/classical_cefeids_phot",
        "ref_file": "ogle_collection/classical_cefeids.txt",
        "p_col":    "P_1",
        "p0": P0_CEFEID,
        "p1": P1_CEFEID,
    },
}


def load_reference(ref_file, p_col):
    df_ref = pd.read_csv(ref_file, skiprows=6, sep=r"\s+")
    ref = {}
    for _, row in df_ref.iterrows():
        ref[str(row["ID"])] = {
            "mag_I":  row["I"],
            "p_true": row[p_col],
        }
    return ref


def plot_periodogram(data, p0, p1, p_num, L, K, candidate_periods, p_true, nombre, star_type):
    """Calcula y grafica el periodograma de entropía usando las funciones de le."""
    testing_periods = le.get_test_periods(p0, p1, p_num)

    t = data["t"].values
    u = data["u"].values

    entropies = np.array([
        le.get_entropy(
            le.get_probabilities_unitcube(le.get_phases(t, u, p), u, L, K)
        )
        for p in testing_periods
    ]) / np.log(L * K)

    _, ax = plt.subplots(figsize=(11, 4))
    ax.plot(testing_periods, entropies, lw=0.6, color="gray", zorder=1)

    for i, p in enumerate(candidate_periods):
        ax.axvline(p, color="steelblue", lw=1.0, ls="--", alpha=0.8,
                   label=f"Cand {i+1}: {p:.5f} d" if i < 5 else "_nolegend_")

    ax.axvline(p_true, color="red", lw=1.3,
               label=f"P_true = {p_true:.5f} d", zorder=2)

    ax.set_xlabel("Período (días)")
    ax.set_ylabel("Entropía normalizada")
    ax.set_title(f"Periodograma — {nombre} ({star_type})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_lightcurve(data, candidate_periods, true_period, nombre, star_type, max_cols=3):
    """Grafica curvas de luz en fase de 0 a 2 (dos ciclos)."""
    n_panels = len(candidate_periods) + 1  # +1 por p_true
    n_cols = min(max_cols, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5*n_cols, 3.5*n_rows),
        sharey=True
    )
    axes = np.atleast_1d(axes).flatten()

    t = data["t"].values
    u = data["u"].values

    def _scatter_two_cycles(ax, period, color, title):
        phases = le.get_phases(t, u, period)
        ph_ext = np.concatenate([phases, phases + 1])
        u_ext  = np.concatenate([u, u])
        ax.scatter(ph_ext, u_ext, s=5, color=color)
        ax.set_xlim(0, 2)
        ax.set_title(title)
        ax.set_xlabel("Fase")

    _scatter_two_cycles(axes[0], true_period, "darkorange", f"P_true = {true_period:.5f} d")
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Magnitud")

    for i, p in enumerate(candidate_periods):
        _scatter_two_cycles(axes[i+1], p, "steelblue", f"Candidate {i+1}: P = {p:.5f} d")

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{nombre} ({star_type})")
    plt.tight_layout()
    plt.show()


# ── Validar ACTIVE_TYPES ──────────────────────────────────────────────────────
invalid = set(ACTIVE_TYPES) - set(COLLECTIONS)
if invalid:
    raise ValueError(f"Tipos no reconocidos en ACTIVE_TYPES: {invalid}")

# ── Bucle principal ───────────────────────────────────────────────────────────
all_results = []

for tipo, cfg in COLLECTIONS.items():
    if tipo not in ACTIVE_TYPES:
        continue

    print(f"\n{'='*50}")
    print(f"Procesando: {tipo.upper()}")
    print(f"{'='*50}")

    try:
        ref = load_reference(cfg["ref_file"], cfg["p_col"])
    except Exception as e:
        print(f"  [ERROR] No se pudo cargar la referencia {cfg['ref_file']}: {e}")
        continue

    archivos = [f for f in os.listdir(cfg["dat_dir"]) if f.endswith(".dat")]

    if N_SAMPLE is not None and N_SAMPLE < len(archivos):
        rng = random.Random(RANDOM_SEED)
        archivos = rng.sample(archivos, N_SAMPLE)
        print(f"  Muestra: {N_SAMPLE} de {len(archivos) + N_SAMPLE} archivos disponibles")
    else:
        print(f"  Procesando todos los archivos: {len(archivos)}")

    L = GRID_PARTS[tipo]["L"]
    K = GRID_PARTS[tipo]["K"]

    for archivo in tqdm(archivos, desc=tipo):
        nombre = archivo.split(".")[0]
        ruta   = os.path.join(cfg["dat_dir"], archivo)

        if nombre not in ref:
            print(f"  [WARN] {nombre} no encontrado en {cfg['ref_file']}, omitiendo.")
            continue

        p_true = ref[nombre]["p_true"]
        mag_I  = ref[nombre]["mag_I"]

        if not (cfg["p0"] <= p_true <= cfg["p1"]):
            print(f"  [SKIP] {nombre} — p_true={p_true:.4f} fuera de [{cfg['p0']}, {cfg['p1']}]")
            continue

        try:
            data = pd.read_csv(ruta, sep=" ", lineterminator="\n", names=("t", "u", "inc_u"))
        except Exception as e:
            print(f"  [ERROR] {nombre} — no se pudo leer el .dat: {e}")
            continue

        try:
            # ── CAMBIO: t y u se pasan por separado, sin depender de etiquetas
            candidate_periods, candidate_entropies = le.find_best_period(
                data["t"].values, data["u"].values,
                p0=cfg["p0"], p1=cfg["p1"],
                p_num=P_NUM,
                L=L, K=K,
                n_candidates=N_CANDIDATES,
                peak_prominence=PEAK_PROMINENCE,
                peak_distance=PEAK_DISTANCE,
            )
            # ── FIN CAMBIO ────────────────────────────────────────────────────
        except Exception as e:
            print(f"  [ERROR] {nombre} — fallo en find_best_period: {e}")
            continue

        if PLOT_ENTROPIES:
            plot_periodogram(data, cfg["p0"], cfg["p1"], P_NUM, L, K,
                             candidate_periods, p_true, nombre, tipo)

        if PLOT_LIGHTCURVE:
            plot_lightcurve(data, candidate_periods, p_true, nombre, tipo)

        error    = tst.period_error(candidate_periods, p_true)
        multiple = tst.which_multiple(candidate_periods, p_true)
        candidate_number = tst.which_candidate(candidate_periods, p_true)

        all_results.append({
            "id":               nombre,
            "tipo":             tipo,
            "mag_I":            mag_I,
            "p_true":           p_true,
            "p_candidates":     candidate_periods.tolist(),
            "best_candidate":   candidate_periods[0],
            "error":            error,
            "multiple":         multiple,
            "success":          error < 0.05,
            "candidate_number": candidate_number,   # 1-based; None si no recuperó
        })

df_results = pd.DataFrame(all_results)
print("\nListo. Resultados en df_results.")
print(df_results.groupby("tipo")["success"].mean().rename("recovery_rate"))

# ── Exportar resultados ───────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("resultados_masivos", exist_ok=True)

df_export = df_results.copy()
if "p_candidates" in df_export.columns:
    df_export["p_candidates"] = df_export["p_candidates"].apply(lambda x: ",".join(map(str, x)))

df_export.to_csv(f"resultados_masivos/resultados_{timestamp}.csv", index=False)

params = {
    "timestamp":       timestamp,
    "P0_BINARY":       P0_BINARY,
    "P1_BINARY":       P1_BINARY,
    "P0_RRLYRAE":      P0_RRLYRAE,
    "P1_RRLYRAE":      P1_RRLYRAE,
    "P0_CEFEID":       P0_CEFEID,
    "P1_CEFEID":       P1_CEFEID,
    "P_NUM":           P_NUM,
    "N_SAMPLE":        N_SAMPLE,
    "RANDOM_SEED":     RANDOM_SEED,
    "ACTIVE_TYPES":    str(ACTIVE_TYPES),
    "PLOT_ENTROPIES":  PLOT_ENTROPIES,
    "PLOT_LIGHTCURVE": PLOT_LIGHTCURVE,
    "N_CANDIDATES":    N_CANDIDATES,
    "PEAK_PROMINENCE": PEAK_PROMINENCE,
    "PEAK_DISTANCE":   PEAK_DISTANCE,
}

pd.DataFrame([params]).to_csv(
    f"resultados_masivos/params_{timestamp}.csv", index=False
)

print(f"Guardado en: resultados_masivos/resultados_{timestamp}.csv")
print(f"Parámetros en: resultados_masivos/params_{timestamp}.csv")

# ── Gráfica de distribución de múltiplos ─────────────────────────────────────
tipos_activos = [t for t in ACTIVE_TYPES if t in df_results["tipo"].unique()]
n_tipos = len(tipos_activos)

if n_tipos > 0:
    fig, axes = plt.subplots(1, n_tipos, figsize=(5 * n_tipos, 5))
    axes = np.atleast_1d(axes)

    for col, tipo in enumerate(tipos_activos):
        df_tipo = df_results[df_results["tipo"] == tipo]
        ax = axes[col]

        mul_counts = df_tipo["multiple"].value_counts(dropna=False).sort_index()
        labels = [str(m) if m is not None else "Fallo" for m in mul_counts.index]
        colors = ["salmon" if l == "Fallo" else "mediumseagreen" for l in labels]

        ax.bar(labels, mul_counts.values, color=colors, edgecolor="white")
        ax.set_xlabel("Múltiplo detectado")
        ax.set_ylabel("Número de estrellas")

        global_rr = df_tipo["success"].mean()
        ax.set_title(f"{tipo.upper()} — Múltiplos\n(Recovery: {global_rr:.2%})")

        for i, val in enumerate(mul_counts.values):
            ax.text(i, val + 0.3, str(val), ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(
        f"resultados_masivos/recovery_bars_{timestamp}.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()
    print(f"Gráficas guardadas en: resultados_masivos/recovery_bars_{timestamp}.png")

df_recovered = df_results[df_results["success"] == True].copy()

if not df_recovered.empty and n_tipos > 0:
    fig, axes = plt.subplots(1, n_tipos, figsize=(5 * n_tipos, 5))
    axes = np.atleast_1d(axes)

    for col, tipo in enumerate(tipos_activos):
        df_tipo_rec = df_recovered[df_recovered["tipo"] == tipo]
        ax = axes[col]

        if df_tipo_rec.empty:
            ax.set_title(f"{tipo.upper()} — Sin recoveries")
            ax.axis("off")
            continue

        all_cands = list(range(1, N_CANDIDATES + 1))
        cand_counts = df_tipo_rec["candidate_number"].value_counts()
        counts = [cand_counts.get(c, 0) for c in all_cands]

        ax.bar(all_cands, counts, color="steelblue", edgecolor="white")

        avg = df_tipo_rec["candidate_number"].mean()
        ax.axvline(avg, color="red", linestyle="--", linewidth=1.5,
                   label=f"Promedio: {avg:.2f}")

        ax.set_xlabel("# Candidato que recuperó la curva")
        ax.set_ylabel("Número de recoveries")
        ax.set_xticks(all_cands)
        ax.set_title(
            f"{tipo.upper()} — Candidato necesario\n"
            f"({len(df_tipo_rec)} recuperadas de {len(df_results[df_results['tipo']==tipo])})"
        )
        ax.legend(fontsize=9)

        for c, v in zip(all_cands, counts):
            if v > 0:
                ax.text(c, v + 0.2, str(v), ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(
        f"resultados_masivos/candidate_number_{timestamp}.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()
    print(f"Gráfica de candidatos guardada en: resultados_masivos/candidate_number_{timestamp}.png")
