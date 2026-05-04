import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_test_periods(period_i, period_f, n):
    """
    Grilla fina (linspace) de 0 a 30 días, dispersa y aleatoria después.
    No requiere star_type.
    """
    THRESHOLD  = 30.0
    FINE_WEIGHT = 2.0

    if period_f <= THRESHOLD:
        return np.linspace(period_i, period_f, n)

    if period_i >= THRESHOLD:
        u = np.sort(np.random.uniform(0, 1, n))
        u = u ** 1.8
        return period_i + u * (period_f - period_i)

    # Rango mixto
    fine_fraction     = (THRESHOLD - period_i) / (period_f - period_i)
    weighted_fraction = (fine_fraction * FINE_WEIGHT) / (
        fine_fraction * FINE_WEIGHT + (1 - fine_fraction)
    )

    n_fine   = int(round(n * weighted_fraction))
    n_sparse = n - n_fine

    fine_periods = np.linspace(period_i, THRESHOLD, n_fine,
                               endpoint=(n_sparse == 0)) if n_fine > 0 else np.array([])

    if n_sparse > 0:
        u     = np.sort(np.random.uniform(0, 1, n_sparse))
        u     = u ** 1.8
        noise = np.random.uniform(-0.5 / n_sparse, 0.5 / n_sparse, n_sparse)
        u     = np.sort(np.clip(u + noise, 0, 1))
        sparse_periods = THRESHOLD + u * (period_f - THRESHOLD)
    else:
        sparse_periods = np.array([])

    return np.concatenate([fine_periods, sparse_periods])


def get_phases(t_data, u_data, trial_period):
    threshold = np.percentile(u_data, 2)
    idx       = np.argmin(np.abs(u_data - threshold))
    t0        = t_data[idx]
    return ((t_data - t0) / trial_period) % 1


def get_probabilities_unitcube(phases, u_data, L, K):
    """
    Calcula probabilidades en el cuadrado unitario (Cincotta).
    Fase y magnitud normalizadas entre 0 y 1.
    L: divisiones en eje de fase (horizontal)
    K: divisiones en eje de magnitud (vertical)
    """
    phi  = np.array(phases)                              # ya en [0, 1)
    u    = np.array(u_data, dtype=float)

    # Normalizar magnitud a [0, 1]
    u_min, u_max = u.min(), u.max()
    if u_max > u_min:
        u_norm = (u - u_min) / (u_max - u_min)
    else:
        u_norm = np.zeros_like(u)

    N_total = len(phi)
    H, _, _ = np.histogram2d(phi, u_norm,
                              bins=[L, K],
                              range=[[0, 1], [0, 1]])
    Mu = H / N_total
    return Mu


def get_entropy(Mu):
    Mu_nz = Mu[Mu > 0]
    return (-Mu_nz * np.log(Mu_nz)).sum()


def entropy_thresholds(L, K):
    """
    Umbrales teóricos de Cincotta para entropía normalizada S/ln(LK):

      S_alias ≈ ln(2K) / ln(L*K)   → alias de muestreo (1 día y múltiplos)
      S_real  ≈ ln(L)   / ln(L*K)   → período real (curva suave)

    Devuelve entropías normalizadas.
    """
    ln_LK   = np.log(L * K)
    S_alias = np.log(2 * K)/ln_LK          # ≈ ln(2K)
    S_real  = np.log(L)/ln_LK              # ≈ ln(L)
    return S_real, S_alias

def find_best_period(data, p0, p1, p_num, L=12, K=7,
                     eps=0.01, plot_entropies=False, plot_periods=False,
                     name=None):
    """
    Encuentra el mejor período usando mínima entropía en el cuadrado unitario.

    Parámetros
    ----------
    L   : divisiones en fase (horizontal)
    K   : divisiones en magnitud (vertical)
    eps : ventana en días alrededor de cada alias (default 0.01)
    """
    testing_periods = get_test_periods(p0, p1, p_num)

    if plot_periods:
        plt.hist(testing_periods, bins=50)
        plt.xlabel("Período (días)")
        plt.title("Distribución de períodos de prueba")
        plt.show()

    # ── Antialiasing ──────────────────────────────────────────────────────────
    alias_centers = (
        [n * 0.5 for n in range(1, int(p1 / 0.5) + 2)] +
        [n / 3   for n in range(1, int(p1 * 3)  + 2)]
    )
    alias_centers = [a for a in alias_centers if p0 <= a <= p1]

    aliasing_mask = np.zeros(len(testing_periods), dtype=bool)
    for ac in alias_centers:
        aliasing_mask |= np.abs(testing_periods - ac) < eps

    testing_periods = testing_periods[~aliasing_mask]

    # ── Calcular entropías ────────────────────────────────────────────────────
    entropies = np.zeros(len(testing_periods))
    for i, p in tqdm(enumerate(testing_periods), total=len(testing_periods)):
        phases        = get_phases(data["t"], data["u"], p)
        Mu            = get_probabilities_unitcube(phases, data["u"], L, K)
        entropies[i]  = get_entropy(Mu)

    # Normalizar entropias
    entropies = entropies / np.log(L * K)

    # Umbrales teóricos normalizados
    S_real, S_alias= entropy_thresholds(L, K)

    # ── Mejor período ─────────────────────────────────────────────────────────
    min_idx     = np.argmin(entropies)
    best_period = testing_periods[min_idx]
    min_entropy = entropies[min_idx]

    final_phases = get_phases(data["t"], data["u"], best_period)
    phases2      = final_phases + 1

    # ── Gráficas ──────────────────────────────────────────────────────────────
    if plot_entropies:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        title_base = f"{name} — P = {best_period:.5f} d" if name else f"P = {best_period:.5f} d"

        for ax, x, xlabel in [
            (ax1, testing_periods,       "Período de prueba (días)"),
            (ax2, 1 / testing_periods,   "Frecuencia (días⁻¹)"),
        ]:
            ax.plot(x, entropies, lw=0.8, color="steelblue")
            ax.axhline(S_real,  color="green",  linestyle="--", lw=1.2,
                       label=f"S_real ≈ ln(L) = {S_real:.3f}")
            ax.axhline(S_alias, color="red",    linestyle="--", lw=1.2,
                       label=f"S_alias ≈ ln(2K) = {S_alias:.3f}")
            ax.axvline(x[min_idx] if ax is ax1 else 1 / best_period,
                       color="orange", linestyle=":", lw=1.5,
                       label=f"Mejor período")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Entropía (nats)")
            ax.legend(fontsize=8)

        ax1.set_title(f"Periodograma — {title_base}")
        ax2.set_title("Periodograma en frecuencia")
        plt.tight_layout()
        plt.show()

    return best_period, min_entropy, final_phases, phases2, testing_periods, entropies