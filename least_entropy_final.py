import numpy as np
from scipy.signal import find_peaks

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
    """
    Calcula las fases para un periodo de prueba dado.
    """
    threshold = np.percentile(u_data, 2)
    idx       = np.argmin(np.abs(u_data - threshold))
    t0        = t_data[idx]
    return ((t_data - t0) / trial_period) % 1


def get_probabilities_unitcube(phases, u_data, L, K):
    """
    Calcula probabilidades en el cuadrado unitario.
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


def find_best_period(data, p0, p1, p_num, L=7, K=7,
                     alias_eps=0.01,
                     n_candidates=1,
                    peak_prominence=0.03,
                    peak_distance=30):
    """
    Encuentra el mejor período usando mínima entropía en el cuadrado unitario.

    Parámetros
    ----------
    L   : divisiones en fase (horizontal)
    K   : divisiones en magnitud (vertical)
    eps : ventana en días alrededor de cada alias (default 0.01)
    """
    testing_periods = get_test_periods(p0, p1, p_num)

    # ── Antialiasing ──────────────────────────────────────────────────────────
    # ── Por defecto toma múltiplos de medio día y 1/3 de día
    alias_centers = (
        [n * 0.5 for n in range(1, int(p1 / 0.5) + 2)] +
        [n / 3   for n in range(1, int(p1 * 3)  + 2)]
    )
    alias_centers = [a for a in alias_centers if p0 <= a <= p1]

    aliasing_mask = np.zeros(len(testing_periods), dtype=bool)
    for ac in alias_centers:
        aliasing_mask |= np.abs(testing_periods - ac) < alias_eps

    testing_periods = testing_periods[~aliasing_mask]

    # ── Calcular entropías ────────────────────────────────────────────────────
    entropies = np.zeros(len(testing_periods))
    for i, p in enumerate(testing_periods):
        phases        = get_phases(data["t"], data["u"], p)
        Mu            = get_probabilities_unitcube(phases, data["u"], L, K)
        entropies[i]  = get_entropy(Mu)

    # Normalizar entropias
    entropies = entropies / np.log(L * K)

    # ── Buscar mínimos locales ───────────────────────────────
    peaks, properties = find_peaks(
        -entropies,
        prominence=peak_prominence,
        distance=peak_distance
    )

    if len(peaks) == 0:
        peaks = [np.argmin(entropies)]

    candidate_periods = testing_periods[peaks]
    candidate_entropies = entropies[peaks]

    order = np.argsort(candidate_entropies)

    candidate_periods = candidate_periods[order][:n_candidates]
    candidate_entropies = candidate_entropies[order][:n_candidates]

    return candidate_periods, candidate_entropies