import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Literal

def get_test_periods(period_i, period_f, n, star_type: Literal["cefeid", "rrlyrae", "binary"]):

    if star_type == "rrlyrae":
        return np.linspace(period_i, period_f, n)

    THRESHOLD = 30.0
    FINE_WEIGHT = 2.0  # controla cuánto más densos son los puntos antes del threshold

    if period_f <= THRESHOLD:
        return np.linspace(period_i, period_f, n)

    if period_i >= THRESHOLD:
        u = np.sort(np.random.uniform(0, 1, n))
        u = u ** 1.8
        return period_i + u * (period_f - period_i)

    # rango mixto
    fine_fraction = (THRESHOLD - period_i) / (period_f - period_i)

    # redistribución ponderada (sin cambiar n total)
    weighted_fraction = (fine_fraction * FINE_WEIGHT) / (
        fine_fraction * FINE_WEIGHT + (1 - fine_fraction)
    )

    n_fine = int(round(n * weighted_fraction))
    n_sparse = n - n_fine

    # región fina
    if n_fine > 0:
        fine_periods = np.linspace(
            period_i,
            THRESHOLD,
            n_fine,
            endpoint=(n_sparse == 0)
        )
    else:
        fine_periods = np.array([])

    # región dispersa
    if n_sparse > 0:
        u = np.sort(np.random.uniform(0, 1, n_sparse))
        u = u ** 1.8

        noise = np.random.uniform(-0.5/n_sparse, 0.5/n_sparse, n_sparse)
        u = np.clip(u + noise, 0, 1)
        u = np.sort(u)

        sparse_periods = THRESHOLD + u * (period_f - THRESHOLD)
    else:
        sparse_periods = np.array([])

    return np.concatenate([fine_periods, sparse_periods])

def get_phases(t_data, u_data, trial_period, star_type: Literal["cefeid", "rrlyrae", "binary"]):
    threshold = np.percentile(u_data, 2)  # percentil 2% = los más bajos
    idx = np.argmin(np.abs(u_data - threshold))  # índice del valor más cercano al percentil
    t0 = t_data[idx]

    return ((t_data - t0) / trial_period) % 1

def get_probabilities(phases, u_data, t_parts, u_parts):
    t = np.array(phases)
    u = np.array(u_data)
    
    N_total = len(t)
    
    t_min, t_max = t.min(), t.max()
    u_min, u_max = u.min(), u.max()
    
    t_edges = np.linspace(t_min, t_max, t_parts + 1)
    u_edges = np.linspace(u_min, u_max, u_parts + 1)
    
    H, _, _ = np.histogram2d(t, u, bins=[t_edges, u_edges])
    
    Mu = H / N_total
    
    return Mu, (t_edges, u_edges)

def get_entropy(Mu):
    Mu = Mu[np.nonzero(Mu)]
    S = -Mu*np.log(Mu)
    entropy = S.sum()

    return entropy

def find_best_period(data, p0, p1, p_num, star_type: Literal["cefeid", "rrlyrae", "binary"], t_parts=4, u_parts=4, plot_entropies=False, eps=0, plot_periods=False,name=None):
    testing_periods = get_test_periods(p0, p1, p_num, star_type)

    if plot_periods:
        plt.hist(testing_periods)
        plt.show()

    if eps == 0:
        eps_by_type = {
            "rrlyrae": 0.01,
            "cefeid":  0.01,
            "binary":  0.05,
        }
        eps = eps_by_type[star_type]

    # Centros de aliases dentro del rango
    alias_centers = (
        [n * 0.5 for n in range(1, int(p1 / 0.5) + 2)] +
        [n / 3   for n in range(1, int(p1 * 3)  + 2)]
    )

    YEARLY_EPS    = 20.0 # días — año, medio año y 2/3 de año
    SMALL_YR_EPS  =  7.0  # días — 1/3 y 1/5 de año

    yearly_centers        = []
    half_yearly_centers   = []
    two_third_yr_centers  = []
    third_yr_centers      = []
    fifth_yr_centers      = []

    
    if star_type == "binary":
        yearly_centers       = [n * 365.25        for n in range(1, int(p1 / 365.25)        + 2)]
        half_yearly_centers  = [n * 182.625       for n in range(1, int(p1 / 182.625)       + 2)]
        #two_third_yr_centers = [n * 243.5         for n in range(1, int(p1 / 243.5)         + 2)]
        #third_yr_centers     = [n * (365.25 / 3)  for n in range(1, int(p1 / (365.25 / 3))  + 2)]
        #fifth_yr_centers     = [n * (365.25 / 5)  for n in range(1, int(p1 / (365.25 / 5))  + 2)]
        alias_centers += (yearly_centers + half_yearly_centers + two_third_yr_centers
                        + third_yr_centers + fifth_yr_centers)
    

    alias_centers        = [a for a in alias_centers        if p0 <= a <= p1]
    yearly_centers       = set(a for a in yearly_centers       if p0 <= a <= p1)
    half_yearly_centers  = set(a for a in half_yearly_centers  if p0 <= a <= p1)
    two_third_yr_centers = set(a for a in two_third_yr_centers if p0 <= a <= p1)
    third_yr_centers     = set(a for a in third_yr_centers     if p0 <= a <= p1)
    fifth_yr_centers     = set(a for a in fifth_yr_centers     if p0 <= a <= p1)

    large_yr_set = yearly_centers | half_yearly_centers | two_third_yr_centers
    small_yr_set = third_yr_centers | fifth_yr_centers

    # Eliminar vecindad ±eps días alrededor de cada alias
    aliasing_mask = np.zeros(len(testing_periods), dtype=bool)
    for ac in alias_centers:
        if ac in large_yr_set:
            window = YEARLY_EPS
        elif ac in small_yr_set:
            window = SMALL_YR_EPS
        else:
            window = eps
        aliasing_mask |= np.abs(testing_periods - ac) < window

    testing_periods = testing_periods[~aliasing_mask]

    entropies = np.zeros_like(testing_periods)
    for i, p in tqdm(enumerate(testing_periods)):
        phases = get_phases(data["t"], data["u"], p, star_type)
        Probabilities, _ = get_probabilities(phases, data["u"], t_parts=t_parts, u_parts=u_parts)
        entropies[i] = get_entropy(Probabilities)

    entropies = entropies / entropies.max()

    min_idx = np.argmin(entropies)  # Encuentra el índice del mínimo
    best_period = testing_periods[min_idx]  # Obtiene el período en ese índice
    min_entropy = entropies[min_idx]  # Obtiene el valor mínimo

    final_phases = get_phases(data["t"], data["u"], best_period, star_type=star_type)
    phases2 = final_phases+1

    if plot_entropies:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(testing_periods, entropies)
        ax1.set_title("Periodograma")
        ax1.set_xlabel("Período de prueba (días)")
        ax1.set_ylabel("Entropía")

        ax2.plot(1 / testing_periods, entropies)
        ax2.set_title("Periodograma en frecuencia")
        ax2.set_xlabel("Frecuencia (días⁻¹)")
        ax2.set_ylabel("Entropía")

        plt.tight_layout()
        plt.show()

    return best_period, min_entropy, final_phases, phases2










