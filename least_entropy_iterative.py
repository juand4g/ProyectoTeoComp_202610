import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Literal

def get_test_periods(period_i, period_f, n, star_type: Literal["cefeid", "rrlyrae", "binary"]):
    if star_type == "rrlyrae":
        return np.linspace(period_i, period_f, n)

    THRESHOLD = 30.0

    if period_f <= THRESHOLD:
        # Todo el rango es fino: linspace normal
        return np.linspace(period_i, period_f, n)

    if period_i >= THRESHOLD:
        # Todo el rango es "largo": densidad decreciente con ruido
        u = np.sort(np.random.uniform(0, 1, n))
        # Raíz cuadrada concentra puntos al inicio (períodos cortos) y dispersa al final
        u = u ** 1.8  
        return period_i + u * (period_f - period_i)

    # Rango mixto: parte fina hasta THRESHOLD, parte dispersa después
    fine_fraction = (THRESHOLD - period_i) / (period_f - period_i)
    n_fine   = 2*int(n * fine_fraction)
    n_sparse = n - n_fine

    fine_periods = np.linspace(period_i, THRESHOLD, n_fine)

    u = np.sort(np.random.uniform(0, 1, n_sparse))
    u = u ** 1.8  # Decrecimiento no lineal con dispersión
    # Añadir ruido suave para que no sea completamente determinista
    noise = np.random.uniform(-0.5/n_sparse, 0.5/n_sparse, n_sparse)
    u = np.clip(u + noise, 0, 1)
    u = np.sort(u)
    sparse_periods = THRESHOLD + u * (period_f - THRESHOLD)

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

def find_best_period(data,
                     p0,
                     p1,
                     p_num,
                     star_type: Literal["cefeid", "rrlyrae", "binary"],
                     t_parts=4,
                     u_parts=4,
                     plot_entropies=False,
                     eps=0,
                     plot_periods=False,
                     name=None,
                     it=3,
                     N_mins=1,
                     refine_window_fraction=0.05):

    def compute_entropies(periods, t_parts, u_parts):
        entropies = np.zeros_like(periods)
        for i, p in enumerate(periods):
            phases = get_phases(data["t"], data["u"], p, star_type)
            Probabilities, _ = get_probabilities(
                phases, data["u"],
                t_parts=t_parts,
                u_parts=u_parts
            )
            entropies[i] = get_entropy(Probabilities)

        entropies = entropies / entropies.max()
        return entropies


    def get_local_minima(periods, entropies, N_mins, window):
        minima_idx = []

        for i in range(len(entropies)):
            left = max(0, i - window)
            right = min(len(entropies), i + window + 1)

            if entropies[i] == np.min(entropies[left:right]):
                minima_idx.append(i)

        minima_idx = sorted(minima_idx, key=lambda i: entropies[i])

        return minima_idx[:N_mins]


    if eps == 0:
        eps_by_type = {
            "rrlyrae": 0.01,
            "cefeid":  0.01,
            "binary":  0.05,
        }
        eps = eps_by_type[star_type]


    current_p0 = p0
    current_p1 = p1
    best_period = None
    min_entropy = None


    for iteration in range(it):

        testing_periods = get_test_periods(
            current_p0,
            current_p1,
            p_num,
            star_type
        )

        entropies = compute_entropies(
            testing_periods,
            t_parts,
            u_parts
        )

        window = max(2, len(testing_periods) // (5 * N_mins))
        minima_idx = get_local_minima(
            testing_periods,
            entropies,
            N_mins,
            window
        )

        candidate_periods = testing_periods[minima_idx]

        if iteration == it - 1:
            best_idx = np.argmin(entropies[minima_idx])
            best_period = candidate_periods[best_idx]
            min_entropy = entropies[minima_idx][best_idx]
            break

        local_half_width = (
            (current_p1 - current_p0)
            * refine_window_fraction
            / (iteration + 1)
        )

        new_periods = []

        points_per_min = max(5, p_num // len(candidate_periods))

        for p_center in candidate_periods:
            local_p0 = max(p0, p_center - local_half_width)
            local_p1 = min(p1, p_center + local_half_width)

            new_periods.append(
                np.linspace(local_p0, local_p1, points_per_min)
            )

        testing_periods = np.unique(np.concatenate(new_periods))

        current_p0 = testing_periods.min()
        current_p1 = testing_periods.max()

        t_parts += 1
        u_parts += 1


    final_phases = get_phases(
        data["t"],
        data["u"],
        best_period,
        star_type
    )

    phases2 = final_phases + 1


    if plot_entropies:
        plt.plot(testing_periods, entropies)
        plt.xlabel("Período (días)")
        plt.ylabel("Entropía normalizada")
        plt.title("Refinamiento final del periodograma")
        plt.show()


    return best_period, min_entropy, final_phases, phases2










