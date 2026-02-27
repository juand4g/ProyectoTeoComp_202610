import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_test_periods(period_i,period_f,n):
    return np.linspace(period_i,period_f,n)

def get_phases(t_data, trial_period):
    t0 = t_data.iloc[0]
    return ((t_data - t0) / trial_period) % 1  

def get_probabilities(phases, u_data, n_grid, m_grid):
    t = np.array(phases)
    u = np.array(u_data)
    
    N_total = len(t)
    
    t_min, t_max = t.min(), t.max()
    u_min, u_max = u.min(), u.max()
    
    t_edges = np.linspace(t_min, t_max, n_grid + 1)
    u_edges = np.linspace(u_min, u_max, m_grid + 1)
    
    # Crear la grilla 2D y contar puntos en cada celda
    # np.histogram2d devuelve:
    #   H: matriz 2D con los conteos
    #   x_edges, y_edges: los bordes de los bins
    H, _, _ = np.histogram2d(t, u, bins=[t_edges, u_edges])
    
    Mu = H / N_total
    
    return Mu, (t_edges, u_edges)

def get_entropy(Mu):
    Mu = Mu[np.nonzero(Mu)]
    S = -Mu*np.log(Mu)
    entropy = S.sum()

    return entropy

def find_best_period(data, p0,p1, p_num, n=4, m=4, plot_entropies=False ):
    testing_periods = get_test_periods(p0,p1,p_num)
    
    entropies = np.zeros_like(testing_periods)

    for i, p in tqdm(enumerate(testing_periods)):
        phases = get_phases(data["t"], p)
        Probabilities,_ = get_probabilities(phases,data["u"],n_grid=n, m_grid=m)
        entropies[i] = get_entropy(Probabilities)

    min_idx = np.argmin(entropies)  # Encuentra el índice del mínimo
    best_period = testing_periods[min_idx]  # Obtiene el período en ese índice
    min_entropy = entropies[min_idx]  # Obtiene el valor mínimo

    final_phases = get_phases(data["t"], best_period)

    if plot_entropies:
        plt.plot(testing_periods,entropies)
        plt.show()

    return best_period, min_entropy, final_phases









