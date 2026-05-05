import pandas as pd
import numpy as np

def test(star_id: str, period_exp):
    """
    Devuelve el error absoluto y relativo del periodo experimental hallado.
    """
    test_data = pd.read_csv("resultados_cubillos.csv")
    
    try:
        matches = test_data[test_data["Ident_Estrella"] == star_id]

        if matches.empty:
            raise KeyError(f"Estrella '{star_id}' no encontrada en los datos de prueba.")
        
        if len(matches) > 1:
            raise ValueError(f"Se encontraron {len(matches)} coincidencias para '{star_id}'.")
        
        period_teo = matches["Periodo_ME_dias"].iloc[0]

        if pd.isna(period_teo):
            raise ValueError(f"El período de '{star_id}' es NaN.")
        
        abs_error = np.abs(period_exp-period_teo)
        rel_error = np.abs(period_exp-period_teo)/period_teo

        return period_teo,abs_error,rel_error
    
    except KeyError as e:
        raise KeyError(f"Error buscando estrella: {e}") from e
    except ValueError as e:
        raise ValueError(f"Error en los datos: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error inesperado buscando '{star_id}': {e}") from e 

def period_error(p_found, p_true, multiples=[1/2, 1, 2, 3, 4, 5, 6, 7]):

    p_found = np.atleast_1d(p_found)

    best_error = np.inf

    for p in p_found:
        ratios = [abs(p - m*p_true)/(m*p_true) for m in multiples]
        best_error = min(best_error, min(ratios))

    return best_error


def which_multiple(p_found, p_true, multiples=[1/2, 1, 2, 3, 4, 5, 6, 7]):

    p_found = np.atleast_1d(p_found)

    best_multiple = None
    best_error = np.inf

    for p in p_found:
        errors = {m: abs(p - m*p_true)/(m*p_true) for m in multiples}
        m_best = min(errors, key=errors.get)

        if errors[m_best] < best_error:
            best_error = errors[m_best]
            best_multiple = m_best

    return best_multiple if best_error < 0.05 else None


# ── NUEVO ─────────────────────────────────────────────────────────────────────
def which_candidate(p_found, p_true, multiples=[1/2, 1, 2, 3, 4, 5, 6, 7], threshold=0.05):
    """
    Retorna el índice 1-based del primer candidato (menor número) que pasa el
    test de período. Si ninguno lo pasa, retorna None.
    """
    p_found = np.atleast_1d(p_found)

    for i, p in enumerate(p_found):
        errors = [abs(p - m * p_true) / (m * p_true) for m in multiples]
        if min(errors) < threshold:
            return i + 1  # 1-based

    return None
# ── FIN NUEVO ─────────────────────────────────────────────────────────────────