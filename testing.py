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
    ratios = [abs(p_found - m * p_true) / (m * p_true) for m in multiples]
    return min(ratios)   

def which_multiple(p_found, p_true, multiples=[1/2, 1, 2, 3, 4, 5, 6, 7]):
    errors = {m: abs(p_found - m * p_true) / (m * p_true) for m in multiples}
    best = min(errors, key=errors.get)
    return best if errors[best] < 0.05 else None