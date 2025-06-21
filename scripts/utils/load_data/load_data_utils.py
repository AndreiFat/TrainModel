import pandas as pd


def load_data(file_path, sep=';'):
    """
    Încarcă un fișier CSV într-un DataFrame Pandas.

    Args:
        file_path (str): Calea către fișierul CSV.
        sep (str): Separatorul de coloane al fișierului CSV. Default este ';'.

    Returns:
        pd.DataFrame: DataFrame-ul încărcat. None dacă fișierul nu este găsit.
    """
    print(f"Încărcarea datelor din '{file_path}'...")
    try:
        df = pd.read_csv(file_path, sep=sep)
        return df
    except FileNotFoundError:
        print(f"Eroare: Fișierul '{file_path}' nu a fost găsit. Asigură-te că calea este corectă.")
        return None
