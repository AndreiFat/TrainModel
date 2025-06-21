import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import os


def encode_categorical_features(df, categorical_cols):
    """
    Aplică One-Hot Encoding coloanelor categorice specificate.

    Args:
        df (pd.DataFrame): DataFrame-ul de prelucrat.
        categorical_cols (list): Lista de nume ale coloanelor categorice.

    Returns:
        tuple: (pd.DataFrame: DataFrame-ul cu coloanele categorice codificate One-Hot,
                list: O listă cu numele noilor coloane create prin One-Hot Encoding).
    """
    print(f"Aplicarea One-Hot Encoding pentru coloanele: {categorical_cols}...")

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('necunoscut')
            df[col] = df[col].astype(str).str.lower().str.strip()
        else:
            print(
                f"Avertisment: Coloana '{col}' (categorică) nu a fost găsită în DataFrame. O vom adăuga cu 'necunoscut'.")
            df[col] = 'necunoscut'

    df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, dtype=int)

    encoded_col_names = [col_name for col_name in df_encoded.columns if
                         any(col_name.startswith(f"{c}_") for c in categorical_cols)]

    print(f"One-Hot Encoding aplicat. Au fost create {len(encoded_col_names)} noi coloane.")
    return df_encoded, encoded_col_names


def scale_numerical_features(df, numerical_cols, scaler_type='MinMax', scaler=None, save_path=None):
    """
    Scalează coloanele numerice specificate.
    Dacă un scaler este furnizat (pentru setul de testare), îl utilizează.
    Altfel, antrenează un scaler nou și, opțional, îl salvează.

    Args:
        df (pd.DataFrame): DataFrame-ul de prelucrat.
        numerical_cols (list): Lista de nume ale coloanelor numerice.
        scaler_type (str): Tipul de scaler ('MinMax' sau 'Standard'). Default 'MinMax'.
        scaler (sklearn.preprocessing.BaseScaler, optional): Un scaler pre-antrenat.
                                                              Folosit pentru setul de testare/date noi.
        save_path (str, optional): Calea pentru a salva scaler-ul antrenat. Dacă None, nu se salvează.

    Returns:
        tuple: (pd.DataFrame: DataFrame-ul cu coloanele numerice scalate,
                sklearn.preprocessing.BaseScaler: Scaler-ul antrenat/folosit).
    """
    print(f"Scalarea coloanelor numerice ({scaler_type} Scaling)...")

    for col in numerical_cols:
        if col not in df.columns:
            print(
                f"Avertisment: Coloana numerică '{col}' nu a fost găsită în DataFrame. Aceasta va fi omisă de la scalare.")
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

    existing_numerical_cols = [col for col in numerical_cols if col in df.columns]

    if scaler is None:
        if scaler_type == 'MinMax':
            scaler = MinMaxScaler()
        elif scaler_type == 'Standard':
            scaler = StandardScaler()
        else:
            raise ValueError("scaler_type trebuie să fie 'MinMax' sau 'Standard'.")

        df[existing_numerical_cols] = scaler.fit_transform(df[existing_numerical_cols])
        print(f"Un nou scaler de tip '{scaler_type}' a fost antrenat.")
        if save_path:
            joblib.dump(scaler, save_path)
            print(f"Scaler-ul a fost salvat în '{save_path}'.")
    else:
        df[existing_numerical_cols] = scaler.transform(df[existing_numerical_cols])
        print(f"Scaler-ul existent de tip '{scaler_type}' a fost utilizat pentru transformare.")

    return df, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Împarte datele în seturi de antrenament și testare.

    Args:
        X (pd.DataFrame): Caracteristicile (features).
        y (pd.DataFrame sau pd.Series): Variabilele țintă (labels).
        test_size (float): Proporția setului de testare.
        random_state (int): Seed pentru reproductibilitate.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"Împărțirea datelor în seturi de antrenament ({100 * (1 - test_size)}%) și testare ({100 * test_size}%)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(
        f"Dimensiuni seturi: X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")
    return X_train, X_test, y_train, y_test


def save_dataframes(dfs_dict, base_path):
    """
    Salvează un dicționar de DataFrame-uri în fișiere CSV.

    Args:
        dfs_dict (dict): Un dicționar unde cheile sunt numele fișierelor (fără extensie)
                         și valorile sunt DataFrame-uri.
        base_path (str): Calea directorului unde vor fi salvate fișierele.
    """
    print(f"\nSalvarea seturilor de date în '{base_path}'...")
    os.makedirs(base_path, exist_ok=True)
    for name, df_to_save in dfs_dict.items():
        file_path = os.path.join(base_path, f"{name}.csv")
        df_to_save.to_csv(file_path, index=False, sep=";")
        print(f"  - '{file_path}' salvat.")

