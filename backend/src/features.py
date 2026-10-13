import pandas as pd
import numpy as np

CATEGORICAL_COLS = ['SEX', 'INSR_TYPE', 'TYPE_VEHICLE', 'MAKE', 'USAGE']
HIGH_CARDINALITY_COLS = ['MAKE']
NOISE_COLS = ['OBJECT_ID', 'INSR_BEGIN', 'INSR_END', 'EFFECTIVE_YR', 'CLAIM_PAID']
TARGET_COL = 'CLAIM_PAID'
LOG_TARGET_COL = 'CLAIM_PAID_LOG'


def clean_data(df):
    """Initial cleaning and logical validation (Hito 2)."""
    # 1 & 2. Duplicates
    df = df.drop_duplicates().copy()
    df = df.drop_duplicates(subset=['OBJECT_ID', 'INSR_BEGIN', 'INSR_END'], keep='first').copy()

    # 3. Date conversion
    df['INSR_BEGIN'] = pd.to_datetime(df['INSR_BEGIN'], errors='coerce', format='mixed')
    df['INSR_END'] = pd.to_datetime(df['INSR_END'], errors='coerce', format='mixed')
    df['EFFECTIVE_YR'] = pd.to_numeric(df['EFFECTIVE_YR'], errors='coerce').fillna(0).astype('int64')

    # 4. Consistency and Duration
    df = df.dropna(subset=['INSR_BEGIN', 'INSR_END']).copy()
    df = df[df['INSR_BEGIN'] <= df['INSR_END']].copy()
    df['DURATION'] = (df['INSR_END'] - df['INSR_BEGIN']).dt.days

    # 5. Valid ranges and Fill nulls
    df = df[(df['PROD_YEAR'] >= 1950) & (df['PROD_YEAR'] <= 2026)].copy()
    df = df[df['PREMIUM'] > 0].copy()
    df = df[df['INSURED_VALUE'] > 0].copy()

    cols_to_fill = ['SEATS_NUM', 'CARRYING_CAPACITY', 'CCM_TON']
    df[cols_to_fill] = df[cols_to_fill].fillna(0)
    df = df.dropna(subset=['MAKE']).copy()

    return df


def filter_training_data(df):
    """Specific filtering for Training phase: actual claims and IQR (Hito 2)."""
    # 6. Separate claims (not null)
    df_claims = df[df['CLAIM_PAID'].notnull()].copy()

    # A. Business Rule: Claim vs Insured Value
    df_claims = df_claims[df_claims['CLAIM_PAID'] <= (df_claims['INSURED_VALUE'] * 1.1)].copy()

    # B. Statistical Rule: IQR Outliers
    q1 = df_claims['CLAIM_PAID'].quantile(0.25)
    q3 = df_claims['CLAIM_PAID'].quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + 1.5 * iqr
    df_claims = df_claims[df_claims['CLAIM_PAID'] <= upper_limit].copy()

    return df_claims


def add_target(df):
    """Crea la variable objetivo en escala logarítmica (log1p)."""
    if TARGET_COL in df.columns:
        df = df.copy()
        df[LOG_TARGET_COL] = np.log1p(df[TARGET_COL])
    return df


def select_features(df):
    """Elimina las columnas de ruido (IDs, fechas y el target original)."""
    return df.drop(columns=[c for c in NOISE_COLS if c in df.columns], errors='ignore')


def fit_target_encoding(df, cols=None, target_col=LOG_TARGET_COL, smoothing=20.0):
    """Calcula la codificación por media del target (suavizada) sobre el TRAIN.

    Sustituye categorías de alta cardinalidad (p. ej. MAKE, 746 valores) por la
    media del target, evitando la explosión de columnas del One-Hot Encoding.
    """
    cols = cols if cols is not None else HIGH_CARDINALITY_COLS
    global_mean = float(df[target_col].mean())
    mappings = {}
    for col in cols:
        if col not in df.columns:
            continue
        stats = df.groupby(col)[target_col].agg(['mean', 'count'])
        smooth = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
        mappings[col] = smooth.to_dict()
    return {'cols': list(mappings.keys()), 'mappings': mappings, 'global_mean': global_mean}


def apply_target_encoding(df, encoding):
    """Sustituye las columnas categóricas por su valor codificado."""
    if not encoding:
        return df
    df = df.copy()
    for col in encoding['cols']:
        if col in df.columns:
            df[col] = df[col].map(encoding['mappings'][col]).fillna(encoding['global_mean']).astype(float)
    return df


def one_hot_encode(df, skip_cols=None):
    """One-Hot Encoding de las categóricas que no se hayan codificado por target."""
    skip = set(skip_cols or [])
    df = df.copy()
    existing = [c for c in CATEGORICAL_COLS if c in df.columns and c not in skip]
    for c in existing:
        df[c] = df[c].astype(str)
    return pd.get_dummies(df, columns=existing, drop_first=True, dtype=int)


def prepare_features(df):
    """Compatibilidad: target log + limpieza + One-Hot Encoding completo."""
    return one_hot_encode(select_features(add_target(df)))
