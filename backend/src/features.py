import pandas as pd
import numpy as np

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

def prepare_features(df):
    """Final transformations: log, encoding and dropping noise (Hito 2 & 3)."""
    # Log transformation
    if 'CLAIM_PAID' in df.columns:
        df['CLAIM_PAID_LOG'] = np.log1p(df['CLAIM_PAID'])

    # Drop columns that are no longer useful (Your Hito 2 step)
    cols_to_drop = ['OBJECT_ID', 'INSR_BEGIN', 'INSR_END', 'EFFECTIVE_YR', 'CLAIM_PAID']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # Categorical Conversion (Aseguramos que solo actúe sobre las que existen)
    categorical_cols = ['SEX', 'INSR_TYPE', 'TYPE_VEHICLE', 'MAKE', 'USAGE']
    existing_categorical = [c for c in categorical_cols if c in df.columns]
    
    if 'SEX' in df.columns:
        df['SEX'] = df['SEX'].astype(str)
    
    # One-Hot Encoding (OHE)
    df_model = pd.get_dummies(df, columns=existing_categorical, drop_first=True, dtype=int)
    
    return df_model