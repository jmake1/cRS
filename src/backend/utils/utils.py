# Copyright 2025 jmake1
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import pandas as pd
from typing import Optional, Dict

EPS = 1e-12

import pandas as pd
import numpy as np
from math import log, ceil

def generate_lambda_table(start_lambda: float = 0.9, end_lambda: float = 0.99, step: float = 0.01, weight_percentil: float = 0.99):
    
    lambdas = []
    n_effs = []
    half_lives = []
    ns = []
    
    if not (0 < start_lambda < 1 and 0 < end_lambda < 1):
        raise ValueError("Lambda must be between 0 and 1")
    if start_lambda >= end_lambda:
        raise ValueError("start_lambda must be < end_lambda")
    if step <= 0:
        raise ValueError("step must be positive")
    
    lambda_val = start_lambda
    while lambda_val <= (end_lambda + EPS):
        
        n_eff = 1 / (1 - lambda_val)
        half_life = log(0.5) / log(lambda_val) if lambda_val > 0 else np.inf
        n = ceil(log(1-weight_percentil) / log(lambda_val)) if lambda_val > 0 else np.inf
        
        lambdas.append(lambda_val)
        n_effs.append(n_eff)
        half_lives.append(half_life)
        ns.append(n)
        
        lambda_val += step
    
    df = pd.DataFrame({
        'lambda': lambdas,
        'n_eff': n_effs,
        'half_life': half_lives,
        f'n_{weight_percentil*100}_percent': ns
    })
    
    return df.set_index('lambda').sort_index(ascending=False)

def ensure_utc_inplace(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"Coluna '{col}' não encontrada.")
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    else:
        if getattr(df[col].dt, "tz", None) is None:
            df[col] = df[col].dt.tz_localize("UTC")
        else:
            df[col] = df[col].dt.tz_convert("UTC")
    return df

def day_floor_utc(d) -> pd.Timestamp:
    t = pd.Timestamp(d)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.normalize()

def day_ceil_exclusive_utc(d) -> pd.Timestamp:
    return day_floor_utc(d) + pd.Timedelta(days=1)

def slice_by_day_range(
    df: pd.DataFrame,
    start_date: Optional[object] = None,
    end_date: Optional[object] = None,
    col: str = "datetime",
) -> pd.DataFrame:
    df_out = df.copy()
    ensure_utc_inplace(df_out, col)
    if start_date is None and end_date is None:
        return df_out

    lo_incl = day_floor_utc(start_date) if start_date is not None else df_out[col].min()
    hi_excl = day_ceil_exclusive_utc(end_date) if end_date is not None else (
        df_out[col].max().normalize() + pd.Timedelta(days=1)
    )
    mask = (df_out[col] >= lo_incl) & (df_out[col] < hi_excl)
    return df_out.loc[mask].copy()


def wide_close(df: pd.DataFrame, values: str = "close") -> pd.DataFrame:
    return df.pivot(index="datetime", columns="symbol", values=values).sort_index()

def log_returns(px: pd.DataFrame) -> pd.DataFrame:
    return np.log(px + EPS).diff().dropna(how="all")

def compute_beta_ewma_uni(r_alt: pd.DataFrame,
                             r_f: pd.DataFrame,
                             lam: float = 0.99,
                             min_obs: int = 100) -> Dict[str, pd.DataFrame]:

    r_alt = r_alt.astype(float)
    r_f = r_f.astype(float).reindex(r_alt.index)
    idx = r_alt.index
    assets = r_alt.columns
    factors = list(r_f.columns)
    T = len(idx)
    n_assets = len(assets)
    k = len(factors)
    alpha = 1.0 - lam

    if k == 0:
        return {}

    mA = r_alt.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
    mF = r_f.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()

    A_center = (r_alt.values - mA.values)    
    F_center = (r_f.values - mF.values)       

    prod_ff = F_center[:, :, None] * F_center[:, None, :]
    prod_ff_flat = prod_ff.reshape(T, k * k)
    cols_ff = [f"{fi}__{fj}" for fi in factors for fj in factors]
    df_prod_ff = pd.DataFrame(prod_ff_flat, index=idx, columns=cols_ff)

    s_ff_flat = df_prod_ff.ewm(alpha=alpha, adjust=False, ignore_na=True, min_periods=min_obs).mean().values
    s_ff = s_ff_flat.reshape(T, k, k)

    raw_if = A_center[:, :, None] * F_center[:, None, :]

    raw_if_flat = raw_if.reshape(T, n_assets * k)

    cols_if = [
        f"{asset}__{factor}"
        for asset in assets
        for factor in factors
    ]

    df_if = pd.DataFrame(raw_if_flat, index=idx, columns=cols_if)

    s_if_flat = df_if.ewm(alpha=alpha, adjust=False, ignore_na=True, min_periods=min_obs).mean().values
    s_if = s_if_flat.reshape(T, n_assets, k)


    betas_arr = np.full((T, n_assets, k), np.nan, dtype=float) 
    for t in range(T):
        S_ff_raw = s_ff[t]
        if np.all(np.isnan(S_ff_raw)):
            continue

        ridge = 1e-3 * np.trace(S_ff_raw) / k
        S_ff_t = S_ff_raw + ridge * np.eye(k)

        S_ff_inv = np.linalg.inv(S_ff_t)

        betas_arr[t] = s_if[t] @ S_ff_inv

    betas_out = {}
    for j, fj in enumerate(factors):
        df_beta = pd.DataFrame(betas_arr[:, :, j], index=idx, columns=assets)
        betas_out[fj] = df_beta

    return betas_out

def residuals_with_beta_freeze_uni(
    r_alt: pd.DataFrame,
    F: pd.DataFrame,
    betas: Dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:

    F = F.reindex(r_alt.index)

    pred = pd.DataFrame(0.0, index=r_alt.index, columns=r_alt.columns)

    for f in F.columns:
        beta_f = (
            betas[f]
            .reindex(index=r_alt.index, columns=r_alt.columns)
            .shift(1)
        )

        pred = pred.add(beta_f.mul(F[f], axis=0))

    res = r_alt - pred
    return res, pred

def winsorize(R: pd.DataFrame, q: float = 0.01) -> pd.DataFrame:
    if q <= 0.0:
        return R
    lo = R.quantile(q, axis=0, method="single")
    hi = R.quantile(1 - q, axis=0, method="single")
    return R.clip(lower=lo, upper=hi, axis=1)