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

import pandas as pd
import numpy as np
from typing import Dict
import logging
from backend.models.hmm import RSHMM
from backend.utils import (
    ensure_utc_inplace,
    slice_by_day_range,
    wide_close,
    log_returns,
    compute_beta_ewma_uni,
    residuals_with_beta_freeze_uni,
    winsorize, 
)
import itertools

class RSEW:
    
    def __init__(self, beta_lambda: float, factors: list, start_date: object, end_date: object, min_obs: int = None, winsor: float = 0) -> None:
        self.beta_lambda = beta_lambda
        self.factors = factors
        self.start_date = pd.Timestamp(start_date).tz_localize('UTC')
        self.end_date = pd.Timestamp(end_date).tz_localize('UTC')
        self.min_obs = min_obs
        self.winsor = winsor
        
        self.hmm_model = RSHMM()
        
        self.logger = logging.getLogger(__name__)
        
    def _check_df(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("input data is empty")
        for c in ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']:
            if c not in df.columns:
                raise KeyError(f"{c} column not present in data")
        for c in ['datetime', 'symbol']:
            if df[c].isnull().any():
                raise ValueError(f"{c} column contains empty or null values")
        if df.duplicated(subset=['symbol', 'datetime']).any():
            raise ValueError("duplicated (symbol, datetime) in data")
        symbols = df['symbol'].unique().tolist()
        for symbol in self.factors:
            if not symbol in symbols:
                raise ValueError(f"Factor {symbol} not available in data")
        
    def get_factors(self):
        return self.factors
    
    def get_factor_comb(self):
        return self.factor_comb
    
    def get_universe(self, factors: list = []):
        return [sym for sym in self.universe if sym not in factors]

    def compute_pred_snapshot(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        self._check_df(df)
        df_all = slice_by_day_range(ensure_utc_inplace(df.copy(), "datetime"), start_date=(self.start_date), end_date=(self.end_date))
        self._check_df(df_all)
        
        logger = self.logger
        start_date = self.start_date
        end_date = self.end_date
        beta_lambda = self.beta_lambda
        min_obs = self.min_obs
        factors = self.factors
        factor_comb = [list(comb) for r in range(1, len(factors) + 1) for comb in itertools.combinations(factors, r)]
        
        px = wide_close(df_all, "close")
        r = log_returns(px)
        r = winsorize(r, self.winsor)
        
        logger.info(f"Computing snapshot, factors: {factors}, factor_comb: {factor_comb}, lambda: {beta_lambda}, min_obs: {min_obs}, start: {start_date}, end: {end_date}, symbols: {len(r.columns)}, lines: {len(r)}")

        betas = {}
        preds = {}
        resids = {}
        r2s = {}
        alphas = {}
        sigmas = {}
        phis = {}
        zs = {}
        zcums = {}
        residcums = {}
        for factor in factor_comb:
            logger.info(f"Computing factor {factor}...")
            universe = sorted([c for c in r.columns if c not in factor])
            r_f = r[factor].astype(float)
            r_alt = r[universe].astype(float)
            beta = compute_beta_ewma_uni(r_alt, r_f, beta_lambda, min_obs)
                    
            resid, pred = residuals_with_beta_freeze_uni(r_alt, r_f, beta)
            
            var_e_end = resid.ewm(alpha=(1-beta_lambda), adjust=False, min_periods=min_obs).var()
            var_a_end = r[universe].ewm(alpha=(1-beta_lambda), adjust=False).var()
            r2 = (1.0 - (var_e_end / var_a_end))
            
            u, v = resid, resid.shift(1)
            m_u = u.ewm(alpha=(1-beta_lambda), adjust=False, min_periods=min_obs).mean()
            m_v = v.ewm(alpha=(1-beta_lambda), adjust=False, min_periods=min_obs).mean()
            m_uv = (u * v).ewm(alpha=(1-beta_lambda), adjust=False, min_periods=min_obs).mean()
            cov_uv = m_uv - m_u * m_v
            var_v = (v.pow(2).ewm(alpha=(1-beta_lambda), adjust=False).mean() - m_v.pow(2))
            phi = (cov_uv / var_v)
            
            alpha = resid.ewm(alpha=(1-beta_lambda), adjust=False, min_periods=min_obs).mean()
            resid_pow = resid.pow(2)
            sigma = (resid_pow.ewm(alpha=(1-beta_lambda), adjust=False, min_periods=min_obs).mean().pow(0.5))
            z = resid.div(sigma.shift(1), axis=0)
            zcum = z.cumsum()
            residcum = resid.cumsum()
            
            betas[str(factor)] = beta
            preds[str(factor)] = pred
            resids[str(factor)] = resid
            r2s[str(factor)] = r2
            alphas[str(factor)] = alpha
            sigmas[str(factor)] = sigma
            phis[str(factor)] = phi
            zs[str(factor)] = z
            zcums[str(factor)] = zcum
            residcums[str(factor)] = residcum
            
        self.betas = betas
        self.preds = preds
        self.resids = resids
        self.r2s = r2s
        self.alphas = alphas
        self.sigmas = sigmas
        self.phis = phis
        self.zs = zs
        self.zcums = zcums
        self.residcums = residcums
        self.factor_comb = factor_comb
        self.universe = sorted(r.columns)
        self.r = r
        
        logger.info(f"Adding adicional info...")
        
        logger.info(f"Finished snapshot computation, effective window: {start_date}-{end_date}")
    
    def render_snapshot(self, factors: list = None, date: object = None) -> pd.DataFrame:
        if factors == None:
            factors = self.factors
        if date == None:
            date = self.end_date
        else:
            date = pd.Timestamp(date).tz_localize("UTC")
        universe = sorted([c for c in self.universe if c not in factors])
        out = pd.DataFrame({
            "alpha": self.alphas[str(factors)][universe].loc[date].reindex(universe),
            "sigma": self.sigmas[str(factors)][universe].loc[date].reindex(universe),
            "phi": self.phis[str(factors)][universe].loc[date].reindex(universe),
            "r2": self.r2s[str(factors)][universe].loc[date].reindex(universe),
            "z": self.zs[str(factors)][universe].loc[date].reindex(universe),
            "obs": self.resids[str(factors)][universe].count().reindex(universe)
        })

        betas = pd.DataFrame(index=universe)
        for f in factors:
            betas[f'beta_{f}'] = self.betas[str(factors)][f][universe].loc[date].reindex(universe)
        
        out = out.join(betas)
        
        return out.dropna()
    
    def render_factor_snapshot(self, date: object = None) -> pd.DataFrame:
        if date == None:
            date = self.end_date
            
        factors = self.factors
        factor_comb = self.factor_comb
        
        beta_columns = [f"beta_{f}" for f in factors]
        
        out_list = []
        for factor in factor_comb:
            universe = sorted([c for c in self.universe if (c not in factor) and (c in factors)])
            pre_out = pd.DataFrame({
                "alpha": self.alphas[str(factor)][universe].loc[date].reindex(universe),
                "sigma": self.sigmas[str(factor)][universe].loc[date].reindex(universe),
                "phi": self.phis[str(factor)][universe].loc[date].reindex(universe),
                "r2": self.r2s[str(factor)][universe].loc[date].reindex(universe),
                "z": self.zs[str(factor)][universe].loc[date].reindex(universe),
                "obs": self.resids[str(factor)][universe].count().reindex(universe)
            })

            betas = pd.DataFrame(index=universe, columns=beta_columns)
            for f in factor:
                betas[f'beta_{f}'] = self.betas[str(factor)][f][universe].loc[date].reindex(universe)
            
            pre_out = pre_out.join(betas)
            out_list.append(pre_out.reset_index())
            
        out = pd.concat(out_list)

        return out

    def regression_view(self,
        symbol: str,
        factors: list = None,
    ) -> Dict[str, object]:
        
        if factors == None:
            factors = self.factors

        ri_eval = self.r[symbol]
        pred_eval = self.preds[str(factors)][symbol]

        dfp = pd.concat({"real": ri_eval, "pred": pred_eval}, axis=1)

        eps = self.resids[str(factors)][symbol].rename("eps")
        eps_cum = eps.cumsum().rename("eps_cum")
        
        z_dyn = self.zs[str(factors)][symbol].rename("z_eps")
        z_cum = z_dyn.cumsum().rename("z_cum")

        out = pd.concat([dfp, eps, eps_cum, z_dyn, z_cum], axis=1)

        meta = {
            "symbol": symbol,
            "factors": factors,
            "beta_lambda": self.beta_lambda,
        }
        return {"df": out, "meta": meta}
                        
    def hmm_train(self,
              n_states: int = 9,
              min_len: int = 30,
              random_state: int = 42, 
              offset: int = 0,
              multi: bool = False,
              iter: int = 1000,
              tol: float = 0.005, 
              min_covar: float = 0.0001, 
              mix: int = 2,
              type: str = 'rs_hmm'):
        
        data = []
        if multi:
            for f in self.factor_comb:
                data.append(self.zs[str(f)])
        else:
            data.append(self.zs[str(self.factors)])
                        
        self.hmm_model.train(data, n_states, min_len, random_state, offset, iter, tol, min_covar, mix, type)

    def hmm_view(self, symbol: str, factors: list = None, type: str = 'rs_hmm', min_len: int = 30) -> tuple[pd.DataFrame, Dict]:
        if factors == None:
            factors = self.factors
        return self.hmm_model.view(self.zs[str(factors)], symbol, type, min_len)