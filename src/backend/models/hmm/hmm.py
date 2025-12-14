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
from backend.io import load_model, save_model
from scipy.special import logsumexp
from typing import Dict, Tuple
import logging
from hmmlearn.hmm import GaussianHMM, GMMHMM


class RSHMM:
    
    def __init__(self):
        
        self.model = None
        self.meta = None
        
        self.logger = logging.getLogger(__name__)

    def _entry(self, z):
            
            s = z.dropna()
            valid_idx = s.index
            l = [s.values]
            X = np.column_stack(l)
            
            return X, valid_idx
            
    def prepare_data(self,
                        data: list,
                        offset,
                        min_len):
        
        seqs = []
        lengths = []
        used_meta = []
            
        for d in data:

            Z = d.iloc[offset:]
            
            for sym in Z.columns:
                s = Z[sym]
                
                X, valid_idx = self._entry(s)
                
                seq_len = X.shape[0]
                if seq_len >= min_len:
                    seqs.append(X.astype(np.float64))
                    lengths.append(seq_len)
                    used_meta.append((sym, valid_idx[0], valid_idx[-1]))
                        
        return seqs, lengths, used_meta
                        
    def train(self, data,
            n_states: int = 9,
            min_len: int = 30,
            random_state: int = 42, 
            offset: int = 0,
            iter: int = 1000,
            tol: float = 0.005, 
            min_covar: float = 0.0001, 
            mix: int = 2,
            type: str = 'rs_hmm'):
                        
        seqs, lengths, used_meta = self.prepare_data(data, offset=offset, min_len=min_len)
                    
        if not seqs:
            raise RuntimeError("No valid sequences to train.")

        X = np.vstack(seqs)
        if type == 'rs_hmm':
            model = GaussianHMM(
                n_components=n_states,
                min_covar=min_covar,
                covariance_type="diag",
                n_iter=iter,
                random_state=random_state,
                algorithm='viterbi',
                params='stmc',
                init_params='stmc',
                implementation='log',
                tol=tol
            )
        elif type == 'rs_gmmhmm':
            model = GMMHMM(
                n_components=n_states,
                min_covar=min_covar,
                n_mix=mix,
                covariance_type="diag",
                n_iter=iter,
                random_state=random_state,
                algorithm='viterbi',
                params='stmcw',
                init_params='stmcw',
                implementation='log',
                tol=tol
            )
        model.fit(X, lengths)

        self.model = model
        self.meta = {
            "lengths": lengths,
            "used": used_meta,
            "n_states": n_states,
            "random_state": random_state,
            "offset": offset,
            "iter": iter,
            "tol": tol,
            "min_cov": min_covar,
            "mix": mix,
            "type": type
        }
        
        save_model(type, self.model, self.meta)
        return model
        
    def _stationary_from_transmat(self, A: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eig(A.T)
        i = np.argmin(np.abs(vals - 1.0))
        v = np.real(vecs[:, i])
        v = v / v.sum()
        v = np.clip(v, 0, None)
        return (v / v.sum()).astype(float)

    def _cov_to_std(self, model, k: int, feature: int = 0) -> float:
        ct  = getattr(model, "covariance_type", "diag")
        cov = getattr(model, "covars_", None)
        if cov is None:
            raise ValueError("Model has no covars_")

        if ct == "diag":
            vk = np.asarray(cov[k], dtype=float).reshape(-1)
            return float(np.sqrt(vk[feature]))

        elif ct == "full":
            Mk = np.asarray(cov[k], dtype=float)
            return float(np.sqrt(Mk[feature, feature]))

        elif ct == "tied":
            M = np.asarray(cov, dtype=float)
            return float(np.sqrt(M[feature, feature]))

        elif ct == "spherical":
            v = float(np.asarray(cov[k], dtype=float))
            return float(np.sqrt(v))

        else:
            raise ValueError(f"Unsupported covariance_type: {ct}")


    def view(self, data: pd.DataFrame, symbol: str, type: str = 'rs_hmm', min_len: int = 30) -> tuple[pd.DataFrame, Dict]:
        
        model = self.model    
        meta_train = self.meta
        
        if model == None or meta_train == None or type != meta_train['type']:
            model, meta_train = load_model(type)
            if model == None or meta_train == None:
                return pd.DataFrame(), {}
            self.model = model
            self.meta = meta_train
            
        z = data

        z = z.iloc[meta_train['offset']:]
        if z is None or z.empty:
            return pd.DataFrame(), {}

        z = z[symbol].dropna().rename("z")
        if len(z) < min_len:
            return pd.DataFrame(), {}
        
        X, valid_idx = self._entry(z)
        if X.shape[0] < 1:
            return pd.DataFrame(), {}
        post = model.predict_proba(X)
        states = model.predict(X)

        K = model.n_components
        means = [float(np.asarray(model.means_[k], dtype=float).reshape(-1)[0]) for k in range(K)]
        stds = [self._cov_to_std(model, k, feature=0) for k in range(K)]
        A = np.asarray(model.transmat_, dtype=float)
        pi = self._stationary_from_transmat(A)
        exp_dur = [float(1.0 / max(1e-12, (1.0 - A[k, k]))) for k in range(K)]
        order = np.argsort(means).tolist()

        post_cols = [f"s{k}" for k in range(K)]
        out = pd.concat(
            [
                z,
                pd.Series(states, index=valid_idx, name="state"),
                pd.DataFrame(post, index=valid_idx, columns=post_cols),
            ],
            axis=1,
        )
        
        pT = np.asarray(post[-1], dtype=float)
        pT = np.clip(pT, 0.0, 1.0)
        s = pT.sum()
        pT = (pT / s) if s > 0 else np.full(K, 1.0 / K)

        p1 = pT @ A
        p1 = np.clip(p1, 0.0, 1.0)
        p1 = p1 / p1.sum()

        freq = pd.infer_freq(z.index)
        if freq is not None and len(z) >= 1:
            t_next = z.index[-1] + pd.tseries.frequencies.to_offset(freq)
        elif len(z) >= 2:
            t_next = z.index[-1] + (z.index[-1] - z.index[-2])
        else:
            t_next = z.index[-1]

        f1_cols = [f"s{k}" for k in range(K)]
        row = pd.Series(index=out.columns, dtype=float)
        row[f1_cols] = p1      
        row["state"] = np.nan   
        row["z"] = np.nan       
        row["is_forecast"] = 1.0

        out = pd.concat([out, pd.DataFrame([row], index=[t_next])], axis=0)

        meta = {
            "symbol": symbol,
            "t0": z.index.min().isoformat(),
            "t1": z.index.max().isoformat(),
            "n_obs": int(len(z)),
            "n_states": int(K),
            "state_means": means,
            "state_stds": stds,
            "trans_mat": A.tolist(),
            "stationary": pi.tolist(),
            "exp_duration": exp_dur,
            "state_order_by_mean": order,
            **(meta_train if isinstance(meta_train, dict) else {}),
        }
        return out, meta


    def view_forward(self, data: pd.DataFrame, symbol: str, type: str = 'rs_hmm', min_len: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        model = self.model
        meta_train = self.meta

        if model is None or meta_train is None or type != meta_train.get('type'):
            model, meta_train = load_model(type)
            if model is None or meta_train is None:
                return pd.DataFrame(), pd.DataFrame(), {}
            self.model = model
            self.meta = meta_train

        z = data
        z = z.iloc[meta_train['offset']:]
        if z is None or z.empty:
            return pd.DataFrame(), pd.DataFrame(), {}

        z = z[symbol].dropna().rename("z")
        if len(z) < min_len:
            return pd.DataFrame(), pd.DataFrame(), {}

        X, valid_idx = self._entry(z)
        if X.shape[0] < 1:
            return pd.DataFrame(), pd.DataFrame(), {}

        def _safe_log(a):
            return np.log(np.clip(a, 1e-300, None))

        logB = model._compute_log_likelihood(X)
        K = model.n_components
        logA = _safe_log(np.asarray(model.transmat_))
        logpi = _safe_log(np.asarray(model.startprob_))

        T = logB.shape[0]
        log_alpha = np.full((T, K), -np.inf)

        log_alpha[0] = logpi + logB[0]
        log_alpha[0] -= logsumexp(log_alpha[0])

        for t in range(1, T):
            prev = log_alpha[t-1][:, None] + logA
            log_alpha[t] = logsumexp(prev, axis=0) + logB[t]
            log_alpha[t] -= logsumexp(log_alpha[t])

        filtered = np.exp(log_alpha)
        filtered = filtered / filtered.sum(axis=1, keepdims=True)
        states = filtered.argmax(axis=1).astype(int)

        post_cols = [f"s{k}" for k in range(K)]
        out = pd.concat(
            [
                z,
                pd.Series(states, index=valid_idx, name="state"),
                pd.DataFrame(filtered, index=valid_idx, columns=post_cols),
            ],
            axis=1,
        )

        A = np.asarray(model.transmat_, dtype=float)
        p1_all = filtered @ A 
        p1_all = np.clip(p1_all, 0.0, None)
        row_sums = p1_all.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        p1_all = p1_all / row_sums

        f_cols = [f"s{k}" for k in range(K)]
        forecasts_df = pd.DataFrame(p1_all, index=valid_idx, columns=f_cols)
        forecasts_df["state"] = forecasts_df[f_cols].values.argmax(axis=1).astype(int)

        freq = pd.infer_freq(z.index)
        if freq is not None and len(z) >= 1:
            t_next_index = z.index + pd.tseries.frequencies.to_offset(freq)
        elif len(z) >= 2:
            delta = z.index[-1] - z.index[-2]
            t_next_index = z.index + delta
        else:
            t_next_index = z.index

        forecasts_df["t_plus_1"] = t_next_index

        pT = np.asarray(filtered[-1], dtype=float)
        pT = np.clip(pT, 0.0, 1.0)
        ssum = pT.sum()
        pT = (pT / ssum) if ssum > 0 else np.full(K, 1.0 / K)
        p1_final = pT @ A
        p1_final = np.clip(p1_final, 0.0, 1.0)
        p1_final = p1_final / p1_final.sum()

        if freq is not None and len(z) >= 1:
            t_next = z.index[-1] + pd.tseries.frequencies.to_offset(freq)
        elif len(z) >= 2:
            t_next = z.index[-1] + (z.index[-1] - z.index[-2])
        else:
            t_next = z.index[-1]

        row = pd.Series(index=out.columns, dtype=float)
        row[post_cols] = p1_final
        row["state"] = np.nan
        row["z"] = np.nan
        row["is_forecast"] = 1.0
        out = pd.concat([out, pd.DataFrame([row], index=[t_next])], axis=0)

        means = [float(np.asarray(model.means_[k], dtype=float).reshape(-1)[0]) for k in range(K)]
        stds = [self._cov_to_std(model, k, feature=0) for k in range(K)]
        pi = self._stationary_from_transmat(A)
        exp_dur = [float(1.0 / max(1e-12, (1.0 - A[k, k]))) for k in range(K)]
        order = np.argsort(means).tolist()

        meta = {
            "symbol": symbol,
            "t0": z.index.min().isoformat(),
            "t1": z.index.max().isoformat(),
            "n_obs": int(len(z)),
            "n_states": int(K),
            "state_means": means,
            "state_stds": stds,
            "trans_mat": A.tolist(),
            "stationary": pi.tolist(),
            "exp_duration": exp_dur,
            "state_order_by_mean": order,
            **(meta_train if isinstance(meta_train, dict) else {}),
        }

        return out, forecasts_df, meta