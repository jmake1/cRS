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

import streamlit as st
from backend.models.rs import RSEW
from backend.utils import generate_lambda_table
from app.core import ohlcv_data_form, plot_hmm_view, plot_regression_view
import pandas as pd
from datetime import timedelta


def main():
    st.set_page_config(page_title="cRS", page_icon="📈", layout="wide")
    st.title("📈 cRS")
    
    ohlcv_data_form()
    
    if "ohlcv_raw" not in st.session_state:
        st.warning("📁 No data present. Please load some data first."); st.stop()
    
    ohlcv_raw = st.session_state.get("ohlcv_raw")
    dt_series = pd.to_datetime(ohlcv_raw["datetime"], utc=True, errors="coerce")
    min_dt, max_dt = dt_series.min().date(), dt_series.max().date()
    symbols = ohlcv_raw['symbol'].unique().tolist()
    
    st.sidebar.header("Symbols")
    symbols_exc = st.sidebar.multiselect("Discard selected symbols", symbols, default=[])
    symbols_inc = st.sidebar.multiselect("Only work with selected symbols", help="Must not exist in symbols to discard", options=symbols, default=[])
    
    st.sidebar.header("Timeline")
    start_date = st.sidebar.date_input("Start datetime (incl)", value=min_dt, min_value=min_dt, max_value=max_dt)
    end_date   = st.sidebar.date_input("End datetime (incl)", value=(max_dt - timedelta(days=1)), min_value=min_dt, max_value=max_dt)
    min_obs = st.sidebar.number_input('Min obs', 0, 1000000, 0, 1)
    if start_date > end_date:
        st.sidebar.error("Start datetime > End datetime")
        
    st.sidebar.header("Factorization")
    factors = st.sidebar.multiselect("Choose factors to consider", help="Must exist in symbols to include (if not empty) / Must not exist in symbols to discard", options=symbols, default=[])
    beta_lambda = st.sidebar.number_input("λβ (EWMA β)", 0.00000, 1.00000, 0.99, 0.00001, format="%.5f", help="Consult the lambda table for optimal choices")

    run = st.sidebar.button("Run")
    
    if run:
        
        df = ohlcv_raw
        df = df[~df['symbol'].isin(symbols_exc)]
        if symbols_inc != []:
            df = df[df['symbol'].isin(symbols_inc)]
            
        st.session_state['rs_cfg'] = {
            "beta_lambda": float(beta_lambda),
            "factors": factors,
            "start_date": start_date, 
            "end_date": end_date,
            "min_obs": min_obs,
        }
        
        RS = RSEW(beta_lambda=float(beta_lambda), factors=factors, start_date=start_date, end_date=end_date, min_obs=min_obs)

        RS.compute_pred_snapshot(df=df)

        st.session_state['rs_model'] = RS
        
        st.rerun()
        
    if 'rs_model' in st.session_state:
        RS = st.session_state['rs_model']
        
        st.sidebar.header("HMM")
        offset = st.sidebar.number_input("Offset", 0, 100, 0, 1)
        states = st.sidebar.number_input("States", 1, 100, 2, 1)
        mix_states = st.sidebar.number_input("Mix States (for HMM with Gaussian Mixture Emissions)", 1, 100, 2, 1)
        iter = st.sidebar.number_input("Iterations", 1, 10000, 500, 1)
        tol = st.sidebar.number_input("Tol", 0.0001, 0.1000, 0.0005, 0.0001, format="%.4f")
        min_cov = st.sidebar.number_input("Min Cov", 0.00001, 0.10000, 0.00100, 0.00001, format="%.5f")
        multi = st.sidebar.checkbox('Multi-mode', help="Wether to use all factor combinations to train", value=False)
        model_type = st.sidebar.select_slider('Type', help="HMM - HMM with Gaussian Emissions, GMMHMM - HMM with Gaussian Mixture Emissions", options=['rs_hmm', 'rs_gmmhmm'], value='rs_hmm')
        train_hmm = st.sidebar.button("Train HMM")
        
        if train_hmm:
            RS.hmm_train(n_states=states, offset=offset, multi=multi, iter=iter, tol=tol, min_covar=min_cov, mix=mix_states, type=model_type)
            
    
        with st.expander("Factor Pred", expanded=True):
            pred_snap = RS.render_factor_snapshot()
            st.dataframe(pred_snap, width='stretch')
            
        factors = st.selectbox("Factors", help="From computed factor combinations", options=RS.get_factor_comb())
        with st.expander("Pred", expanded=True):
            pred_snap = RS.render_snapshot(factors=factors)
            st.dataframe(pred_snap, width='stretch')
            
        with st.expander("Plots", expanded=True):
            sym = st.selectbox("Symbol", RS.get_universe(factors=factors))
            plot_regression_view(RS, sym)
            st.subheader("HMM Regimes (plot)")
            plot_hmm_view(RS, sym)

        
if __name__ == "__main__":
    main()