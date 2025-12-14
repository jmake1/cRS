import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from backend.models.rs import RSEW


def plot_regression_view(RS: RSEW, sym: str, factors: list = None):

    res = RS.regression_view(
        symbol=sym,
        factors=factors
    )
    
    dfp, meta = res["df"], res["meta"]
    
    format = st.select_slider("Format", ["default", "z-score"], value="default")

    if dfp is None or dfp.empty:
        st.warning(f"No data for selected symbol ({meta.get('err','')}).")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dfp.index, y=dfp["real"], name="Real (r_i)", mode="lines"))
        fig.add_trace(go.Scatter(x=dfp.index, y=dfp["pred"], name="Pred (ŷ)", mode="lines"))
        x_last = dfp.index[-1]
        fig.add_trace(go.Scatter(
            x=[x_last], y=[dfp["real"].iloc[-1]],
            mode="markers", name="last (real)", marker=dict(size=8)
        ))

        fig.update_layout(
            title=f"{sym} — Real vs Pred",
            hovermode="x unified",
            xaxis=dict(
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, step="day", stepmode="backward", label="7D"),
                        dict(count=1, step="month", stepmode="backward", label="1M"),
                        dict(count=3, step="month", stepmode="backward", label="3M"),
                        dict(step="all", label="All"),
                    ])
                ),
            ),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, width='stretch', config={"scrollZoom": True})

    if format == "default":
        if "eps" in dfp.columns:
            figz = go.Figure()
            figz.add_trace(go.Scatter(x=dfp.index, y=dfp["eps"], name="eps", mode="lines"))
            for y0 in [0, 0.5, -0.5]:
                figz.add_hline(y=y0, line_dash="dot")
            figz.update_layout(
                title="Residual EPS",
                hovermode="x unified",
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    rangeselector=dict(
                        buttons=list([
                            dict(count=7, step="day", stepmode="backward", label="7D"),
                            dict(count=1, step="month", stepmode="backward", label="1M"),
                            dict(step="all", label="All"),
                        ])
                    ),
                ),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(figz, width='stretch', config={"scrollZoom": True})
            
        if "eps_cum" in dfp.columns:
            figc = go.Figure()
            figc.add_trace(go.Scatter(x=dfp.index, y=dfp["eps_cum"], name="Σ rcum", mode="lines"))
            figc.add_hline(y=0, line_dash="dot")

            figc.update_layout(
                title="Cumulative residual",
                hovermode="x unified",
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    rangeselector=dict(
                        buttons=list([
                            dict(count=7, step="day", stepmode="backward", label="7D"),
                            dict(count=1, step="month", stepmode="backward", label="1M"),
                            dict(count=3, step="month", stepmode="backward", label="3M"),
                            dict(step="all", label="All"),
                        ])
                    ),
                ),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(figc, width='stretch', config={"scrollZoom": True})
        
    else: 
        if "z_eps" in dfp.columns:
            figz = go.Figure()
            figz.add_trace(go.Scatter(x=dfp.index, y=dfp["z_eps"], name="z_eps", mode="lines"))
            for y0 in [0, 2, -2]:
                figz.add_hline(y=y0, line_dash="dot")
            figz.update_layout(
                title="Residual z-score",
                hovermode="x unified",
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    rangeselector=dict(
                        buttons=list([
                            dict(count=7, step="day", stepmode="backward", label="7D"),
                            dict(count=1, step="month", stepmode="backward", label="1M"),
                            dict(step="all", label="All"),
                        ])
                    ),
                ),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(figz, width='stretch', config={"scrollZoom": True})
            
        if "z_cum" in dfp.columns:
            figc = go.Figure()
            figc.add_trace(go.Scatter(x=dfp.index, y=dfp["z_cum"], name="Σ z_dyn", mode="lines"))
            if "z_cum_freeze" in dfp.columns:
                figc.add_trace(go.Scatter(x=dfp.index, y=dfp["z_cum_freeze"], name="Σ z_freeze", mode="lines"))
            figc.add_hline(y=0, line_dash="dot")

            figc.update_layout(
                title="Cumulative Z (Σ z)",
                hovermode="x unified",
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    rangeselector=dict(
                        buttons=list([
                            dict(count=7, step="day", stepmode="backward", label="7D"),
                            dict(count=1, step="month", stepmode="backward", label="1M"),
                            dict(count=3, step="month", stepmode="backward", label="3M"),
                            dict(step="all", label="All"),
                        ])
                    ),
                ),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(figc, width='stretch', config={"scrollZoom": True})

def plot_hmm_view(RS: RSEW, sym: str, factors: list = None):

    type = st.select_slider('Type', options=['rs_hmm', 'rs_gmmhmm'], value='rs_hmm')
    
    dfp, meta = RS.hmm_view(symbol=sym, factors=factors, type=type)

    if dfp is None or dfp.empty:
        st.warning(f"No data for selected symbol or HMM model not trained.")
    else:
        K = int(meta.get("n_states", 3))
        st.caption(
            f"{sym} | janela: {meta.get('t0','?')} → {meta.get('t1','?')} | "
            f"n={meta.get('n_obs',0)}"
        )
            
        means = meta.get("state_means", [])
        stds  = meta.get("state_stds", [])
        order = meta.get("state_order_by_mean", list(range(K)))

        palette = ["#4C78A8","#F58518","#E45756","#72B7B2","#54A24B",
                "#EECA3B","#B279A2","#FF9DA6","#9D755D","#BAB0AC"]
        state_color = {k: palette[i % len(palette)] for i, k in enumerate(order)}

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dfp.index, y=dfp["z"], name="z", mode="lines"))
        for k in range(K):
            m = dfp["state"] == k
            if m.any():
                fig.add_trace(go.Scatter(
                    x=dfp.index[m], y=dfp.loc[m, "z"],
                    mode="markers", name=f"State {k}",
                    marker=dict(size=6, color=state_color[k]), opacity=0.7
                ))
        for y0 in [0, 2, -2]:
            fig.add_hline(y=y0, line_dash="dot")
        fig.update_layout(
            title=f"{sym} — Z and States (HMM)",
            hovermode="x unified",
            xaxis=dict(
                rangeslider=dict(visible=True),
                rangeselector=dict(buttons=list([
                    dict(count=7, step="day", stepmode="backward", label="7D"),
                    dict(count=1, step="month", stepmode="backward", label="1M"),
                    dict(step="all", label="All"),
                ])),
            ),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, width='stretch', config={"scrollZoom": True})

        figp = go.Figure()
        for k in range(K):
            col = f"s{k}"
            if col in dfp.columns:
                figp.add_trace(go.Scatter(x=dfp.index, y=dfp[col], name=f"p(State {k})", mode="lines", marker=dict(size=6, color=state_color[k])))
        figp.update_layout(
            title="Probabilities P(state | z_t)",
            hovermode="x unified",
            xaxis=dict(rangeslider=dict(visible=True)),
            yaxis=dict(range=[0, 1]),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(figp, width='stretch', config={"scrollZoom": True})

        means = meta.get("state_means", [])
        stds = meta.get("state_stds", [])
        stat = meta.get("stationary", [])
        dur = meta.get("exp_duration", [])
        order = meta.get("state_order_by_mean", list(range(K)))
        df_states = pd.DataFrame({
            "state": list(range(K)),
            "mean(z)": np.round(means, 4),
            "std(z)": np.round(stds, 4),
            "π* (stationary)": np.round(stat, 4),
            "E[duration]": np.round(dur, 2),
        }).set_index("state").loc[order]
        st.subheader("States")
        st.dataframe(df_states)

        A = np.array(meta.get("trans_mat", [[np.nan]*K]*K), dtype=float)
        dfA = pd.DataFrame(np.round(A, 4), index=[f"s{k}" for k in range(K)], columns=[f"s{k}" for k in range(K)])
        st.subheader("Transition Matrix (A)")
        st.dataframe(dfA)