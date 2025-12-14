import streamlit as st
from backend.io import load_ohlcv, update_ohlcv

def ohlcv_data_form():
    st.sidebar.header("OHLCV Data")
    with st.sidebar.form("ohlcv_data_form"):
        sources = st.multiselect("Sources", ['bybit', 'mexc', 'binance'], default=['bybit'])
        instruments = st.multiselect("Instruments", ['spot', 'perp'], default=['spot'])
        update = st.checkbox("Update")
        interval = st.radio("Interval", ['1h','4h','1d','1w','1M'], index=3)
        load = st.form_submit_button("Load Data")

    if load:
        try:
            if update:
                update_ohlcv(sources=sources, instruments=instruments, interval=interval)
            st.session_state.ohlcv_raw = load_ohlcv(sources=sources, instruments=instruments, interval=interval)
        except FileNotFoundError:
            st.error("There is no data available locally, please update.")
        except Exception as e:
            raise e