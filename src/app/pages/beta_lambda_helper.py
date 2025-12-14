import streamlit as st
from backend.utils import generate_lambda_table


def main():
    st.set_page_config(page_title="lambda_helper", page_icon="📈", layout="wide")
    st.title("📈 Lambda Helper")
    
    st.header("Beta Lambda helper table")
    st.subheader("Beta Lambda λ")
    st.write("Decaying factor for EWM (alpha = 1-λ)")
    st.write("Smaller lambdas -> EWM more responsive, less memory")
    st.write("Higher lambdas -> EWM less responsive, more memory")
    st.subheader("N_eff")
    st.write("Equivalent number of equally-weighted observations (n_eff = 1 / (1 - λ))")
    st.write("λ=0.9 → n_eff=10 → Similar to 10-period simple moving average")
    st.subheader("Half_life")
    st.write("Number of periods for weight to decay to half (half_life = log(0.5) / log(λ))")
    st.write("λ=0.9 → half_life=6.58 → After 7 periods, weight is <50%")
    st.subheader("N_percentil")
    st.write("Minimum periods to capture percentil of total weight (n_99 = ceil(log(0.01) / log(λ)))")
    st.write("λ=0.9 → n_99=44 → 44 periods contain 99% of weight")
    
    st.dataframe(generate_lambda_table(start_lambda=0.80, end_lambda=0.99, step=0.01, weight_percentil=0.99))
    
if __name__ == "__main__":
    main()