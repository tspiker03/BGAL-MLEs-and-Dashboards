import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import gamma
import matplotlib.pyplot as plt
from MLE_BGAL import w, r_MLE, delta_MLE, mu_MLE

st.title("Stock Price Analysis Dashboard")

# Stock selection
stock_symbol = st.text_input("Enter the stock ticker symbol")

# Period selection
period_options = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
selected_period = st.selectbox("Select the period", period_options)

# Frequency selection
frequency_options = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
selected_frequency = st.selectbox("Select the frequency", frequency_options)

if stock_symbol:
    data = yf.download(stock_symbol, period=selected_period, interval=selected_frequency)
    vix_data = yf.download("^VIX", period=selected_period, interval=selected_frequency)

    if not data.empty and not vix_data.empty:
        df = pd.DataFrame(data["Close"])
        df["Volatility"] = vix_data["Close"]
        df["Log Returns"] = np.log(df["Close"]) - np.log(df["Close"].shift(1))
        st.subheader("Closing Price")
        st.line_chart(df["Close"])

        # Check for stationarity and perform multiple differencing if needed
        max_diff = 3
        diff_data = df["Volatility"].dropna()
        diff_column = "Volatility"
        for i in range(max_diff):
            result = sm.tsa.stattools.adfuller(diff_data)
            if result[1] <= 0.05:
                break
            diff_data = diff_data.diff().dropna()
            diff_column = f"Volatility_Diff_{i + 1}"
            df[diff_column] = diff_data

        st.subheader("ACF Plot of VIX")
        fig, ax = plt.subplots()
        sm.graphics.tsa.plot_acf(diff_data, lags=30, ax=ax)
        st.pyplot(fig)

        ar_model = st.number_input("Enter the AR model (integer)", min_value=1, value=1, step=1)

        if ar_model:
            model_data = df[[diff_column]].dropna()
            model = sm.tsa.AutoReg(model_data, lags=ar_model, trend="c")
            results = model.fit()
            alpha = results.params[0]
            betas = results.params[1:]

            if ar_model == 1:
                df.loc[model_data.index, "G"] = df.loc[model_data.index, "Volatility"] / (
                            df.loc[model_data.index, "Volatility"].shift(1) ** betas[0] * np.exp(alpha))
            else:
                lags = list(range(1, ar_model + 1))
                beta_terms = " * ".join(
                    [f"(df.loc[model_data.index, 'Volatility'].shift({lag}) ** betas[{i}])" for i, lag in
                     enumerate(lags)])
                df.loc[model_data.index, "G"] = eval(
                    f"df.loc[model_data.index, 'Volatility'] / ({beta_terms} * np.exp(alpha))")

            g = df["G"].dropna()
            a, loc, scale = gamma.fit(g)

            st.subheader("Gamma QQ Plot")
            fig, ax = plt.subplots()
            sm.qqplot(g, dist=gamma(a, loc, scale), line="45", ax=ax)
            st.pyplot(fig)

            st.subheader("G versus Time")
            st.line_chart(df["G"])

            st.subheader("ACF of G")
            fig, ax = plt.subplots()
            sm.graphics.tsa.plot_acf(g, lags=30, ax=ax)
            st.pyplot(fig)

            st.subheader("Histogram of G")
            fig, ax = plt.subplots()
            ax.hist(g, bins=30, density=True)
            ax.set_xlabel("G")
            ax.set_ylabel("Density")
            st.pyplot(fig)

            df = df[np.isfinite(df["G"])]

            X = pd.DataFrame({"X1": 1 / np.sqrt(df["G"]), "X2": np.sqrt(df["G"])})
            y = df["Log Returns"] / np.sqrt(df["G"])
            model = sm.OLS(y, X, hasconst=False)
            results = model.fit()
            a, c = results.params

            df["Z"] = df["Log Returns"] / np.sqrt(df["G"]) - a * (1 / np.sqrt(df["G"])) - c * np.sqrt(df["G"])

            st.subheader("Z(t) QQ Plot")
            fig, ax = plt.subplots()
            sm.qqplot(df["Z"].dropna(), line="s", ax=ax)
            st.pyplot(fig)

            st.subheader("Log Returns versus Time")
            st.line_chart(df["Log Returns"])

            st.subheader("ACF of Log Returns")
            fig, ax = plt.subplots()
            sm.graphics.tsa.plot_acf(df["Log Returns"].dropna(), lags=30, ax=ax)
            st.pyplot(fig)

            st.subheader("Histogram of Log Returns")
            fig, ax = plt.subplots()
            ax.hist(df["Log Returns"].dropna(), bins=30, density=True)
            ax.set_xlabel("Log Returns")
            ax.set_ylabel("Density")
            st.pyplot(fig)

            st.subheader("Log Returns versus G")
            fig, ax = plt.subplots()
            ax.scatter(df["G"], df["Log Returns"])
            ax.set_xlabel("G")
            ax.set_ylabel("Log Returns")
            st.pyplot(fig)

            if not df.empty and "G" in df.columns and "a" in locals() and "c" in locals() and "alpha" in locals() and "betas" in locals():
                st.subheader("Summary")
                st.write("Gamma Distribution MLE Parameters:")
                st.write(f"Shape (a): {a:.4f}")
                st.write(f"Location: {loc:.4f}")
                st.write(f"Scale: {scale:.4f}")

                X = df["G"].dropna().values
                y = df["Log Returns"].dropna().values

                delta_hat = delta_MLE(X, y)
                mu_hat = mu_MLE(X, y, delta_hat)
                sigma_hat = r_MLE(w(X, y, delta_hat))

                st.write(f"Delta_hat: {delta_hat:.4f}")
                st.write(f"Mu_hat: {mu_hat:.4f}")
                st.write(f"Sigma_hat: {sigma_hat:.4f}")

                size = len(df)
                df["G~"] = np.random.gamma(a, scale, size)

                Z = np.random.standard_normal(size)
                df["Y~"] = delta_hat + mu_hat * df["G~"] + sigma_hat * np.sqrt(df["G~"]) * Z

                st.subheader("Histogram of G~")
                fig, ax = plt.subplots()
                ax.hist(df["G~"].dropna(), density=True)
                ax.set_xlabel("G~")
                ax.set_ylabel("Density")
                st.pyplot(fig)

                st.subheader("Histogram of Y~")
                fig, ax = plt.subplots()
                ax.hist(df["Y~"].dropna(), density=True)
                ax.set_xlabel("Y~")
                ax.set_ylabel("Density")
                st.pyplot(fig)

                st.subheader("G~ versus Y~")
                fig, ax = plt.subplots()
                ax.scatter(df["G~"], df["Y~"])
                ax.set_xlabel("G~")
                ax.set_ylabel("Y~")
                st.pyplot(fig)

                st.write("Time Series Parameters:")
                beta_terms = []
                for i in range(1, len(betas) + 1):
                    beta_terms.append(f"\\beta_{{{i}}} \\cdot X(t-{i})")
                latex_equation = "$X(t)$ = $\\alpha$ + " + " + ".join(beta_terms)
                st.latex(latex_equation)

                st.write("Coefficient Values:")
                st.write(f"$\\alpha$: {alpha:.4f}")
                for i, beta in enumerate(betas, start=1):
                    st.write(f"$\\beta_{{{i}}}$: {beta:.4f}")

                st.write("Regression Coefficients for Z(t):")
                st.write(f"a: {a:.4f}")
                st.write(f"c: {c:.4f}")