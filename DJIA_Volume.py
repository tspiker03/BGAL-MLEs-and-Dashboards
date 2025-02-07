import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import gamma
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
from MLE_BGAL import w, r_MLE, delta_MLE, mu_MLE

# Create media directory if it doesn't exist
if not os.path.exists('media'):
    os.makedirs('media')

# Store figures in session state
if 'figures' not in st.session_state:
    st.session_state.figures = []


def save_plots():
    """Save all plots stored in session state"""
    # Save as individual PNGs
    for name, fig in st.session_state.figures:
        filepath = os.path.join('media', f'DJI_Volume_{selected_period}_{selected_frequency}_{name}.png')
        fig.savefig(filepath)
    
    # Save all plots as a single PDF
    pdf_path = os.path.join('media', f'DJI_Volume_{selected_period}_{selected_frequency}_report.pdf')
    with PdfPages(pdf_path) as pdf:
        # Add title page
        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        title = f'Dow Jones Industrial Average Volume Analysis Report\nPeriod: {selected_period}\nFrequency: {selected_frequency}'
        plt.text(0.5, 0.5, title, transform=fig.transFigure, ha='center', va='center')
        pdf.savefig()
        plt.close()
        
        # Add summary page
        if 'summary_data' in st.session_state:
            fig = plt.figure(figsize=(8.5, 11))
            fig.clf()
            summary = st.session_state.summary_data
            plt.text(0.5, 0.5, summary, transform=fig.transFigure, ha='center', va='center')
            pdf.savefig()
            plt.close()
        
        # Add all plots
        for name, fig in st.session_state.figures:
            pdf.savefig(fig)
    
    st.success(f"Saved {len(st.session_state.figures)} plots to media folder and generated PDF report")


# Clear previous figures at the start of each run
st.session_state.figures = []

st.title("Stock Price and Volume Analysis Dashboard")
st.write("Index: Dow Jones Industrial Average (^DJI)")
st.write("Using DJIA Trading Volume")

# Set stock symbol
stock_symbol = "^DJI"  # Dow Jones Industrial Average

# Period selection
period_options = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
selected_period = st.selectbox("Select the period", period_options)

# Frequency selection
frequency_options = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
selected_frequency = st.selectbox("Select the frequency", frequency_options)

if stock_symbol:
    data = yf.download(stock_symbol, period=selected_period, interval=selected_frequency)

    if not data.empty:
        # Handle multi-index columns
        df = pd.DataFrame()
        df["Close"] = data.loc[:, ("Close", data.columns.get_level_values(1)[0])]
        df["Volume"] = data.loc[:, ("Volume", data.columns.get_level_values(1)[0])]
        #df['Volume'] = (df['Volume']+df['Volume'].mean())/df['Volume'].std()
        df["Log Returns"] = np.log(df["Close"]) - np.log(df["Close"].shift(1))
        # Drop rows with NaN values
        df = df.dropna()
        
        st.subheader("Closing Price")
        st.line_chart(df["Close"])
        
        st.subheader("Trading Volume")
        st.line_chart(df["Volume"])

        # Use log volume for stationarity testing
        df["Log Volume"] = np.log(df["Volume"])
        diff_data = df["Log Volume"].dropna()
        diff_column = "Log Volume"

        # Check for stationarity and perform multiple differencing if needed
        max_diff = 3
        for i in range(max_diff):
            result = sm.tsa.stattools.adfuller(diff_data)
            if result[1] <= 0.05:
                break
            diff_data = diff_data.diff().dropna()
            diff_column = f"Volume_Diff_{i + 1}"
            df[diff_column] = diff_data

            st.subheader("ACF Plot of DJIA Log Volume")
            fig, ax = plt.subplots()
            sm.graphics.tsa.plot_acf(diff_data, lags=30, ax=ax)
            ax.set_title("Autocorrelation Function of DJIA Log Volume")
            st.session_state.figures.append(('volume_acf', fig))
            st.pyplot(fig)

        ar_model = st.number_input("Enter the AR model (integer)", min_value=1, value=1, step=1)

        if ar_model:
            # Clean and prepare data for AR model
            model_data = df[[diff_column]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(model_data) > 0:
                model = sm.tsa.AutoReg(model_data, lags=ar_model, trend="c")
                results = model.fit()
            alpha = results.params[0]
            betas = results.params[1:]

            if ar_model == 1:
                df.loc[model_data.index, "G"] = df.loc[model_data.index, "Log Volume"]/ (
                            df.loc[model_data.index, "Log Volume"].shift(1) ** betas[0] * np.exp(alpha))
            else:
                lags = list(range(1, ar_model + 1))
                beta_terms = " * ".join(
                    [f"(df.loc[model_data.index, 'Log Volume'].shift({lag}) ** betas[{i}])" for i, lag in
                     enumerate(lags)])
                df.loc[model_data.index, "G"] = eval(
                    f"df.loc[model_data.index, 'Log Volume'] / ({beta_terms} * np.exp(alpha))")

            # Clean up any invalid values
            df["G"] = df["G"].replace([np.inf, -np.inf], np.nan)
            g = df["G"].dropna()

            # Ensure G is positive and finite before gamma fitting
            g = g[g > 0]
            g = g[np.isfinite(g)]

            if len(g) > 0:
                a1, loc, scale = gamma.fit(g)

                st.subheader("G QQ Plot")
                fig, ax = plt.subplots()
                sm.qqplot(g, dist=gamma(a1, loc, scale), line="45", ax=ax)
                ax.set_title("Q-Q Plot of G vs Gamma Distribution")
                st.session_state.figures.append(('g_qq_plot', fig))
                st.pyplot(fig)

                st.subheader("G versus Time")
                st.line_chart(df["G"])

                st.subheader("ACF of G")
                fig, ax = plt.subplots()
                sm.graphics.tsa.plot_acf(g, lags=30, ax=ax)
                ax.set_title("Autocorrelation Function of G")
                st.session_state.figures.append(('g_acf', fig))
                st.pyplot(fig)

                st.subheader("Histogram of G")
                fig, ax = plt.subplots()
                ax.hist(g, bins=20, density=True)
                ax.set_title("Distribution of G")
                ax.set_xlabel("G")
                ax.set_ylabel("Density")
                st.session_state.figures.append(('g_histogram', fig))
                st.pyplot(fig)

                df = df[np.isfinite(df["G"])]

                X = pd.DataFrame({"X1": 1 / np.sqrt(df["G"]), "X2": np.sqrt(df["G"])})
                y = df["Log Returns"] / np.sqrt(df["G"])
                model = sm.OLS(y, X, hasconst=False)
                results = model.fit()
                a, c = results.params

                X = df["G"].dropna().values
                y = df["Log Returns"].dropna().values
                delta_hat = delta_MLE(X, y)
                mu_hat = mu_MLE(X, y, delta_hat)
                sigma_hat = r_MLE(w(X, y, delta_hat))

                size = len(df)
                df["G~"] = np.random.gamma(a1, scale, size)
                Z = np.random.standard_normal(size)
                df["Y~"] = delta_hat + mu_hat * df["G~"] + sigma_hat * np.sqrt(df["G~"]) * Z
                df["Z"] = df["Log Returns"] - delta_hat - mu_hat * (df["G~"]) - sigma_hat * np.sqrt(df["G~"])

                st.subheader("Z(t) QQ Plot, Error Term after fitting BGGL model")
                fig, ax = plt.subplots()
                sm.qqplot(df["Z"].dropna(), line="s", ax=ax)
                ax.set_title("Q-Q Plot of Z(t)")
                st.session_state.figures.append(('z_qq_plot', fig))
                st.pyplot(fig)

                st.subheader("Log Returns of DJI versus Time")
                st.line_chart(df["Log Returns"])

                st.subheader("ACF of Log Returns of DJI")
                fig, ax = plt.subplots()
                sm.graphics.tsa.plot_acf(df["Log Returns"].dropna(), lags=30, ax=ax)
                ax.set_title("Autocorrelation Function of Log Returns")
                st.session_state.figures.append(('log_returns_acf', fig))
                st.pyplot(fig)

                st.subheader("Histogram of Log Returns of DJI")
                fig, ax = plt.subplots()
                ax.hist(df["Log Returns"].dropna(), bins=20, density=True)
                ax.set_title("Distribution of Log Returns")
                ax.set_xlabel("Log Returns")
                ax.set_ylabel("Density")
                st.session_state.figures.append(('log_returns_histogram', fig))
                st.pyplot(fig)

                st.subheader("Log Returns versus G_t")
                fig, ax = plt.subplots()
                ax.scatter(df["G"], df["Log Returns"])
                ax.set_title("Log Returns vs G_t")
                ax.set_xlabel("G_t")
                ax.set_ylabel("Log Returns")
                st.session_state.figures.append(('log_returns_vs_g', fig))
                st.pyplot(fig)

                st.subheader("Summary")
                summary_text = f"""Gamma Distribution MLE Parameters:
Shape (a): {a1:.4f}
Location: {loc:.4f}
Scale: {scale:.4f}

Model Parameters:
Delta_hat: {delta_hat:.4f}
Mu_hat: {mu_hat:.4f}
Sigma_hat: {sigma_hat:.4f}"""

                st.text(summary_text)
                st.session_state.summary_data = summary_text

                st.subheader("Histogram of Simulated Volume")
                fig, ax = plt.subplots()
                ax.hist(df["G~"].dropna(), density=True, bins=20)
                ax.set_title("Distribution of Simulated Volume")
                ax.set_xlabel("G~")
                ax.set_ylabel("Density")
                st.session_state.figures.append(('g_tilde_histogram', fig))
                st.pyplot(fig)

                st.subheader("Histogram of Simulated BGAL")
                fig, ax = plt.subplots()
                ax.hist(df["Y~"].dropna(), density=True, bins=20)
                ax.set_title("Distribution of Simulated BGAL")
                ax.set_xlabel("Simulated BGAL")
                ax.set_ylabel("Density")
                st.session_state.figures.append(('y_tilde_histogram', fig))
                st.pyplot(fig)

                st.subheader("G~ versus Simulated BGAL")
                fig, ax = plt.subplots()
                ax.scatter(df["G~"], df["Y~"])
                ax.set_title("Simulated BGAL vs Simulated Volume")
                ax.set_xlabel("G~")
                ax.set_ylabel("Simulated BGAL")
                st.session_state.figures.append(('g_tilde_vs_y_tilde', fig))
                st.pyplot(fig)

                st.subheader("QQ Plot of Log Returns vs Simulated BGAL")
                fig, ax = plt.subplots()
                # Sort both arrays
                sorted_y = np.sort(df["Log Returns"].dropna())
                sorted_y_tilde = np.sort(df["Y~"].dropna())
                # Create QQ plot by plotting one against the other
                ax.scatter(sorted_y, sorted_y_tilde, alpha=0.5)
                # Add 45-degree line for reference
                min_val = min(sorted_y.min(), sorted_y_tilde.min())
                max_val = max(sorted_y.max(), sorted_y_tilde.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='45° line')
                ax.set_title("Q-Q Plot: Log Returns vs Simulated BGAL")
                ax.set_xlabel("Sample Quantiles (Log Returns)")
                ax.set_ylabel("Theoretical Quantiles (Simulated BGAL)")
                ax.legend()
                st.session_state.figures.append(('y_vs_y_tilde_qq', fig))
                st.pyplot(fig)

                # Add save button at the bottom of the page
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save All Plots"):
                        save_plots()
                with col2:
                    if st.button("Generate PDF Report"):
                        save_plots()
                        st.success(f"PDF report generated: DJI_Volume_{selected_period}_{selected_frequency}_report.pdf")
                    