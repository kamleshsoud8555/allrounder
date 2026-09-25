import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import timedelta

st.set_page_config(page_title="Advanced Market Trend Predictor", page_icon="📈", layout="wide")

st.title("📈 Indian Market Trend Predictor")
st.markdown("### Stocks | MCX | Nifty 50 | Sensex | Volatility | Support & Resistance")

# Symbol Mapping
name_to_symbol = {
    # Indices
    "NIFTY": "^NSEI",
    "NIFTY 50": "^NSEI",
    "NIFTY50": "^NSEI",
    "NSEI": "^NSEI",
    "SENSEX": "^BSESN",
    "BSE SENSEX": "^BSESN",
    "BSESN": "^BSESN",

    # Stocks
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "INFOSYS": "INFY.NS",
    "SBIN": "SBIN.NS",
    "SBI": "SBIN.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "ITC": "ITC.NS",
    "WIPRO": "WIPRO.NS",
    "LT": "LT.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "MARUTI": "MARUTI.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "AXISBANK": "AXISBANK.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "AIRTEL": "BHARTIARTL.NS",

    # MCX
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "CRUDE": "CL=F",
    "CRUDE OIL": "CL=F",
    "CRUDEOIL": "CL=F",
    "NATURAL GAS": "NG=F",
    "NATGAS": "NG=F",
    "COPPER": "HG=F",
}

user_input = st.text_input("Enter Name (Example: Nifty 50, Sensex, Gold, Reliance)", value="Nifty 50")
user_input_clean = user_input.upper().strip()

if st.button("Get Future Trend", type="primary") or user_input:
    try:
        if user_input_clean in name_to_symbol:
            ticker = name_to_symbol[user_input_clean]
        elif user_input_clean.endswith((".NS", ".BO", "=F")) or user_input_clean.startswith("^"):
            ticker = user_input_clean
        else:
            ticker = user_input_clean + ".NS"

        with st.spinner(f"Analyzing {user_input}..."):
            data = yf.download(ticker, period="1y", progress=False, auto_adjust=True)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if data.empty or "Close" not in data.columns:
            st.error(f"❌ Could not find data for **{user_input}**")
            st.info("Try: Nifty 50, Sensex, Gold, Silver, Crude Oil, Reliance, TCS")
        else:
            data = data.dropna()
            current_price = float(data["Close"].iloc[-1])

            # Header
            if ticker == "^NSEI":
                st.success("✅ Analyzing **Nifty 50**")
            elif ticker == "^BSESN":
                st.success("✅ Analyzing **Sensex**")
            elif ticker.endswith("=F"):
                st.success(f"✅ International Futures for **{user_input}**")
            else:
                st.success(f"✅ Found: **{ticker}**")

            # ========== BASIC METRICS ==========
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Level / Price", f"{current_price:.2f}")

            recent = data["Close"].iloc[-20:].mean()
            previous = data["Close"].iloc[-40:-20].mean()

            if recent > previous * 1.03:
                trend = "📈 Strong Bullish"
            elif recent > previous * 1.01:
                trend = "📈 Mild Bullish"
            elif recent < previous * 0.97:
                trend = "📉 Strong Bearish"
            elif recent < previous * 0.99:
                trend = "📉 Mild Bearish"
            else:
                trend = "↔️ Sideways / Neutral"

            col2.metric("Current Trend", trend)

            # Forecast
            df = data.reset_index()
            df["Days"] = np.arange(len(df))
            model = LinearRegression()
            model.fit(df[["Days"]], df["Close"])
            future_days = np.arange(len(df), len(df) + 30).reshape(-1, 1)
            future_pred = model.predict(future_days)
            future_price = float(future_pred[-1])
            change = ((future_price - current_price) / current_price) * 100
            col3.metric("30-Day Forecast", f"{future_price:.2f}", f"{change:+.1f}%")

            # ========== SUPPORT & RESISTANCE ==========
            st.subheader("🛡️ Support & Resistance Levels")

            # Simple but effective method using recent highs and lows
            high_20 = data["High"].tail(20).max()
            low_20 = data["Low"].tail(20).min()
            high_50 = data["High"].tail(50).max()
            low_50 = data["Low"].tail(50).min()

            # Pivot Point style
            last_high = data["High"].iloc[-1]
            last_low = data["Low"].iloc[-1]
            last_close = data["Close"].iloc[-1]
            pivot = (last_high + last_low + last_close) / 3
            r1 = (2 * pivot) - last_low
            s1 = (2 * pivot) - last_high
            r2 = pivot + (last_high - last_low)
            s2 = pivot - (last_high - last_low)

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Nearest Support (S1)", f"{s1:.2f}")
            sc2.metric("Pivot Point", f"{pivot:.2f}")
            sc3.metric("Nearest Resistance (R1)", f"{r1:.2f}")

            st.caption(f"Additional levels → Strong Support: {low_50:.2f} | Strong Resistance: {high_50:.2f}")

            # ========== VOLATILITY SECTION ==========
            st.subheader("📊 Volatility Metrics")

            returns = data["Close"].pct_change().dropna()
            vol_20 = returns.tail(20).std() * np.sqrt(252) * 100
            vol_30 = returns.tail(30).std() * np.sqrt(252) * 100

            vcol1, vcol2, vcol3 = st.columns(3)
            vcol1.metric("20-Day Historical Volatility", f"{vol_20:.2f}%")
            vcol2.metric("30-Day Historical Volatility", f"{vol_30:.2f}%")

            # India VIX
            if ticker == "^NSEI":
                try:
                    vix_data = yf.download("^INDIAVIX", period="5d", progress=False, auto_adjust=True)
                    if isinstance(vix_data.columns, pd.MultiIndex):
                        vix_data.columns = vix_data.columns.get_level_values(0)
                    if not vix_data.empty:
                        india_vix = float(vix_data["Close"].iloc[-1])
                        vcol3.metric("India VIX", f"{india_vix:.2f}")
                    else:
                        vcol3.metric("India VIX", "Not available")
                except:
                    vcol3.metric("India VIX", "Not available")
            else:
                vcol3.metric("India VIX", "Only for Nifty")

            # ========== PUT-CALL RATIO SECTION ==========
            st.subheader("📉 Put-Call Ratio (PCR)")
            st.info("""
            **Live Put-Call Ratio** requires full option chain data which is not reliably available for free.  
            
            You can check the latest Nifty PCR from:
            - [NSE Option Chain](https://www.nseindia.com/option-chain)
            - [Sensibull](https://web.sensibull.com)
            - [Opstra](https://opstra.definedge.com)
            
            **How to read PCR:**
            - PCR > 1.0 → More Puts → Market may be oversold (bullish bias)
            - PCR < 0.7 → More Calls → Market may be overbought (bearish bias)
            """)

            # Chart
            st.subheader("Price / Index Chart + 30-Day Forecast")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Actual", line=dict(color="#1f77b4", width=2)))

            # Add Support & Resistance lines
            fig.add_hline(y=s1, line_dash="dot", line_color="green", annotation_text="Support S1")
            fig.add_hline(y=r1, line_dash="dot", line_color="red", annotation_text="Resistance R1")

            last_date = df["Date"].iloc[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
            fig.add_trace(go.Scatter(x=future_dates, y=future_pred, name="30-Day Forecast", line=dict(color="orange", width=2, dash="dash")))

            fig.update_layout(title=f"{user_input} - Actual vs Forecast + S/R Levels", xaxis_title="Date", yaxis_title="Level / Price", height=550)
            st.plotly_chart(fig, use_container_width=True)

            st.warning("""
            **⚠️ Disclaimer**  
            This is only an educational tool based on historical data.  
            It is **NOT financial advice**. Markets are risky.
            """)

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Try: Nifty 50, Sensex, Gold, Silver, Crude Oil, Reliance, TCS")
