import streamlit as st
import yfinance as yf
import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands

from sklearn.ensemble import RandomForestClassifier

# ==========================================
# NSE AI STOCK ANALYSER
# ==========================================

st.set_page_config(
    page_title="NSE AI Analyzer",
    page_icon="📈",
    layout="centered"
)

st.title("📈 NSE AI Stock Analyzer")
st.caption("AI-powered technical analysis of NSE stocks")

# ------------------------------------------
# STOCK INPUT
# ------------------------------------------

symbol = st.text_input(
    "Enter NSE stock symbol",
    value="RELIANCE"
).strip().upper()

analyze = st.button(
    "🔍 Analyze Stock",
    use_container_width=True
)

if analyze and symbol:

    ticker = symbol + ".NS"

    with st.spinner("Downloading NSE data and training AI..."):

        data = yf.download(
            ticker,
            period="5y",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if data.empty:
            st.error(
                "Could not download data. "
                "Please check the NSE symbol."
            )
            st.stop()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.dropna()

        # ----------------------------------
        # TECHNICAL INDICATORS
        # ----------------------------------

        data["SMA20"] = data["Close"].rolling(20).mean()
        data["SMA50"] = data["Close"].rolling(50).mean()

        data["EMA20"] = EMAIndicator(
            close=data["Close"],
            window=20
        ).ema_indicator()

        data["RSI"] = RSIIndicator(
            close=data["Close"],
            window=14
        ).rsi()

        macd = MACD(
            close=data["Close"],
            window_slow=26,
            window_fast=12,
            window_sign=9
        )

        data["MACD"] = macd.macd()
        data["MACD_SIGNAL"] = macd.macd_signal()

        bb = BollingerBands(
            close=data["Close"],
            window=20,
            window_dev=2
        )

        data["BB_HIGH"] = bb.bollinger_hband()
        data["BB_LOW"] = bb.bollinger_lband()

        data["Return"] = data["Close"].pct_change()

        # ----------------------------------
        # AI TARGET
        # ----------------------------------

        data["Future_Close"] = data["Close"].shift(-1)

        data["Target"] = (
            data["Future_Close"] > data["Close"]
        ).astype(int)

        data = data.dropna()

        features = [
            "Close",
            "Volume",
            "SMA20",
            "SMA50",
            "EMA20",
            "RSI",
            "MACD",
            "MACD_SIGNAL",
            "BB_HIGH",
            "BB_LOW",
            "Return"
        ]

        X = data[features]
        y = data["Target"]

        split = int(len(data) * 0.8)

        X_train = X.iloc[:split]
        y_train = y.iloc[:split]

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        )

        model.fit(X_train, y_train)

        latest = data.iloc[-1]

        latest_features = pd.DataFrame(
            [latest[features]],
            columns=features
        )

        prediction = model.predict(
            latest_features
        )[0]

        probabilities = model.predict_proba(
            latest_features
        )[0]

        down_probability = probabilities[0] * 100
        up_probability = probabilities[1] * 100

        # ----------------------------------
        # VALUES
        # ----------------------------------

        price = float(latest["Close"])
        rsi = float(latest["RSI"])
        sma20 = float(latest["SMA20"])
        sma50 = float(latest["SMA50"])
        ema20 = float(latest["EMA20"])

        macd_value = float(latest["MACD"])
        macd_signal = float(latest["MACD_SIGNAL"])

        direction = (
            "UP 📈"
            if prediction == 1
            else "DOWN 📉"
        )

        # ----------------------------------
        # SIGNAL
        # ----------------------------------

        if (
            prediction == 1
            and rsi < 70
            and price > sma20
            and sma20 > sma50
            and macd_value > macd_signal
        ):
            signal = "BUY 🟢"

        elif prediction == 0:
            signal = "AVOID 🔴"

        else:
            signal = "WATCH 🟡"

    # ======================================
    # DISPLAY
    # ======================================

    st.divider()

    st.header(f"{symbol} — AI Analysis")

    st.metric(
        "Latest Price",
        f"₹{price:,.2f}"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "AI Direction",
            direction
        )

    with col2:
        st.metric(
            "AI Signal",
            signal
        )

    st.subheader("🤖 AI Probability")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "UP Probability",
            f"{up_probability:.2f}%"
        )

    with col2:
        st.metric(
            "DOWN Probability",
            f"{down_probability:.2f}%"
        )

    st.subheader("📊 Technical Indicators")

    indicators = pd.DataFrame({
        "Indicator": [
            "RSI",
            "SMA 20",
            "SMA 50",
            "EMA 20",
            "MACD",
            "MACD Signal"
        ],
        "Value": [
            f"{rsi:.2f}",
            f"₹{sma20:,.2f}",
            f"₹{sma50:,.2f}",
            f"₹{ema20:,.2f}",
            f"{macd_value:.4f}",
            f"{macd_signal:.4f}"
        ]
    })

    st.dataframe(
        indicators,
        hide_index=True,
        use_container_width=True
    )

    # ======================================
    # PRICE CHART
    # ======================================

    st.subheader("📈 Price Chart")

    chart_data = data[
        ["Close", "SMA20", "SMA50"]
    ].tail(180)

    st.line_chart(chart_data)

    st.divider()

    st.warning(
        "This AI system is experimental and does "
        "not guarantee future stock prices. "
        "Do your own research before investing."
    )
