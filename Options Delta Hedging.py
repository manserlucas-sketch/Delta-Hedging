# delta_hedging_simulator.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(page_title="Delta Hedging Simulator", layout="wide")

# ----------------------------
# 1. Black-Scholes & Greeks
# ----------------------------
def calculate_d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def calculate_d2(d1, sigma, T):
    return d1 - sigma * np.sqrt(T)

def black_scholes_price(option_type, S, K, T, r, sigma):
    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(d1, sigma, T)
    if option_type == "Call":
        return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

def calculate_delta(option_type, S, K, T, r, sigma):
    d1 = calculate_d1(S, K, T, r, sigma)
    return stats.norm.cdf(d1) if option_type == "Call" else stats.norm.cdf(d1) - 1

# ----------------------------
# 2. GBM Simulation
# ----------------------------
def simulate_gbm(S0, mu, sigma, T, N):
    dt = T / N
    S = np.zeros(N+1)
    S[0] = S0
    for t in range(1, N+1):
        Z = np.random.normal()
        S[t] = S[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    return S

def generate_two_paths_condition(S0, mu, sigma, T, N):
    while True:
        path1 = simulate_gbm(S0, mu, sigma, T, N)
        path2 = simulate_gbm(S0, mu, sigma, T, N)
        if (path1[-1] > S0 and path2[-1] < S0) or (path1[-1] < S0 and path2[-1] > S0):
            return path1, path2

# ----------------------------
# 3. Delta Hedging Simulator
# ----------------------------
def delta_hedging(path, option_type, position, K, T, r, sigma, hedge_freq=1, txn_cost=0.0):
    N = len(path) - 1
    dt = T / N
    shares = 0.0
    cash = 0.0
    pnl = []

    # Option premium at t=0
    premium = black_scholes_price(option_type, path[0], K, T, r, sigma)
    if position.lower() == "long":
        cash -= premium
        sign = 1
    else:  # short
        cash += premium
        sign = -1

    for t in range(N):
        tau = T - t*dt
        if tau <= 0:
            tau = 1e-10  # prevent division by zero

        # compute delta and adjust sign for position
        delta = calculate_delta(option_type, path[t], K, tau, r, sigma) * sign

        # target hedge shares (rebalance every hedge_freq steps)
        if t % hedge_freq == 0:
            hedge_target = -delta
            trade_shares = hedge_target - shares
            cash -= trade_shares * path[t] + abs(trade_shares) * txn_cost
            shares = hedge_target

        # portfolio value
        option_val = black_scholes_price(option_type, path[t], K, tau, r, sigma) * sign
        portfolio = option_val + shares*path[t] + cash
        pnl.append(portfolio)

    # final step: settle option and liquidate shares
    S_T = path[-1]
    if option_type == "Call":
        payoff = max(S_T - K, 0) * sign
    else:
        payoff = max(K - S_T, 0) * sign
    portfolio += shares*S_T + cash  # ensure all positions liquidated
    pnl.append(portfolio)
    return np.array(pnl)

# ----------------------------
# 4. Streamlit UI
# ----------------------------
st.title("🎯 Delta Hedging Simulator")

# Sidebar inputs
st.sidebar.header("Option & Market Parameters")
S0 = st.sidebar.number_input("Spot Price S0", value=100.0)
K = st.sidebar.number_input("Strike Price K", value=100.0)
T = st.sidebar.number_input("Time to Expiration (Years)", value=1.0, min_value=0.01)
r = st.sidebar.number_input("Risk-free Rate (%)", value=0.0)/100
sigma = st.sidebar.number_input("Volatility (%)", value=20.0)/100
mu = st.sidebar.number_input("Expected Return μ (%)", value=0.0)/100

option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"])
position = st.sidebar.selectbox("Position", ["Long", "Short"])
txn_cost = st.sidebar.number_input("Transaction Cost per Share", value=0.0)
hedge_freq = st.sidebar.selectbox("Hedge Frequency", [1, 5, 21], format_func=lambda x: f"Every {x} day(s)")

N = 252  # trading days

# Generate GBM paths
path_up, path_down = generate_two_paths_condition(S0, mu, sigma, T, N)
time = np.linspace(0, T, N+1)

# Run hedging simulation
pnl_up = delta_hedging(path_up, option_type, position, K, T, r, sigma, hedge_freq, txn_cost)
pnl_down = delta_hedging(path_down, option_type, position, K, T, r, sigma, hedge_freq, txn_cost)

# ----------------------------
# 5. Plots
# ----------------------------
st.header("📈 Simulation Results")

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Stock paths
ax[0].plot(time, path_up, label="Stock Path Up")
ax[0].plot(time, path_down, label="Stock Path Down")
ax[0].set_ylabel("Stock Price")
ax[0].set_title("Simulated Stock Price Paths (GBM)")
ax[0].legend()
ax[0].grid(True)

# Hedged PnL
ax[1].plot(time, pnl_up[:N+1], label="Hedged Portfolio PnL Up")
ax[1].plot(time, pnl_down[:N+1], label="Hedged Portfolio PnL Down")
ax[1].set_ylabel("Portfolio Value ($)")
ax[1].set_xlabel("Time (Years)")
ax[1].set_title("Delta-Hedged Portfolio Value Over Time")
ax[1].legend()
ax[1].grid(True)

st.pyplot(fig)

# ----------------------------
# 6. Key Metrics
# ----------------------------
st.header("📊 Hedging Metrics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Final PnL (Up)", f"${pnl_up[-1]:.2f}")
with col2:
    st.metric("Final PnL (Down)", f"${pnl_down[-1]:.2f}")
with col3:
    st.metric("Total Transaction Costs (Up)", f"${abs(pnl_up[-1]-pnl_up[0]-S0):.2f}")
with col4:
    st.metric("Total Transaction Costs (Down)", f"${abs(pnl_down[-1]-pnl_down[0]-S0):.2f}")

st.markdown("*Option premium is included in the initial cash account.*")
