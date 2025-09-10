# delta_hedging_full_v2.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(page_title="Greeks & Delta Hedging Simulator v2", layout="wide")

# ----------------------------
# Black-Scholes & Greeks (per-unit)
# ----------------------------
def d1_fn(S0, K, tau, r, sigma):
    tau = max(tau, 1e-12)
    return (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))

def d2_fn(d1, sigma, tau):
    return d1 - sigma * np.sqrt(tau)

def bs_price(option_type, S0, K, tau, r, sigma):
    tau = max(tau, 1e-12)
    d1 = d1_fn(S0, K, tau, r, sigma)
    d2 = d2_fn(d1, sigma, tau)
    if option_type == "Call":
        return S0 * stats.norm.cdf(d1) - K * np.exp(-r * tau) * stats.norm.cdf(d2)
    else:
        return K * np.exp(-r * tau) * stats.norm.cdf(-d2) - S0 * stats.norm.cdf(-d1)

def greeks_all(option_type, S0, K, tau, r, sigma):
    tau = max(tau, 1e-12)
    d1 = d1_fn(S0, K, tau, r, sigma)
    d2 = d2_fn(d1, sigma, tau)
    phi_d1 = stats.norm.pdf(d1)
    # Delta
    delta = stats.norm.cdf(d1) if option_type == "Call" else stats.norm.cdf(d1) - 1
    # Gamma
    gamma = phi_d1 / (S0 * sigma * np.sqrt(tau))
    # Vega
    vega = S0 * phi_d1 * np.sqrt(tau)
    # Theta (per year)
    if option_type == "Call":
        theta = ( - (S0 * phi_d1 * sigma) / (2 * np.sqrt(tau)) - r * K * np.exp(-r * tau) * stats.norm.cdf(d2) )
    else:
        theta = ( - (S0 * phi_d1 * sigma) / (2 * np.sqrt(tau)) + r * K * np.exp(-r * tau) * stats.norm.cdf(-d2) )
    # Rho
    if option_type == "Call":
        rho = K * tau * np.exp(-r * tau) * stats.norm.cdf(d2)
    else:
        rho = -K * tau * np.exp(-r * tau) * stats.norm.cdf(-d2)
    return delta, gamma, vega, theta, rho

GREEK_DESCRIPTIONS = {
    "Delta": "Δ — sensitivity of option price to a small change in the underlying price (dPrice/dS).",
    "Gamma": "Γ — rate of change of Delta as the underlying price changes (dΔ/dS).",
    "Vega": "ν — sensitivity of option price to volatility changes (dPrice/dσ).",
    "Theta": "Θ — time decay; sensitivity of option price to passage of time (dPrice/dt).",
    "Rho": "ρ — sensitivity of option price to a change in the risk-free rate (dPrice/dr)."
}

# ----------------------------
# GBM + optional Merton jumps
# ----------------------------
def simulate_gbm_jump(S0, mu, sigma, T, N, jump_on=False, lam=0.0, mu_j=0.0, sigma_j=0.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = T / N
    path = np.zeros(N+1)
    path[0] = S0
    for i in range(1, N+1):
        Z = np.random.normal()
        diffusion = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        Jmult = 1.0
        if jump_on:
            k = np.random.poisson(lam * dt)
            if k > 0:
                # compound jump: product of k lognormal multipliers
                jump_log = np.random.normal(mu_j, sigma_j) * k
                Jmult = np.exp(jump_log)
        path[i] = path[i-1] * np.exp(diffusion) * Jmult
    return path

def gen_two_paths_until_one_above_below(S0, mu, sigma, T, N, jump_on, lam, mu_j, sigma_j, seed=None, max_tries=1000):
    tries = 0
    rng = np.random.RandomState(seed) if seed is not None else None
    while tries < max_tries:
        if rng is None:
            p1 = simulate_gbm_jump(S0, mu, sigma, T, N, jump_on, lam, mu_j, sigma_j)
            p2 = simulate_gbm_jump(S0, mu, sigma, T, N, jump_on, lam, mu_j, sigma_j)
        else:
            p1 = simulate_gbm_jump(S0, mu, sigma, T, N, jump_on, lam, mu_j, sigma_j, seed=rng.randint(0,2**31-1))
            p2 = simulate_gbm_jump(S0, mu, sigma, T, N, jump_on, lam, mu_j, sigma_j, seed=rng.randint(0,2**31-1))
        cond1 = (p1[-1] > K and p2[-1] < K)
        cond2 = (p1[-1] < K and p2[-1] > K)
        if cond1 or cond2:
            if p1[-1] > K:
                return p1, p2
            else:
                return p2, p1
        tries += 1
    # fallback
    return p1, p2

# ----------------------------
# Hedging accounting: produce day-by-day DataFrame and final PnL per user formula
# ----------------------------
def delta_hedge_table_and_pnl(path, option_type, side, K, T, r, sigma, n_contracts, contract_size,
                              hedge_freq_days, txn_cost_per_share):
    N = len(path) - 1
    dt = 1.0 / N  # because tau = (N-day)/N
    # arrays
    days = np.arange(0, N+1)
    stock_price = path.copy()
    delta_unit = np.zeros(N+1)
    shares_in_portfolio = np.zeros(N+1)
    shares_purchased = np.zeros(N+1)
    cost_shares = np.zeros(N+1)
    tx_costs = np.zeros(N+1)
    cumulative_cash_outflow = np.zeros(N+1)

    total_shares_controlled = n_contracts * contract_size
    sign = 1.0 if side == "Long" else -1.0

    # premium per unit and total
    premium_per_unit = bs_price(option_type, stock_price[0], K, 1.0, r, sigma)  # tau at day 0 = 1
    total_premium = premium_per_unit * total_shares_controlled

    # initial settings: no shares, no cash flows yet; we record initial cumulative cash outflow as -premium if long (since user defined cumulative positive outflow)
    cumulative_cash_outflow[0] = total_premium if sign == 1 else 0.0  # user wanted cumulative outflow to include premium for long (negative cash outflow)
    # Loop days (we will compute Greeks and trades based on tau = (N - day)/N)
    for day in range(0, N+1):
        tau = (N - day) / N
        tau = max(tau, 1e-12)
        # Greeks per unit
        d, g, v, th, rh = greeks_all(option_type, stock_price[day], K, tau, r, sigma)
        delta_unit[day] = d
        gamma_unit[day] = g
        vega_unit[day] = v
        theta_unit[day] = th
        rho_unit[day] = rh

        # we perform rebalancing at the start of the day based on delta computed using today's tau
        if day == 0:
            prev_shares = 0.0
        else:
            prev_shares = shares_in_portfolio[day-1]

        if day % hedge_freq_days == 0:
            # target hedge shares (aggregate) = - sign * delta_unit * total_shares_controlled
            target_shares = - sign * delta_unit[day] * total_shares_controlled
            trade_shares = target_shares - prev_shares
            shares_purchased[day] = trade_shares
            cost_shares[day] = trade_shares * stock_price[day]
            tx_costs[day] = abs(trade_shares) * txn_cost_per_share
            cumulative_cash_outflow[day] = cumulative_cash_outflow[day-1] + (abs(cost_shares[day]) + tx_costs[day]) if day>0 else cumulative_cash_outflow[0] + (abs(cost_shares[day]) + tx_costs[day])
            shares_in_portfolio[day] = target_shares
        else:
            shares_purchased[day] = 0.0
            cost_shares[day] = 0.0
            tx_costs[day] = 0.0
            cumulative_cash_outflow[day] = cumulative_cash_outflow[day-1] if day>0 else cumulative_cash_outflow[0]
            shares_in_portfolio[day] = prev_shares

    # final settlement values
    S_T = stock_price[-1]
    if option_type == "Call":
        payoff_per_unit = max(S_T - K, 0.0)
    else:
        payoff_per_unit = max(K - S_T, 0.0)
    payoff_total = payoff_per_unit * sign * total_shares_controlled

    # user-specified PnL formula:
    # Long: PnL = - total_premium - cumulative_cash_outflow_last + payoff_total
    # Short: PnL = + total_premium - cumulative_cash_outflow_last - payoff_total
    cumulative_outflow_last = cumulative_cash_outflow[-1]
    if side == "Long":
        pnl_by_formula = - total_premium - cumulative_outflow_last + payoff_per_unit * total_shares_controlled
    else:
        pnl_by_formula = + total_premium - cumulative_outflow_last - payoff_per_unit * total_shares_controlled

    # Build DataFrame
    df = pd.DataFrame({
        "Day": days,
        "Stock Price": np.round(stock_price, 6),
        "Delta (per unit)": np.round(delta_unit, 6),
        "Shares in Portfolio": np.round(shares_in_portfolio, 6),
        "Shares Purchased (today)": np.round(shares_purchased, 6),
        "Cost of Shares Purchase": np.round(cost_shares, 6),
        "Transaction Cost": np.round(tx_costs, 6),
        "Cumulative Cash Outflow": np.round(cumulative_cash_outflow, 6),
    })
    df.set_index("Day", inplace=True)
    info = {
        "premium_per_unit": premium_per_unit,
        "total_premium": total_premium,
        "payoff_per_unit": payoff_per_unit,
        "payoff_total_signed": payoff_total,
        "pnl_by_formula": pnl_by_formula,
        "cumulative_outflow_last": cumulative_outflow_last
    }
    return df, info

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Greeks & Delta Hedging Simulator (per-scenario, S0 naming)")

# Sidebar inputs
st.sidebar.header("Market & Option Parameters")
S0 = st.sidebar.number_input("Spot Price S0", value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price K", value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Expiration (Years)", value=1.0, min_value=0.01, step=0.01)
r = st.sidebar.number_input("Risk-free Rate (%)", value=0.0, step=0.1) / 100.0
sigma = st.sidebar.number_input("Volatility σ (%)", value=20.0, step=0.5) / 100.0
mu = st.sidebar.number_input("Expected return μ (%) (for GBM)", value=0.0, step=0.1) / 100.0

st.sidebar.subheader("Contracts & Costs")
n_contracts = st.sidebar.number_input("Number of Option Contracts", min_value=1, value=10, step=1)
contract_size = st.sidebar.number_input("Contract Size (shares per contract)", min_value=1, value=100, step=1)
txn_cost_per_share = st.sidebar.number_input("Transaction cost per share ($)", min_value=0.0, value=0.0, step=0.01)

st.sidebar.subheader("Hedging")
hedge_freq_days = st.sidebar.selectbox("Rebalance every (days)", [1, 5, 21], index=0, format_func=lambda x: f"Every {x} day(s)")

st.sidebar.subheader("Stochastic scenario")
N = st.sidebar.number_input("Trading days per year (N)", min_value=10, value=252, step=1)

use_jumps = st.sidebar.checkbox("Use Merton jump-diffusion", value=False)
if use_jumps:
    lam = st.sidebar.number_input("Jump intensity λ (per year)", min_value=0.0, value=0.5, step=0.1)
    mu_j = st.sidebar.number_input("Jump log-mean (μ_J)", value=-0.05, step=0.01)
    sigma_j = st.sidebar.number_input("Jump log-stdev (σ_J)", value=0.1, step=0.01)
else:
    lam = 0.0; mu_j = 0.0; sigma_j = 0.0

seed = st.sidebar.number_input("Random seed (0 for random)", value=0, step=1)
seed_use = None if seed == 0 else int(seed)

# ----------------------------
# Compute results for each of 4 cases independently
# ----------------------------
cases = [("Call", "Long"), ("Call", "Short"), ("Put", "Long"), ("Put", "Short")]
results = {}

with st.spinner("Running scenario simulations (each case independently)..."):
    for (opt, side) in cases:
        # generate two paths for this case independently
        path_up, path_down = gen_two_paths_until_one_above_below(S0, mu, sigma, T, N, use_jumps, lam, mu_j, sigma_j, seed_use)
        df_up, info_up = delta_hedge_table_and_pnl(path_up, opt, side, K, T, r, sigma, n_contracts, contract_size, hedge_freq_days, txn_cost_per_share)
        df_down, info_down = delta_hedge_table_and_pnl(path_down, opt, side, K, T, r, sigma, n_contracts, contract_size, hedge_freq_days, txn_cost_per_share)
        results[(opt, side)] = {
            "path_up": path_up, "path_down": path_down,
            "df_up": df_up, "df_down": df_down,
            "info_up": info_up, "info_down": info_down
        }

st.success("Simulations complete.")

# ----------------------------
# Compact plotting layout
# ----------------------------
st.header("Compact Visuals")

# Choose which case to inspect
col_sel1, col_sel2 = st.columns([1,1])
with col_sel1:
    sel_opt = st.selectbox("Inspect Option Type", ["Call", "Put"])
with col_sel2:
    sel_side = st.selectbox("Inspect Side", ["Long", "Short"])
sel = (sel_opt, sel_side)
res = results[sel]

time = np.linspace(0, T, N+1)

fig = plt.figure(constrained_layout=True, figsize=(14, 10))
gs = fig.add_gridspec(3, 3)

# top-left: stock paths compact
ax0 = fig.add_subplot(gs[0, :])
label_up = f"Path A (ST={res['path_up'][-1]:.2f}) - UP (ST > K)"
label_down = f"Path B (ST={res['path_down'][-1]:.2f}) - DOWN (ST < K)"
ax0.plot(time, res['path_up'], label=label_up, linewidth=1.5)
ax0.plot(time, res['path_down'], label=label_down, linewidth=1.5, linestyle='--')
ax0.axhline(K, color='gray', linestyle=':', linewidth=1)
ax0.set_title("Stock Price Paths (compact)")
ax0.legend(loc='upper left', fontsize='small')
ax0.grid(True)

# middle-left: portfolio values
ax1 = fig.add_subplot(gs[1, :])
ax1.plot(time, res['df_up']["Cumulative Cash Outflow"].values, label='Cumulative Cash Outflow (Up)')
ax1.plot(time, res['df_down']["Cumulative Cash Outflow"].values, label='Cumulative Cash Outflow (Down)')
ax1.set_title("Cumulative Cash Outflow over Time")
ax1.legend(fontsize='small'); ax1.grid(True)

# bottom: Greeks time series (5 small plots)
ax_g1 = fig.add_subplot(gs[2, 0])
ax_g2 = fig.add_subplot(gs[2, 1])
ax_g3 = fig.add_subplot(gs[2, 2])

# Delta (use up scenario as example)
ax_g1.plot(time, res['df_up']["Delta (per unit)"].values, label='Delta (up)')
ax_g1.plot(time, res['df_down']["Delta (per unit)"].values, label='Delta (down)', linestyle='--')
ax_g1.set_title("Delta evolution"); ax_g1.grid(True); ax_g1.legend(fontsize='x-small')

# Gamma
ax_g2.plot(time, res['df_up']["Gamma (per unit)"].values, label='Gamma (up)')
ax_g2.plot(time, res['df_down']["Gamma (per unit)"].values, label='Gamma (down)', linestyle='--')
ax_g2.set_title("Gamma evolution"); ax_g2.grid(True); ax_g2.legend(fontsize='x-small')

# Vega (overlay up/down)
ax_g3.plot(time, res['df_up']["Vega (per unit)"].values, label='Vega (up)')
ax_g3.plot(time, res['df_down']["Vega (per unit)"].values, label='Vega (down)', linestyle='--')
ax_g3.set_title("Vega evolution"); ax_g3.grid(True); ax_g3.legend(fontsize='x-small')

# create a smaller figure for Theta and Rho below
fig2, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(10,2.5), constrained_layout=True)
ax_t.plot(time, res['df_up']["Theta (per yr per unit)"].values, label='Theta (up)')
ax_t.plot(time, res['df_down']["Theta (per yr per unit)"].values, label='Theta (down)', linestyle='--')
ax_t.set_title("Theta evolution"); ax_t.grid(True); ax_t.legend(fontsize='x-small')

ax_r.plot(time, res['df_up']["Rho (per unit)"].values, label='Rho (up)')
ax_r.plot(time, res['df_down']["Rho (per unit)"].values, label='Rho (down)', linestyle='--')
ax_r.set_title("Rho evolution"); ax_r.grid(True); ax_r.legend(fontsize='x-small')

st.pyplot(fig)
st.pyplot(fig2)

# descriptions for Greeks (short sentences)
st.markdown("**Greeks descriptions (general):**")
for name, desc in GREEK_DESCRIPTIONS.items():
    st.write(f"**{name}** — {desc}")

# ----------------------------
# Detailed boards for all 4 cases (expanders)
# ----------------------------
st.header("Detailed daily boards (each case independent)")

for opt, side in cases:
    with st.expander(f"{opt} - {side}", expanded=False):
        res = results[(opt, side)]
        colL, colR = st.columns(2)
        with colL:
            st.subheader("Up scenario (ST > K)")
            st.dataframe(res['df_up'], height=300)
            st.markdown(f"- Premium per unit: ${res['info_up']['premium_per_unit']:.6f}")
            st.markdown(f"- Total premium: ${res['info_up']['total_premium']:.2f}")
            st.markdown(f"- Payoff per unit at T: ${res['info_up']['payoff_per_unit']:.6f}")
            st.markdown(f"- Final PnL (by formula): ${res['info_up']['pnl_by_formula']:.2f}")
        with colR:
            st.subheader("Down scenario (ST < K)")
            st.dataframe(res['df_down'], height=300)
            st.markdown(f"- Premium per unit: ${res['info_down']['premium_per_unit']:.6f}")
            st.markdown(f"- Total premium: ${res['info_down']['total_premium']:.2f}")
            st.markdown(f"- Payoff per unit at T: ${res['info_down']['payoff_per_unit']:.6f}")
            st.markdown(f"- Final PnL (by formula): ${res['info_down']['pnl_by_formula']:.2f}")

# ----------------------------
# Summary metrics for selected case
# ----------------------------
st.header("Selected-case summary")
res_sel = results[(sel_opt, sel_side)]
st.metric("Selected case", f"{sel_opt} - {sel_side}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Final PnL (Up)", f"${res_sel['info_up']['pnl_by_formula']:.2f}")
col2.metric("Final PnL (Down)", f"${res_sel['info_down']['pnl_by_formula']:.2f}")
col3.metric("Total Premium (Up)", f"${res_sel['info_up']['total_premium']:.2f}")
col4.metric("Total Premium (Down)", f"${res_sel['info_down']['total_premium']:.2f}")

st.markdown("""
**Notes**
- Each case (Call/Put × Long/Short) was simulated independently: two paths were generated for that case until one ended above K and the other below K.
- Time-to-maturity used per day: `tau = (N - day) / N`.
- Final PnL follows the user-specified formula (premium & cumulative cash outflow + payoff).
- Transaction costs are flat $/share applied to each trade.
- Jumps (Merton) are simulated in the underlying when enabled; Black–Scholes is still used for per-day marking/greeks.
""")
