# delta_hedging_full_v4.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(page_title="Greeks & Delta Hedging Simulator v4", layout="wide")

# ----------------------------
# Black-Scholes & Greeks
# ----------------------------
def d1_fn(S0, K, tau, r, sigma):
    return (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))

def d2_fn(d1, sigma, tau):
    return d1 - sigma * np.sqrt(tau)

def bs_price(option_type, S0, K, tau, r, sigma):
    # exact payoff at expiry
    if tau <= 0:
        return max(0.0, S0 - K) if option_type == "Call" else max(0.0, K - S0)
    d1 = d1_fn(S0, K, tau, r, sigma)
    d2 = d2_fn(d1, sigma, tau)
    if option_type == "Call":
        return S0 * stats.norm.cdf(d1) - K * np.exp(-r * tau) * stats.norm.cdf(d2)
    else:
        return K * np.exp(-r * tau) * stats.norm.cdf(-d2) - S0 * stats.norm.cdf(-d1)

def greeks_all(option_type, S0, K, tau, r, sigma):
    if tau <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    d1 = d1_fn(S0, K, tau, r, sigma)
    d2 = d2_fn(d1, sigma, tau)
    phi_d1 = stats.norm.pdf(d1)
    delta = stats.norm.cdf(d1) if option_type == "Call" else stats.norm.cdf(d1) - 1
    gamma = phi_d1 / (S0 * sigma * np.sqrt(tau))
    vega = S0 * phi_d1 * np.sqrt(tau)
    theta = ( - (S0 * phi_d1 * sigma) / (2 * np.sqrt(tau)) - r * K * np.exp(-r * tau) * stats.norm.cdf(d2) ) if option_type == "Call" else ( - (S0 * phi_d1 * sigma) / (2 * np.sqrt(tau)) + r * K * np.exp(-r * tau) * stats.norm.cdf(-d2) )
    rho = (K * tau * np.exp(-r * tau) * stats.norm.cdf(d2)) if option_type == "Call" else (-K * tau * np.exp(-r * tau) * stats.norm.cdf(-d2))
    return delta, gamma, vega, theta, rho

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
                jump_log = np.random.normal(mu_j, sigma_j) * k
                Jmult = np.exp(jump_log)
        path[i] = path[i-1] * np.exp(diffusion) * Jmult
    return path

def gen_two_paths_until_one_above_below(S0, mu, sigma, T, N, jump_on, lam, mu_j, sigma_j, K, seed=None, max_tries=1000):
    rng = np.random.RandomState(seed) if seed is not None else None
    for _ in range(max_tries):
        s1 = simulate_gbm_jump(S0, mu, sigma, T, N, jump_on, lam, mu_j, sigma_j, seed=(rng.randint(0,2**31-1) if rng else None))
        s2 = simulate_gbm_jump(S0, mu, sigma, T, N, jump_on, lam, mu_j, sigma_j, seed=(rng.randint(0,2**31-1) if rng else None))
        if (s1[-1] > K and s2[-1] < K) or (s1[-1] < K and s2[-1] > K):
            return (s1, s2) if s1[-1] > K else (s2, s1)
    # fallback
    return s1, s2

# ----------------------------
# Hedging & accounting (corrected)
# ----------------------------
def delta_hedge_table_and_pnl(path, option_type, side, K, T, N, r, sigma,
                              n_contracts, contract_size, hedge_freq_days, txn_cost_per_share):
    # ensure ints
    N = int(N)
    hedge_freq_days = int(hedge_freq_days)
    n_contracts = int(n_contracts)
    contract_size = int(contract_size)

    total_shares = n_contracts * contract_size
    sign = 1.0 if side == "Long" else -1.0
    days = np.arange(0, N+1)
    S_path = path.copy()

    # arrays
    delta_unit = np.zeros(N+1); gamma_unit = np.zeros(N+1)
    vega_unit = np.zeros(N+1); theta_unit = np.zeros(N+1); rho_unit = np.zeros(N+1)
    shares_in_portfolio = np.zeros(N+1)
    shares_purchased = np.zeros(N+1); cost_shares = np.zeros(N+1)
    tx_costs = np.zeros(N+1); cumulative_cash_outflow = np.zeros(N+1)
    cash_account = np.zeros(N+1)
    option_value = np.zeros(N+1); portfolio_value = np.zeros(N+1)

    dt = T / N

    # initial premium and cash
    premium_per_unit = bs_price(option_type, S_path[0], K, T, r, sigma)
    total_premium = premium_per_unit * total_shares
    # cash: long pays (outflow), short receives (inflow)
    cash_account[0] = -sign * total_premium
    cumulative_cash_outflow[0] = 0.0  # track net cash spent on hedging trades (exclude premium)
    # initial option mark-to-market (position included)
    option_value[0] = sign * premium_per_unit * total_shares
    portfolio_value[0] = option_value[0] + shares_in_portfolio[0] * S_path[0] + cash_account[0]

    # loop days
    for day in range(0, N+1):
        tau = (N - day) / N * T  # time remaining in years
        # Greeks
        if tau <= 0:
            d = g = v = th = rh = 0.0
        else:
            d,g,v,th,rh = greeks_all(option_type, S_path[day], K, tau, r, sigma)
        delta_unit[day] = d; gamma_unit[day] = g; vega_unit[day] = v; theta_unit[day] = th; rho_unit[day] = rh

        prev_shares = shares_in_portfolio[day-1] if day > 0 else 0.0
        # rebalancing
        if day % hedge_freq_days == 0:
            target_shares = - sign * delta_unit[day] * total_shares
            trade_shares = target_shares - prev_shares
            shares_purchased[day] = trade_shares
            cost_shares[day] = trade_shares * S_path[day]          # signed: positive if buy, negative if sell
            tx_costs[day] = abs(trade_shares) * txn_cost_per_share
            # update cumulative outflow (net): include signed cost_shares and tx costs
            cumulative_cash_outflow[day] = (cumulative_cash_outflow[day-1] if day>0 else 0.0) + cost_shares[day] + tx_costs[day]
            # update cash account: trade cash flows (buy reduces cash, sell increases cash), minus tx costs
            cash_account[day] = (cash_account[day-1] if day>0 else cash_account[0]) - cost_shares[day] - tx_costs[day]
            shares_in_portfolio[day] = target_shares
        else:
            shares_purchased[day] = 0.0
            cost_shares[day] = 0.0
            tx_costs[day] = 0.0
            cumulative_cash_outflow[day] = cumulative_cash_outflow[day-1] if day>0 else 0.0
            cash_account[day] = cash_account[day-1] if day>0 else cash_account[0]
            shares_in_portfolio[day] = prev_shares

        # optional: accrue interest on cash (end-of-day)
        if r != 0.0 and day < N:
            cash_account[day] = cash_account[day] * np.exp(r * dt)

        # mark-to-market option position (position sign included)
        option_val_unit = bs_price(option_type, S_path[day], K, tau, r, sigma)
        option_value[day] = sign * option_val_unit * total_shares
        # portfolio (MTM)
        portfolio_value[day] = option_value[day] + shares_in_portfolio[day] * S_path[day] + cash_account[day]

    # FINAL LIQUIDATION (ensure we close underlying hedge and include its cash flows)
    S_T = S_path[-1]
    final_shares = shares_in_portfolio[-1]
    if abs(final_shares) > 0:
        final_trade = -final_shares
        final_cost = final_trade * S_T
        final_tx = abs(final_trade) * txn_cost_per_share
        # add these final trades to last-day records
        shares_purchased[-1] += final_trade
        cost_shares[-1] += final_cost
        tx_costs[-1] += final_tx
        cumulative_cash_outflow[-1] += final_cost + final_tx
        # update cash account (do not accrue further)
        cash_account[-1] = cash_account[-1] - final_cost - final_tx
        shares_in_portfolio[-1] = 0.0

    # final payoff (signed)
    payoff_per_unit = max(S_T - K, 0.0) if option_type == "Call" else max(K - S_T, 0.0)
    payoff_total_signed = payoff_per_unit * total_shares * sign

    # final portfolio by accounting (cash after liquidation + option payoff signed)
    final_portfolio_value = cash_account[-1] + payoff_total_signed
    # User-specified formula (matches after liquidation): for Long: -premium - cumulative_outflow + payoff_total
    cumulative_outflow_last = cumulative_cash_outflow[-1]
    if side == "Long":
        pnl_by_formula = - total_premium - cumulative_outflow_last + payoff_per_unit * total_shares
    else:
        pnl_by_formula = + total_premium - cumulative_outflow_last - payoff_per_unit * total_shares

    # consistency check (they should be equal or very close)
    # final_portfolio_value should equal pnl_by_formula (within rounding) if logic is consistent
    # Build DataFrame
    df = pd.DataFrame({
        "Day": days,
        "Stock Price": np.round(S_path, 6),
        "Tau (yrs)": np.round((N - days) / N * T, 8),
        "Delta (per unit)": np.round(delta_unit, 6),
        "Gamma (per unit)": np.round(gamma_unit, 8),
        "Vega (per unit)": np.round(vega_unit, 6),
        "Theta (per yr per unit)": np.round(theta_unit, 6),
        "Rho (per unit)": np.round(rho_unit, 6),
        "Shares in Portfolio": np.round(shares_in_portfolio, 6),
        "Shares Purchased (today)": np.round(shares_purchased, 6),
        "Cost of Shares Purchase": np.round(cost_shares, 6),
        "Transaction Cost": np.round(tx_costs, 6),
        "Cumulative Cash Outflow": np.round(cumulative_cash_outflow, 6),
        "Cash Account": np.round(cash_account, 6),
        "Option Value (position)": np.round(option_value, 6),
        "Portfolio Value (MTM)": np.round(portfolio_value, 6)
    }).set_index("Day")

    info = {
        "premium_per_unit": premium_per_unit,
        "total_premium": total_premium,
        "payoff_per_unit": payoff_per_unit,
        "payoff_total_signed": payoff_total_signed,
        "cumulative_outflow_last": cumulative_outflow_last,
        "final_portfolio_value": final_portfolio_value,
        "pnl_by_formula": pnl_by_formula
    }
    return df, info

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Greeks & Delta Hedging Simulator v4 (corrected accounting + PnL time series)")
st.sidebar.header("Market & Option Parameters")

S0 = st.sidebar.number_input("Spot Price S0", value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price K", value=100.0, step=1.0)
T = float(st.sidebar.number_input("Time to Expiration (Years)", value=1.0, step=0.01))
r = float(st.sidebar.number_input("Risk-free Rate (%)", value=0.0, step=0.1)) / 100.0
sigma = float(st.sidebar.number_input("Volatility σ (%)", value=20.0, step=0.5)) / 100.0
mu = float(st.sidebar.number_input("Expected Return μ (%)", value=0.0, step=0.1)) / 100.0

st.sidebar.subheader("Contracts & Costs")
n_contracts = int(st.sidebar.number_input("Number of Option Contracts", min_value=1, value=10, step=1))
contract_size = int(st.sidebar.number_input("Contract Size (shares/contract)", min_value=1, value=100, step=1))
txn_cost_per_share = float(st.sidebar.number_input("Transaction cost per share ($)", min_value=0.0, value=0.0, step=0.01))

st.sidebar.subheader("Hedging")
hedge_freq_days = int(st.sidebar.selectbox("Rebalance every (days)", [1,5,21], index=0))

st.sidebar.subheader("Stochastic scenario")
N = int(st.sidebar.number_input("Trading days per year (N)", min_value=10, value=252, step=1))

use_jumps = st.sidebar.checkbox("Use Merton jump-diffusion", value=False)
if use_jumps:
    lam = float(st.sidebar.number_input("Jump intensity λ (per year)", min_value=0.0, value=0.5, step=0.1))
    mu_j = float(st.sidebar.number_input("Jump log-mean μ_J", value=-0.05, step=0.01))
    sigma_j = float(st.sidebar.number_input("Jump log-stdev σ_J", value=0.1, step=0.01))
else:
    lam = mu_j = sigma_j = 0.0

seed = int(st.sidebar.number_input("Random seed (0=random)", value=0, step=1))
seed_use = None if seed == 0 else int(seed)

# ----------------------------
# Run independent cases
# ----------------------------
cases = [("Call", "Long"), ("Call", "Short"), ("Put", "Long"), ("Put", "Short")]
results = {}

with st.spinner("Running simulations for 4 independent cases..."):
    for idx, (opt, side) in enumerate(cases):
        seed_case = None if seed_use is None else seed_use + idx*10
        path_up, path_down = gen_two_paths_until_one_above_below(S0, mu, sigma, T, N, use_jumps, lam, mu_j, sigma_j, K, seed_case)
        df_up, info_up = delta_hedge_table_and_pnl(path_up, opt, side, K, T, N, r, sigma,
                                                   n_contracts, contract_size, hedge_freq_days, txn_cost_per_share)
        df_down, info_down = delta_hedge_table_and_pnl(path_down, opt, side, K, T, N, r, sigma,
                                                       n_contracts, contract_size, hedge_freq_days, txn_cost_per_share)
        results[(opt, side)] = {"path_up": path_up, "path_down": path_down,
                                "df_up": df_up, "df_down": df_down,
                                "info_up": info_up, "info_down": info_down}

st.success("Simulations complete.")

# ----------------------------
# PLOTS: PnL progression for selected case
# ----------------------------
st.header("PnL progression over time (per selected case, up vs down)")

sel_opt = st.selectbox("Select Option Type", ["Call", "Put"])
sel_side = st.selectbox("Select Side", ["Long", "Short"])
res = results[(sel_opt, sel_side)]
time = np.linspace(0, T, N+1)

fig, ax = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
# stock paths
ax[0].plot(time, res["path_up"], label=f"Up Path (ST={res['path_up'][-1]:.2f})")
ax[0].plot(time, res["path_down"], label=f"Down Path (ST={res['path_down'][-1]:.2f})", linestyle='--')
ax[0].axhline(K, color='gray', linestyle=':', linewidth=1)
ax[0].set_title("Stock Price Paths")
ax[0].legend(fontsize='small'); ax[0].grid(True)

# portfolio (MTM) = PnL progression
ax[1].plot(time, res["df_up"]["Portfolio Value (MTM)"].values, label="Portfolio Value (Up)")
ax[1].plot(time, res["df_down"]["Portfolio Value (MTM)"].values, label="Portfolio Value (Down)", linestyle='--')
ax[1].set_title("PnL progression (hedged portfolio MTM) — Up vs Down")
ax[1].legend(fontsize='small'); ax[1].grid(True)

st.pyplot(fig)

# ----------------------------
# Detailed boards
# ----------------------------
st.header("Detailed daily boards (each case independent)")
for opt, side in cases:
    with st.expander(f"{opt} - {side}", expanded=False):
        r = results[(opt, side)]
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Up scenario")
            st.dataframe(r["df_up"], height=300)
            st.markdown(f"- Premium per unit: ${r['info_up']['premium_per_unit']:.6f}")
            st.markdown(f"- Final portfolio (accounting): ${r['info_up']['final_portfolio_value']:.2f}")
            st.markdown(f"- Final PnL (formula): ${r['info_up']['pnl_by_formula']:.2f}")
        with c2:
            st.subheader("Down scenario")
            st.dataframe(r["df_down"], height=300)
            st.markdown(f"- Premium per unit: ${r['info_down']['premium_per_unit']:.6f}")
            st.markdown(f"- Final portfolio (accounting): ${r['info_down']['final_portfolio_value']:.2f}")
            st.markdown(f"- Final PnL (formula): ${r['info_down']['pnl_by_formula']:.2f}")

st.markdown("""
**Notes**
- Each scenario (Up/Down) is simulated independently per case.
- `Portfolio Value (MTM)` is the hedged portfolio mark-to-market: option position (signed) + hedge shares + cash account (incl. premium & trades & liquidation).
- Final PnL by formula equals the final accounting value after we force liquidation at maturity (so results are consistent).
""")
