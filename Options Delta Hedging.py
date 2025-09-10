# delta_hedging_full.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(page_title="Greeks & Delta Hedging Simulator", layout="wide")

# ----------------------------
# 1. Black-Scholes & Greeks
# ----------------------------
def d1_fn(S, K, tau, r, sigma):
    # tau is time to expiry in years (>= small positive)
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))

def d2_fn(d1, sigma, tau):
    return d1 - sigma * np.sqrt(tau)

def bs_price(option_type, S, K, tau, r, sigma):
    tau = max(tau, 1e-12)
    d1 = d1_fn(S, K, tau, r, sigma)
    d2 = d2_fn(d1, sigma, tau)
    if option_type == "Call":
        return S * stats.norm.cdf(d1) - K * np.exp(-r * tau) * stats.norm.cdf(d2)
    else:
        return K * np.exp(-r * tau) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

def greeks_all(option_type, S, K, tau, r, sigma):
    tau = max(tau, 1e-12)
    d1 = d1_fn(S, K, tau, r, sigma)
    d2 = d2_fn(d1, sigma, tau)
    phi_d1 = stats.norm.pdf(d1)
    # Delta
    delta = stats.norm.cdf(d1) if option_type == "Call" else stats.norm.cdf(d1) - 1
    # Gamma
    gamma = phi_d1 / (S * sigma * np.sqrt(tau))
    # Vega (per 1 unit vol)
    vega = S * phi_d1 * np.sqrt(tau)
    # Theta (per year)
    if option_type == "Call":
        theta = ( - (S * phi_d1 * sigma) / (2 * np.sqrt(tau)) - r * K * np.exp(-r * tau) * stats.norm.cdf(d2) )
    else:
        theta = ( - (S * phi_d1 * sigma) / (2 * np.sqrt(tau)) + r * K * np.exp(-r * tau) * stats.norm.cdf(-d2) )
    # Rho (per 1 unit rate)
    if option_type == "Call":
        rho = K * tau * np.exp(-r * tau) * stats.norm.cdf(d2)
    else:
        rho = -K * tau * np.exp(-r * tau) * stats.norm.cdf(-d2)

    return delta, gamma, vega, theta, rho

# Short textual descriptions for Greeks
GREEK_DESCRIPTIONS = {
    "Delta": "Sensitivity of option price to a small change in the underlying stock price (dPrice / dS).",
    "Gamma": "Rate of change of Delta with respect to the underlying price (dDelta / dS).",
    "Vega": "Sensitivity of option price to a small change in volatility (dPrice / dσ).",
    "Theta": "Time decay: sensitivity of the option price to the passage of time (dPrice / dt).",
    "Rho": "Sensitivity of the option price to a small change in the risk-free interest rate (dPrice / dr)."
}

# ----------------------------
# 2. Stochastic Engines (GBM + optional Merton jumps)
# ----------------------------
def simulate_gbm_jump(S0, mu, sigma, T, N, jump_on=False, lam=0.0, mu_j=0.0, sigma_j=0.0, seed=None):
    """
    Simulate a single path with GBM and optional Merton jumps.
    lam: jump intensity (per year)
    mu_j, sigma_j: log-jump distribution parameters (mu_j is mean of log(J), sigma_j stdev)
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / N
    S = np.zeros(N + 1)
    S[0] = S0
    for t in range(1, N + 1):
        Z = np.random.normal()
        # diffusion part
        S_prev = S[t-1]
        diffusion = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        Jmult = 1.0
        if jump_on:
            # number of jumps this step (Poisson with mean lam*dt)
            k = np.random.poisson(lam * dt)
            if k > 0:
                # compound jump: product of k independent lognormal multipliers
                # log J ~ Normal(mu_j, sigma_j^2)
                # so multiplier = exp(sum of k normal rvs)
                jump_log = np.random.normal(mu_j, sigma_j) * k
                Jmult = np.exp(jump_log)
        S[t] = S_prev * np.exp(diffusion) * Jmult
    return S

def generate_two_paths_condition_K(S0, mu, sigma, T, N, jump_on=False, lam=0.0, mu_j=0.0, sigma_j=0.0):
    """
    Generate two fully stochastic paths such that one finishes above K and the other below K.
    We'll accept the current global K value supplied at call-time by passing it in.
    """
    # We'll need the strike to decide; to avoid global dependency, caller must ensure K is accessible.
    # This wrapper will be used below with K parameter known.
    raise NotImplementedError("Use the version called in the app below which has access to K.")

# ----------------------------
# 3. Hedging and accounting engine
# ----------------------------
def delta_hedge_accounting(path, option_type, side, K, T, r, sigma, n_contracts, contract_size,
                           hedge_freq_days, txn_cost_per_share, jump_on=False):
    """
    Run delta-hedging for given path and produce a DataFrame with requested columns:
    Day, Stock Price, Delta, Shares in Portfolio (hedge shares), Shares Purchased (today),
    Cost of Shares Purchase, Transaction Cost, Cumulative Cash Flow, Cash Account, Option Value, Portfolio Value
    side: "Long" or "Short" (relative to option)
    """
    N = len(path) - 1
    dt = T / N
    days = np.arange(0, N+1)  # day 0..N
    total_shares_controlled = n_contracts * contract_size

    # initialize arrays
    stock_price = path.copy()
    delta_arr = np.zeros(N+1)
    shares_in_portfolio = np.zeros(N+1)   # hedge shares (positive = long underlying)
    shares_purchased = np.zeros(N+1)      # net shares bought on that day (today's trade)
    cost_shares = np.zeros(N+1)           # cost_shares = shares_purchased * price
    tx_costs = np.zeros(N+1)
    cumulative_cash_outflow = np.zeros(N+1)  # cumulative cash outflow (as user requested)
    cash_account = np.zeros(N+1)           # cash account (incl premium and trades) - accrues interest
    option_value = np.zeros(N+1)
    portfolio_value = np.zeros(N+1)

    sign = 1.0 if side == "Long" else -1.0

    # initial premium (per option unit)
    premium_per_unit = bs_price(option_type, stock_price[0], K, T, r, sigma)
    total_premium = premium_per_unit * total_shares_controlled  # premium times number of shares controlled
    # cash: long pays premium, short receives premium
    cash_account[0] = -sign * total_premium
    cumulative_cash_outflow[0] = (-sign * total_premium) if sign==1 else 0.0
    # initial option value (position sign included)
    option_value[0] = sign * premium_per_unit * total_shares_controlled
    # no shares initially
    shares_in_portfolio[0] = 0.0

    # loop days
    for t in range(1, N+1):
        tau = T - (t-1) * dt  # time to expiry used to compute delta at beginning of day t-1
        tau = max(tau, 1e-12)
        # compute greeks/delta for the option per unit
        delta_unit = greeks_all(option_type, stock_price[t-1], K, tau, r, sigma)[0]
        delta_portfolio = sign * delta_unit * total_shares_controlled  # aggregate delta of the option position
        delta_arr[t-1] = delta_unit  # record delta per unit for day index t-1

        # decide if we rebalance today (rebalance at day indices that are multiples of hedge_freq_days)
        if (t-1) % hedge_freq_days == 0:
            # target hedge shares = - delta_portfolio
            hedge_target = - delta_portfolio
            # shares currently in portfolio:
            prev_shares = shares_in_portfolio[t-1]
            trade_shares = hedge_target - prev_shares
            # trade
            shares_purchased[t] = trade_shares
            cost_shares[t] = trade_shares * stock_price[t-1]
            tx_costs[t] = abs(trade_shares) * txn_cost_per_share
            # update cumulative cash outflow: previous + cost_shares + tx_costs (per user's definition)
            cumulative_cash_outflow[t] = cumulative_cash_outflow[t-1] + cost_shares[t] + tx_costs[t]
            # update cash account: pays for purchase (or receives proceeds if trade_shares negative)
            cash_account[t] = cash_account[t-1] - cost_shares[t] - tx_costs[t]
            # set new shares
            shares_in_portfolio[t] = hedge_target
        else:
            # no trade today: carry forward
            shares_purchased[t] = 0.0
            cost_shares[t] = 0.0
            tx_costs[t] = 0.0
            cumulative_cash_outflow[t] = cumulative_cash_outflow[t-1]
            # cash accrues interest from previous day to this day
            cash_account[t] = cash_account[t-1]
            shares_in_portfolio[t] = shares_in_portfolio[t-1]

        # accrue interest on cash between days (simple continuous compounding for dt)
        cash_account[t] = cash_account[t] * np.exp(r * dt)

        # update option value at end of day t-1 (or use price at t for mark-to-market)
        option_val_unit = bs_price(option_type, stock_price[t], K, T - t*dt, r, sigma)
        option_value[t] = sign * option_val_unit * total_shares_controlled

        # portfolio value at day t (option + underlying position + cash)
        portfolio_value[t] = option_value[t] + shares_in_portfolio[t] * stock_price[t] + cash_account[t]

        # record delta at final day too (for completeness)
        if t == N:
            delta_unit_final = greeks_all(option_type, stock_price[t], K, max(T - t*dt, 1e-12), r, sigma)[0]
            delta_arr[t] = delta_unit_final

    # final settlement at maturity: ensure final row shows payoff and liquidated shares
    S_T = stock_price[-1]
    # option payoff per unit
    if option_type == "Call":
        payoff_unit = max(S_T - K, 0.0)
    else:
        payoff_unit = max(K - S_T, 0.0)
    payoff_total = sign * payoff_unit * total_shares_controlled

    # liquidate shares at S_T: add proceeds
    liquidation = shares_in_portfolio[-1] * S_T
    final_cash = cash_account[-1]  # after accruals
    final_portfolio = payoff_total + liquidation + final_cash

    # overwrite the last row of portfolio_value to be final_portfolio (settlement)
    portfolio_value[-1] = final_portfolio
    option_value[-1] = payoff_total
    cumulative_cash_outflow[-1] = cumulative_cash_outflow[-2]  # last trade already accounted
    cash_account[-1] = final_cash

    # build DataFrame with requested columns for display
    df = pd.DataFrame({
        "Day": np.arange(0, N+1),
        "Stock Price": stock_price,
        "Delta (per unit)": np.round(delta_arr, 6),
        "Shares in Portfolio": np.round(shares_in_portfolio, 6),
        "Shares Purchased (today)": np.round(shares_purchased, 6),
        "Cost of Shares Purchase": np.round(cost_shares, 6),
        "Transaction Cost": np.round(tx_costs, 6),
        "Cumulative Cash Outflow": np.round(cumulative_cash_outflow, 6),
        "Cash Account": np.round(cash_account, 6),
        "Option Value (position)": np.round(option_value, 6),
        "Portfolio Value": np.round(portfolio_value, 6)
    })
    # ensure Day is index for better display
    df.set_index("Day", inplace=True)
    return df, {
        "premium_per_unit": premium_per_unit,
        "total_premium": total_premium,
        "final_portfolio_value": final_portfolio,
        "payoff_total": payoff_total,
        "liquidation": liquidation,
        "final_cash": final_cash
    }

# ----------------------------
# 4. Streamlit UI
# ----------------------------
st.title("📊 Greeks & Delta Hedging Simulator (single option, 4 position boards)")

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

st.sidebar.subheader("Simulation options")
seed = st.sidebar.number_input("Random seed (0 for random)", value=0, step=1)
if seed == 0:
    seed_use = None
else:
    seed_use = int(seed)

# ----------------------------
# 5. Show Greeks & descriptions
# ----------------------------
st.header("Option Snapshot and Greeks")
col1, col2 = st.columns([2, 1])

delta_u, gamma_u, vega_u, theta_u, rho_u = greeks_all("Call", S0, K, T, r, sigma)  # for display; use call as example
# compute for selected S? show both call and put maybe
call_price = bs_price("Call", S0, K, T, r, sigma)
put_price = bs_price("Put", S0, K, T, r, sigma)

with col1:
    st.subheader("Option snapshot (per unit)")
    st.write(f"Spot S0 = {S0:.2f}    Strike K = {K:.2f}    T = {T:.3f} years    r = {r:.3f}    σ = {sigma:.3f}")
    st.write(f"Call price (per unit): ${call_price:.4f}    |    Put price (per unit): ${put_price:.4f}")

with col2:
    st.subheader("Greeks (Call example)")
    st.markdown(f"**Delta:** {delta_u:.6f} — {GREEK_DESCRIPTIONS['Delta']}")
    st.markdown(f"**Gamma:** {gamma_u:.6f} — {GREEK_DESCRIPTIONS['Gamma']}")
    st.markdown(f"**Vega:** {vega_u:.6f} — {GREEK_DESCRIPTIONS['Vega']}")
    st.markdown(f"**Theta:** {theta_u:.6f} (per year) — {GREEK_DESCRIPTIONS['Theta']}")
    st.markdown(f"**Rho:** {rho_u:.6f} — {GREEK_DESCRIPTIONS['Rho']}")

# ----------------------------
# 6. Generate two trajectories (one ends above K, the other below K)
# ----------------------------
# We'll repeatedly sample two paths until one ends above K and the other below K.
def gen_two_paths_until_one_above_below(S0, mu, sigma, T, N, jump_on, lam, mu_j, sigma_j, seed=None, max_tries=1000):
    tries = 0
    rng = np.random.RandomState(seed) if seed is not None else None
    while tries < max_tries:
        if rng is None:
            p1 = simulate_gbm_jump(S0, mu, sigma, T, N, jump_on, lam, mu_j, sigma_j)
            p2 = simulate_gbm_jump(S0, mu, sigma, T, N, jump_on, lam, mu_j, sigma_j)
        else:
            p1 = simulate_gbm_jump(S0, mu, sigma, T, N, jump_on, lam, mu_j, sigma_j, seed=rng.randint(0, 2**31-1))
            p2 = simulate_gbm_jump(S0, mu, sigma, T, N, jump_on, lam, mu_j, sigma_j, seed=rng.randint(0, 2**31-1))
        cond1 = p1[-1] > K and p2[-1] < K
        cond2 = p1[-1] < K and p2[-1] > K
        if cond1 or cond2:
            # determine which is up vs down relative to strike K
            if p1[-1] > K:
                path_up, path_down = p1, p2
            else:
                path_up, path_down = p2, p1
            return path_up, path_down
        tries += 1
    st.warning("Could not generate two paths with opposite final relation to strike K after many tries. Returning two GBM draws.")
    # fallback
    return p1, p2

path_up, path_down = gen_two_paths_until_one_above_below(S0, mu, sigma, T, N, use_jumps, lam, mu_j, sigma_j, seed_use)

time = np.linspace(0, T, N+1)

# ----------------------------
# 7. Compute boards for four cases
# ----------------------------
st.header("Delta hedging boards for: Buy Call / Sell Call / Buy Put / Sell Put")
# compute for each of four cases using the same two trajectories and same parameters
cases = [
    ("Call", "Long"),
    ("Call", "Short"),
    ("Put",  "Long"),
    ("Put",  "Short")
]

# We'll compute for both trajectories for each case and display two expandable tables per case
all_results = {}
for opt_type, side in cases:
    # Up scenario
    df_up, info_up = delta_hedge_accounting(path_up, opt_type, side, K, T, r, sigma,
                                            n_contracts, contract_size,
                                            hedge_freq_days, txn_cost_per_share, use_jumps)
    # Down scenario
    df_down, info_down = delta_hedge_accounting(path_down, opt_type, side, K, T, r, sigma,
                                                n_contracts, contract_size,
                                                hedge_freq_days, txn_cost_per_share, use_jumps)
    all_results[(opt_type, side, "up")] = (df_up, info_up)
    all_results[(opt_type, side, "down")] = (df_down, info_down)

# ----------------------------
# 8. Plots: stock paths and portfolio values (for selected case) + legends corrected
# ----------------------------
st.header("Simulation Results & Visuals")

colA, colB = st.columns([2, 1])
with colB:
    st.subheader("Select case to visualize")
    sel_opt = st.selectbox("Option Type", ["Call", "Put"])
    sel_side = st.selectbox("Side", ["Long", "Short"])
    sel_case = (sel_opt, sel_side)

# prepare stock path labels based on final relative to K
label_up = f"Path A (ST={path_up[-1]:.2f}) - UP (ST > K)"
label_down = f"Path B (ST={path_down[-1]:.2f}) - DOWN (ST < K)"

# compute portfolio values for selected case
df_up_sel, info_up_sel = all_results[(sel_opt, sel_side, "up")]
df_down_sel, info_down_sel = all_results[(sel_opt, sel_side, "down")]

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
# stock paths
axes[0].plot(time, path_up, label=label_up)
axes[0].plot(time, path_down, label=label_down)
axes[0].axhline(K, color="gray", linestyle="--", label="Strike K")
axes[0].set_ylabel("Stock Price")
axes[0].set_title("Simulated Stock Price Paths (GBM or jump-diffusion)")
axes[0].legend()
axes[0].grid(True)

# portfolio values (from DataFrame column "Portfolio Value")
axes[1].plot(time, df_up_sel["Portfolio Value"].values, label=f"{sel_opt} {sel_side} - Up scenario")
axes[1].plot(time, df_down_sel["Portfolio Value"].values, label=f"{sel_opt} {sel_side} - Down scenario")
axes[1].set_ylabel("Portfolio Value ($)")
axes[1].set_title("Delta-hedged Portfolio Value Over Time (includes premium & txns & accruals)")
axes[1].legend()
axes[1].grid(True)

# hedging error = portfolio value - theoretical option value (position) (should be near zero if perfect replication)
hedge_error_up = df_up_sel["Portfolio Value"].values - df_up_sel["Option Value (position)"].values
hedge_error_down = df_down_sel["Portfolio Value"].values - df_down_sel["Option Value (position)"].values
axes[2].plot(time, hedge_error_up, label="Hedge Error (Up)")
axes[2].plot(time, hedge_error_down, label="Hedge Error (Down)")
axes[2].set_ylabel("Hedging Error ($)")
axes[2].set_xlabel("Time (Years)")
axes[2].set_title("Hedging Error over Time (portfolio - option value)")
axes[2].legend()
axes[2].grid(True)

st.pyplot(fig)

# ----------------------------
# 9. Show the 4 boards in expanders (each board: two tabs for Up/Down)
# ----------------------------
st.header("Detailed daily boards (per case, per scenario)")

for opt_type, side in cases:
    with st.expander(f"{opt_type} - {side}", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Up scenario (ST > K)")
            df_up, info_up = all_results[(opt_type, side, "up")]
            st.dataframe(df_up)  # wide; user can scroll
            st.markdown(f"**Summary Up:** premium per unit = ${info_up['premium_per_unit']:.6f}, "
                        f"total premium = ${info_u_
