# delta_hedging_full_v3.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(page_title="Greeks & Delta Hedging Simulator v3", layout="wide")

# ----------------------------
# Black-Scholes & Greeks
# ----------------------------
def d1_fn(S0, K, tau, r, sigma):
    return (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))

def d2_fn(d1, sigma, tau):
    return d1 - sigma * np.sqrt(tau)

def bs_price(option_type, S0, K, tau, r, sigma):
    if tau <= 0:
        return max(0.0, S0-K) if option_type=="Call" else max(0.0, K-S0)
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
    delta = stats.norm.cdf(d1) if option_type=="Call" else stats.norm.cdf(d1)-1
    gamma = phi_d1 / (S0 * sigma * np.sqrt(tau))
    vega = S0 * phi_d1 * np.sqrt(tau)
    theta = (- (S0 * phi_d1 * sigma) / (2*np.sqrt(tau)) - r*K*np.exp(-r*tau)*stats.norm.cdf(d2)) if option_type=="Call" else (- (S0 * phi_d1 * sigma) / (2*np.sqrt(tau)) + r*K*np.exp(-r*tau)*stats.norm.cdf(-d2))
    rho = K * tau * np.exp(-r*tau) * stats.norm.cdf(d2) if option_type=="Call" else -K * tau * np.exp(-r*tau) * stats.norm.cdf(-d2)
    return delta, gamma, vega, theta, rho

GREEK_DESCRIPTIONS = {
    "Delta": "Δ — sensitivity of option price to the underlying price.",
    "Gamma": "Γ — rate of change of Delta w.r.t underlying price.",
    "Vega": "ν — sensitivity of option price to volatility.",
    "Theta": "Θ — time decay of the option price.",
    "Rho": "ρ — sensitivity to risk-free interest rate changes."
}

# ----------------------------
# GBM with optional Merton jumps
# ----------------------------
def simulate_gbm_jump(S0, mu, sigma, T, N, jump_on=False, lam=0.0, mu_j=0.0, sigma_j=0.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = T / N
    path = np.zeros(N+1)
    path[0] = S0
    for i in range(1, N+1):
        Z = np.random.normal()
        diffusion = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
        Jmult = 1.0
        if jump_on:
            k = np.random.poisson(lam*dt)
            if k>0:
                Jmult = np.exp(np.random.normal(mu_j, sigma_j)*k)
        path[i] = path[i-1] * np.exp(diffusion) * Jmult
    return path

def gen_two_paths_until_one_above_below(S0, mu, sigma, T, N, jump_on, lam, mu_j, sigma_j, K, seed=None, max_tries=1000):
    rng = np.random.RandomState(seed) if seed is not None else None
    for _ in range(max_tries):
        p1 = simulate_gbm_jump(S0, mu, sigma, T, N, jump_on, lam, mu_j, sigma_j, rng.randint(0,2**31-1) if rng else None)
        p2 = simulate_gbm_jump(S0, mu, sigma, T, N, jump_on, lam, mu_j, sigma_j, rng.randint(0,2**31-1) if rng else None)
        if (p1[-1]>K and p2[-1]<K) or (p1[-1]<K and p2[-1]>K):
            return (p1, p2) if p1[-1]>K else (p2, p1)
    return p1, p2

# ----------------------------
# Delta hedging table & PnL
# ----------------------------
def delta_hedge_table_and_pnl(path, option_type, side, K, N, r, sigma, n_contracts, contract_size, hedge_freq_days, txn_cost_per_share):
    total_shares = n_contracts * contract_size
    sign = 1.0 if side=="Long" else -1.0
    days = np.arange(0, N+1)
    stock_price = path.copy()
    delta_unit = np.zeros(N+1); gamma_unit = np.zeros(N+1)
    vega_unit = np.zeros(N+1); theta_unit = np.zeros(N+1); rho_unit = np.zeros(N+1)
    shares_in_portfolio = np.zeros(N+1)
    shares_purchased = np.zeros(N+1); cost_shares = np.zeros(N+1)
    tx_costs = np.zeros(N+1); cumulative_cash_outflow = np.zeros(N+1)

    # premium
    premium_per_unit = bs_price(option_type, stock_price[0], K, 1.0, r, sigma)
    total_premium = premium_per_unit * total_shares
    cumulative_cash_outflow[0] = total_premium if sign==1 else 0.0

    for day in range(N+1):
        tau = (N - day) / N
        if tau <=0:
            delta_unit[day] = gamma_unit[day] = vega_unit[day] = theta_unit[day] = rho_unit[day] = 0.0
        else:
            d,g,v,th,rh = greeks_all(option_type, stock_price[day], K, tau, r, sigma)
            delta_unit[day] = d; gamma_unit[day]=g; vega_unit[day]=v; theta_unit[day]=th; rho_unit[day]=rh

        prev_shares = shares_in_portfolio[day-1] if day>0 else 0.0
        if day % hedge_freq_days ==0:
            target_shares = -sign * delta_unit[day] * total_shares
            trade_shares = target_shares - prev_shares
            shares_purchased[day] = trade_shares
            cost_shares[day] = trade_shares * stock_price[day]
            tx_costs[day] = abs(trade_shares) * txn_cost_per_share
            cumulative_cash_outflow[day] = cumulative_cash_outflow[day-1] + abs(cost_shares[day]) + tx_costs[day] if day>0 else cumulative_cash_outflow[0]+abs(cost_shares[day])+tx_costs[day]
            shares_in_portfolio[day] = target_shares
        else:
            shares_in_portfolio[day] = prev_shares
            shares_purchased[day] = cost_shares[day] = tx_costs[day] = 0.0
            cumulative_cash_outflow[day] = cumulative_cash_outflow[day-1] if day>0 else cumulative_cash_outflow[0]

    # final payoff
    S_T = stock_price[-1]
    payoff_per_unit = max(S_T-K,0) if option_type=="Call" else max(K-S_T,0)
    payoff_total = payoff_per_unit * total_shares * sign
    cumulative_outflow_last = cumulative_cash_outflow[-1]
    pnl = -total_premium - cumulative_outflow_last + payoff_total if side=="Long" else total_premium - cumulative_outflow_last - payoff_total

    df = pd.DataFrame({
        "Day": days,
        "Stock Price": stock_price,
        "Tau (yrs)": (N-days)/N,
        "Delta (per unit)": delta_unit,
        "Gamma (per unit)": gamma_unit,
        "V
