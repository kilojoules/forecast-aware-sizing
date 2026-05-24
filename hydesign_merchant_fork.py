"""Minimal fork of hydesign.ems.ems.ems_cplex_parts for merchant battery.

Single structural change vs upstream: drop the P_HPP_t >= 0 lower bound so
that the plant can import from grid (battery grid-charging). Peak-hour
penalty and cycling penalty zeroed via API args.

Also exposes `enforce_terminal_soc` (default True per hydesign) for
controlled comparison: with terminal off + DoD=1.0, the LP collapses to
our `lp_linear_actions`. With hydesign defaults (DoD=0.9, terminal on),
it captures off-the-shelf operational conservatism.

`ems_cplex_merchant` is the batched wrapper (110-h chunks per hydesign's
own batching), needed for full-year sizing under CPLEX Community Edition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from docplex.mp.model import Model


def ems_cplex_parts_merchant(
    wind_ts,
    solar_ts,
    price_ts,
    P_batt_MW,
    E_batt_MWh_t,
    hpp_grid_connection,
    battery_depth_of_discharge,
    charge_efficiency,
    peak_hr_quantile=0.9,
    cost_of_battery_P_fluct_in_peak_price_ratio=0.0,
    n_full_power_hours_expected_per_day_at_peak_price=0,
    enforce_terminal_soc=True,
):
    """Bidirectional-grid fork of hydesign.ems.ems.ems_cplex_parts.
    Only structural change: P_HPP_t lb = -hpp_grid_connection (was 0).
    """
    N_t = len(price_ts.index)
    N_days = N_t / 24
    e_peak_day_expected = (
        n_full_power_hours_expected_per_day_at_peak_price * hpp_grid_connection
    )
    e_peak_period_expected = e_peak_day_expected * N_days
    price_peak = np.quantile(price_ts.values, peak_hr_quantile)
    peak_hours_index = np.where(price_ts >= price_peak)[0]

    price_ts_to_max = price_peak - price_ts
    price_ts_to_max.loc[price_ts_to_max < 0] = 0
    price_ts_to_max.iloc[:-1] = (
        0.5 * price_ts_to_max.iloc[:-1].values
        + 0.5 * price_ts_to_max.iloc[1:].values
    )

    mdl = Model(name="EMS_merchant")
    mdl.context.cplex_parameters.threads = 1
    mdl.context.cplex_parameters.emphasis.mip = 1

    time = price_ts.index
    SOCtime = time.append(pd.Index([time[-1] + pd.Timedelta("1hour")]))

    # ===== ONLY STRUCTURAL CHANGE =====
    P_HPP_t = mdl.continuous_var_dict(
        time, lb=-hpp_grid_connection, ub=hpp_grid_connection,
        name="HPP power output (bidirectional)"
    )
    # ==================================
    P_curtailment_t = mdl.continuous_var_dict(time, lb=0, name="Curtailment")
    P_charge_discharge = mdl.continuous_var_dict(
        time, lb=-P_batt_MW / charge_efficiency, ub=P_batt_MW * charge_efficiency,
        name="Battery power"
    )
    E_SOC_t = mdl.continuous_var_dict(SOCtime, lb=0, name="Energy level")
    penalty = mdl.continuous_var(name="penalty", lb=-1e12)
    e_penalty = mdl.continuous_var(name="e_penalty", lb=-1e12)

    fabs = mdl.piecewise(-1, [(0, 0)], 1)

    mdl.maximize(
        mdl.sum(price_ts[t] * P_HPP_t[t] for t in time) - penalty
        - mdl.sum(
            fabs(
                P_charge_discharge[t + pd.Timedelta("1hour")]
                - P_charge_discharge[t]
            )
            * cost_of_battery_P_fluct_in_peak_price_ratio
            * price_ts_to_max[t]
            for t in time[:-1]
        )
    )

    mdl.add_constraint(
        e_penalty
        == (
            e_peak_period_expected
            - mdl.sum(P_HPP_t[time[i]] for i in peak_hours_index)
        )
    )
    f1 = mdl.piecewise(0, [(0, 0)], 1)
    mdl.add_constraint(penalty == price_peak * f1(e_penalty))

    mdl.add_constraint(E_SOC_t[SOCtime[0]] == 0.5 * E_batt_MWh_t[time[0]])
    if enforce_terminal_soc:
        mdl.add_constraint(E_SOC_t[SOCtime[-1]] == 0.5 * E_batt_MWh_t[time[0]])

    f2 = mdl.piecewise(charge_efficiency, [(0, 0)], 1 / charge_efficiency)

    for t in time:
        tt = t + pd.Timedelta("1hour")
        dt = 1
        mdl.add_constraint(
            P_HPP_t[t]
            == wind_ts[t] + solar_ts[t] - P_curtailment_t[t] + P_charge_discharge[t]
        )
        mdl.add_constraint(E_SOC_t[tt] == E_SOC_t[t] - f2(P_charge_discharge[t]) * dt)
        mdl.add_constraint(
            E_SOC_t[t] >= (1 - battery_depth_of_discharge) * E_batt_MWh_t[t]
        )
        mdl.add_constraint(E_SOC_t[t] <= E_batt_MWh_t[t])
        mdl.add_constraint(P_charge_discharge[t] <= P_batt_MW * charge_efficiency)
        mdl.add_constraint(P_charge_discharge[t] >= -P_batt_MW / charge_efficiency)

    sol = mdl.solve(log_output=False)
    if sol is None:
        raise RuntimeError("CPLEX returned no solution (likely Community Edition cap)")

    def get(d):
        return pd.DataFrame.from_dict(
            sol.get_value_dict(d), orient="index"
        ).loc[:, 0]

    return (
        get(P_HPP_t),
        get(P_curtailment_t),
        get(P_charge_discharge),
        get(E_SOC_t),
        np.full(N_t, sol.get_value(penalty) / N_t),
    )


def _split_batches(n: int, batch_size: int):
    out, s = [], 0
    while s < n:
        out.append(list(range(s, min(s + batch_size, n))))
        s += batch_size
    return out


def ems_cplex_merchant(
    wind_ts,
    solar_ts,
    price_ts,
    P_batt_MW,
    E_batt_MWh_t,
    hpp_grid_connection,
    battery_depth_of_discharge,
    charge_efficiency,
    peak_hr_quantile=0.9,
    cost_of_battery_P_fluct_in_peak_price_ratio=0.0,
    n_full_power_hours_expected_per_day_at_peak_price=0,
    enforce_terminal_soc=True,
    batch_size=110,
):
    """Batched merchant EMS — 110-h chunks (matches hydesign default).

    With enforce_terminal_soc=True, each batch starts and ends at 0.5*b_E,
    which is the hydesign off-the-shelf behavior: implicit short-horizon
    cycle balance.
    """
    n = len(wind_ts)
    batches = _split_batches(n, batch_size)
    P_HPP = np.zeros(n)
    P_curt = np.zeros(n)
    P_chg_dis = np.zeros(n)
    E_SOC = np.zeros(n + 1)
    penalty = np.zeros(n)
    for batch in batches:
        out = ems_cplex_parts_merchant(
            wind_ts=wind_ts.iloc[batch],
            solar_ts=solar_ts.iloc[batch],
            price_ts=price_ts.iloc[batch],
            P_batt_MW=P_batt_MW,
            E_batt_MWh_t=E_batt_MWh_t.iloc[batch],
            hpp_grid_connection=hpp_grid_connection,
            battery_depth_of_discharge=battery_depth_of_discharge,
            charge_efficiency=charge_efficiency,
            peak_hr_quantile=peak_hr_quantile,
            cost_of_battery_P_fluct_in_peak_price_ratio=cost_of_battery_P_fluct_in_peak_price_ratio,
            n_full_power_hours_expected_per_day_at_peak_price=n_full_power_hours_expected_per_day_at_peak_price,
            enforce_terminal_soc=enforce_terminal_soc,
        )
        P_HPP_b, P_curt_b, P_chg_dis_b, E_SOC_b, penalty_b = out
        nb = len(batch)
        P_HPP[batch] = P_HPP_b.to_numpy()
        P_curt[batch] = P_curt_b.to_numpy()
        P_chg_dis[batch] = P_chg_dis_b.to_numpy()
        # E_SOC may have nb+1 (with terminal) or nb (terminal dropped by solver
        # when unconstrained) entries; take the leading nb start-of-hour values.
        e_arr = E_SOC_b.to_numpy()
        E_SOC[batch] = e_arr[:nb]
        penalty[batch] = penalty_b
    E_SOC[-1] = E_SOC[0]
    return P_HPP, P_curt, P_chg_dis, E_SOC, penalty
