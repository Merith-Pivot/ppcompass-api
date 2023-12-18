import pandas as pd
import numpy as np
from itertools import product

from scipy.stats.mstats import hmean
from sklearn import linear_model, model_selection, preprocessing

import prophet


def apply_prophet(brand_data, features=["ap", "price_per_volume", "tdp"]):
    prophet_model = prophet.Prophet(
        interval_width=0.95,
        changepoint_prior_scale=0.01,
        weekly_seasonality=False,
        yearly_seasonality=True,
        mcmc_samples=300,
    )
    for f in features:
        prophet_model.add_regressor(f)
    prophet_model.fit(brand_data)
    forecast = prophet_model.predict(brand_data)
    brand_data["trend"] = forecast["trend"]
    brand_data["seasonality"] = forecast["yearly"]
    brand_data["residual"] = (
        brand_data["y"] - brand_data["trend"] - brand_data["seasonality"]
    )
    return brand_data


def get_model_data(brand_data, features=["ap", "price_per_volume", "tdp"]):
    scaler_X = preprocessing.StandardScaler()
    scaler_y = preprocessing.StandardScaler()
    raw_X = brand_data[features + ["residual"]]
    raw_y = brand_data[["y"]]
    X, y = raw_X, raw_y
    X.iloc[:, :] = scaler_X.fit_transform(raw_X)
    y.iloc[:, :] = scaler_y.fit_transform(raw_y)
    return X, y, scaler_X, scaler_y


def get_return(X):
    model_ap = linear_model.LinearRegression()
    X_sensitivity = X.filter(["ap"])
    model_ap.fit(X_sensitivity, X[["residual"]])
    a = model_ap.coef_[0][0]
    return a


def get_risk(X, y):
    model = linear_model.RidgeCV()
    cv_model = model_selection.cross_validate(
        model,
        X,
        y,
        cv=model_selection.RepeatedKFold(n_splits=3, n_repeats=5),
        return_estimator=True,
        n_jobs=2,
    )

    coefs = pd.DataFrame(
        [model.coef_[0] for model in cv_model["estimator"]],
        columns=X.columns,
    )
    risk = coefs.mean().abs().transform(lambda x: x / x.sum())["residual"]

    return risk, coefs.mean().abs().drop(["residual"]).sort_values(ascending=False)


def get_risk_return(data, brand):
    X, y, scaler_X, scaler_y = (
        data.query("brand == @brand")
        .reset_index(drop=True)
        .pipe(apply_prophet)
        .pipe(get_model_data)
    )

    ret = get_return(X)
    rsk, drivers = get_risk(X, y)
    return ret, rsk, scaler_X, scaler_y, drivers


def get_last_year(data, brand):
    brand_data = data.query("brand == @brand")
    differential = brand_data.iloc[[-1, -12], 16].values
    growth_rate = differential[0] / differential[1]
    return brand_data.iloc[-12:].drop(columns=["brand", "ds"]).sum(), growth_rate


def get_brands_mapping(raw_data):
    data_arr = []
    for brand in raw_data["brand"].unique():
        ret, rsk, scaler_X, scaler_y, drivers = get_risk_return(raw_data, brand)
        last_year, growth_rate = get_last_year(raw_data, brand)
        y_scaled = (last_year["y"] / 12 - scaler_y.mean_[0]) / scaler_y.scale_[0]
        x_scaled = (last_year["ap"] / 12 - scaler_X.mean_[0]) / scaler_X.scale_[0]
        intercept = y_scaled - ret * x_scaled
        data_arr.append(
            [
                brand,
                ret,
                rsk,
                list(drivers.index),
                last_year["y"],
                last_year["net_sales"],
                last_year["ap"],
                last_year["price_per_volume"],
                last_year["tdp"],
                intercept,
                last_year["net_sales"] / last_year["y"],
                scaler_X.mean_[0],
                scaler_X.scale_[0],
                scaler_y.mean_[0],
                scaler_y.scale_[0],
                growth_rate,
                raw_data.query("brand == @brand")["ds"].to_list(),
                raw_data.query("brand == @brand")["net_sales"].to_list(),
                raw_data.query("brand == @brand")["y"].to_list(),
                raw_data.query("brand == @brand")["ap"].to_list(),
                raw_data.query("brand == @brand")["tdp"].to_list(),
                raw_data.query("brand == @brand")["price_per_volume"].to_list(),
            ]
        )

    return (
        pd.DataFrame(
            data=data_arr,
            columns=[
                "brand",
                "return",
                "risk",
                "drivers",
                "ly_sales",
                "ly_net_sales",
                "ly_budget",
                "ly_price",
                "ly_tdp",
                "intercept",
                "price",
                "x_mean",
                "x_std",
                "y_mean",
                "y_std",
                "growth_rate",
                "date_array",
                "ns_array",
                "vs_array",
                "ap_array",
                "tdp_array",
                "ppv_array",
            ],
            index=raw_data["brand"].unique(),
        )
        .assign(
            **{
                "return_score": lambda df_: (df_["return"] - df_["return"].mean())
                / df_["return"].std(),
                "risk_score": lambda df_: (df_["risk"] - df_["risk"].mean())
                / df_["risk"].std(),
            },
        )
        .assign(
            **{
                "return_score": lambda df_: 100
                * np.exp(df_["return_score"])
                / (1 + np.exp(df_["return_score"])),
                "risk_score": lambda df_: 100
                * np.exp(df_["risk_score"])
                / (1 + np.exp(df_["risk_score"])),
            },
        )
    )


def get_scenarios_ns(budget_df, brands_mapping):
    ns = (
        (
            (
                (
                    (
                        budget_df.drop(columns=["budget", "risk", "return"]) / 12
                        - brands_mapping["x_mean"].T
                    )
                    / brands_mapping["x_std"].T
                )
                * brands_mapping["return"].T
                + brands_mapping["intercept"].T
            )
            * brands_mapping["y_std"].T
            + brands_mapping["y_mean"].T
        )
        * 12
        - brands_mapping["ly_sales"].T
        + (brands_mapping["ly_sales"].T * brands_mapping["growth_rate"].T)
    ) * brands_mapping["price"].T
    ns.columns = [c + "_ns" for c in ns.columns]
    # ns["total_ns"] = ns.sum(axis=1)
    # return pd.concat((budget_df, ns), axis=1)
    return budget_df.assign(**{"total_ns": ns.sum(axis=1)})


def get_scenarios_return(budget_df, brands_mapping):
    return (
        budget_df.div(budget_df["budget"], axis=0)
        .multiply(brands_mapping["return_score"])
        .sum(axis=1)
    )


def get_scenarios_risk(budget_df, brands_mapping):
    ratios = (
        budget_df.div(budget_df["budget"], axis=0)
        .drop(columns=["budget", "return"])
        .values
    )
    brand_risk = brands_mapping["risk_score"].to_list()
    risks = [hmean(brand_risk, weights=ratio) for ratio in ratios]
    return risks


def correct_scenarios_risk(budget_df):
    mid = len(budget_df) // 2
    return budget_df.assign(
        **{"risk": lambda df_: (df_["risk"] * 0.5 / df_.loc[mid, "risk"]).fillna(0.5)}
    )


def get_scenarios_df(data_path="data/template.csv", step=10, min_val=-100, max_val=100):
    raw_data = pd.read_csv(data_path)
    brands_mapping = get_brands_mapping(raw_data)
    ratios = np.arange(min_val + 100, max_val + 100 + 1, step) / 100
    scenarios = []
    for brand in brands_mapping.index:
        scenarios.append(ratios * brands_mapping.loc[brand, "ly_budget"])
    return (
        pd.DataFrame(
            product(*scenarios),
            columns=brands_mapping.index,
        ).assign(
            **{
                "budget": lambda df_: df_.sum(axis=1),
            }
        )
    ), brands_mapping.T


def get_budget_df(data_path="data/template.csv", step=10, min_val=-100, max_val=100):
    raw_data = pd.read_csv(data_path)
    brands_mapping = get_brands_mapping(raw_data)
    ratios = np.arange(min_val + 100, max_val + 100 + 1, step) / 100
    scenarios = []
    for brand in brands_mapping.index:
        scenarios.append(ratios * brands_mapping.loc[brand, "ly_budget"])

    return (
        pd.DataFrame(product(*scenarios), columns=brands_mapping.index)
        .assign(**{"budget": lambda df_: df_.sum(axis=1)})
        .assign(
            **{
                "return": lambda df_: df_.pipe(get_scenarios_return, brands_mapping),
                "risk": lambda df_: df_.pipe(get_scenarios_risk, brands_mapping),
            }
        )
        .pipe(correct_scenarios_risk)
        .pipe(get_scenarios_ns, brands_mapping)
    ), brands_mapping


def get_scenarios_types(temp_df):
    if len(temp_df) == 1:
        temp_df = pd.concat((temp_df, temp_df), axis=0).reset_index(drop=True)
    temp_df["type"] = "Normal"
    temp_df.loc[len(temp_df) - 1, "type"] = "Conservative"
    agg_pos = temp_df["return"].astype(float).argmax()
    def_pos = temp_df["risk"].astype(float).argmin()
    pro_pos = (
        ((temp_df["total_ns"] - temp_df["budget"]) / temp_df["total_ns"])
        .astype(float)
        .argmax()
    )
    if agg_pos == def_pos:
        temp_df.loc[agg_pos, "type"] = "Aggressive"
        temp_df = pd.concat((temp_df, temp_df.T[[agg_pos]].T), axis=0).reset_index(
            drop=True
        )
        temp_df.loc[len(temp_df) - 1, "type"] = "Defensive"
    else:
        temp_df.loc[agg_pos, "type"] = "Aggressive"
        temp_df.loc[def_pos, "type"] = "Defensive"
    if pro_pos == agg_pos:
        temp_df = pd.concat((temp_df, temp_df.T[[agg_pos]].T), axis=0).reset_index(
            drop=True
        )
        temp_df.loc[len(temp_df) - 1, "type"] = "Profitable"
    elif pro_pos == def_pos:
        temp_df = pd.concat((temp_df, temp_df.T[[def_pos]].T), axis=0).reset_index(
            drop=True
        )
        temp_df.loc[len(temp_df) - 1, "type"] = "Profitable"
    else:
        temp_df.loc[pro_pos, "type"] = "Profitable"

    return temp_df
