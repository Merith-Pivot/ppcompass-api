import urllib.parse
import json
import jwt
import os
import datetime

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly

import pandas as pd
import numpy as np
from scipy.stats.mstats import hmean

from pymongo import MongoClient
from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from io import BytesIO

from app.backend import (
    get_scenarios_return,
    get_scenarios_risk,
    get_scenarios_ns,
    get_scenarios_types,
)

import dotenv

dotenv.load_dotenv(".env")


colors_brand = [
    "#444e96",
    "#1b9e77",
    "#b884b9",
    "#e7298a",
    "#ff917d",
    "#ffa600",
    "#66a61e",
    "#0cdea0",
]

colors_grid = [
    "#ebebeb",
    "#f3f3f3",
    "#f7f7f7",
    "#fbfbfb",
]

colors_drivers = [
    "#b7c0ee",
    "#e8bf39",
    "#ae5f00",
    "#db53a5",
    "#2f0147",
]

NEXTAUTH_SECRET = os.getenv("NEXTAUTH_SECRET")
username = urllib.parse.quote_plus(os.getenv("username_mongo"))
password = urllib.parse.quote_plus(os.getenv("password_mongo"))
address = os.getenv("address")
mongo = MongoClient("mongodb+srv://%s:%s@%s" % (username, password, address))["ppc-db"]

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_client(request):
    token = request.headers.get("Authorization")[7:]
    return jwt.decode(
        token, NEXTAUTH_SECRET, algorithms="HS256", options={"verify_signature": False}
    )["data"]["name"]


@app.get("/brands")
async def get_brands(request: Request):
    client = get_client(request)
    mapping = pd.DataFrame([e for e in mongo["brands"].find({"client": client})])
    mapping["budget_share"] = mapping["ly_budget"] / mapping["ly_budget"].sum()
    mapping["ns_share"] = mapping["ly_net_sales"] / mapping["ly_net_sales"].sum()
    res = [
        v
        for _, v in mapping.filter(
            items=[
                "brand",
                "ly_net_sales",
                "ly_budget",
                "growth_rate",
                "return_score",
                "risk_score",
                "client",
                "budget_share",
                "ns_share",
            ]
        )
        .T.to_dict()
        .items()
    ]
    return res


@app.get("/portfolio")
async def get_portfolio(request: Request):
    client = get_client(request)
    mapping = [e for e in mongo["brands"].find({"client": client})]
    res = pd.DataFrame(mapping)[["ly_net_sales", "ly_budget"]].sum()
    res["ly_ap_ns"] = res["ly_budget"] / res["ly_net_sales"]
    return res


def local_get_scenarios(client, budget, threshold=0.01):
    budget = float(budget)
    marge = budget * threshold
    scenarios_json = [
        e
        for e in mongo["scenarios"].find(
            {
                "client": client,
                "budget": {"$gt": budget - marge - 1, "$lt": budget + marge + 1},
            }
        )
    ]
    brands_mapping = (
        pd.DataFrame([e for e in mongo["brands"].find({"client": client})])
        .drop(columns="_id")
        .set_index("brand")
    )
    conservative = {"_id": "", "client": client}
    budget_temp = 0
    for b in brands_mapping.index:
        conservative[b] = brands_mapping.loc[b, "ly_budget"]
        budget_temp += conservative[b]
    conservative["budget"] = budget_temp
    scenarios_json = scenarios_json + [conservative]
    res = (
        pd.DataFrame(scenarios_json)
        .drop(columns=["_id", "client"])
        .assign(
            **{
                "return": lambda df_: df_.pipe(get_scenarios_return, brands_mapping),
                "risk": lambda df_: df_.pipe(get_scenarios_risk, brands_mapping),
            }
        )
        .pipe(get_scenarios_ns, brands_mapping)
        .pipe(get_scenarios_types)
    )

    res.iloc[:, : len(brands_mapping)] = np.round(
        res.iloc[:, : len(brands_mapping)].divide(brands_mapping["ly_budget"], axis=1)
        * 100
        - 100,
        0,
    )
    return res, brands_mapping


@app.get("/scenarios")
async def get_scenarios(request: Request, budget, threshold=0.01):
    client = get_client(request)
    res, brands_mapping = local_get_scenarios(client, budget, threshold)
    fig = px.scatter(
        res,
        x="risk",
        y="return",
        color="type",
        template="plotly_white",
        size_max=20,
        size=len(res) * [20],
        hover_data=res.columns[: len(brands_mapping)],
        opacity=1,
        color_discrete_map={
            "Normal": "#eedecf",
            "Profitable": "#eacb28",
            "Aggressive": "#b81515",
            "Defensive": "#2791dd",
            "Conservative": "#723700",
            "Customized": "#ef7f40",
        },
    )
    fig.update_layout(
        xaxis_range=[max(res["risk"]) + 5, min(res["risk"]) - 5],
        yaxis_range=[min(res["return"]) - 5, max(res["return"]) + 5],
        # legend=dict(orientation="h", xanchor="center", y=1.1, x=0.5),
        legend_title_text="Type of scenario",
    )
    fig.add_hrect(
        y0=-100,
        y1=50,
        line_width=0,
        fillcolor=colors_grid[2],
        opacity=0.25,
        layer="below",
    )
    fig.add_vrect(
        x0=-100,
        x1=50,
        line_width=0,
        fillcolor=colors_grid[0],
        opacity=0.35,
        layer="below",
    )
    fig.add_hrect(
        y0=50,
        y1=200,
        line_width=0,
        fillcolor=colors_grid[0],
        opacity=0.35,
        layer="below",
    )
    fig.add_vrect(
        x0=50,
        x1=200,
        line_width=0,
        fillcolor=colors_grid[1],
        opacity=0,
        layer="below",
    )
    fig.add_hline(
        y=50,
        line_dash="dot",
    )
    fig.add_vline(
        x=50,
        line_dash="dot",
    )
    fig.add_hline(
        y=max(res["return"]) + 5,
        line_dash="dot",
        annotation_text="High Return<br>High Risk",
        annotation_position="bottom left",
        opacity=0,
    )
    fig.add_hline(
        y=max(res["return"]) + 5,
        line_dash="dot",
        annotation_text="High Return<br>Low Risk",
        annotation_position="bottom right",
        opacity=0,
    )
    fig.add_hline(
        y=min(res["return"]) - 5,
        line_dash="dot",
        annotation_text="Low Return<br>High Risk",
        annotation_position="top left",
        opacity=0,
    )
    fig.add_hline(
        y=min(res["return"]) - 5,
        line_dash="dot",
        annotation_text="Low Return<br>Low Risk",
        annotation_position="top right",
        opacity=0,
    )
    fig.update_layout(
        margin=dict(
            l=0,
            r=0,
            t=35,
            b=0,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#4D4D4D",
        font_family="Roboto",
        title_font_color="#4D4D4D",
        legend_title_font_color="#4D4D4D",
        legend_title_text="Type of scenario",
    )

    fig.update_xaxes(
        title="← Risk ―",
        showgrid=False,
        ticks="inside",
        showline=True,
        zeroline=False,
        linecolor="black",
    )
    fig.update_yaxes(
        title="― Return →",
        showgrid=False,
        ticks="inside",
        showline=True,
        zeroline=False,
        linecolor="black",
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


@app.get("/scenario")
async def get_scenario(
    request: Request, budgets, budget, scenario_type, threshold=0.01
):
    scenario_type = scenario_type.capitalize()
    client = get_client(request)
    ny_budget = json.loads(budgets)
    brands_mapping = (
        pd.DataFrame([e for e in mongo["brands"].find({"client": client})])
        .drop(columns="_id")
        .set_index("brand")
    )
    res = brands_mapping[
        ["ly_budget", "ly_net_sales", "ly_mvc", "return_score", "risk_score", "drivers"]
    ]

    if scenario_type:
        temp_df = local_get_scenarios(client, budget, threshold)[0]
        res["ny_change"] = temp_df.query("type == @scenario_type").values[0][: len(res)]
    else:
        res["ny_change"] = ny_budget
    res["ny_budget"] = (res["ny_change"] + 100) / 100 * res["ly_budget"]
    ns = (
        (
            (
                (
                    (
                        (res[["ny_budget"]].T / 12 - brands_mapping["x_mean"].T)
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
        )
        * brands_mapping["price"].T
    ).values

    res["ny_net_sales"] = ns[0]
    res["ny_mvc"] = res["ly_mvc"] / res["ly_net_sales"] * res["ny_net_sales"]
    res["mvc_change"] = res["ny_mvc"] / res["ly_mvc"] * 100 - 100
    res["budget_share"] = res["ny_budget"] / res["ny_budget"].sum()
    return_score = sum(res["budget_share"] * res["return_score"])
    risk_score = hmean(res["risk_score"], weights=res["budget_share"])
    # portfolio
    res = res.T
    res["portfolio"] = res.sum(axis=1)
    res = res.T
    res["ny_ns_change"] = (res["ny_net_sales"] / res["ly_net_sales"]) * 100 - 100
    res["ny_budget_change"] = (res["ny_budget"] / res["ly_budget"]) * 100 - 100
    res.loc["portfolio", "return_score"] = return_score
    res.loc["portfolio", "risk_score"] = risk_score
    res["ap_ns_ratio"] = res["ny_budget"] / res["ny_net_sales"]

    return res.T.to_dict()  # new budget = sum of ny_budget


@app.get("/brand_graph")
async def graph_brands(request: Request, brand, feature):
    client = get_client(request)
    bm = mongo["brands"].find_one({"client": client, "brand": brand})
    if feature == "Net Sales":
        feature_name = "ns_array"
    if feature == "Volume Sales":
        feature_name = "vs_array"
    elif feature == "A&P":
        feature_name = "ap_array"
    elif feature == "Distribution":
        feature_name = "tdp_array"
    elif feature == "Price":
        feature_name = "ppv_array"
    temp_df = pd.DataFrame(
        index=["date", "feature"], data=[bm["date_array"], bm[feature_name]]
    ).T
    fig = px.line(
        temp_df,
        x="date",
        y="feature",
        template="plotly_white",
        color_discrete_sequence=["#6BA5DB"],
        markers=True,
    )
    fig.update_layout(
        margin=dict(
            l=0,
            r=0,
            t=35,
            b=0,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFBF9",
        font_color="#4D4D4D",
        font_family="Roboto",
        title_font_color="#4D4D4D",
        legend_title_font_color="#4D4D4D",
    )
    fig.update_yaxes(
        title=feature,
        showgrid=False,
        ticks="inside",
        showline=True,
        linecolor="black",
        zeroline=False,
    )
    fig.update_xaxes(
        title="Date",
        showgrid=False,
        ticks="inside",
        showline=True,
        linecolor="black",
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


@app.get("/brands_matrix")
async def brands_matrix(request: Request, size_bubble="ly_budget"):
    client = get_client(request)
    bm = [e for e in mongo["brands"].find({"client": client})]
    temp_df = pd.DataFrame(bm)
    fig = px.scatter(
        temp_df,
        x="risk_score",
        y="return_score",
        size=size_bubble,
        template="plotly_white",
        size_max=60,
        color="brand",
        color_discrete_map={b: c for b, c in zip(temp_df["brand"], colors_brand)},
    )
    fig.update_layout(
        xaxis_range=[max(temp_df["risk_score"]) + 50, min(temp_df["risk_score"]) - 50],
        yaxis_range=[
            min(temp_df["return_score"]) - 50,
            max(temp_df["return_score"]) + 50,
        ],
        legend=dict(orientation="h", xanchor="center", y=1.1, x=0.5),
    )
    fig.add_hrect(
        y0=-100,
        y1=50,
        line_width=0,
        fillcolor=colors_grid[2],
        opacity=0.25,
        layer="below",
    )
    fig.add_vrect(
        x0=-100,
        x1=50,
        line_width=0,
        fillcolor=colors_grid[0],
        opacity=0.35,
        layer="below",
    )
    fig.add_hrect(
        y0=50,
        y1=200,
        line_width=0,
        fillcolor=colors_grid[0],
        opacity=0.35,
        layer="below",
    )
    fig.add_vrect(
        x0=50,
        x1=200,
        line_width=0,
        fillcolor=colors_grid[1],
        opacity=0,
        layer="below",
    )
    fig.add_hline(
        y=50,
        line_dash="dot",
    )
    fig.add_vline(
        x=50,
        line_dash="dot",
    )
    fig.add_hline(
        y=max(temp_df["return_score"]) + 50,
        line_dash="dot",
        annotation_text="High Return<br>High Risk",
        annotation_position="bottom left",
        opacity=0,
    )
    fig.add_hline(
        y=max(temp_df["return_score"]) + 50,
        line_dash="dot",
        annotation_text="High Return<br>Low Risk",
        annotation_position="bottom right",
        opacity=0,
    )
    fig.add_hline(
        y=min(temp_df["return_score"]) - 50,
        line_dash="dot",
        annotation_text="Low Return<br>High Risk",
        annotation_position="top left",
        opacity=0,
    )
    fig.add_hline(
        y=min(temp_df["return_score"]) - 50,
        line_dash="dot",
        annotation_text="Low Return<br>Low Risk",
        annotation_position="top right",
        opacity=0,
    )
    fig.update_layout(
        margin=dict(
            l=0,
            r=0,
            t=35,
            b=0,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#4D4D4D",
        font_family="Roboto",
        title_font_color="#4D4D4D",
        legend_title_font_color="#4D4D4D",
    )

    fig.update_xaxes(
        title="← Risk ―",
        showgrid=False,
        zeroline=False,
        ticks="inside",
        showline=True,
        linecolor="black",
    )
    fig.update_yaxes(
        title="― Return →",
        showgrid=False,
        zeroline=False,
        ticks="inside",
        showline=True,
        linecolor="black",
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


@app.get("/histogram")
async def histogram(request: Request, y="return"):
    client = get_client(request)
    bm = [e for e in mongo["brands"].find({"client": client})]
    temp_df = pd.DataFrame(bm)
    fig = px.bar(
        temp_df,
        x="brand",
        y=y + "_score",
        template="plotly_white",
        color="brand",
        color_discrete_map={b: c for b, c in zip(temp_df["brand"], colors_brand)},
    )

    fig.add_hline(
        y=50,
        line_dash="dot",
        annotation_text="Portfolio's Average",
        annotation_font_color="#B4B4B4",
        line_color="#B4B4B4",
        annotation_position="top left",
        opacity=1,
    )

    fig.update_layout(
        margin=dict(
            l=0,
            r=0,
            t=35,
            b=0,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFBF9",
        font_color="#4D4D4D",
        font_family="Roboto",
        title_font_color="#4D4D4D",
        legend_title_font_color="#4D4D4D",
        showlegend=False,
    )
    fig.update_xaxes(
        title=None,  # "Brand",
        showgrid=False,
        showline=True,
        linecolor="black",
    )
    fig.update_yaxes(
        title=None,  # y.capitalize(),
        showgrid=False,
        ticks="inside",
        showline=True,
        linecolor="black",
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


@app.get("/risk_details")
async def risk_details(request: Request):
    client = get_client(request)
    bm = [e for e in mongo["brands"].find({"client": client})]
    temp_df = pd.DataFrame(bm)
    fig = px.bar(
        temp_df,
        x="brand",
        y="risk",
        template="plotly_white",
        color="brand",
        color_discrete_map={b: c for b, c in zip(temp_df["brand"], colors_brand)},
    )

    fig.update_layout(
        margin=dict(
            l=0,
            r=0,
            t=35,
            b=0,
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFBF9",
        font_color="#4D4D4D",
        font_family="Roboto",
        title_font_color="#4D4D4D",
        legend_title_font_color="#4D4D4D",
    )
    fig.update_xaxes(
        title=None,  # "Brand",
        showgrid=False,
        showline=True,
        linecolor="black",
    )
    fig.update_yaxes(
        title=None,  # "Percentage of uncertainty",
        tickformat=",.0%",
        showgrid=False,
        showline=True,
        linecolor="black",
        ticks="inside",
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


@app.get("/sensitivity")
async def sensitivity(request: Request):
    client = get_client(request)
    bm = [e for e in mongo["brands"].find({"client": client})]
    temp_df = pd.DataFrame(bm)
    X = np.linspace(
        (-1+ temp_df["ly_budget"]).to_list(), (1 + temp_df["ly_budget"]).to_list()
    )
    y = (
        (
            (
                (
                    (
                        pd.DataFrame(X) / 12 - temp_df["x_mean"].T
                    )
                    / temp_df["x_std"].T
                )
                * temp_df["return"].T
                + temp_df["intercept"].T
            )
            * temp_df["y_std"].T
            + temp_df["y_mean"].T
        )
        * 12
        - temp_df["ly_sales"].T
    ) * temp_df["price"].T
    y.columns = temp_df["brand"]
    fig = px.line(
        y.melt(),
        x=np.array((len(y.columns) * [np.linspace(-1, 1)])).flatten(),
        y="value",
        color="brand",
        template="plotly_white",
        color_discrete_map={b: c for b, c in zip(temp_df["brand"], colors_brand)},
    )

    fig.update_layout(
        margin=dict(
            l=0,
            r=0,
            t=35,
            b=0,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFBF9",
        font_color="#4D4D4D",
        font_family="Roboto",
        title_font_color="#4D4D4D",
        legend_title_font_color="#4D4D4D",
        legend=dict(orientation="h", xanchor="center", y=1.1, x=0.5),
        legend_title_text="Brand",
    )
    fig.update_xaxes(
        title="A&P",
        showgrid=False,
        ticks="inside",
        showline=True,
        linecolor="black",
        tickformat="€",
        tickmode="array",
        tickvals=[-1, -0.5, 0, 0.5, 1],
        ticktext=["-1€", "-0.50€", "0€", "+0.50€", "+1€"],
    )
    fig.update_yaxes(
        title="Sales",
        showgrid=False,
        ticks="inside",
        showline=True,
        linecolor="black",
        tickmode="array",
        tickvals=[-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10],
        ticktext=[
            "-10€",
            "-7.50€",
            "-5€",
            "-2.50€",
            "0€",
            "+2.50€",
            "+5€",
            "+7.50€",
            "+10€",
        ],
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


@app.get("/return_details")
async def return_details(request: Request, brand):
    client = get_client(request)
    bm = mongo["brands"].find_one({"client": client, "brand": brand})
    temp_df = pd.DataFrame(
        index=["date", "A&P", "Net Sales"],
        data=[bm["date_array"], bm["ap_array"], bm["ns_array"]],
    ).T.melt("date")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    traces = px.line(
        temp_df,
        x="date",
        y="value",
        template="plotly_white",
        color="variable",
        markers=True,
        color_discrete_sequence=["#6BA5DB", "#DB946B"],
    ).data
    fig.add_trace(traces[0], secondary_y=False)
    fig.add_trace(traces[1], secondary_y=True)
    fig.update_layout(
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFBF9",
        font_color="#4D4D4D",
        font_family="Roboto",
        title_font_color="#4D4D4D",
        legend_title_font_color="#4D4D4D",
        legend=dict(orientation="h", xanchor="center", y=1.1, x=0.5),
        template="plotly_white",
        legend_title_text="Curve",
    )
    fig.update_xaxes(
        title="Date",
        showgrid=False,
        ticks="inside",
        showline=True,
        zeroline=False,
        linecolor="black",
    )
    fig.update_yaxes(
        title_text="A&P",
        secondary_y=False,
        showgrid=False,
        showline=True,
        zeroline=False,
        ticks="inside",
        linecolor="black",
    )
    fig.update_yaxes(
        title_text="Net Sales",
        secondary_y=True,
        showgrid=False,
        showline=True,
        zeroline=False,
        ticks="inside",
        linecolor="black",
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


@app.get("/histogram_scenario")
async def histogram_scenario(
    request: Request, budgets, budget, scenario_type, threshold=0.01
):
    scenario_type = scenario_type.capitalize()
    client = get_client(request)
    ny_budget = np.array([float(b) for k, b in json.loads(budgets).items()])
    brands_mapping = (
        pd.DataFrame([e for e in mongo["brands"].find({"client": client})])
        .drop(columns="_id")
        .set_index("brand")
    )
    res = brands_mapping[["ly_budget", "ly_net_sales"]]

    if scenario_type:
        temp_df = local_get_scenarios(client, budget, threshold)[0]
        res["ny_change"] = temp_df.query("type == @scenario_type").values[0][: len(res)]
    else:
        res["ny_change"] = ny_budget
    res["ny_budget"] = (res["ny_change"] + 100) / 100 * res["ly_budget"]

    ns = (
        (
            (
                (
                    (
                        (res[["ny_budget"]].T / 12 - brands_mapping["x_mean"].T)
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
        )
        * brands_mapping["price"].T
    ).values

    res["ny_net_sales"] = ns[0]
    res["ap_ns_ratio"] = res["ny_budget"] / res["ny_net_sales"]
    # portfolio
    temp_df = (
        pd.DataFrame(res)
        .filter(["ny_budget", "ny_net_sales"])
        .reset_index()
        .melt("brand")
        .replace({"ny_budget": "New Year A&P", "ny_net_sales": "New Year Net Sales"})
    )
    # return temp_df
    # temp_df["ny_ns_share"] = temp_df["ny_net_sales"] / temp_df["ny_net_sales"].sum()
    # temp_df["ny_budget_share"] = temp_df["ny_budget"] / temp_df["ny_budget"].sum()
    # rest_df = (
    #     temp_df.reset_index()
    #     .filter(items=["brand", "ny_budget", "ny_net_sales"])
    #     .melt("brand")
    # )
    # res_df = (
    #     temp_df.reset_index()
    #     .filter(items=["brand", "ny_budget_share", "ny_ns_share"])
    #     .melt("brand")
    #     .rename(columns={"value": "value2"})
    #     .drop(columns=["brand", "variable"])
    # )
    # res_df = pd.concat((res_df, rest_df), axis=1).assign(
    #     **{
    #         "value": lambda df_: [
    #             str(np.round(v / 1000000, 2)) + "M" for v in df_["value"]
    #         ]
    #     }
    # )
    fig = px.bar(
        temp_df,
        y="value",
        x="brand",
        color="variable",
        orientation="v",
        barmode="group",
        # text="value",
        color_discrete_sequence=["#6BA5DB", "#DB946B"],
        template="plotly_white",
    )

    fig.update_layout(
        margin=dict(
            l=0,
            r=0,
            t=35,
            b=0,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFBF9",
        font_color="#4D4D4D",
        font_family="Roboto",
        title_font_color="#4D4D4D",
        legend_title_font_color="#4D4D4D",
        legend=dict(orientation="h", xanchor="center", y=1.1, x=0.5),
        legend_title_text="Brand",
    )
    fig.update_xaxes(
        title="Brand",
        showgrid=False,
        showline=True,
        linecolor="black",
    )
    fig.update_yaxes(
        title="Percentage share",
        showgrid=False,
        showline=True,
        linecolor="black",
        ticks="inside",
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


@app.get("/exploration")
async def exploration(request: Request, budgets, budget, scenario_type, threshold=0.01):
    scenario_type = scenario_type.capitalize()
    client = get_client(request)
    ny_budget = json.loads(budgets)
    brands_mapping = (
        pd.DataFrame([e for e in mongo["brands"].find({"client": client})])
        .drop(columns="_id")
        .set_index("brand")
    )
    res = brands_mapping[["ly_budget", "ly_net_sales", "return_score", "risk_score"]]

    if scenario_type:
        temp_df = local_get_scenarios(client, budget, threshold)[0]
        res["ny_change"] = temp_df.query("type == @scenario_type").values[0][: len(res)]
    else:
        res["ny_change"] = ny_budget
    res["ny_budget"] = (res["ny_change"] + 100) / 100 * res["ly_budget"]
    ns = (
        (
            (
                (
                    (
                        (res[["ny_budget"]].T / 12 - brands_mapping["x_mean"].T)
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
        )
        * brands_mapping["price"].T
    ).values

    res["ny_net_sales"] = ns[0]
    res["budget_share"] = res["ny_budget"] / res["ny_budget"].sum()
    return_score = sum(res["budget_share"] * res["return_score"])
    risk_score = hmean(res["risk_score"], weights=res["budget_share"])
    # portfolio
    res = res.T
    res["portfolio"] = res.sum(axis=1)
    res = res.T
    res["ny_ns_change"] = (res["ny_net_sales"] / res["ly_net_sales"]) * 100 - 100
    res["ny_budget_change"] = (res["ny_budget"] / res["ly_budget"]) * 100 - 100
    res.loc["portfolio", "return_score"] = return_score
    res.loc["portfolio", "risk_score"] = risk_score
    res["ap_ns_ratio"] = res["ny_budget"] / res["ny_net_sales"]

    changes = res["ny_change"]
    hovertemplate = "".join(
        [
            f"<br>{b}=%{{" + f"customdata[{i}]}}"
            for i, b in enumerate(brands_mapping.index)
        ]
    )
    custom_data = {
        "customdata": [[changes[o] for o in brands_mapping.index]],
        "hovertemplate": "type=Customized<br>risk=%{x}<br>return=%{y}<br>size=%{marker.size}"
        + hovertemplate
        + "<extra></extra>",
        "legendgroup": "Customized",
        "marker": {
            "color": "#ef7f40",
            "size": [5],
            "opacity": [1],
            "sizemode": "area",
            "sizeref": 0.025,
            "symbol": "circle",
        },
        "mode": "markers",
        "name": "Customized",
        "showlegend": True,
        "x": [risk_score],
        "xaxis": "x",
        "y": [return_score],
        "yaxis": "y",
        "type": "scattergl",
    }
    return custom_data


@app.get("/data_metadata")
async def data_metadata(request: Request):
    client = get_client(request)

    columns = list(
        mongo["brands_data"]
        .find_one(
            {
                "client": client,
            },
            {"_id": 0, "client": 0, "Date": 0, "Brand": 0},
        )
        .keys()
    )

    brands = [e["brand"] for e in mongo["brands"].find({"client": client})]

    return {"columns": columns, "brands": brands}


@app.get("/datas")
async def datas(request: Request, brands, columns, start_date, end_date):
    client = get_client(request)
    columns = json.loads(columns) + ["Brand", "Date", "_id"]
    brands = json.loads(brands)

#fillna
    brands_data = [
        e
        for e in mongo["brands_data"].find(
            {
                "client": client,
                "Brand": {"$in": brands},
                "Date": {
                    "$gte": datetime.datetime.strptime(start_date, "%Y-%m-%d"),
                    "$lte": datetime.datetime.strptime(end_date, "%Y-%m-%d"),
                },
            },
            {c: (1 if c != "_id" else 0) for c in columns},
        )
    ]
    return brands_data

