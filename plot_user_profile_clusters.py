import os

import chart_studio
import chart_studio.plotly as py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from dotenv import load_dotenv

# --- CONFIGURATION ---
METHOD = "tsne"  # Options: 'tsne', 'umap', 'pca'
USER_ID = "test_user"
API_BASE = "http://localhost:8000"
LABELED_MODE = True


def load_environment():
    load_dotenv(dotenv_path=os.path.join("backend", ".env"))


def fetch_user_profile_visualization(user_id: str, method: str) -> list:
    api_url = f"{API_BASE}/profiles/{user_id}/visualization?method={method}"
    response = requests.get(api_url)
    response.raise_for_status()
    return response.json()["points"]


def prepare_plot_data(points: list, labeled_mode: bool):
    df = pd.DataFrame(points)
    clusters_df = df[df["type"] == "cluster"].copy()
    movies_df = df[df["type"] == "movie"].copy()

    # For each cluster, select up to 30 movies with highest similarity to the centroid
    selected_movies = []
    n_cluster_movies = 0 if labeled_mode else 30
    for _, cluster_row in clusters_df.iterrows():
        cluster_x, cluster_y = cluster_row["x"], cluster_row["y"]
        cluster_label = cluster_row["label"]
        cluster_movies = movies_df.copy()
        cluster_movies["distance"] = np.sqrt(
            (cluster_movies["x"] - cluster_x) ** 2
            + (cluster_movies["y"] - cluster_y) ** 2
        )
        top_movies = cluster_movies.nsmallest(n_cluster_movies, "distance").copy()
        top_movies["cluster_id"] = cluster_label
        selected_movies.append(top_movies)
    selected_movies_df = (
        pd.concat(selected_movies)
        if selected_movies
        else pd.DataFrame(columns=movies_df.columns)
    )

    # Combine clusters and selected movies
    plot_df = pd.concat([clusters_df, selected_movies_df])
    plot_df["show_label"] = plot_df["type"] == "cluster"

    # Add random movies not already selected for any cluster
    already_selected = (
        set(selected_movies_df["label"]) if not selected_movies_df.empty else set()
    )
    random_candidates = movies_df[~movies_df["label"].isin(already_selected)]
    x_min, x_max = clusters_df["x"].min(), clusters_df["x"].max()
    y_min, y_max = clusters_df["y"].min(), clusters_df["y"].max()
    x_margin = 0.15 * (x_max - x_min)
    y_margin = 0.15 * (y_max - y_min)
    x_min_plot, x_max_plot = x_min - x_margin, x_max + x_margin
    y_min_plot, y_max_plot = y_min - y_margin, y_max + y_margin
    if not labeled_mode:
        in_plot_mask = (
            (random_candidates["x"] >= x_min_plot)
            & (random_candidates["x"] <= x_max_plot)
            & (random_candidates["y"] >= y_min_plot)
            & (random_candidates["y"] <= y_max_plot)
        )
        random_candidates = random_candidates[in_plot_mask]
    n_random = 10000 if labeled_mode else min(150, len(random_candidates))
    n_random = min(n_random, len(random_candidates))
    random_movies = (
        random_candidates.sample(n=n_random, random_state=42).copy()
        if n_random > 0
        else pd.DataFrame(columns=movies_df.columns)
    )
    if not random_movies.empty:
        random_movies["type"] = "movie"
        random_movies["show_label"] = False
    plot_df = pd.concat([plot_df, random_movies])

    # Update type labels for legend clarity
    clusters_df["type"] = "centroid"
    selected_movies_df["type"] = "movie (clustered)"
    if not random_movies.empty:
        random_movies["type"] = "movie"
        random_movies["show_label"] = False
    plot_df = pd.concat([clusters_df, selected_movies_df, random_movies])

    clusters_df["cluster_id"] = clusters_df["label"]
    if not random_movies.empty:
        random_movies["cluster_id"] = "random"
        random_movies["show_label"] = False
    plot_df = pd.concat([clusters_df, selected_movies_df, random_movies])

    if labeled_mode:
        plot_df.loc[
            plot_df["type"].isin(["movie", "movie (clustered)"]), "show_label"
        ] = True
    size_map = {
        "centroid": 15,
        "movie (clustered)": 2,
        "movie": 0.5 if labeled_mode else 2,
    }
    cluster_labels = clusters_df["label"].tolist()
    cluster_colors = px.colors.qualitative.Plotly[: len(cluster_labels)]
    color_discrete_map = {
        label: color for label, color in zip(cluster_labels, cluster_colors)
    }
    if labeled_mode:
        color_discrete_map["movie"] = "#FFFFFF"
    else:
        color_discrete_map["random"] = "#888888"
    n_movies = plot_df[plot_df["type"].isin(["movie", "movie (clustered)"])].shape[0]
    return (
        plot_df,
        clusters_df,
        selected_movies_df,
        random_movies,
        size_map,
        color_discrete_map,
        n_movies,
    )


def create_plot(
    plot_df,
    clusters_df,
    selected_movies_df,
    random_movies,
    size_map,
    color_discrete_map,
    n_movies,
    labeled_mode,
):
    symbol_map = {
        "centroid": "circle-open",
        "movie (clustered)": "circle",
        "movie": "circle" if labeled_mode else "cross",
    }
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="type" if labeled_mode else "cluster_id",
        symbol="type",
        symbol_map=symbol_map,
        color_discrete_map=color_discrete_map,
        text=plot_df["label"].where(plot_df["show_label"], ""),
        size=[size_map.get(t, 4) for t in plot_df["type"]],
        hover_name="label",
        hover_data={
            "year": True,
            "genres": True,
            "rating": True,
            "duration": True,
        },
        title=f"User Profile Clusters Visualization for {USER_ID} ({METHOD.upper()} projection from 3072D embeddings)\nTotal movies plotted: {n_movies}",
        labels={"x": "X", "y": "Y", "type": "type", "cluster_id": "Cluster"},
        opacity=0.7,
    )
    fig.update_traces(
        textposition="top center",
        marker=dict(line=dict(width=1, color="DarkSlateGrey")),
    )
    for _, row in clusters_df.iterrows():
        cluster_color = color_discrete_map.get(row["label"], "rgba(0,100,255,0.15)")
        if cluster_color.startswith("#"):
            h = cluster_color.lstrip("#")
            rgb = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
            fillcolor = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.15)"
        else:
            fillcolor = cluster_color
        fig.add_shape(
            type="circle",
            xref="x",
            yref="y",
            x0=row["x"] - 3.5,
            y0=row["y"] - 3.5,
            x1=row["x"] + 3.5,
            y1=row["y"] + 3.5,
            fillcolor=fillcolor,
            line_color="rgba(0,0,0,0)",
            layer="below",
        )
    movie_points = plot_df[plot_df["type"] != "cluster (centroid)"]
    fig.add_trace(
        go.Histogram2dContour(
            x=movie_points["x"],
            y=movie_points["y"],
            colorscale="Blues",
            reversescale=True,
            showscale=False,
            contours=dict(coloring="fill"),
            opacity=0.17,
            line_width=0,
            ncontours=6,
            hoverinfo="skip",
        )
    )
    x_min, x_max = plot_df["x"].min(), plot_df["x"].max()
    y_min, y_max = plot_df["y"].min(), plot_df["y"].max()
    x_margin = 0.1 * (x_max - x_min)
    y_margin = 0.1 * (y_max - y_min)
    fig.update_layout(
        xaxis_range=[x_min - x_margin, x_max + x_margin],
        yaxis_range=[y_min - y_margin, y_max + y_margin],
        dragmode="pan",
    )
    return fig


def upload_plot(fig):
    username = os.environ.get("PLOTLY_USERNAME")
    api_key = os.environ.get("PLOTLY_API_KEY")
    chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
    url = py.plot(fig, filename="user_profile_clusters", auto_open=False)
    print("Shareable link:", url)
    return url


def main():
    load_environment()
    points = fetch_user_profile_visualization(USER_ID, METHOD)
    (
        plot_df,
        clusters_df,
        selected_movies_df,
        random_movies,
        size_map,
        color_discrete_map,
        n_movies,
    ) = prepare_plot_data(points, LABELED_MODE)
    fig = create_plot(
        plot_df,
        clusters_df,
        selected_movies_df,
        random_movies,
        size_map,
        color_discrete_map,
        n_movies,
        LABELED_MODE,
    )
    fig.show(config={"scrollZoom": True})
    upload_plot(fig)


if __name__ == "__main__":
    main()
