import os
from dash import Dash, dcc, html, Input, Output, no_update, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from rdkit.Chem import PandasTools, MolFromSmiles, Draw
from ast import literal_eval
from collections import Counter
import pickle as pkl
from io import BytesIO
import base64
from PIL import Image


def pil_to_b64(pil):
    with BytesIO() as buffer:
        pil.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return "data:image/png;base64, " + encoded_image

df = pd.read_pickle("../../../results/CheF_v5_fb_tsne.pkl")

with open("../../../results/all_labels.txt", "r") as f:
    sorted_terms = f.read().splitlines()

fig = go.Figure(data=[
    go.Scattergl(
            x=df[df["summarizations"].map(lambda x: not any("antiviral" == word for word in x))][f"fp_tsne_x"],
            y=df[df["summarizations"].map(lambda x: not any("antiviral" == word for word in x))][f"fp_tsne_y"],
        mode="markers",
        marker=dict(
            opacity=0.1,
            color="#000000",
            size=2,
            line={"color": "#000000", "width":0.1}, # black
        ),
        hoverinfo="skip",
    ),
    go.Scattergl(
        # check if "antiviral" is in the df['summarizations'] list
            x=df[df["summarizations"].map(lambda x: any("antiviral" == word for word in x))][f"fp_tsne_x"],
            y=df[df["summarizations"].map(lambda x: any("antiviral" == word for word in x))][f"fp_tsne_y"],
        mode="markers",
        marker=dict(
            opacity=1,
            color="#AB63FA",
            size=5,
            line=dict(width=0.5, color='DarkSlateGrey'),
        ),
        hoverinfo="skip",
    ),
])

# turn off native plotly.js hover effects - make sure to use
# hoverinfo="none" rather than "skip" which also halts events.
fig.update_traces(hoverinfo="none", hovertemplate=None)

fig.update_layout(
    plot_bgcolor='rgba(255,255,255,0.1)'
)

# make plot large square
fig.update_layout(
    width=900,
    height=900,
    # autosize=False,
    margin=dict(l=0, r=0, b=0, t=0, pad=0),
    # dark theme
    template="ggplot2",

)

# remove legend of both traces
fig.update_layout(showlegend=False)

# hide axis and tick marks, and numbers
# fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)
# fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)
# increase font size of axis ticks
fig.update_xaxes(tickfont=dict(size=28))
fig.update_yaxes(tickfont=dict(size=28))




app = Dash(external_stylesheets=[dbc.themes.MINTY])
app.title = "CheF-100k Interactive"
# app._favicon = "umap.ico"
server = app.server

# make app background color match ggplot2
app.layout = html.Div([
    # make plot on left side of page
    html.Div([
        # make graph autosize to browser window
        dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip"),
    ], style={'width': '60%', 'display': 'inline-block', 'text-align': 'center', 'vertical-align': 'middle', 'height': '1000px', 'padding-left':'10%'}),
    # make smaller and on right side of plot. Near top 10% of page. Pad on right side
    html.Div([
        # header
        # html.H1("CheF-100k Interactive Fingerprint t-SNE"),
        html.H1("CheF-100k Interactive Structural t-SNE"),
        # dcc.Dropdown(["Fingerprint t-SNE", "ChemBERTa t-SNE", "ChemBERTa UMAP"], 'Fingerprint t-SNE', id='data-dropdown',),
        # html.Div(id='data-to-plot-output-container', children="Plotting Fingerprint t-SNE"),

        # create spacer
        html.Div(style={'height': '20px'}),
        dcc.Dropdown(sorted_terms, 'antiviral', id='term-dropdown',),
        html.Div(id='dd-output-container', children="Showing molecules with label: antiviral"),
    ], style={'width': '38%', 'display': 'inline-block', 'text-align': 'center', 'vertical-align': 'top', 'margin-top': '10%', 'padding-left':'5%', 'padding-right': '15%'}),
])



@callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("graph-basic-2", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    # num = pt["pointNumber"]

    # make sure to select from correct data, as there are two overlayed plots
    # This doesn't work with my subsetting of the dataframe during subgraph plotting
    # df_row = df.iloc[num]

    # get df row with same xy coordinates as hoverData. Kind of a cheat but it works!!!
    df_row = df[(df[f"fp_tsne_x"] == pt["x"]) & (df[f"fp_tsne_y"] == pt["y"])].iloc[0]

    img_src = pil_to_b64(Draw.MolToImage(MolFromSmiles(df_row['smiles']), size=(200, 200)))
    cid = df_row['cid']
    smi = df_row['smiles']
    summ = df_row['summarizations']


    children = [
        html.Div([
            html.Img(src=img_src, style={"width": "100%"}),
            html.H2(f"{cid}", style={"color": "darkblue", "overflow-wrap": "break-word"}),
            # wrap smi so it stays within box. It is one long string that won't normally wrap
            html.P(f"{smi}", style={'width': '200px', 'overflow-wrap':'break-word'}),
            html.P(f"{summ}", style={'width': '200px', 'overflow-wrap':'break-word'}),
        ], style={'width': '200px', 'white-space': 'normal'})
    ]

    return True, bbox, children




@callback(
    Output('dd-output-container', 'children'),
    Output('graph-basic-2', 'figure'),
    Input('term-dropdown', 'value'),
)
def update_output(value):
    
    fig = go.Figure(data=[
        go.Scattergl(
            x=df[df["summarizations"].map(lambda x: not any(value == word for word in x))][f"fp_tsne_x"],
            y=df[df["summarizations"].map(lambda x: not any(value == word for word in x))][f"fp_tsne_y"],

            mode="markers",
            marker=dict(
                opacity=0.1,
                color="#000000",
                size=2,
            line={"color": "#000000", "width":0.1}, # black
            ),
            hoverinfo="skip",
        ),
        go.Scattergl(
            x=df[df["summarizations"].map(lambda x: any(value == word for word in x))][f"fp_tsne_x"],
            y=df[df["summarizations"].map(lambda x: any(value == word for word in x))][f"fp_tsne_y"],
            mode="markers",
            marker=dict(
                opacity=1,
                color="#AB63FA",
                size=5,
                # line={"color": "#000000"}, # black
                # show borders
                line=dict(width=0.5, color='DarkSlateGrey'),
            ),
            hoverinfo="skip",
        ),
    ])
    # turn off native plotly.js hover effects - make sure to use
    # hoverinfo="none" rather than "skip" which also halts events.
    fig.update_traces(hoverinfo="none", hovertemplate=None)

    fig.update_layout(
        plot_bgcolor='rgba(255,255,255,0.1)'
    )

    # make plot large square
    fig.update_layout(
        width=900,
        height=900,
        # autosize=False,
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        template="ggplot2",
    )

    # remove legend of both traces
    fig.update_layout(showlegend=False)

    # hide axis and tick marks, and numbers
    # fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)
    # fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)
    fig.update_xaxes(tickfont=dict(size=28))
    fig.update_yaxes(tickfont=dict(size=28))

    return f'Showing molecules with label: {value}', fig


if __name__ == "__main__":
    # app.run_server(debug=False)
    app.run(debug=True)