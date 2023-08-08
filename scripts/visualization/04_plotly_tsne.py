from dash import Dash, dcc, html, Input, Output, no_update, callback
import plotly.graph_objects as go
import pandas as pd
from rdkit.Chem import PandasTools
from ast import literal_eval



df = pd.read_csv("tsne_plotly.csv")
df["summarizations"] = df["summarizations"].apply(literal_eval)

df["color"] = df["summarizations"].apply(lambda x: "red" if "antiviral" in x else "black")
df["opacity"] = df["summarizations"].apply(lambda x: 1 if "antiviral" in x else 0.3)
df["size"] = df["summarizations"].apply(lambda x: 5 if "antiviral" in x else 2)
fig = go.Figure(data=[
    go.Scatter(
        x=df["chemberta_tsne_x"],
        y=df["chemberta_tsne_y"],
        mode="markers",
        marker=dict(
            opacity=df["opacity"],
            color=df["color"],
            size=df["size"],
            line={"color": "#000000"}, # black
            # colorscale='viridis',
            # color=df["MW"],
            # size=df["MW"],
            # colorbar={"title": "t-SNE of ChemBERTa embeddings w/ ChAPL Embeddings"},
            # reversescale=True,
            # sizeref=45,
            # sizemode="diameter",
        ),
        hoverinfo="skip",
    )
])

# # color points red if they contain "opioid" in summarizations
# fig.add_trace(
#     go.Scatter(
#         x=df["chemberta_tsne_x"],
#         y=df["chemberta_tsne_y"],
#         mode="markers",
#         marker=dict(
#             color=df["color"],
#             size=10,
#             opacity=1,
#             line={"color": "#444"},
#         ),
#         hoverinfo="skip",
#     )
# )

# turn off native plotly.js hover effects - make sure to use
# hoverinfo="none" rather than "skip" which also halts events.
fig.update_traces(hoverinfo="none", hovertemplate=None)

fig.update_layout(
    # xaxis=dict(title='t-SNE x'),
    # yaxis=dict(title='t-SNE y'),
    plot_bgcolor='rgba(255,255,255,0.1)'
)

# make plot large square
fig.update_layout(
    width=1600,
    height=800,
    autosize=False,
    margin=dict(l=0, r=0, b=0, t=0, pad=0),

)

# remove legend of both traces
fig.update_layout(showlegend=False)

# hide axis and tick marks, and numbers
fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)
fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)

# make point size small
# fig.update_traces(marker=dict(size=1.5))

app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
    dcc.Tooltip(id="graph-tooltip"),
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
    num = pt["pointNumber"]

    df_row = df.iloc[num]
    img_src = df_row['im_url']
    cid = df_row['cid']
    smi = df_row['smiles']
    summ = df_row['summarizations']


    children = [
        html.Div([
            html.Img(src=img_src, style={"width": "100%"}),
            html.H2(f"{cid}", style={"color": "darkblue", "overflow-wrap": "break-word"}),
            html.P(f"{smi}"),
            html.P(f"{summ}"),
        ], style={'width': '200px', 'white-space': 'normal'})
    ]

    return True, bbox, children


if __name__ == "__main__":
    app.run(debug=True)
