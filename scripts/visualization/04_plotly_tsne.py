from dash import Dash, dcc, html, Input, Output, no_update, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from rdkit.Chem import PandasTools
from ast import literal_eval
from collections import Counter
import pickle as pkl

# NOTE used to create pkl files. Using pkl for faster loading
# df = pd.read_csv("tsne_fingerprint_plotly.csv")
# # df = pd.read_csv("tsne_chemberta_plotly.csv")
# df["summarizations"] = df["summarizations"].apply(literal_eval)
# df.to_pickle("tsne_fingerprint_plotly.pkl")
# # get terms in sorted order
# all_terms = Counter()
# for i in df["summarizations"]:
#     all_terms.update(i)
# # get just keys of sorted terms
# sorted_terms = [i[0] for i in all_terms.most_common()]
# with open("sorted_terms.pkl", "wb") as f:
#     # pickle
#     pkl.dump(sorted_terms, f)

df = pd.read_pickle("tsne_fingerprint_plotly.pkl")
with open("sorted_terms.pkl", "rb") as f:
    # pickle
    sorted_terms = pkl.load(f)




# plot
df["color"] = df["summarizations"].apply(lambda x: "#FF6692" if "antiviral" in x else "black")
df["opacity"] = df["summarizations"].apply(lambda x: 1 if "antiviral" in x else 0.3)
df["size"] = df["summarizations"].apply(lambda x: 5 if "antiviral" in x else 1.5)
fig = go.Figure(data=[
    go.Scattergl(
        # x=df["chemberta_tsne_x"],
        # y=df["chemberta_tsne_y"],
        x=df["fp_tsne_x"],
        y=df["fp_tsne_y"],
        mode="markers",
        marker=dict(
            opacity=df["opacity"],
            color=df["color"],
            size=df["size"],
            line={"color": "#000000"}, # black
        ),
        hoverinfo="skip",
    )
])
# turn off native plotly.js hover effects - make sure to use
# hoverinfo="none" rather than "skip" which also halts events.
fig.update_traces(hoverinfo="none", hovertemplate=None)

fig.update_layout(
    plot_bgcolor='rgba(255,255,255,0.1)'
)

# make plot large square
fig.update_layout(
    # width=1600,
    height=800,
    # autosize=False,
    # margin=dict(l=0, r=0, b=0, t=0, pad=0),
    # dark theme
    template="ggplot2",

)

# remove legend of both traces
fig.update_layout(showlegend=False)

# hide axis and tick marks, and numbers
fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)
fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)





app = Dash(external_stylesheets=[dbc.themes.MINTY])

# make app background color match ggplot2
app.layout = html.Div([
    # make plot on left side of page
    html.Div([
        dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip"),
    ], style={'width': '85%', 'display': 'inline-block', 'text-align': 'center', 'vertical-align': 'middle', 'height': '100%'}),
    # make smaller and on right side of plot. Near top 10% of page. Pad on right side
    html.Div([
        dcc.Dropdown(sorted_terms, 'antiviral', id='term-dropdown',),
        html.Div(id='dd-output-container'),
    ], style={'width': '15%', 'display': 'inline-block', 'text-align': 'center', 'vertical-align': 'top', 'margin-top': '10%', 'padding-right': '3%'}),
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




@callback(
    Output('dd-output-container', 'children'),
    Output('graph-basic-2', 'figure'),
    Input('term-dropdown', 'value')
)
def update_output(value):
    # replot with new value
    df["color"] = df["summarizations"].apply(lambda x: "#FF6692" if value in x else "black")
    df["opacity"] = df["summarizations"].apply(lambda x: 1 if value in x else 0.3)
    df["size"] = df["summarizations"].apply(lambda x: 5 if value in x else 1.5)
    fig = go.Figure(data=[
        go.Scattergl(
            # x=df["chemberta_tsne_x"],
            # y=df["chemberta_tsne_y"],
            x=df["fp_tsne_x"],
            y=df["fp_tsne_y"],
            mode="markers",
            marker=dict(
                opacity=df["opacity"],
                color=df["color"],
                size=df["size"],
                line={"color": "#000000"}, # black
            ),
            hoverinfo="skip",
        )
    ])
    # turn off native plotly.js hover effects - make sure to use
    # hoverinfo="none" rather than "skip" which also halts events.
    fig.update_traces(hoverinfo="none", hovertemplate=None)

    fig.update_layout(
        plot_bgcolor='rgba(255,255,255,0.1)'
    )

    # make plot large square
    fig.update_layout(
        # width=1600,
        height=800,
        # autosize=False,
        # margin=dict(l=0, r=0, b=0, t=0, pad=0),
        template="ggplot2",
    )

    # remove legend of both traces
    fig.update_layout(showlegend=False)

    # hide axis and tick marks, and numbers
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)

    return f'Showing molecules with label: {value}', fig


if __name__ == "__main__":
    app.run(debug=True)
