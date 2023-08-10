from dash import Dash, dcc, html, Input, Output, no_update, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from rdkit.Chem import PandasTools
from ast import literal_eval
from collections import Counter
import pickle as pkl

# NOTE used to create pkl files. Using pkl for faster loading
# all_terms = Counter()
# for i in df["summarizations"]:
#     all_terms.update(i)
# # get just keys of sorted terms
# sorted_terms = [i[0] for i in all_terms.most_common()]
# with open("sorted_terms.pkl", "wb") as f:
#     # pickle
#     pkl.dump(sorted_terms, f)

df = pd.read_pickle("umap_chemberta_plotly.pkl")
with open("sorted_terms.pkl", "rb") as f:
    # pickle
    sorted_terms = pkl.load(f)




# # # plot
# df["color"] = df["summarizations"].map(lambda x: "#FF6692" if "antiviral" in x else "black")
# df["opacity"] = df["summarizations"].map(lambda x: 1 if "antiviral" in x else 0.3)
# df["size"] = df["summarizations"].map(lambda x: 5 if "antiviral" in x else 1.5)
# fig = go.Figure(data=[
#     go.Scattergl(
        
#         x=df["cb_umap_x"],
#         y=df["cb_umap_y"],
#         mode="markers",
#         marker=dict(
#             opacity=df["opacity"],
#             color=df["color"],
#             size=df["size"],
#             line={"color": "#000000"}, # black


#         ),
#         hoverinfo="skip",
#     )
# ])

df_term_true = df[df["summarizations"].map(lambda x: True if "antiviral" in x else False)]
df_term_false = df[df["summarizations"].map(lambda x: False if "antiviral" in x else True)]

fig = go.Figure(data=[
    go.Scattergl(
        x=df[df["summarizations"].map(lambda x: False if "antiviral" in x else True)]["cb_umap_x"],
        y=df[df["summarizations"].map(lambda x: False if "antiviral" in x else True)]["cb_umap_y"],
        mode="markers",
        marker=dict(
            opacity=0.3,
            color="#000000",
            size=2,
            line={"color": "#000000"}, # black
        ),
        hoverinfo="skip",
    ),
    go.Scattergl(
        # check if "antiviral" is in the df['summarizations'] list
        x=df[df["summarizations"].map(lambda x: True if "antiviral" in x else False)]["cb_umap_x"],
        y=df[df["summarizations"].map(lambda x: True if "antiviral" in x else False)]["cb_umap_y"],
        mode="markers",
        marker=dict(
            opacity=1,
            color="#FF6692",
            size=5,
            line=dict(width=0.7, color='DarkSlateGrey'),
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
    width=1200,
    height=1200,
    # autosize=False,
    margin=dict(l=0, r=0, b=0, t=0, pad=0),
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
    ], style={'width': '60%', 'display': 'inline-block', 'text-align': 'center', 'vertical-align': 'middle', 'height': '100%', 'padding-left':'10%'}),
    # make smaller and on right side of plot. Near top 10% of page. Pad on right side
    html.Div([
        # header
        html.H1("ChAPL-100k Interactive ChemBERTa UMAP"),
        dcc.Dropdown(sorted_terms, 'antiviral', id='term-dropdown',),
        html.Div(id='dd-output-container'),
    ], style={'width': '40%', 'display': 'inline-block', 'text-align': 'center', 'vertical-align': 'top', 'margin-top': '10%', 'padding-left':'5%', 'padding-right': '15%'}),
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

    # make sure to select from correct data, as there are two overlayed plots
    # This doesn't work with my subsetting of the dataframe during subgraph plotting
    # df_row = df.iloc[num]

    # get df row with same xy coordinates as hoverData. Kind of a cheat but it works!!!
    df_row = df[(df["cb_umap_x"] == pt["x"]) & (df["cb_umap_y"] == pt["y"])].iloc[0]

    img_src = df_row['im_url']
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
    Input('term-dropdown', 'value')
)
def update_output(value):
    # # # replot with new value
    # df["color"] = df["summarizations"].map(lambda x: "#FF6692" if value in x else "black")
    # df["opacity"] = df["summarizations"].map(lambda x: 1 if value in x else 0.3)
    # df["size"] = df["summarizations"].map(lambda x: 5 if value in x else 1.5)
    # fig = go.Figure(data=[
    #     go.Scattergl(
    #         x=df["cb_umap_x"],
    #         y=df["cb_umap_y"],
    #         mode="markers",
    #         marker=dict(
    #             opacity=df["opacity"],
    #             color=df["color"],
    #             size=df["size"],
    #             line={"color": "#000000"}, # black
    #         ),
    #         hoverinfo="skip",
    #     )
    # ])

    df_tmp = df[df["summarizations"].map(lambda x: True if value in x else False)]

    fig = go.Figure(data=[
        go.Scattergl(
            x=df[df["summarizations"].map(lambda x: False if value in x else True)]["cb_umap_x"],
            y=df[df["summarizations"].map(lambda x: False if value in x else True)]["cb_umap_y"],
            mode="markers",
            marker=dict(
                opacity=0.3,
                color="#000000",
                size=2,
                line={"color": "#000000"}, # black
            ),
            hoverinfo="skip",
        ),
        go.Scattergl(
            x=df[df["summarizations"].map(lambda x: True if value in x else False)]["cb_umap_x"],
            y=df[df["summarizations"].map(lambda x: True if value in x else False)]["cb_umap_y"],
            mode="markers",
            marker=dict(
                opacity=1,
                color="#FF6692",
                size=5,
                # line={"color": "#000000"}, # black
                # show borders
                line=dict(width=0.7, color='DarkSlateGrey'),
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
        width=1200,
        height=1200,
        # autosize=False,
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        template="ggplot2",
    )

    # remove legend of both traces
    fig.update_layout(showlegend=False)

    # hide axis and tick marks, and numbers
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)

    return f'Showing molecules with label: {value}', fig


if __name__ == "__main__":
    app.run_server(debug=True,port=8051)