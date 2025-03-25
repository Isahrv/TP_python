#================= Importation des librairies ================#
import dash
from dash import dcc, html, Output, Input, callback
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from calendar import month_abbr, month_name
import plotly.express as px
import dash.dash_table as dt

#================= Configuration de l'application ================#
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

#================= Traitement des données ================#
df = pd.read_csv("./data.csv", index_col=0)
df = df[['CustomerID', 'Gender', 'Location', 'Product_Category', 'Quantity', 'Avg_Price', 'Transaction_Date', 'Month', 'Discount_pct']]

df['CustomerID'] = df['CustomerID'].fillna(0).astype(int)
df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])

df['Total_price'] = df['Quantity'] * df['Avg_Price'] * (1 - (df['Discount_pct'] / 100)).round(3)

def calculer_chiffre_affaire(data):
    return data['Total_price'].sum()

def frequence_meilleure_vente(data, top=10, ascending=False):
    resultat = pd.crosstab(
        [data['Gender'], data['Product_Category']], 
        'Total vente', 
        values=data['Total_price'], 
        aggfunc= lambda x : len(x), 
        rownames=['Sexe', 'Categorie du produit'],
        colnames=['']
    ).reset_index().groupby(
        ['Sexe'], as_index=False, group_keys=True
    ).apply(
        lambda x: x.sort_values('Total vente', ascending=ascending).iloc[:top, :]
    ).reset_index(drop=True).set_index(['Sexe', 'Categorie du produit'])

    return resultat

def indicateur_du_mois(data, current_month = 12, freq=True, abbr=False): 
    previous_month = current_month - 1 if current_month > 1 else 12
    if freq : 
        resultat = data['Month'][(data['Month'] == current_month) | (data['Month'] == previous_month)].value_counts()
        resultat = resultat.sort_index()
        resultat.index = [(month_abbr[i] if abbr else month_name[i]) for i in resultat.index]
        return resultat
    else:
        resultat = data[(data['Month'] == current_month) | (data['Month'] == previous_month)].groupby('Month').apply(calculer_chiffre_affaire)
        resultat.index = [(month_abbr[i] if abbr else month_name[i]) for i in resultat.index]
        return resultat

def barplot_top_10_ventes(data) :
    df_plot = frequence_meilleure_vente(data, ascending=True)
    graph = px.bar(
        df_plot,
        x='Total vente', 
        y=df_plot.index.get_level_values(1),
        color=df_plot.index.get_level_values(0), 
        barmode='group',
        title="Frequence des 10 meilleures ventes",
        labels={"x": "Fréquence", "y": "Categorie du produit", "color": "Sexe"},
        width=680, height=600
    ).update_layout(
        margin = dict(t=60)
    )
    return graph

# Evolution chiffre d'affaire
def plot_evolution_chiffre_affaire(data) :
    df_plot = data.groupby(pd.Grouper(key='Transaction_Date', freq='W')).apply(calculer_chiffre_affaire)[:-1]
    chiffre_evolution = px.line(
        x=df_plot.index, y=df_plot,
        title="Evolution du chiffre d'affaire par semaine",
        labels={"x": "Semaine", "y": "Chiffre d'affaire"},
    ).update_layout( 
        width=1000, height=400,
        margin=dict(t=60, b=0),
        
    )
    return chiffre_evolution

## Chiffre d'affaire du mois
def plot_chiffre_affaire_mois(data) :
    df_plot = indicateur_du_mois(data, freq=False)
    indicateur = go.Figure(
        go.Indicator(
            mode = "number+delta",
            value = df_plot[1],
            delta = {'reference': df_plot[0]},
            domain = {'row': 0, 'column': 1},
            title=f"{df_plot.index[1]}",
        )
    ).update_layout(
        width=200, height=200, 
        margin=dict(l=0, r=20, t=20, b=0)
    )
    return indicateur

# Ventes du mois
def plot_vente_mois(data, abbr=False) :
    df_plot = indicateur_du_mois(data, freq=True, abbr=abbr)
    indicateur = go.Figure(
        go.Indicator(
            mode = "number+delta",
            value = df_plot[1],
            delta = {'reference': df_plot[0]},
            domain = {'row': 0, 'column': 1},
            title=f"{df_plot.index[1]}",
        )
    ).update_layout( 
        width=200, height=200, 
        margin=dict(l=0, r=20, t=20, b=0)
    )
    return indicateur

#================= Interface graphique ================#

graph1 = dcc.Graph(
    id='graph-1',
    figure=barplot_top_10_ventes(df),
    responsive=True,
    className="h-100"
)

graph2 = dcc.Graph(
    id='graph-2',
    figure=plot_evolution_chiffre_affaire(df),
    responsive=True,
    className="h-100"
)

columns_to_display = ["Transaction_Date", "Gender", "Location", "Product_Category", "Quantity", "Avg_Price", "Discount_pct"]

graph3 = dt.DataTable(
    id='graph-3',
    columns=[{"name": col, "id": col} for col in columns_to_display],
    data=df.assign(Transaction_Date=df['Transaction_Date'].dt.strftime('%Y-%m-%d'))
    .sort_values(by="Transaction_Date", ascending=False)
    .head(100)
    .to_dict('records'),page_size=10,
    filter_action="native",
    style_table={
        "width": "80%",          
        "border": "1px solid #ccc"  
    }
)

fig1 = plot_chiffre_affaire_mois(df)
fig2 = plot_vente_mois(df)

dropdown_options = [{'label': loc, 'value': loc} for loc in df['Location'].dropna().unique()]

app.layout = dbc.Container([
    dbc.Col(
        dbc.Row([
            dbc.Col(children='ECAP Store', className='h2'),
            dbc.Col(html.Div([
                dcc.Dropdown(
                    id='zone-dropdown',
                    options=dropdown_options, 
                    searchable=True,
                    placeholder='Choisissez des zones...',
                    disabled=False
                )
            ])),
], style={'backgroundColor': '#ADD8E6', 'padding': '15px'})),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col(dcc.Graph(id="fig1", figure=fig1), md=6),
                dbc.Col(dcc.Graph(id="fig2", figure=fig2), md=6),
                ], style={'height': '30vh'}),
            dbc.Row([dbc.Col(graph1, md = 12, style={'height': '90%'})]
                    , style={'height': '90vh'}),
        ], md=5),
        dbc.Col([
            dbc.Row([dbc.Col(graph2, md = 12, style={'height': '100%'})]
                    , style={'height': '60vh'}),
            dbc.Row([
                dbc.Col(html.H5("Table des 100 dernières ventes")),
                dbc.Col(graph3, md = 12, style={'height': '100%'})]
                , style={'height': '60vh'}),
        ], md=7),
    ], style={'height': '85vh'}),
], fluid=True)
    
@callback(
    [Output('graph-1', 'figure'),
     Output('graph-2', 'figure'),
     Output('graph-3', 'data'),
     Output('fig1', 'figure'),
     Output('fig2', 'figure')],
    [Input('zone-dropdown', 'value')]
)
def update_graphs(selected_zone):
    df_filtered = df[df['Location'].isin([selected_zone] if isinstance(selected_zone, str) else selected_zone)] if selected_zone else df
    return (
        barplot_top_10_ventes(df_filtered),
        plot_evolution_chiffre_affaire(df_filtered),
        df_filtered.assign(Transaction_Date=df_filtered['Transaction_Date'].dt.strftime('%Y-%m-%d'))
    .sort_values(by="Transaction_Date", ascending=False)
    .head(100)
    .to_dict('records'),plot_chiffre_affaire_mois(df_filtered),
        plot_vente_mois(df_filtered)
    )

if __name__ == '__main__':
    app.run(debug=True)