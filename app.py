# Define the Dash App and it's properties here 

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from pandas.io.formats import style
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import numpy as np

new_issue_concession = pd.read_csv('../NIC/data/new_issue_concession.csv')
new_issue_concession['PricingDate'] = pd.to_datetime(new_issue_concession['PricingDate'])
new_issue_concession.rename(columns={'Size (m)': 'Size_m', 'Book Size (m)': 'Book_Amount'}, inplace=True)
new_issue_concession['Nic']=pd.to_numeric(new_issue_concession['Nic'], errors='coerce')
new_issue_concession['month']=pd.PeriodIndex(new_issue_concession.PricingDate, freq='m')
new_issue_concession['avg_nic_per_month_all_sectors'] = new_issue_concession.groupby('month').Nic.transform('mean')
new_issue_concession['avg_nic_per_month_sector'] = new_issue_concession.groupby(['month', 'Sector']).Nic.transform('mean')
new_issue_concession['Month_Volume'] = new_issue_concession.groupby(['month'])['Size_m'].sum()

treasury = pd.read_csv('../NIC/data/treasury.csv')
treasury['Date'] = pd.to_datetime(treasury['Date'])
treasury['month'] = pd.PeriodIndex(treasury.Date, freq='m')
avg_treasury = treasury.groupby('month').agg({'two_year':'mean', 'ten_year':'mean'}).reset_index().set_index('month').sort_values('month', ascending =False)

nic_treasury_merged = pd.merge(new_issue_concession, avg_treasury[['two_year', 'ten_year']], on='month', how='left')
nic_treasury_merged_avg = nic_treasury_merged[['month', 'Sector','avg_nic_per_month_all_sectors','avg_nic_per_month_sector','two_year', 'ten_year']]
month_averages = nic_treasury_merged_avg.drop_duplicates(subset = ['month', 'Sector'], keep = 'first')

nic_dedupe = new_issue_concession[['month','Sector', 'avg_nic_per_month_sector']].drop_duplicates(subset = ['month', 'Sector'], keep = 'first')

nic_sector_month =avg_treasury.reset_index()
nic_sector_month_melted= pd.melt(nic_sector_month, id_vars='month', value_vars=['two_year','ten_year'])

test = pd.DataFrame(np.vstack([nic_dedupe, nic_sector_month_melted]), columns=nic_dedupe.columns)

test2=test.set_index('month')

tripple_a = ['Aaa','AAA']
double_a = ['Aa1','AA+','Aa2','AA','Aa3','AA-']
single_a = ['A1','A2','A3','A+','A','A-']
tripple_b =['Baa1','Baa2','Baa3','BBB+','BBB','BBB-']

conditions = [
    (nic_treasury_merged['S&P'].isin(tripple_a)) | (nic_treasury_merged['Moodys'].isin(tripple_a)),
    (nic_treasury_merged['S&P'].isin(double_a)) | (nic_treasury_merged['Moodys'].isin(double_a)),
    (nic_treasury_merged['S&P'].isin(single_a)) | (nic_treasury_merged['Moodys'].isin(single_a)),
    (nic_treasury_merged['S&P'].isin(tripple_b)) | (nic_treasury_merged['Moodys'].isin(tripple_b))
]

values = ['AAA', 'AA', 'A', 'BBB']

nic_treasury_merged['ratings'] = np.select(conditions, values)


nic_by_ratings= nic_treasury_merged[['month','Nic','ratings']]
include_ratings=['AAA', 'AA', 'A', 'BBB']

nic_by_ratings_time_series=nic_by_ratings[nic_by_ratings['ratings'].isin(include_ratings)]

xcx=nic_by_ratings_time_series.groupby(['month', 'ratings'])['Nic'].mean().reset_index().set_index('month')

nic_by_rating=px.bar(xcx, x=xcx.index.strftime("%Y-%m"), y='Nic', color= 'ratings', category_orders={"ratings": ['AAA', 'AA', 'A', 'BBB']}, template="plotly_dark")
nic_by_rating.update_layout(xaxis=dict(
        title='Timeline',
        titlefont_size=14,
        tickfont_size=12
        ))
nic_by_rating.update_yaxes(title=dict(text='New Issue Concession (Basis Points)'))


nic_vols=nic_treasury_merged.groupby('month').agg({'Nic': 'mean', 'Size_m':'sum'}).reset_index().set_index('month')


CHART_THEME = "plotly_dark"

# Create figure with secondary y-axis
nic_vols_fig = make_subplots(specs=[[{"secondary_y": True}]])
nic_vols_fig.layout.template = CHART_THEME
# Add traces
nic_vols_fig.add_trace(
    go.Bar(x=nic_vols.index.strftime("%Y-%m"), y=nic_vols['Size_m']*1e6, name="Volume US$"),
    secondary_y=False,
)
nic_vols_fig.add_trace(
    go.Scatter(x=nic_vols.index.strftime("%Y-%m"), y=nic_vols['Nic'], name="New Issue Concession (bp)"),
    secondary_y=True,
)

nic_vols_fig.layout.height=500
nic_vols_fig.update_layout(margin = dict(t=50, b=50, l=25, r=25))  # optm the chart space

nic_vols_fig.update_layout(
    title='Volume and Average New Issue Concession By Month',
    #xaxis=dict(tickmode = 'array', tickvals = nic_vols.index, ticktext = nic_vols['month']),
    yaxis=dict(
        title='Value:$US',
        titlefont_size=14,
        tickfont_size=12,
        ))
nic_vols_fig.update_yaxes(title_text='New Issue Concession (basis points)', secondary_y =True)


app_nic4 = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP], 
                meta_tags=[{"name": "viewport", "content": "width=device-width"}],
                suppress_callback_exceptions=True)

sector_options = test2['Sector'].unique()

colors = {"background": "#000000", "text": "#FFFFFF"}


app_nic4.layout = html.Div([
  #html.Img(src=logo_link, style={'margin':'30px 0px 0px 0px' }),
  html.H1('NEW ISSUE CONCESSIONS TRENDS FOR US INVESTMENT GRADE'),
  html.Div(
    children=[
    html.Div(
        children=[
          html.H3('NEW ISSUE CONCESSION BY RATING'),
          dcc.Dropdown(),
          dcc.Graph(id='scatter', figure=nic_by_rating),
        ],
        style={'width':'350px', 'height':'650px', 'display':'inline-block', 
               'vertical-align':'top', 'border':'1px solid black', 'padding':'20px'}),
    html.Div(
      children=[
        dcc.Dropdown(id='select-type',
                     options = [{'label': i, 'value': i} for i in sector_options],
                     value = ['two_year', 'ten_year'],
                     multi=True                    
                     
        ),  
        dcc.Graph(id='graph'),
          
         dcc.Dropdown(id='select-type2'                 
                      
         ),
        dcc.Graph(id='graph2', figure=nic_vols_fig)
      ],
      style={'width':'700px', 'height':'650px','display':'inline-block'})
    ]),], 
  style={'text-align':'center', 'display':'inline-block', 'width':'100%'}
  )

@app_nic4.callback(Output('graph', 'figure'),
             [Input('select-type', 'value')])

def make_figure(sector_options):
    if  ['two_year', 'ten_year'] in sector_options:
        dff = test2.copy()
    else:
        dff = test2.loc[test2['Sector'].isin(sector_options)]
    
    
    
    fig = px.line(
        dff,
        x=dff.index.strftime("%Y-%m"),
        y=dff['avg_nic_per_month_sector'],
        color = 'Sector'

        )
    fig.update_layout(
        plot_bgcolor=colors["background"],
        paper_bgcolor=colors["background"],
        font_color=colors["text"],
        
    )   
    fig.update_xaxes(title_text="Timeline")
    fig.update_yaxes(title_text="New Issue Concession (basis points)")
 

    return fig



if __name__ == '__main__':
    app_nic4.run_server(debug=False, port=9002)