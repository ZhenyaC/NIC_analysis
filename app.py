# Define the Dash App and it's properties here 

import dash
import dash_core_components as dcc
import dash_html_components as html
#from dash_html_components.Label import Label
from pandas.io.formats import style
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import numpy as np
import dash_bootstrap_components as dbc

new_issue_concession = pd.read_csv('../NIC_analysis/data/new_issue_concession.csv')

exclude_these = [
25010511,
25010383,
25010411,
25010448,
25010449,
25010324,
25010263,
25010260,
25010267,
25010315,
25010175,
25010183,
25010170,
25010169,
25009979,
25010038,
25009961,
25010061,
25009911,
25009935,
25009648,
25009633,
25009580,
25009643,
25009422,
25009505,
25009434,
25009457,
25009423,
25009489,
25009302,
25009130,
25009366,
25009289,
25009314,
25009313,
25009312,
25009271,
25009288,
25009154,
25009092,
25009147,
25008935,
25008970,
25008972,
25008971,
25008949,
25008966,
25008945,
25008944,
25008922,
25008860,
25008539,
25008802,
25008706,
25008340,
25008619,
25008550,
25008450,
25008557,
25008537,
25008570,
25008554,
25008496,
25008236,
25008378,
25008387,
25008321,
25008295,
25008294,
25008026,
25008319,
25008199,
25008200,
25008209,
25008229,
25008179,
25008021,
25008058,
25008057,
25008056,
25008035,
25008022,
25008024,
25007968,
25007932,
25007927,
25007928,
25007887,
25007888,
25007940,
25007959,
25007926,
25007869,
25007933,
25007813,
25007863,
25007826,
25007825,
25007638,
25007640,
25007449,
25007062,
25007372,
25007405,
25007350,
25007331,
25007259,
25007241,
25007240,
25007193,
25007208,
25007205,
25007144,
25007194,
25007195,
25007145,
25007016,
25007015,
25007053,
25007018,
25006857,
25006835,
25004097,
25006702,
25006688,
25006593,
25002422,
25006554,
25006533,
25006454,
25006416,
25006410,
25006224,
25006011,
25006250,
25006206,
25006138,
25006137,
25006070,
25006012,
25006013,
25006014,
25005665,
25005948,
25005884,
25005897,
25005933,
25005934,
25005935,
25005919,
25005847,
25005813,
25005783,
25005740,
25005727,
25005754,
25005708,
25005637,
25005742,
25005711,
25005703,
25005677,
25005676,
25005705,
25005388,
25005538,
25005534,
25005535,
25005478,
25005497,
25005262,
25005420,
25005334,
25005272,
25003136,
25005196,
25005187,
25005138,
25005135,
25005071,
25005095,
25005046,
25005028,
25005029,
25005030,
25004995,
25005025,
25004936,
25004937,
25004938,
25004957,
25004847,
25004860,
25004897,
25004898,
25004899,
25004868,
25004754,
25004669,
25004670,
25004671,
25004613,
25004608,
25004575,
25004538,
25004483,
25004405,
25004414,
25004395,
25004393,
25004278,
25004242,
25004161,
25004205,
25004134,
25004079,
25004027,
25004002,
25003910,
25003790,
25003827,
25003808,
25003755,
25003777,
25003645,
25003644,
25003623,
25003600,
25003598,
25003550,
25003549,
25003558,
25003223,
25003543,
25003544,
25003517,
25003500,
25003365,
25003327,
25003271,
25003421,
25002968,
25003226,
25003197,
25001987,
25002921,
25003084,
25003070,
25003057,
25003073,
25002878,
25003014,
25003029,
25003028,
25002608,
25002957,
25002981,
25002720,
25002827,
25002725,
25002813,
25002621,
25002732,
25002629,
25002657,
25009173,
25010696,
25010692,
25010663,
25010664,
25010665,
25010596,
25010770,
25011011,
25010940,
25011010,
25011009,
25010933,
25010857,
25010817,
25010816,
25010815,
25010966,
25011159,
25011158,
25011109,
25011120,
25011119,
25010990,
25011340,
25011339,
25011319,
25011318,
25011317,
25011223,
25011279,
25011221,
25011550,
25011416,
25011618,
25011417,
25011166,
25011068,
25011510,
25011611,
25011551,
25011637,
25011636,
25011635,
25011583,
25011842,
25011939,
25011797,
25011905,
25011968,
25012018,
25012004,
25012185,
25012125,
25012505,
25012360,
25012443,
25012359,
25012556,
25012336,
25012590,
25012933,
25012776,
25012798,
25012174,
25012986,
25012973,
25012688,
25012758,
25012982,
25012981,
25012980,
25010801,
25012963,
25015431,
25014767,
25014766,
25014765,
25014614,
25014613,
25014612,
25014925,
25014200,
25014461,
25014460,
25014315,
25013872,
25013971,
25013258,
25013257,
25014548,
25014791,
25014857,
25014941,
25013897,
25013877,
25013663,
25013668,
25013660,
25013541,
25013615,
25015383,
25015361,
25015353,
25015354,
25015351,
25015165,
25015187,
25015180,
25015177,
25015237,
25015005,
25015050,
25014993,
25014774,
25014660,
25014661,
25014662,
25014472,
25014263,
25013929,
25014050,
25013661,
25013672,
25013646,
25013483,
25013388,
25013335,
25013101,
25015549,
25019124,
25018633,
25018632,
25018661,
25019369,
25018552,
25019195,
25019058,
25019124,
25018971,
25018872,
25018580,
25018771,
25018633,
25018632,
25018565,
25018361,
25018306,
25018280,
25018301,
25018240,
25018031,
25018193,
25017990,
25018032,
25018142,
25018141,
25018008,
25017586,
25017484,
25017559,
25017705,
25017603,
25017404,
25017366,
25017278,
25017277,
25016592,
25016969,
25017124,
25016974,
25016922,
25017012,
25017085,
25016920,
25017005,
25016824,
25016999,
25016998,
25016921,
25016645,
25016849,
25016850,
25016757,
25016522,
25016568,
25016539,
25016515,
25016322,
25016174,
25015930,
25015838,
25015958,
25015948,
25015941,
25015919,
25015918,
25015920,
25015786,
25015883,
25015836,
25015835,
25015833,
25015794,
25015694,
25015569,
25015549,
25015383,
25015361,
25015353,
25015354,
25015351,
25015165,
25019569,
25019469,
25019470,
25019471,
25020686,
25020685,
25021705,
25021551,
25021358,
25021365,
25021349,
25020967,
25020686,
25020685,
25020518,
25020053,
25019811,
25019628,
25019569,
25019469,
25019470,
25019471,
25019574,
25023931,
25023802,
25024054,
25024311,
25024317,
25024141,
25024525,
25023654,
25023631,
25023586,
25023631,
25023586,
25023392,
25023061,
25023105,
25023102,
25022977,
25022947,
25022890,
25022731,
25022858,
25022736,
25022613,
25022131,
25022125,
25022266,
25022410,
25021970,
25021881,
25019363,
25021705,
25021551,
25021358,
25021365,
25021349
]
new_issue_concession=new_issue_concession.loc[(~new_issue_concession['DealId'].isin(exclude_these))]
new_issue_concession['PricingDate'] = pd.to_datetime(new_issue_concession['PricingDate'])
new_issue_concession.rename(columns={'Size (m)': 'Size_m', 'Book Size (m)': 'Book_Amount'}, inplace=True)
new_issue_concession['Nic']=pd.to_numeric(new_issue_concession['Nic'], errors='coerce')
new_issue_concession['month']=pd.PeriodIndex(new_issue_concession.PricingDate, freq='m')
new_issue_concession['avg_nic_per_month_all_sectors'] = new_issue_concession.groupby('month').Nic.transform('mean')
new_issue_concession['avg_nic_per_month_sector'] = new_issue_concession.groupby(['month', 'Sector']).Nic.transform('mean')
new_issue_concession['Month_Volume'] = new_issue_concession.groupby(['month'])['Size_m'].sum()

treasury = pd.read_excel('../NIC_analysis/data/treasury.xlsx')
treasury['Date'] = pd.to_datetime(treasury['Date'])
treasury['month'] = pd.PeriodIndex(treasury.Date, freq='m')
avg_treasury=treasury.groupby('month').agg({'two_year':'mean','ten_year':'mean'}).reset_index().set_index('month').sort_values('month', ascending =False)

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


app = dash.Dash(__name__)
server=app.server
sector_options = test2['Sector'].unique()

colors = {"background": "#000000", "text": "#FFFFFF"}

app.layout = html.Div([
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
                     options = [str(i) for i in sector_options],
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

@app.callback(Output('graph', 'figure'),
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
    app.run_server(debug=True)