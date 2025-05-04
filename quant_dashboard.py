import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import io
import dash_bootstrap_components as dbc

# Define the column name mapping
column_name_mapping = {
    'Info - Ticker': 'Ticker',
    
    # Quality Rank KPIs
    'ROE - Current': 'ROE',
    'ROA - Current': 'ROA',
    'ROIC - Current': 'ROIC',
    'FCF - Millions': 'FCF',
    'Total Equity  - Millions': 'Equity',
    'Market Cap - Current': 'MCAP',

    # Momentum Rank KPIs
    'Performance - Perform. 1d': 'Perf 1d',
    'Performance - Perform. 3m': 'Perf 3m',
    'Performance - Perform. 6m': 'Perf 6m',
    'Performance - Perform. 1y': 'Perf 1y',
    'RS Rank - L - 0-100': 'RS Rank',
    'RS Rank - L - Industry': 'RS Rank Ind',
    'F-Score - Point': 'F-Score',

    # Value Rank KPIs
    'P/E - Current': 'PE',
    'P/B - Current': 'PB',
    'P/S - Current': 'PS',
    'P/FCF - Current': 'P/FCF',
    'EV/EBITDA - Current': 'EV/EBITDA',
    'Ord. Div Yield - Current': 'Div Yield'
}

# Define the columns to display for each strategy
quality_columns = ['Ticker', 'Company', 'RS Rank', 'q_rank', 'ROE', 'ROA', 'ROIC', 'FCFROE', 'MCAP']
value_columns = ['Ticker', 'Company', 'RS Rank', 'value_rank', 'PE', 'EV/EBITDA', 'P/FCF', 'Div Yield']
momentum_columns = ['Ticker', 'Company', 'RS Rank', 'F-Score']

# Create the Dash app with callback exception suppression
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Define the layout
app.layout = html.Div([
    html.Div([
        html.H1("Quantitative Strategy Screener", className="mb-4 mt-3 text-center"),
        
        # File upload section
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select a Borsdata CSV File')
                ]),
                style={
                    'width': '100%',
                    'height': '80px',
                    'lineHeight': '80px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '10px',
                    'textAlign': 'center',
                    'margin': '20px 0',
                    'backgroundColor': '#f8f9fa'
                },
                multiple=False
            ),
        ], className="mb-4"),
        
        # Equity input section
        html.Div([
            html.Label("Total Equity (SEK):", className="mr-2 h5"),
            dcc.Input(
                id='equity-input',
                type='number',
                value=750000,
                className="form-control",
                style={'max-width': '300px', 'margin': '10px 0'}
            ),
            html.Button('Apply', id='apply-equity', className="btn btn-primary", style={'margin': '10px 0'}),
        ], className="mb-4 p-3", style={'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
        
        # Strategy tabs
        dbc.Tabs([
            dbc.Tab([
                html.Div(id='quality-strategy-table-container', className="mt-3 p-3"),
            ], label="Quality Strategy"),
            
            dbc.Tab([
                html.Div(id='growth-strategy-table-container', className="mt-3 p-3"),
            ], label="Growth Strategy"),
            
            dbc.Tab([
                html.Div(id='value-strategy-table-container', className="mt-3 p-3"),
            ], label="Value Strategy"),
            
            dbc.Tab([
                html.Div(id='momentum-strategy-table-container', className="mt-3 p-3"),
            ], label="Momentum Strategy"),
            
            dbc.Tab([
                html.Div(id='aggregated-positions-container', className="mt-3 p-3"),
            ], label="Aggregated Positions"),
        ], className="mb-4"),
        
        # Store components to hold the processed data
        dcc.Store(id='processed-data'),
        dcc.Store(id='quality-selected-stocks', data=[]),
        dcc.Store(id='growth-selected-stocks', data=[]),
        dcc.Store(id='value-selected-stocks', data=[]),
        dcc.Store(id='momentum-selected-stocks', data=[]),
        dcc.Store(id='current-equity', data=750000),
    ], style={'padding': '20px', 'maxWidth': '1200px', 'margin': '0 auto'})
])

# Function to parse uploaded file
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename.lower():
            # Handle CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return df
        elif 'xls' in filename.lower():
            # Handle Excel file
            df = pd.read_excel(io.BytesIO(decoded))
            return df
        else:
            print(f"Unsupported file format: {filename}")
            return None
    except Exception as e:
        print(f"Error parsing file: {e}")
        return None

# Function to process data for quality strategy
def quantitative_quality_strategy(df, num_stocks=40):
    data = df.copy(deep=True)

    # Calculate FCFROE
    data['FCFROE'] = data['FCF - Millions'] / data['Total Equity  - Millions']
    
    # Calculate q_rank
    data['q_rank'] = pd.DataFrame(data['ROE - Current'].rank() + 
                                  data['ROA - Current'].rank() + 
                                  data['ROIC - Current'].rank() + 
                                  data['FCFROE'].rank()).rank()

    # Filter data
    data = data[data['Info - Sector'] != 'Financials']
    data = data[data['Info - Country'] == 'Sweden']
    data = data[~data['Info - List'].isin(['NGM', 'Spotlight'])]
    data = data[data['Market Cap - Current'] >= 500]

    # Sort and select top stocks
    data = data.sort_values(ascending=False, by=['q_rank']).head(num_stocks)
    
    # Rename columns
    data_renamed = data.rename(columns=column_name_mapping)
    
    return data_renamed

def quantitative_growth_strategy(df, num_stocks=40):
    data = df.copy(deep=True)
    growth_columns = ['Earnings g. - Growth 1y', 'Earnings g. - Growth 3y', 
                     'Revenue g. - Growth 1y', 'Revenue g. - Growth 3y']

    # Filter criteria
    data = data[data['Info - Sector'] != 'Financials']
    data = data[data['Info - Sector'] != 'Real Estate']  # Exclude real estate too as mentioned
    data = data[data['Info - Country'] == 'Sweden']
    data = data[data['Market Cap - Current'] >= 500]
    
    # Filter for companies that have been profitable for the last 5 years
    # Since we don't have 5 years of PE data explicitly, we'll use current PE to ensure profitability
    data = data[data['P/E - Current'] > 0]
    
    # Convert growth metrics to numeric and handle NaN values - with debugging
    for kpi in growth_columns:
        # Convert percentage strings to floats if necessary
        if data[kpi].dtype == object:
            # Check if the values contain '%' signs
            if any(str(val).endswith('%') for val in data[kpi].dropna().head()):
                # Convert percentage strings to float values
                data[kpi] = data[kpi].astype(str).str.rstrip('%').astype(float) / 100
        
        # Convert to numeric
        data[kpi] = pd.to_numeric(data[kpi], errors='coerce')
        # Fill NaN values with 0
        data[kpi].fillna(0, inplace=True)
    
    # Rank by growth metrics (higher is better)
    data['earnings_1y_rank'] = data['Earnings g. - Growth 1y'].rank(ascending=False)
    data['earnings_3y_rank'] = data['Earnings g. - Growth 3y'].rank(ascending=False)
    data['revenue_1y_rank'] = data['Revenue g. - Growth 1y'].rank(ascending=False)
    data['revenue_3y_rank'] = data['Revenue g. - Growth 3y'].rank(ascending=False)
    
    # Calculate combined growth rank
    data['growth_rank'] = (data['earnings_1y_rank'] + 
                          data['earnings_3y_rank'] + 
                          data['revenue_1y_rank'] + 
                          data['revenue_3y_rank']).rank(ascending=True)
    
    # Select top 40 companies by growth rank
    data = data.sort_values(by='growth_rank').head(num_stocks)
    
    data_renamed = data.rename(columns=column_name_mapping)
    
    return data_renamed

# Function to process data for value strategy
def quantitative_value_strategy(df, num_stocks=40):
    data = df.copy(deep=True)

    # Calculate rankings
    # Convert 'Ord. Div Yield - Current' to numeric, coercing errors
    da = pd.to_numeric(data['Ord. Div Yield - Current'], errors='coerce')
    da = da.fillna(0.00000001).replace(0, 0.00000001)
    data['Ord. Div Yield - Current'].fillna(0, inplace=True)
    data['DA_rank'] = (1 / da).rank()
    
    max_rank = data['P/E - Current'].count()
    for kpi in ['P/E - Current', 'P/B - Current', 'P/S - Current', 'P/FCF - Current', 'EV/EBITDA - Current']:
        # Convert the column to a numeric type and handle NaN values
        data[kpi] = pd.to_numeric(data[kpi], errors='coerce')
        data[kpi].fillna(float('inf'), inplace=True)
        # Perform ranking
        data[kpi+' rank'] = data[kpi].rank()
        data.loc[data[kpi] <= 0, kpi+' rank'] = max_rank

    data['value_rank'] = (
        data['P/E - Current rank'] + 
        data['P/B - Current rank'] + 
        data['P/S - Current rank'] + 
        data['P/FCF - Current rank'] + 
        data['EV/EBITDA - Current rank'] + 
        data['DA_rank']
    ).rank()

    # Filter data
    data = data[data['Info - Country'] == 'Sweden']
    data = data[data['Info - Sector'] != 'Financials']
    data = data[~data['Info - List'].isin(['NGM', 'Spotlight'])]
    data = data[data['Market Cap - Current'] >= 500]

    # Sort and select top stocks
    data = data.sort_values(ascending=True, by=['value_rank']).head(num_stocks)
    
    # Rename columns
    data_renamed = data.rename(columns=column_name_mapping)
    
    return data_renamed

# Function to process data for momentum strategy
def quantitative_momentum_strategy(df, num_stocks=40):
    data = df.copy(deep=True)

    # Filter data
    data = data[data['Info - Country'] == 'Sweden']
    data = data[~data['Info - List'].isin(['NGM', 'Spotlight'])]
    data = data[data['Market Cap - Current'] >= 500]
    data = data[data['F-Score - Point'] > 2]

    # Sort by RS Rank
    data = data.sort_values(ascending=False, by=['RS Rank - L - 0-100']).head(num_stocks)
    
    # Rename columns
    data_renamed = data.rename(columns=column_name_mapping)
    
    return data_renamed

# Callback to process uploaded data
@app.callback(
    Output('processed-data', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_processed_data(contents, filename):
    if contents is None:
        return None
    
    df = parse_contents(contents, filename)
    if df is None:
        return None
    
    # Process data for each strategy
    quality_data = quantitative_quality_strategy(df)
    growth_data = quantitative_growth_strategy(df)
    value_data = quantitative_value_strategy(df)
    momentum_data = quantitative_momentum_strategy(df)
    
    # Store all processed data
    processed_data = {
        'quality': quality_data.to_dict('records'),
        'growth': growth_data.to_dict('records'),
        'value': value_data.to_dict('records'),
        'momentum': momentum_data.to_dict('records')
    }
    
    return processed_data

@app.callback(
    [Output('quality-selected-stocks', 'data', allow_duplicate=True),
     Output('growth-selected-stocks', 'data', allow_duplicate=True),
     Output('value-selected-stocks', 'data', allow_duplicate=True),
     Output('momentum-selected-stocks', 'data', allow_duplicate=True)],
    [Input('processed-data', 'data')],
    prevent_initial_call=True
)
def initialize_selected_stocks(processed_data):
    if processed_data is None:
        return [], [], [], []
    
    # Initialize selected stocks as top 10 for each strategy
    quality_data = pd.DataFrame(processed_data['quality'])
    growth_data = pd.DataFrame(processed_data['growth'])
    value_data = pd.DataFrame(processed_data['value'])
    momentum_data = pd.DataFrame(processed_data['momentum'])
    
    quality_selected = quality_data.sort_values(ascending=False, by=['RS Rank']).head(10)['Ticker'].tolist()
    growth_selected = growth_data.sort_values(ascending=False, by=['RS Rank']).head(10)['Ticker'].tolist()
    value_selected = value_data.sort_values(ascending=False, by=['RS Rank']).head(10)['Ticker'].tolist()
    momentum_selected = momentum_data.sort_values(ascending=False, by=['RS Rank']).head(10)['Ticker'].tolist()
    
    return quality_selected, growth_selected, value_selected, momentum_selected

# Callback to update equity value
@app.callback(
    Output('current-equity', 'data'),
    Input('apply-equity', 'n_clicks'),
    State('equity-input', 'value')
)
def update_equity(n_clicks, equity):
    if n_clicks is None:
        return 750000
    return equity

# Callback to update Quality strategy table
@app.callback(
    Output('quality-strategy-table-container', 'children'),
    [Input('processed-data', 'data'),
     Input('quality-selected-stocks', 'data'),
     Input('current-equity', 'data')],
    prevent_initial_call=True
)
def update_quality_table(processed_data, selected_stocks, equity):
    if processed_data is None:
        return html.Div("Please upload data first.")
    
    # Get quality data
    quality_data = pd.DataFrame(processed_data['quality'])
    
    # Sort by RS Rank
    quality_data = quality_data.sort_values(ascending=False, by=['RS Rank'])
    
    # Add selection column using emoji for visual representation
    selected_stocks = selected_stocks or []
    quality_data['Select'] = quality_data['Ticker'].apply(lambda x: '✅' if x in selected_stocks else '⬜')
    
    # Calculate allocation
    equity_per_strategy = equity / 4
    equity_per_stock = equity_per_strategy / 10
    
    # Add allocation column for selected stocks
    quality_data['Allocation'] = 0
    quality_data.loc[quality_data['Select'] == '✅', 'Allocation'] = equity_per_stock
    
    # Create table
    table = dash_table.DataTable(
        id='quality-table',
        columns=[
            {'name': 'Select', 'id': 'Select'},
            {'name': 'Ticker', 'id': 'Ticker'},
            {'name': 'Company', 'id': 'Company'},
            {'name': 'RS Rank', 'id': 'RS Rank', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Quality Rank', 'id': 'q_rank', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'ROE', 'id': 'ROE', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'ROA', 'id': 'ROA', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'ROIC', 'id': 'ROIC', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Allocation', 'id': 'Allocation', 'type': 'numeric', 'format': {'specifier': ',.0f'}}
        ],
        data=quality_data.to_dict('records'),
        editable=False,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        row_selectable=False,
        cell_selectable=True,
        style_cell={
            'textAlign': 'left',
            'padding': '5px'
        },
        style_header={
            'backgroundColor': 'lightgrey',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            },
            {
                'if': {'filter_query': '{Select} contains "✅"'},
                'backgroundColor': 'rgba(0, 128, 0, 0.1)'
            }
        ]
    )
    
    return [
        html.H3("Quality Strategy", className="mb-3"),
        table
    ]

# Callback for click-to-select in quality table
@app.callback(
    Output('quality-selected-stocks', 'data', allow_duplicate=True),
    Input('quality-table', 'active_cell'),
    State('quality-table', 'data'),
    State('quality-selected-stocks', 'data'),
    prevent_initial_call=True
)
def update_quality_selected_stocks(active_cell, table_data, current_selected):
    if active_cell is None or table_data is None:
        return current_selected or []
    
    # Only process clicks in the 'Select' column
    if active_cell['column_id'] == 'Select':
        row_idx = active_cell['row']
        ticker = table_data[row_idx]['Ticker']
        current_selected = current_selected or []
        
        # Toggle selection
        if ticker in current_selected:
            current_selected.remove(ticker)
        else:
            current_selected.append(ticker)
    
    return current_selected

# Callback to update Growth strategy table
@app.callback(
    Output('growth-strategy-table-container', 'children'),
    [Input('processed-data', 'data'),
     Input('growth-selected-stocks', 'data'),
     Input('current-equity', 'data')],
    prevent_initial_call=True
)
def update_growth_table(processed_data, selected_stocks, equity):
    if processed_data is None:
        return html.Div("Please upload data first.")
    
    # Get growth data
    growth_data = pd.DataFrame(processed_data['growth'])
    
    # Sort by growth rank
    growth_data = growth_data.sort_values(ascending=False, by=['RS Rank'])
    
    # Add selection column using emoji for visual representation
    selected_stocks = selected_stocks or []
    growth_data['Select'] = growth_data['Ticker'].apply(lambda x: '✅' if x in selected_stocks else '⬜')
    
    # Calculate allocation
    equity_per_strategy = equity / 4
    equity_per_stock = equity_per_strategy / 10
    
    # Add allocation column for selected stocks
    growth_data['Allocation'] = 0
    growth_data.loc[growth_data['Select'] == '✅', 'Allocation'] = equity_per_stock
    
    # Create table
    table = dash_table.DataTable(
        id='growth-table',
        columns=[
            {'name': 'Select', 'id': 'Select'},
            {'name': 'Ticker', 'id': 'Ticker'},
            {'name': 'Company', 'id': 'Company'},
            {'name': 'RS Rank', 'id': 'RS Rank', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Growth Rank', 'id': 'growth_rank', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Earnings Growth 1Y', 'id': 'Earnings g. - Growth 1y', 'type': 'numeric', 'format': {'specifier': '.2%'}},
            {'name': 'Earnings Growth 3Y', 'id': 'Earnings g. - Growth 3y', 'type': 'numeric', 'format': {'specifier': '.2%'}},
            {'name': 'Revenue Growth 1Y', 'id': 'Revenue g. - Growth 1y', 'type': 'numeric', 'format': {'specifier': '.2%'}},
            {'name': 'Revenue Growth 3Y', 'id': 'Revenue g. - Growth 3y', 'type': 'numeric', 'format': {'specifier': '.2%'}},
            {'name': 'Allocation', 'id': 'Allocation', 'type': 'numeric', 'format': {'specifier': ',.0f'}}
        ],
        data=growth_data.to_dict('records'),
        editable=False,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        row_selectable=False,
        cell_selectable=True,
        style_cell={
            'textAlign': 'left',
            'padding': '5px'
        },
        style_header={
            'backgroundColor': 'lightgrey',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            },
            {
                'if': {'filter_query': '{Select} contains "✅"'},
                'backgroundColor': 'rgba(0, 128, 0, 0.1)'
            }
        ]
    )
    
    return [
        html.H3("Growth Strategy", className="mb-3"),
        table
    ]

# Callback for click-to-select in growth table
@app.callback(
    Output('growth-selected-stocks', 'data', allow_duplicate=True),
    Input('growth-table', 'active_cell'),
    State('growth-table', 'data'),
    State('growth-selected-stocks', 'data'),
    prevent_initial_call=True
)
def update_growth_selected_stocks(active_cell, table_data, current_selected):
    if active_cell is None or table_data is None:
        return current_selected or []
    
    # Only process clicks in the 'Select' column
    if active_cell['column_id'] == 'Select':
        row_idx = active_cell['row']
        ticker = table_data[row_idx]['Ticker']
        current_selected = current_selected or []
        
        # Toggle selection
        if ticker in current_selected:
            current_selected.remove(ticker)
        else:
            current_selected.append(ticker)
    
    return current_selected

# Callback to update Value strategy table
@app.callback(
    Output('value-strategy-table-container', 'children'),
    [Input('processed-data', 'data'),
     Input('value-selected-stocks', 'data'),
     Input('current-equity', 'data')],
    prevent_initial_call=True
)
def update_value_table(processed_data, selected_stocks, equity):
    if processed_data is None:
        return html.Div("Please upload data first.")
    
    # Get value data
    value_data = pd.DataFrame(processed_data['value'])
    
    # Sort by RS Rank
    value_data = value_data.sort_values(ascending=False, by=['RS Rank'])
    
    # Add selection column using emoji for visual representation
    selected_stocks = selected_stocks or []
    value_data['Select'] = value_data['Ticker'].apply(lambda x: '✅' if x in selected_stocks else '⬜')
    
    # Calculate allocation
    equity_per_strategy = equity / 4
    equity_per_stock = equity_per_strategy / 10
    
    # Add allocation column for selected stocks
    value_data['Allocation'] = 0
    value_data.loc[value_data['Select'] == '✅', 'Allocation'] = equity_per_stock
    
    # Create table
    table = dash_table.DataTable(
        id='value-table',
        columns=[
            {'name': 'Select', 'id': 'Select'},
            {'name': 'Ticker', 'id': 'Ticker'},
            {'name': 'Company', 'id': 'Company'},
            {'name': 'RS Rank', 'id': 'RS Rank', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Value Rank', 'id': 'value_rank', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'PE', 'id': 'PE', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'EV/EBITDA', 'id': 'EV/EBITDA', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'P/FCF', 'id': 'P/FCF', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Div Yield', 'id': 'Div Yield', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Allocation', 'id': 'Allocation', 'type': 'numeric', 'format': {'specifier': ',.0f'}}
        ],
        data=value_data.to_dict('records'),
        editable=False,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        row_selectable=False,
        cell_selectable=True,
        style_cell={
            'textAlign': 'left',
            'padding': '5px'
        },
        style_header={
            'backgroundColor': 'lightgrey',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            },
            {
                'if': {'filter_query': '{Select} contains "✅"'},
                'backgroundColor': 'rgba(0, 128, 0, 0.1)'
            }
        ]
    )
    
    return [
        html.H3("Value Strategy", className="mb-3"),
        table
    ]

# Callback for click-to-select in value table
@app.callback(
    Output('value-selected-stocks', 'data', allow_duplicate=True),
    Input('value-table', 'active_cell'),
    State('value-table', 'data'),
    State('value-selected-stocks', 'data'),
    prevent_initial_call=True
)
def update_value_selected_stocks(active_cell, table_data, current_selected):
    if active_cell is None or table_data is None:
        return current_selected or []
    
    # Only process clicks in the 'Select' column
    if active_cell['column_id'] == 'Select':
        row_idx = active_cell['row']
        ticker = table_data[row_idx]['Ticker']
        current_selected = current_selected or []
        
        # Toggle selection
        if ticker in current_selected:
            current_selected.remove(ticker)
        else:
            current_selected.append(ticker)
    
    return current_selected

# Callback to update Momentum strategy table
@app.callback(
    Output('momentum-strategy-table-container', 'children'),
    Input('processed-data', 'data'),
    Input('momentum-selected-stocks', 'data'),
    Input('current-equity', 'data')
)
def update_momentum_table(processed_data, selected_stocks, equity):
    if processed_data is None:
        return html.Div("Please upload data first.")
    
    # Get momentum data
    momentum_data = pd.DataFrame(processed_data['momentum'])
    
    # Sort by RS Rank
    momentum_data = momentum_data.sort_values(ascending=False, by=['RS Rank'])
    
    # Add selection column using emoji for visual representation
    selected_stocks = selected_stocks or []
    momentum_data['Select'] = momentum_data['Ticker'].apply(lambda x: '✅' if x in selected_stocks else '⬜')
    
    # Calculate allocation
    equity_per_strategy = equity / 4
    equity_per_stock = equity_per_strategy / 10
    
    # Add allocation column for selected stocks
    momentum_data['Allocation'] = 0
    momentum_data.loc[momentum_data['Select'] == '✅', 'Allocation'] = equity_per_stock
    
    # Create table
    table = dash_table.DataTable(
        id='momentum-table',
        columns=[
            {'name': 'Select', 'id': 'Select'},
            {'name': 'Ticker', 'id': 'Ticker'},
            {'name': 'Company', 'id': 'Company'},
            {'name': 'RS Rank', 'id': 'RS Rank', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'F-Score', 'id': 'F-Score', 'type': 'numeric', 'format': {'specifier': '.0f'}},
            {'name': 'Allocation', 'id': 'Allocation', 'type': 'numeric', 'format': {'specifier': ',.0f'}}
        ],
        data=momentum_data.to_dict('records'),
        editable=False,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        row_selectable=False,
        cell_selectable=True,
        style_cell={
            'textAlign': 'left',
            'padding': '5px'
        },
        style_header={
            'backgroundColor': 'lightgrey',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            },
            {
                'if': {'filter_query': '{Select} contains "✅"'},
                'backgroundColor': 'rgba(0, 128, 0, 0.1)'
            }
        ]
    )
    
    return [
        html.H3("Momentum Strategy", className="mb-3"),
        table
    ]

# Callback for click-to-select in momentum table
@app.callback(
    Output('momentum-selected-stocks', 'data', allow_duplicate=True),
    Input('momentum-table', 'active_cell'),
    State('momentum-table', 'data'),
    State('momentum-selected-stocks', 'data'),
    prevent_initial_call=True
)
def update_momentum_selected_stocks(active_cell, table_data, current_selected):
    if active_cell is None or table_data is None:
        return current_selected or []
    
    # Only process clicks in the 'Select' column
    if active_cell['column_id'] == 'Select':
        row_idx = active_cell['row']
        ticker = table_data[row_idx]['Ticker']
        current_selected = current_selected or []
        
        # Toggle selection
        if ticker in current_selected:
            current_selected.remove(ticker)
        else:
            current_selected.append(ticker)
    
    return current_selected

# Callback to update Aggregated Positions
@app.callback(
    Output('aggregated-positions-container', 'children'),
    [Input('processed-data', 'data'),
     Input('quality-selected-stocks', 'data'),
     Input('growth-selected-stocks', 'data'),
     Input('value-selected-stocks', 'data'),
     Input('momentum-selected-stocks', 'data'),
     Input('current-equity', 'data')]
)
def update_aggregated_positions(processed_data, quality_selected, growth_selected, value_selected, momentum_selected, equity):
    if processed_data is None:
        return html.Div("Please upload data first.")
    
    # Get data for each strategy
    quality_data = pd.DataFrame(processed_data['quality'])
    growth_data = pd.DataFrame(processed_data['growth'])
    value_data = pd.DataFrame(processed_data['value'])
    momentum_data = pd.DataFrame(processed_data['momentum'])
    
    # Ensure selected stocks are lists
    quality_selected = quality_selected or []
    growth_selected = growth_selected or []
    value_selected = value_selected or []
    momentum_selected = momentum_selected or []
    
    # Filter selected stocks
    quality_selected_data = quality_data[quality_data['Ticker'].isin(quality_selected)][['Ticker', 'Company']]
    growth_selected_data = growth_data[growth_data['Ticker'].isin(growth_selected)][['Ticker', 'Company']]
    value_selected_data = value_data[value_data['Ticker'].isin(value_selected)][['Ticker', 'Company']]
    momentum_selected_data = momentum_data[momentum_data['Ticker'].isin(momentum_selected)][['Ticker', 'Company']]
    
    # Add strategy column
    quality_selected_data['Strategy'] = 'Quality'
    growth_selected_data['Strategy'] = 'Growth'
    value_selected_data['Strategy'] = 'Value'
    momentum_selected_data['Strategy'] = 'Momentum'
    
    # Calculate allocations
    equity_per_strategy = equity / 4
    
    # Calculate equity per stock - handle cases with zero stocks selected
    quality_equity_per_stock = equity_per_strategy / len(quality_selected) if len(quality_selected) > 0 else 0
    growth_equity_per_stock = equity_per_strategy / len(growth_selected) if len(growth_selected) > 0 else 0
    value_equity_per_stock = equity_per_strategy / len(value_selected) if len(value_selected) > 0 else 0
    momentum_equity_per_stock = equity_per_strategy / len(momentum_selected) if len(momentum_selected) > 0 else 0
    
    quality_selected_data['Allocation'] = quality_equity_per_stock
    growth_selected_data['Allocation'] = growth_equity_per_stock
    value_selected_data['Allocation'] = value_equity_per_stock
    momentum_selected_data['Allocation'] = momentum_equity_per_stock
    
    # Combine all selected stocks
    all_selected = pd.concat([quality_selected_data, growth_selected_data, value_selected_data, momentum_selected_data])
    
    if all_selected.empty:
        return html.Div("No stocks selected. Please select stocks from the strategy tables.")
    
    # Aggregate by ticker
    aggregated = all_selected.groupby(['Ticker', 'Company']).agg({
        'Strategy': lambda x: ', '.join(sorted(set(x))),
        'Allocation': 'sum'
    }).reset_index()
    
    # Calculate allocation percentage
    aggregated['Allocation %'] = (aggregated['Allocation'] / equity) * 100
    
    # Sort by allocation
    aggregated = aggregated.sort_values(by='Allocation', ascending=False)
    
    # Create table
    table = dash_table.DataTable(
        id='aggregated-table',
        columns=[
            {'name': 'Ticker', 'id': 'Ticker'},
            {'name': 'Company', 'id': 'Company'},
            {'name': 'Strategies', 'id': 'Strategy'},
            {'name': 'Allocation (SEK)', 'id': 'Allocation', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
            {'name': 'Allocation %', 'id': 'Allocation %', 'type': 'numeric', 'format': {'specifier': '.2f'}}
        ],
        data=aggregated.to_dict('records'),
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        style_cell={
            'textAlign': 'left',
            'padding': '5px'
        },
        style_header={
            'backgroundColor': 'lightgrey',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
    )
    
    # Create summary pie chart
    fig = px.pie(
        aggregated, 
        values='Allocation', 
        names='Ticker', 
        title='Portfolio Allocation by Stock',
        hole=0.3
    )
    
    # Summary statistics
    total_positions = len(aggregated)
    total_allocation = aggregated['Allocation'].sum()
    avg_allocation = total_allocation / total_positions if total_positions > 0 else 0
    max_allocation = aggregated['Allocation'].max() if not aggregated.empty else 0
    max_allocation_ticker = aggregated.loc[aggregated['Allocation'].idxmax(), 'Ticker'] if not aggregated.empty else 'None'
    
    return [
        html.H3("Aggregated Positions", className="mb-3"),
        table,
        html.Div([
            html.H4("Portfolio Allocation", className="mt-4 mb-3"),
            dcc.Graph(figure=fig)
        ])
    ]

if __name__ == '__main__':
    app.run(debug=True)