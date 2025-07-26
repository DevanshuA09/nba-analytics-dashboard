# Ultimate NBA Analytics Dashboard
# Professional portfolio-ready application with advanced features


# Importing necessary modules
import dash
from dash import dcc, html, dash_table, callback_context
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import NBA API modules
try:
    from nba_api.stats.endpoints import (
        leaguedashteamstats, leaguedashplayerstats, 
        shotchartdetail, teamdashboardbygeneralsplits,
        playerdashboardbygeneralsplits
    )
    from nba_api.stats.static import teams
except ImportError:
    print("Please install nba_api: pip install nba_api")

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])
server = app.server

# NBA Team Colors Dictionary
NBA_TEAM_COLORS = {
    'Atlanta Hawks': {'primary': '#E03A3E', 'secondary': '#C4CED4', 'accent': '#26282A'},
    'Boston Celtics': {'primary': '#007A33', 'secondary': '#BA9653', 'accent': '#000000'},
    'Brooklyn Nets': {'primary': '#000000', 'secondary': '#FFFFFF', 'accent': '#777777'},
    'Charlotte Hornets': {'primary': '#1D1160', 'secondary': '#00788C', 'accent': '#A1A1A4'},
    'Chicago Bulls': {'primary': '#CE1141', 'secondary': '#000000', 'accent': '#FFFFFF'},
    'Cleveland Cavaliers': {'primary': '#860038', 'secondary': '#FDBB30', 'accent': '#041E42'},
    'Dallas Mavericks': {'primary': '#00538C', 'secondary': '#002B5E', 'accent': '#B8C4CA'},
    'Denver Nuggets': {'primary': '#0E2240', 'secondary': '#FEC524', 'accent': '#8B2131'},
    'Detroit Pistons': {'primary': '#C8102E', 'secondary': '#1D42BA', 'accent': '#BEC0C2'},
    'Golden State Warriors': {'primary': '#1D428A', 'secondary': '#FFC72C', 'accent': '#26282A'},
    'Houston Rockets': {'primary': '#CE1141', 'secondary': '#000000', 'accent': '#C4CED4'},
    'Indiana Pacers': {'primary': '#002D62', 'secondary': '#FDBB30', 'accent': '#BEC0C2'},
    'LA Clippers': {'primary': '#1D428A', 'secondary': '#C8102E', 'accent': '#BEC0C2'},
    'Los Angeles Lakers': {'primary': '#552583', 'secondary': '#FDB927', 'accent': '#000000'},
    'Memphis Grizzlies': {'primary': '#5D76A9', 'secondary': '#12173F', 'accent': '#F5B112'},
    'Miami Heat': {'primary': '#98002E', 'secondary': '#F9A01B', 'accent': '#000000'},
    'Milwaukee Bucks': {'primary': '#00471B', 'secondary': '#EEE1C6', 'accent': '#0077C0'},
    'Minnesota Timberwolves': {'primary': '#0C2340', 'secondary': '#236192', 'accent': '#9EA2A2'},
    'New Orleans Pelicans': {'primary': '#0C2340', 'secondary': '#C8102E', 'accent': '#85714D'},
    'New York Knicks': {'primary': '#006BB6', 'secondary': '#F58426', 'accent': '#BEC0C2'},
    'Oklahoma City Thunder': {'primary': '#007AC1', 'secondary': '#EF3B24', 'accent': '#002D62'},
    'Orlando Magic': {'primary': '#0077C0', 'secondary': '#C4CED4', 'accent': '#000000'},
    'Philadelphia 76ers': {'primary': '#006BB6', 'secondary': '#ED174C', 'accent': '#002B5C'},
    'Phoenix Suns': {'primary': '#1D1160', 'secondary': '#E56020', 'accent': '#63727A'},
    'Portland Trail Blazers': {'primary': '#E03A3E', 'secondary': '#000000', 'accent': '#BCC4CA'},
    'Sacramento Kings': {'primary': '#5A2D81', 'secondary': '#63727A', 'accent': '#000000'},
    'San Antonio Spurs': {'primary': '#C4CED4', 'secondary': '#000000', 'accent': '#196F3D'},
    'Toronto Raptors': {'primary': '#CE1141', 'secondary': '#000000', 'accent': '#A1A1A4'},
    'Utah Jazz': {'primary': '#002B5C', 'secondary': '#00471B', 'accent': '#F9A01B'},
    'Washington Wizards': {'primary': '#002B5C', 'secondary': '#E31837', 'accent': '#C4CED4'}
}

# Team Logo URLs (using NBA official logos)
NBA_TEAM_LOGOS = {
    'Atlanta Hawks': 'https://cdn.nba.com/logos/nba/1610612737/primary/L/logo.svg',
    'Boston Celtics': 'https://cdn.nba.com/logos/nba/1610612738/primary/L/logo.svg',
    'Brooklyn Nets': 'https://cdn.nba.com/logos/nba/1610612751/primary/L/logo.svg',
    'Charlotte Hornets': 'https://cdn.nba.com/logos/nba/1610612766/primary/L/logo.svg',
    'Chicago Bulls': 'https://cdn.nba.com/logos/nba/1610612741/primary/L/logo.svg',
    'Cleveland Cavaliers': 'https://cdn.nba.com/logos/nba/1610612739/primary/L/logo.svg',
    'Dallas Mavericks': 'https://cdn.nba.com/logos/nba/1610612742/primary/L/logo.svg',
    'Denver Nuggets': 'https://cdn.nba.com/logos/nba/1610612743/primary/L/logo.svg',
    'Detroit Pistons': 'https://cdn.nba.com/logos/nba/1610612765/primary/L/logo.svg',
    'Golden State Warriors': 'https://cdn.nba.com/logos/nba/1610612744/primary/L/logo.svg',
    'Houston Rockets': 'https://cdn.nba.com/logos/nba/1610612745/primary/L/logo.svg',
    'Indiana Pacers': 'https://cdn.nba.com/logos/nba/1610612754/primary/L/logo.svg',
    'LA Clippers': 'https://cdn.nba.com/logos/nba/1610612746/primary/L/logo.svg',
    'Los Angeles Lakers': 'https://cdn.nba.com/logos/nba/1610612747/primary/L/logo.svg',
    'Memphis Grizzlies': 'https://cdn.nba.com/logos/nba/1610612763/primary/L/logo.svg',
    'Miami Heat': 'https://cdn.nba.com/logos/nba/1610612748/primary/L/logo.svg',
    'Milwaukee Bucks': 'https://cdn.nba.com/logos/nba/1610612749/primary/L/logo.svg',
    'Minnesota Timberwolves': 'https://cdn.nba.com/logos/nba/1610612750/primary/L/logo.svg',
    'New Orleans Pelicans': 'https://cdn.nba.com/logos/nba/1610612740/primary/L/logo.svg',
    'New York Knicks': 'https://cdn.nba.com/logos/nba/1610612752/primary/L/logo.svg',
    'Oklahoma City Thunder': 'https://cdn.nba.com/logos/nba/1610612760/primary/L/logo.svg',
    'Orlando Magic': 'https://cdn.nba.com/logos/nba/1610612753/primary/L/logo.svg',
    'Philadelphia 76ers': 'https://cdn.nba.com/logos/nba/1610612755/primary/L/logo.svg',
    'Phoenix Suns': 'https://cdn.nba.com/logos/nba/1610612756/primary/L/logo.svg',
    'Portland Trail Blazers': 'https://cdn.nba.com/logos/nba/1610612757/primary/L/logo.svg',
    'Sacramento Kings': 'https://cdn.nba.com/logos/nba/1610612758/primary/L/logo.svg',
    'San Antonio Spurs': 'https://cdn.nba.com/logos/nba/1610612759/primary/L/logo.svg',
    'Toronto Raptors': 'https://cdn.nba.com/logos/nba/1610612761/primary/L/logo.svg',
    'Utah Jazz': 'https://cdn.nba.com/logos/nba/1610612762/primary/L/logo.svg',
    'Washington Wizards': 'https://cdn.nba.com/logos/nba/1610612764/primary/L/logo.svg'
}

# Functions to get team colors and logo
def get_team_colors(team_name):
    """Get team-specific colors"""
    return NBA_TEAM_COLORS.get(team_name, {
        'primary': '#1f77b4', 
        'secondary': '#ff7f0e', 
        'accent': '#2ca02c'
    })

def get_team_logo(team_name):
    """Get team logo URL"""
    return NBA_TEAM_LOGOS.get(team_name, 'https://cdn.nba.com/logos/nba/1610612744/primary/L/logo.svg')

# Fetching data
@app.callback(Output('data-store', 'data'), [Input('season-dropdown', 'value')])
def load_data(season):
    """Load and cache NBA data for the selected season"""
    try:
        # Team stats
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season, 
            per_mode_detailed='PerGame',
            season_type_all_star='Regular Season'
        ).get_data_frames()[0]
        
        # Player stats
        player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed='PerGame',
            season_type_all_star='Regular Season'
        ).get_data_frames()[0]
        
        # Clean column names and ensure proper data types
        team_stats.columns = [c.lower().replace(' ', '_') for c in team_stats.columns]
        player_stats.columns = [c.lower().replace(' ', '_') for c in player_stats.columns]
        
        # Convert numeric columns to proper types
        numeric_team_cols = ['w', 'l', 'w_pct', 'pts', 'reb', 'ast', 'fg_pct', 'fg3_pct', 'stl', 'blk', 'tov', 'min', 'fga', 'fgm', 'fg3a', 'fg3m', 'fta', 'ftm', 'oreb', 'dreb']
        numeric_player_cols = ['pts', 'reb', 'ast', 'fg_pct', 'fg3_pct', 'stl', 'blk', 'tov', 'min', 'fga', 'fgm', 'fg3a', 'fg3m', 'fta', 'ftm', 'oreb', 'dreb']
        
        for col in numeric_team_cols:
            if col in team_stats.columns:
                team_stats[col] = pd.to_numeric(team_stats[col], errors='coerce')
        
        for col in numeric_player_cols:
            if col in player_stats.columns:
                player_stats[col] = pd.to_numeric(player_stats[col], errors='coerce')
        
        return {
            'team_stats': team_stats.to_dict('records'),
            'player_stats': player_stats.to_dict('records'),
            'last_updated': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error loading data: {e}")
        return {}

def calculate_advanced_metrics(team_data, league_avg):
    """Calculate advanced basketball metrics"""
    metrics = {}
    
    # Offensive Rating (points per 100 possessions)
    possessions = team_data.get('fga', 0) - team_data.get('oreb', 0) + team_data.get('tov', 0) + 0.4 * team_data.get('fta', 0)
    metrics['off_rating'] = (team_data.get('pts', 0) / possessions * 100) if possessions > 0 else 0
    
    # Defensive Rating (opponent points per 100 possessions) - approximation
    metrics['def_rating'] = 110.0  # Placeholder - would need opponent data
    
    # True Shooting Percentage
    ts_denominator = 2 * (team_data.get('fga', 0) + 0.44 * team_data.get('fta', 0))
    metrics['ts_pct'] = team_data.get('pts', 0) / ts_denominator if ts_denominator > 0 else 0
    
    # Effective Field Goal Percentage
    metrics['efg_pct'] = (team_data.get('fgm', 0) + 0.5 * team_data.get('fg3m', 0)) / team_data.get('fga', 0) if team_data.get('fga', 0) > 0 else 0
    
    # Pace (possessions per 48 minutes)
    metrics['pace'] = possessions * 48 / team_data.get('min', 48) if team_data.get('min', 0) > 0 else 0
    
    # Net Rating
    metrics['net_rating'] = metrics['off_rating'] - metrics['def_rating']
    
    return metrics

def calculate_player_efficiency_rating(player_data):
    """Calculate Player Efficiency Rating (PER)"""
    # Simplified PER calculation
    try:
        minutes = player_data.get('min', 0)
        if minutes == 0:
            return 0
            
        # Basic PER components
        uper = (1/minutes) * (
            player_data.get('tov', 0)
        )
        
        return max(0, uper * 15)  # Scale to typical PER range
    except:
        return 0

def create_enhanced_radar_chart(team_data, team_name, team_colors):
    """Create an enhanced radar chart with team colors and better styling"""
    categories = ['Offense', 'Defense', 'Rebounding', 'Playmaking', 'Shooting', 'Efficiency']
    
    # Normalize values (0-100 scale)
    values = [
        min(100, (team_data.get('pts', 0) / 130) * 100),  # Offense
        min(100, 100 - (team_data.get('opp_pts', 110) / 130) * 100),  # Defense (inverted)
        min(100, (team_data.get('reb', 0) / 60) * 100),  # Rebounding
        min(100, (team_data.get('ast', 0) / 35) * 100),  # Playmaking
        min(100, team_data.get('fg_pct', 0) * 200),  # Shooting
        min(100, team_data.get('fg_pct', 0) * 180)  # Efficiency
    ]
    
    fig = go.Figure()
    
    # Add the main trace with team colors
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the shape
        theta=categories + [categories[0]],
        fill='toself',
        name=team_name,
        line=dict(color=team_colors['primary'], width=3),
        fillcolor=f"rgba({int(team_colors['primary'][1:3], 16)}, {int(team_colors['primary'][3:5], 16)}, {int(team_colors['primary'][5:7], 16)}, 0.3)",
        marker=dict(size=8, color=team_colors['secondary'])
    ))
    
    # Add reference circles
    for i in [20, 40, 60, 80]:
        fig.add_trace(go.Scatterpolar(
            r=[i] * len(categories),
            theta=categories,
            mode='lines',
            line=dict(color='rgba(128,128,128,0.2)', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(128,128,128,0.3)',
                gridwidth=1
            ),
            angularaxis=dict(
                gridcolor='rgba(128,128,128,0.3)',
                gridwidth=2
            ),
            bgcolor='rgba(255,255,255,0.8)'
        ),
        showlegend=True,
        title=dict(
            text=f"{team_name} Performance Profile",
            font=dict(size=16, color=team_colors['primary']),
            x=0.5
        ),
        font=dict(size=12),
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_player_efficiency_chart(player_data, team_id, team_colors):
    """Create player efficiency ratings chart"""
    team_players = [p for p in player_data if p.get('team_id') == team_id]
    team_players = sorted(team_players, key=lambda x: x.get('min', 0), reverse=True)[:10]
    
    if not team_players:
        fig = go.Figure()
        fig.add_annotation(text="No player data available", x=0.5, y=0.5)
        return fig
    
    # Calculate PER for each player
    players_with_per = []
    for player in team_players:
        per = calculate_player_efficiency_rating(player)
        players_with_per.append({
            'name': player.get('player_name', 'Unknown'),
            'per': per,
            'minutes': player.get('min', 0),
            'pts': player.get('pts', 0)
        })
    
    # Sort by PER
    players_with_per = sorted(players_with_per, key=lambda x: x['per'], reverse=True)
    
    names = [p['name'].split()[-1] for p in players_with_per]  # Use last names
    per_values = [p['per'] for p in players_with_per]
    minutes = [p['minutes'] for p in players_with_per]
    
    fig = go.Figure()
    
    # Create bubble chart
    fig.add_trace(go.Scatter(
        x=names,
        y=per_values,
        mode='markers',
        marker=dict(
            size=[min(60, m*2) for m in minutes],  # Size based on minutes
            color=per_values,
            colorscale=[
                [0, team_colors['secondary']],
                [0.5, team_colors['primary']],
                [1, team_colors['accent']]
            ],
            showscale=True,
            colorbar=dict(title="PER", x=1.02),
            line=dict(width=2, color='white')
        ),
        text=[f"{p['name']}<br>PER: {p['per']:.1f}<br>MIN: {p['minutes']:.1f}" for p in players_with_per],
        hovertemplate='<b>%{text}</b><extra></extra>',
        name="Player Efficiency"
    ))
    
    
    fig.update_layout(
        title="Player Efficiency Ratings (Bubble size = Minutes played)",
        xaxis_title="Players",
        yaxis_title="Player Efficiency Rating (PER)",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_team_comparison_radar(team_stats_df, selected_team, comparison_team, team_colors):
    """Create radar chart comparing two teams"""
    team1_data = team_stats_df[team_stats_df['team_name'] == selected_team].iloc[0]
    team2_data = team_stats_df[team_stats_df['team_name'] == comparison_team].iloc[0]
    
    categories = ['Points', 'Rebounds', 'Assists', 'FG%', '3P%', 'Defense']
    
    team1_values = [
        (team1_data.get('pts', 0) / 130) * 100,
        (team1_data.get('reb', 0) / 60) * 100,
        (team1_data.get('ast', 0) / 35) * 100,
        team1_data.get('fg_pct', 0) * 200,
        team1_data.get('fg3_pct', 0) * 300,
        100 - (team1_data.get('pts', 0) / 130) * 100  # Mock defense
    ]
    
    team2_values = [
        (team2_data.get('pts', 0) / 130) * 100,
        (team2_data.get('reb', 0) / 60) * 100,
        (team2_data.get('ast', 0) / 35) * 100,
        team2_data.get('fg_pct', 0) * 200,
        team2_data.get('fg3_pct', 0) * 300,
        100 - (team2_data.get('pts', 0) / 130) * 100  # Mock defense
    ]
    
    fig = go.Figure()
    
    # Team 1
    fig.add_trace(go.Scatterpolar(
        r=team1_values + [team1_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=selected_team,
        line=dict(color=team_colors['primary'], width=3),
        fillcolor=f"rgba({int(team_colors['primary'][1:3], 16)}, {int(team_colors['primary'][3:5], 16)}, {int(team_colors['primary'][5:7], 16)}, 0.3)"
    ))
    
    # Team 2 (using neutral colors)
    fig.add_trace(go.Scatterpolar(
        r=team2_values + [team2_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=comparison_team,
        line=dict(color='#ff7f0e', width=3),
        fillcolor='rgba(255, 127, 14, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=True,
        title=f"{selected_team} vs {comparison_team}",
        height=400
    )
    
    return fig

# Professional header with team branding
def create_dynamic_header(team_name):
    """Create header with team logo and colors"""
    team_colors = get_team_colors(team_name)
    team_logo = get_team_logo(team_name)
    
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Img(src=team_logo, height="50px", className="me-3"),
                    dbc.NavbarBrand("NBA Analytics", className="fs-2 fw-bold text-white")
                ], width="auto"),
                dbc.Col([
                    html.P("Advanced Basketball Analytics Dashboard", 
                           className="text-light mb-0 fs-6")
                ], width="auto")
            ], align="center", className="g-0 w-100 justify-content-between")
        ], fluid=True),
        color=team_colors['primary'],
        dark=True,
        className="mb-4",
        style={'background': f'linear-gradient(135deg, {team_colors["primary"]} 0%, {team_colors["secondary"]} 100%)'}
    )

def create_controls_panel():
    """Create enhanced controls panel"""
    seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(2015, 2024)]
    
    # NBA teams list
    nba_teams = [
        'Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets',
        'Chicago Bulls', 'Cleveland Cavaliers', 'Dallas Mavericks', 'Denver Nuggets',
        'Detroit Pistons', 'Golden State Warriors', 'Houston Rockets', 'Indiana Pacers',
        'LA Clippers', 'Los Angeles Lakers', 'Memphis Grizzlies', 'Miami Heat',
        'Milwaukee Bucks', 'Minnesota Timberwolves', 'New Orleans Pelicans', 'New York Knicks',
        'Oklahoma City Thunder', 'Orlando Magic', 'Philadelphia 76ers', 'Phoenix Suns',
        'Portland Trail Blazers', 'Sacramento Kings', 'San Antonio Spurs', 'Toronto Raptors',
        'Utah Jazz', 'Washington Wizards'
    ]
    
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Season", className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id='season-dropdown',
                        options=[{'label': s, 'value': s} for s in seasons],
                        value='2022-23',
                        className="mb-3"
                    )
                ], md=4),
                dbc.Col([
                    html.Label("Primary Team", className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id='team-dropdown',
                        options=[{'label': t, 'value': t} for t in sorted(nba_teams)],
                        value='Golden State Warriors',
                        className="mb-3"
                    )
                ], md=4),
                dbc.Col([
                    html.Label("Compare With", className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id='comparison-team-dropdown',
                        options=[{'label': t, 'value': t} for t in sorted(nba_teams)],
                        value='Los Angeles Lakers',
                        className="mb-3"
                    )
                ], md=4)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        [html.I(className="bi bi-arrow-clockwise me-2"), "Refresh Data"],
                        id="refresh-btn",
                        color="primary",
                        size="sm"
                    )
                ], width="auto"),
                dbc.Col([
                    html.Small(id="last-updated", className="text-muted")
                ], width="auto")
            ], justify="between", align="center")
        ])
    ], className="mb-4")

def create_enhanced_metrics_cards(team_data, team_colors):
    """Create enhanced metrics cards with team colors"""
    cards = []
    
    metrics = [
        {"title": "Win-Loss Record", "value": f"{int(team_data.get('w', 0))}-{int(team_data.get('l', 0))}", "icon": "trophy"},
        {"title": "Win Percentage", "value": f"{team_data.get('w_pct', 0):.1%}", "icon": "percent"},
        {"title": "Points Per Game", "value": f"{team_data.get('pts', 0):.1f}", "icon": "bullseye"},
        {"title": "League Rank", "value": f"#{int(team_data.get('w_pct_rank', 0))}", "icon": "award"}
    ]
    
    for i, metric in enumerate(metrics):
        color_variants = ['primary', 'success', 'info', 'warning']
        cards.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className=f"bi bi-{metric['icon']} me-2"),
                        html.H6(metric['title'], className="mb-0")
                    ], style={'backgroundColor': team_colors['primary'], 'color': 'white'}),
                    dbc.CardBody([
                        html.H2(metric['value'], className="text-center mb-0", 
                               style={'color': team_colors['primary']})
                    ])
                ], className="h-100")
            ], md=3)
        )
    
    return dbc.Row(cards, className="mb-4")

def create_player_comparison_table(player_data, team_id, team_colors):
    """Create an enhanced interactive player comparison table"""
    team_players = [p for p in player_data if p.get('team_id') == team_id]
    
    # Select top players by minutes played
    team_players = sorted(team_players, key=lambda x: x.get('min', 0), reverse=True)[:12]
    
    if not team_players:
        return html.P("No player data available")
    
    # Add PER calculations
    for player in team_players:
        player['per'] = calculate_player_efficiency_rating(player)
    
    df = pd.DataFrame(team_players)
    display_cols = ['player_name', 'pts', 'reb', 'ast', 'fg_pct', 'fg3_pct', 'min', 'per']
    df_display = df[display_cols].round(2)
    
    return dash_table.DataTable(
        data=df_display.to_dict('records'),
        columns=[
            {'name': 'Player', 'id': 'player_name'},
            {'name': 'PPG', 'id': 'pts', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'RPG', 'id': 'reb', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'APG', 'id': 'ast', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'FG%', 'id': 'fg_pct', 'type': 'numeric', 'format': {'specifier': '.1%'}},
            {'name': '3P%', 'id': 'fg3_pct', 'type': 'numeric', 'format': {'specifier': '.1%'}},
            {'name': 'MIN', 'id': 'min', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'PER', 'id': 'per', 'type': 'numeric', 'format': {'specifier': '.1f'}}
        ],
        style_cell={
            'textAlign': 'left', 
            'fontSize': '14px',
            'fontFamily': 'Arial'
        },
        style_header={
            'backgroundColor': team_colors['primary'], 
            'color': 'white', 
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'filter_query': '{per} > 20'},
                'backgroundColor': f'rgba({int(team_colors["secondary"][1:3], 16)}, {int(team_colors["secondary"][3:5], 16)}, {int(team_colors["secondary"][5:7], 16)}, 0.3)',
                'color': 'black',
            },
            {
                'if': {'filter_query': '{per} > 15 && {per} <= 20'},
                'backgroundColor': f'rgba({int(team_colors["primary"][1:3], 16)}, {int(team_colors["primary"][3:5], 16)}, {int(team_colors["primary"][5:7], 16)}, 0.2)',
                'color': 'black',
            }
        ],
        sort_action="native",
        page_size=12,
        style_table={'overflowX': 'auto'}
    )

# Main app layout
app.layout = dbc.Container([
    dcc.Store(id='data-store'),
    html.Div(id='dynamic-header'),
    create_controls_panel(),
    
    # Enhanced Key Metrics Row
    html.Div(id='metrics-cards'),
    
    # Main Analytics Row - Enhanced Radar & Player effieciency comparison
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.Img(id="card-logo-1", height="30px", className="me-2"),
                        html.H5("Enhanced Performance Profile", className="mb-0 d-inline"),
                    ])
                ]),
                dbc.CardBody([
                    dcc.Graph(id="enhanced-radar-chart")
                ])
            ])
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.I(className="bi bi-graph-up me-2"),
                        html.H5("Player Efficiency Ratings", className="mb-0 d-inline"),
                    ])
                ]),
                dbc.CardBody([
                    dcc.Graph(id="player-efficiency-chart")
                ])
            ])
        ], md=6)
    ], className="mb-4"),
    
    # Team Comparison Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.I(className="bi bi-diagram-3 me-2"),
                        html.H5("Team Comparison", className="mb-0 d-inline"),
                    ])
                ]),
                dbc.CardBody([
                    dcc.Graph(id="team-comparison-radar")
                ])
            ])
        ], md=12)
    ], className="mb-4"),
    
    # Enhanced Player Analysis Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.Img(id="card-logo-2", height="30px", className="me-2"),
                        html.H5("Advanced Roster Analysis", className="mb-0 d-inline"),
                        html.Small(" • PER ratings and efficiency metrics", className="text-muted ms-2")
                    ])
                ]),
                dbc.CardBody([
                    html.Div(id="enhanced-player-table")
                ])
            ])
        ], md=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("Team Leaders & Stats", className="mb-0")
                ]),
                dbc.CardBody([
                    html.Div(id="team-leaders-enhanced")
                ])
            ])
        ], md=4)
    ], className="mb-4"),
    
    # Advanced Analytics Dashboard Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("Advanced Team Analytics", className="mb-0"),
                    html.Small("Offensive/Defensive ratings, pace, and efficiency metrics", className="text-muted")
                ]),
                dbc.CardBody([
                    dcc.Graph(id="advanced-analytics-chart")
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Footer with enhanced branding
    html.Hr(),
    html.Footer([
        dbc.Row([
            dbc.Col([
                html.P([
                    "🏀 Built with Python, Dash, Plotly & NBA API | ",
                    html.A("View on GitHub", href="#", className="text-decoration-none"),
                    " | Real-time NBA Analytics Dashboard by Devansu Agarwal"
                ], className="text-center text-muted mb-0")
            ])
        ])
    ], className="mt-4 mb-3")
    
], fluid=True)


@app.callback(
    [Output('dynamic-header', 'children'),
     Output('metrics-cards', 'children'),
     Output('enhanced-radar-chart', 'figure'),
     Output('player-efficiency-chart', 'figure'),
     Output('team-comparison-radar', 'figure'),
     Output('enhanced-player-table', 'children'),
     Output('team-leaders-enhanced', 'children'),
     Output('advanced-analytics-chart', 'figure'),
     Output('card-logo-1', 'src'),
     Output('card-logo-2', 'src'),
     Output('last-updated', 'children')],
    [Input('data-store', 'data'),
     Input('team-dropdown', 'value'),
     Input('comparison-team-dropdown', 'value')]
)
def update_ultimate_dashboard(data, selected_team, comparison_team):
    if not data or 'team_stats' not in data:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="Loading data...", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        empty_header = html.Div("Loading...")
        return empty_header, empty_header, empty_fig, empty_fig, empty_fig, empty_fig, "Loading...", "Loading...", "", "", ""

    try:
        team_stats = pd.DataFrame(data['team_stats'])
        player_stats = data['player_stats']

        if selected_team not in team_stats['team_name'].values:
            selected_team = team_stats['team_name'].iloc[0]

        team_data = team_stats[team_stats['team_name'] == selected_team].iloc[0]
        team_id = team_data['team_id']
        team_colors = get_team_colors(selected_team)
        team_logo = get_team_logo(selected_team)

        dynamic_header = create_dynamic_header(selected_team)
        metrics_cards = create_enhanced_metrics_cards(team_data, team_colors)
        enhanced_radar = create_enhanced_radar_chart(team_data, selected_team, team_colors)
        player_efficiency = create_player_efficiency_chart(player_stats, team_id, team_colors)

        if comparison_team and comparison_team in team_stats['team_name'].values:
            team_comparison = create_team_comparison_radar(team_stats, selected_team, comparison_team, team_colors)
        else:
            team_comparison = enhanced_radar

        enhanced_player_table = create_player_comparison_table(player_stats, team_id, team_colors)

        team_players = [p for p in player_stats if p.get('team_id') == team_id]
        if team_players:
            top_scorer = max(team_players, key=lambda x: x.get('pts', 0))
            top_rebounder = max(team_players, key=lambda x: x.get('reb', 0))
            top_assister = max(team_players, key=lambda x: x.get('ast', 0))
            most_efficient = max(team_players, key=lambda x: calculate_player_efficiency_rating(x))

            leaders_content = [
                dbc.Alert([
                    html.H6("\ud83c\udfc6 Scoring Leader", className="alert-heading"),
                    html.P(f"{top_scorer.get('player_name', 'N/A')}: {top_scorer.get('pts', 0):.1f} PPG")
                ], color="primary", className="mb-2"),
                dbc.Alert([
                    html.H6("\ud83c\udfc0 Rebounding Leader", className="alert-heading"),
                    html.P(f"{top_rebounder.get('player_name', 'N/A')}: {top_rebounder.get('reb', 0):.1f} RPG")
                ], color="success", className="mb-2"),
                dbc.Alert([
                    html.H6("\ud83c\udf1f Assists Leader", className="alert-heading"),
                    html.P(f"{top_assister.get('player_name', 'N/A')}: {top_assister.get('ast', 0):.1f} APG")
                ], color="info", className="mb-2"),
                dbc.Alert([
                    html.H6("\u26a1 Most Efficient", className="alert-heading"),
                    html.P(f"{most_efficient.get('player_name', 'N/A')}: {calculate_player_efficiency_rating(most_efficient):.1f} PER")
                ], color="warning", className="mb-0")
            ]
        else:
            leaders_content = [html.P("Player data not available")]

        advanced_metrics = calculate_advanced_metrics(team_data, team_stats.select_dtypes(include=[np.number]).mean())

        advanced_fig = go.Figure()
        metrics_names = ['Offensive Rating', 'True Shooting %', 'Effective FG%', 'Pace', 'Net Rating']
        metrics_values = [
            advanced_metrics['off_rating'],
            advanced_metrics['ts_pct'] * 100,
            advanced_metrics['efg_pct'] * 100,
            advanced_metrics['pace'],
            advanced_metrics['net_rating']
        ]

        advanced_fig.add_trace(go.Bar(
            x=metrics_names,
            y=metrics_values,
            marker_color=[team_colors['primary'], team_colors['secondary'], team_colors['accent'], team_colors['primary'], team_colors['secondary']],
            text=[f"{v:.1f}" for v in metrics_values],
            textposition='auto'
        ))

        advanced_fig.update_layout(
            title=f"{selected_team} - Advanced Analytics",
            yaxis_title="Rating/Percentage",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        last_updated = f"Last Updated: {datetime.fromisoformat(data['last_updated']).strftime('%Y-%m-%d %H:%M:%S')}"

        return (dynamic_header, metrics_cards, enhanced_radar, player_efficiency,
                team_comparison, enhanced_player_table, leaders_content,
                advanced_fig, team_logo, team_logo, last_updated)

    except Exception as e:
        print(f"Error in dashboard update: {e}")
        error_fig = go.Figure()
        error_fig.add_annotation(text=f"Error loading data: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        error_header = html.Div("Error loading dashboard")
        return error_header, error_header, error_fig, error_fig, error_fig, error_fig, "Error", error_fig, "", "", "Error"

if __name__ == '__main__':
    app.run(debug=True)
