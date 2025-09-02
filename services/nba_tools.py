"""
NBA API Tools for Chatbot Integration
Wraps core nba_api endpoints as callable tools for the AI assistant
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import logging
from nba_api.stats.endpoints import (
    playergamelog, teamgamelog, teamdashboardbygeneralsplits,
    playerdashboardbygeneralsplits, commonteamroster, leaguestandings,
    scoreboardv2
)
from nba_api.stats.static import teams, players
import time

logger = logging.getLogger(__name__)

class NBATools:
    """NBA API tools for chatbot integration"""
    
    def __init__(self, cache_service=None):
        self.cache_service = cache_service
        self.rate_limit_delay = 0.6  # NBA API rate limiting
        
    def _rate_limit(self):
        """Apply rate limiting for NBA API calls"""
        time.sleep(self.rate_limit_delay)
    
    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate cache key for NBA API calls"""
        param_str = "_".join([f"{k}_{v}" for k, v in sorted(params.items())])
        return f"nba_api_{endpoint}_{param_str}"
    
    def _cached_call(self, endpoint: str, params: Dict, fetch_func):
        """Make cached NBA API call"""
        if self.cache_service:
            cache_key = self._get_cache_key(endpoint, params)
            cached_result = self.cache_service.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for {endpoint}")
                return cached_result
        
        # Make API call
        self._rate_limit()
        result = fetch_func()
        
        # Cache result
        if self.cache_service and result:
            cache_key = self._get_cache_key(endpoint, params)
            self.cache_service.set(cache_key, result, ttl=3600)  # 1 hour TTL
        
        return result
    
    def get_player_gamelog(self, player_name: str, season: str = "2023-24", 
                          last_n_games: int = 10) -> Dict[str, Any]:
        """
        Get player game log for the last N games
        
        Args:
            player_name: Name of the player
            season: NBA season (e.g., "2023-24")
            last_n_games: Number of recent games to fetch
            
        Returns:
            Dictionary with player stats and game log
        """
        try:
            # Find player ID
            player_dict = players.find_players_by_full_name(player_name)
            if not player_dict:
                return {"error": f"Player '{player_name}' not found"}
            
            player_id = player_dict[0]['id']
            
            # Get game log
            params = {
                "player_id": player_id,
                "season": season,
                "last_n_games": last_n_games
            }
            
            def fetch_data():
                gamelog = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=season,
                    last_n_games=last_n_games
                )
                return gamelog.get_data_frames()[0]
            
            df = self._cached_call("player_gamelog", params, fetch_data)
            
            if df is None or df.empty:
                return {"error": f"No game log data found for {player_name}"}
            
            # Process and format data
            recent_games = df.head(last_n_games)
            
            # Calculate averages
            stats = {
                "player_name": player_name,
                "season": season,
                "games_played": len(recent_games),
                "recent_averages": {
                    "points": recent_games['PTS'].mean(),
                    "rebounds": recent_games['REB'].mean(),
                    "assists": recent_games['AST'].mean(),
                    "steals": recent_games['STL'].mean(),
                    "blocks": recent_games['BLK'].mean(),
                    "field_goal_pct": recent_games['FG_PCT'].mean(),
                    "three_point_pct": recent_games['FG3_PCT'].mean(),
                    "free_throw_pct": recent_games['FT_PCT'].mean(),
                    "minutes": recent_games['MIN'].mean()
                },
                "game_log": recent_games[['GAME_DATE', 'MATCHUP', 'WL', 'PTS', 'REB', 'AST', 'MIN']].to_dict('records')
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting player gamelog: {e}")
            return {"error": f"Failed to fetch player data: {str(e)}"}
    
    def get_team_stats(self, team_name: str, season: str = "2023-24") -> Dict[str, Any]:
        """
        Get team statistics and standings
        
        Args:
            team_name: Name of the team
            season: NBA season (e.g., "2023-24")
            
        Returns:
            Dictionary with team stats and standings
        """
        try:
            # Find team ID
            team_dict = teams.find_teams_by_full_name(team_name)
            if not team_dict:
                return {"error": f"Team '{team_name}' not found"}
            
            team_id = team_dict[0]['id']
            
            params = {
                "team_id": team_id,
                "season": season
            }
            
            def fetch_data():
                dashboard = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
                    team_id=team_id,
                    season=season
                )
                return dashboard.get_data_frames()[0]
            
            df = self._cached_call("team_stats", params, fetch_data)
            
            if df is None or df.empty:
                return {"error": f"No team data found for {team_name}"}
            
            # Get team record
            team_record = df.iloc[0]
            
            stats = {
                "team_name": team_name,
                "season": season,
                "record": {
                    "wins": team_record['W'],
                    "losses": team_record['L'],
                    "win_percentage": team_record['W_PCT']
                },
                "offensive_stats": {
                    "points_per_game": team_record['PTS'],
                    "field_goal_pct": team_record['FG_PCT'],
                    "three_point_pct": team_record['FG3_PCT'],
                    "free_throw_pct": team_record['FT_PCT'],
                    "assists_per_game": team_record['AST'],
                    "turnovers_per_game": team_record['TOV']
                },
                "defensive_stats": {
                    "opponent_points_per_game": team_record['OPP_PTS'],
                    "opponent_fg_pct": team_record['OPP_FG_PCT'],
                    "opponent_3pt_pct": team_record['OPP_FG3_PCT'],
                    "steals_per_game": team_record['STL'],
                    "blocks_per_game": team_record['BLK']
                },
                "advanced_stats": {
                    "offensive_rating": team_record.get('OFF_RATING', 0),
                    "defensive_rating": team_record.get('DEF_RATING', 0),
                    "net_rating": team_record.get('NET_RATING', 0),
                    "pace": team_record.get('PACE', 0)
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting team stats: {e}")
            return {"error": f"Failed to fetch team data: {str(e)}"}
    
    def get_team_gamelog(self, team_name: str, season: str = "2023-24", 
                        last_n_games: int = 10) -> Dict[str, Any]:
        """
        Get team game log for the last N games
        
        Args:
            team_name: Name of the team
            season: NBA season (e.g., "2023-24")
            last_n_games: Number of recent games to fetch
            
        Returns:
            Dictionary with team game log
        """
        try:
            # Find team ID
            team_dict = teams.find_teams_by_full_name(team_name)
            if not team_dict:
                return {"error": f"Team '{team_name}' not found"}
            
            team_id = team_dict[0]['id']
            
            params = {
                "team_id": team_id,
                "season": season,
                "last_n_games": last_n_games
            }
            
            def fetch_data():
                gamelog = teamgamelog.TeamGameLog(
                    team_id=team_id,
                    season=season,
                    last_n_games=last_n_games
                )
                return gamelog.get_data_frames()[0]
            
            df = self._cached_call("team_gamelog", params, fetch_data)
            
            if df is None or df.empty:
                return {"error": f"No game log data found for {team_name}"}
            
            # Process recent games
            recent_games = df.head(last_n_games)
            
            # Calculate recent performance
            recent_wins = recent_games['WL'].value_counts().get('W', 0)
            recent_losses = recent_games['WL'].value_counts().get('L', 0)
            
            stats = {
                "team_name": team_name,
                "season": season,
                "recent_record": {
                    "wins": recent_wins,
                    "losses": recent_losses,
                    "win_percentage": recent_wins / (recent_wins + recent_losses) if (recent_wins + recent_losses) > 0 else 0
                },
                "recent_averages": {
                    "points": recent_games['PTS'].mean(),
                    "opponent_points": recent_games['OPP_PTS'].mean(),
                    "field_goal_pct": recent_games['FG_PCT'].mean(),
                    "three_point_pct": recent_games['FG3_PCT'].mean(),
                    "rebounds": recent_games['REB'].mean(),
                    "assists": recent_games['AST'].mean()
                },
                "game_log": recent_games[['GAME_DATE', 'MATCHUP', 'WL', 'PTS', 'OPP_PTS']].to_dict('records')
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting team gamelog: {e}")
            return {"error": f"Failed to fetch team game log: {str(e)}"}
    
    def get_todays_games(self) -> Dict[str, Any]:
        """
        Get today's NBA games
        
        Returns:
            Dictionary with today's games
        """
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            
            params = {"date": today}
            
            def fetch_data():
                scoreboard_data = scoreboardv2.ScoreboardV2(game_date=today)
                return scoreboard_data.get_data_frames()
            
            data_frames = self._cached_call("todays_games", params, fetch_data)
            
            if not data_frames or len(data_frames) == 0:
                return {"error": "No games found for today"}
            
            games_df = data_frames[0]  # Game header data
            
            if games_df.empty:
                return {"error": "No games scheduled for today"}
            
            games = []
            for _, game in games_df.iterrows():
                games.append({
                    "game_id": game['GAME_ID'],
                    "home_team": game['HOME_TEAM_NAME'],
                    "away_team": game['VISITOR_TEAM_NAME'],
                    "game_time": game['GAME_STATUS_TEXT'],
                    "home_score": game.get('HOME_TEAM_SCORE', 0),
                    "away_score": game.get('VISITOR_TEAM_SCORE', 0)
                })
            
            return {
                "date": today,
                "games": games,
                "total_games": len(games)
            }
            
        except Exception as e:
            logger.error(f"Error getting today's games: {e}")
            return {"error": f"Failed to fetch today's games: {str(e)}"}
    
    def get_standings(self, season: str = "2023-24", conference: str = None) -> Dict[str, Any]:
        """
        Get NBA standings
        
        Args:
            season: NBA season (e.g., "2023-24")
            conference: "East" or "West" (optional)
            
        Returns:
            Dictionary with standings
        """
        try:
            params = {"season": season, "conference": conference}
            
            def fetch_data():
                standings_data = leaguestandings.LeagueStandings(season=season)
                return standings_data.get_data_frames()[0]
            
            df = self._cached_call("standings", params, fetch_data)
            
            if df is None or df.empty:
                return {"error": "No standings data found"}
            
            # Filter by conference if specified
            if conference:
                df = df[df['Conference'] == conference]
            
            # Sort by win percentage
            df = df.sort_values('W_PCT', ascending=False)
            
            standings = []
            for _, team in df.iterrows():
                standings.append({
                    "rank": team['ConferenceRank'],
                    "team_name": team['TeamName'],
                    "conference": team['Conference'],
                    "wins": team['W'],
                    "losses": team['L'],
                    "win_percentage": team['W_PCT'],
                    "games_behind": team['GB'],
                    "last_10": team['L10']
                })
            
            return {
                "season": season,
                "conference": conference or "All",
                "standings": standings
            }
            
        except Exception as e:
            logger.error(f"Error getting standings: {e}")
            return {"error": f"Failed to fetch standings: {str(e)}"}
    
    def get_team_roster(self, team_name: str, season: str = "2023-24") -> Dict[str, Any]:
        """
        Get team roster
        
        Args:
            team_name: Name of the team
            season: NBA season (e.g., "2023-24")
            
        Returns:
            Dictionary with team roster
        """
        try:
            # Find team ID
            team_dict = teams.find_teams_by_full_name(team_name)
            if not team_dict:
                return {"error": f"Team '{team_name}' not found"}
            
            team_id = team_dict[0]['id']
            
            params = {
                "team_id": team_id,
                "season": season
            }
            
            def fetch_data():
                roster_data = commonteamroster.CommonTeamRoster(
                    team_id=team_id,
                    season=season
                )
                return roster_data.get_data_frames()[0]
            
            df = self._cached_call("team_roster", params, fetch_data)
            
            if df is None or df.empty:
                return {"error": f"No roster data found for {team_name}"}
            
            roster = []
            for _, player in df.iterrows():
                roster.append({
                    "player_id": player['PLAYER_ID'],
                    "player_name": player['PLAYER'],
                    "position": player['POSITION'],
                    "height": player['HEIGHT'],
                    "weight": player['WEIGHT'],
                    "age": player['AGE'],
                    "experience": player['EXP']
                })
            
            return {
                "team_name": team_name,
                "season": season,
                "roster": roster,
                "total_players": len(roster)
            }
            
        except Exception as e:
            logger.error(f"Error getting team roster: {e}")
            return {"error": f"Failed to fetch team roster: {str(e)}"}
    
    def get_head_to_head(self, team1: str, team2: str, season: str = "2023-24") -> Dict[str, Any]:
        """
        Get head-to-head matchup data between two teams
        
        Args:
            team1: First team name
            team2: Second team name
            season: NBA season (e.g., "2023-24")
            
        Returns:
            Dictionary with head-to-head stats
        """
        try:
            # Get both teams' game logs
            team1_log = self.get_team_gamelog(team1, season, last_n_games=82)
            team2_log = self.get_team_gamelog(team2, season, last_n_games=82)
            
            if "error" in team1_log or "error" in team2_log:
                return {"error": "Failed to fetch team data for head-to-head comparison"}
            
            # Find games between the two teams
            team1_games = team1_log.get("game_log", [])
            team2_games = team2_log.get("game_log", [])
            
            # Find common games (simplified approach)
            h2h_games = []
            for game1 in team1_games:
                for game2 in team2_games:
                    if (game1['GAME_DATE'] == game2['GAME_DATE'] and 
                        team2 in game1['MATCHUP'] and team1 in game2['MATCHUP']):
                        h2h_games.append({
                            "date": game1['GAME_DATE'],
                            "home_team": team1 if "vs." in game1['MATCHUP'] else team2,
                            "away_team": team2 if "vs." in game1['MATCHUP'] else team1,
                            "home_score": game1['PTS'] if "vs." in game1['MATCHUP'] else game2['PTS'],
                            "away_score": game2['PTS'] if "vs." in game1['MATCHUP'] else game1['PTS'],
                            "winner": team1 if game1['WL'] == 'W' else team2
                        })
                        break
            
            if not h2h_games:
                return {
                    "team1": team1,
                    "team2": team2,
                    "season": season,
                    "games_played": 0,
                    "message": "No head-to-head games found this season"
                }
            
            # Calculate head-to-head record
            team1_wins = sum(1 for game in h2h_games if game['winner'] == team1)
            team2_wins = len(h2h_games) - team1_wins
            
            return {
                "team1": team1,
                "team2": team2,
                "season": season,
                "games_played": len(h2h_games),
                "team1_wins": team1_wins,
                "team2_wins": team2_wins,
                "games": h2h_games
            }
            
        except Exception as e:
            logger.error(f"Error getting head-to-head data: {e}")
            return {"error": f"Failed to fetch head-to-head data: {str(e)}"}

# Tool registry for chatbot
NBA_TOOLS = {
    "get_player_gamelog": {
        "function": NBATools().get_player_gamelog,
        "description": "Get player game log and recent statistics",
        "parameters": {
            "player_name": "str - Name of the player",
            "season": "str - NBA season (default: 2023-24)",
            "last_n_games": "int - Number of recent games (default: 10)"
        }
    },
    "get_team_stats": {
        "function": NBATools().get_team_stats,
        "description": "Get team statistics and standings",
        "parameters": {
            "team_name": "str - Name of the team",
            "season": "str - NBA season (default: 2023-24)"
        }
    },
    "get_team_gamelog": {
        "function": NBATools().get_team_gamelog,
        "description": "Get team game log and recent performance",
        "parameters": {
            "team_name": "str - Name of the team",
            "season": "str - NBA season (default: 2023-24)",
            "last_n_games": "int - Number of recent games (default: 10)"
        }
    },
    "get_todays_games": {
        "function": NBATools().get_todays_games,
        "description": "Get today's NBA games and schedule",
        "parameters": {}
    },
    "get_standings": {
        "function": NBATools().get_standings,
        "description": "Get NBA standings by conference",
        "parameters": {
            "season": "str - NBA season (default: 2023-24)",
            "conference": "str - East or West (optional)"
        }
    },
    "get_team_roster": {
        "function": NBATools().get_team_roster,
        "description": "Get team roster and player information",
        "parameters": {
            "team_name": "str - Name of the team",
            "season": "str - NBA season (default: 2023-24)"
        }
    },
    "get_head_to_head": {
        "function": NBATools().get_head_to_head,
        "description": "Get head-to-head matchup data between two teams",
        "parameters": {
            "team1": "str - First team name",
            "team2": "str - Second team name",
            "season": "str - NBA season (default: 2023-24)"
        }
    }
}
