"""
Feature extraction service for NBA predictive models
Handles feature engineering for both team win probability and player points models
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging
try:
    from nba_api.stats.endpoints import (
        leaguedashteamstats, leaguedashplayerstats,
        teamdashboardbygeneralsplits, playerdashboardbygeneralsplits,
        teamgamelog, playergamelog
    )
    from nba_api.stats.static import teams, players
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    from .demo_data import DemoDataGenerator

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract features for NBA predictive models"""
    
    def __init__(self):
        self.team_cache = {}
        self.player_cache = {}
        self.game_log_cache = {}
        if not NBA_API_AVAILABLE:
            self.demo_generator = DemoDataGenerator()
        
    def extract_team_features(self, home_team: str, away_team: str, 
                            home_rest_days: int = 1, away_rest_days: int = 1,
                            season: str = "2023-24") -> Optional[np.ndarray]:
        """
        Extract features for team win probability model
        
        Features:
        - Rolling team stats (last 10 games)
        - Home/away advantage
        - Rest days
        - Head-to-head record
        - Team efficiency metrics
        """
        try:
            # Use demo data if NBA API is not available
            if not NBA_API_AVAILABLE:
                return self.demo_generator.generate_team_features()
            
            # Get team IDs
            home_team_id = self._get_team_id(home_team)
            away_team_id = self._get_team_id(away_team)
            
            if not home_team_id or not away_team_id:
                return self.demo_generator.generate_team_features()
            
            # Get rolling team stats
            home_stats = self._get_rolling_team_stats(home_team_id, season, games=10)
            away_stats = self._get_rolling_team_stats(away_team_id, season, games=10)
            
            if home_stats is None or away_stats is None:
                return None
            
            # Calculate feature differences (home - away)
            features = []
            
            # Basic stats differences
            stat_cols = ['PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 
                        'STL', 'BLK', 'TOV', 'OREB', 'DREB']
            
            for col in stat_cols:
                if col in home_stats and col in away_stats:
                    features.append(home_stats[col] - away_stats[col])
                else:
                    features.append(0.0)
            
            # Advanced metrics
            home_off_rating = self._calculate_offensive_rating(home_stats)
            away_off_rating = self._calculate_offensive_rating(away_stats)
            features.append(home_off_rating - away_off_rating)
            
            # Home advantage (binary)
            features.append(1.0)
            
            # Rest days advantage
            rest_advantage = home_rest_days - away_rest_days
            features.append(rest_advantage)
            
            # Recent form (last 5 games win percentage)
            home_recent_form = self._get_recent_form(home_team_id, season, games=5)
            away_recent_form = self._get_recent_form(away_team_id, season, games=5)
            features.append(home_recent_form - away_recent_form)
            
            # Head-to-head record (simplified - use season record)
            h2h_advantage = self._get_head_to_head_advantage(home_team_id, away_team_id, season)
            features.append(h2h_advantage)
            
            features_array = np.array(features)
            # Ensure no NaN values
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
            return features_array
            
        except Exception as e:
            logger.error(f"Error extracting team features: {e}")
            return None
    
    def extract_player_features(self, player_name: str, opponent_team: str,
                              is_home: bool = True, season: str = "2023-24") -> Optional[np.ndarray]:
        """
        Extract features for player points prediction model
        
        Features:
        - Rolling player stats (last 10 games)
        - Opponent defensive rating
        - Home/away advantage
        - Player usage rate and minutes
        - Pace of play
        """
        try:
            # Use demo data if NBA API is not available
            if not NBA_API_AVAILABLE:
                return self.demo_generator.generate_player_features()
            
            # Get player ID
            player_id = self._get_player_id(player_name)
            opponent_team_id = self._get_team_id(opponent_team)
            
            if not player_id or not opponent_team_id:
                return self.demo_generator.generate_player_features()
            
            # Get rolling player stats
            player_stats = self._get_rolling_player_stats(player_id, season, games=10)
            if player_stats is None:
                return None
            
            # Get opponent defensive stats
            opponent_def_stats = self._get_team_defensive_stats(opponent_team_id, season)
            if opponent_def_stats is None:
                opponent_def_stats = {'PTS_ALLOWED': 110.0, 'FG_PCT_ALLOWED': 0.45}
            
            features = []
            
            # Player performance metrics
            features.append(player_stats.get('PTS', 0.0))
            features.append(player_stats.get('MIN', 0.0))
            features.append(player_stats.get('FGA', 0.0))
            features.append(player_stats.get('FG_PCT', 0.0))
            features.append(player_stats.get('FG3A', 0.0))
            features.append(player_stats.get('FG3_PCT', 0.0))
            features.append(player_stats.get('FTA', 0.0))
            features.append(player_stats.get('FT_PCT', 0.0))
            
            # Usage rate (simplified calculation)
            usage_rate = self._calculate_usage_rate(player_stats)
            features.append(usage_rate)
            
            # Recent form (last 5 games average points)
            recent_points = self._get_recent_player_points(player_id, season, games=5)
            features.append(recent_points)
            
            # Opponent defensive strength
            features.append(opponent_def_stats.get('PTS_ALLOWED', 110.0))
            features.append(opponent_def_stats.get('FG_PCT_ALLOWED', 0.45))
            
            # Home advantage
            features.append(1.0 if is_home else 0.0)
            
            # Pace factor (simplified)
            pace_factor = self._get_pace_factor(player_stats)
            features.append(pace_factor)
            
            features_array = np.array(features)
            # Ensure no NaN values
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
            return features_array
            
        except Exception as e:
            logger.error(f"Error extracting player features: {e}")
            return None
    
    def _get_team_id(self, team_name: str) -> Optional[int]:
        """Get team ID from team name"""
        try:
            team_dict = teams.get_teams()
            for team in team_dict:
                if team['full_name'] == team_name:
                    return team['id']
            return None
        except Exception as e:
            logger.error(f"Error getting team ID for {team_name}: {e}")
            return None
    
    def _get_player_id(self, player_name: str) -> Optional[int]:
        """Get player ID from player name"""
        try:
            player_dict = players.get_players()
            for player in player_dict:
                if player['full_name'] == player_name:
                    return player['id']
            return None
        except Exception as e:
            logger.error(f"Error getting player ID for {player_name}: {e}")
            return None
    
    def _get_rolling_team_stats(self, team_id: int, season: str, games: int = 10) -> Optional[Dict]:
        """Get rolling team statistics for last N games"""
        try:
            # Get team game log
            game_log = teamgamelog.TeamGameLog(team_id=team_id, season=season)
            df = game_log.get_data_frames()[0]
            
            if len(df) < games:
                games = len(df)
            
            # Get last N games
            recent_games = df.head(games)
            
            # Calculate averages
            stats = {
                'PTS': recent_games['PTS'].mean(),
                'REB': recent_games['REB'].mean(),
                'AST': recent_games['AST'].mean(),
                'FG_PCT': recent_games['FG_PCT'].mean(),
                'FG3_PCT': recent_games['FG3_PCT'].mean(),
                'FT_PCT': recent_games['FT_PCT'].mean(),
                'STL': recent_games['STL'].mean(),
                'BLK': recent_games['BLK'].mean(),
                'TOV': recent_games['TOV'].mean(),
                'OREB': recent_games['OREB'].mean(),
                'DREB': recent_games['DREB'].mean(),
                'FGA': recent_games['FGA'].mean(),
                'FG3A': recent_games['FG3A'].mean(),
                'FTA': recent_games['FTA'].mean()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting rolling team stats: {e}")
            return None
    
    def _get_rolling_player_stats(self, player_id: int, season: str, games: int = 10) -> Optional[Dict]:
        """Get rolling player statistics for last N games"""
        try:
            # Get player game log
            game_log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
            df = game_log.get_data_frames()[0]
            
            if len(df) < games:
                games = len(df)
            
            # Get last N games
            recent_games = df.head(games)
            
            # Calculate averages
            stats = {
                'PTS': recent_games['PTS'].mean(),
                'MIN': recent_games['MIN'].mean(),
                'FGA': recent_games['FGA'].mean(),
                'FG_PCT': recent_games['FG_PCT'].mean(),
                'FG3A': recent_games['FG3A'].mean(),
                'FG3_PCT': recent_games['FG3_PCT'].mean(),
                'FTA': recent_games['FTA'].mean(),
                'FT_PCT': recent_games['FT_PCT'].mean(),
                'REB': recent_games['REB'].mean(),
                'AST': recent_games['AST'].mean(),
                'STL': recent_games['STL'].mean(),
                'BLK': recent_games['BLK'].mean(),
                'TOV': recent_games['TOV'].mean()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting rolling player stats: {e}")
            return None
    
    def _calculate_offensive_rating(self, stats: Dict) -> float:
        """Calculate offensive rating (points per 100 possessions)"""
        try:
            # Simplified calculation
            pts = stats.get('PTS', 0)
            fga = stats.get('FGA', 1)
            tov = stats.get('TOV', 0)
            oreb = stats.get('OREB', 0)
            fta = stats.get('FTA', 0)
            
            # Estimate possessions
            possessions = fga - oreb + tov + 0.4 * fta
            if possessions <= 0:
                possessions = 1
            
            return (pts / possessions) * 100
            
        except Exception as e:
            logger.error(f"Error calculating offensive rating: {e}")
            return 100.0
    
    def _get_recent_form(self, team_id: int, season: str, games: int = 5) -> float:
        """Get recent form (win percentage in last N games)"""
        try:
            game_log = teamgamelog.TeamGameLog(team_id=team_id, season=season)
            df = game_log.get_data_frames()[0]
            
            if len(df) < games:
                games = len(df)
            
            recent_games = df.head(games)
            wins = (recent_games['WL'] == 'W').sum()
            
            return wins / games
            
        except Exception as e:
            logger.error(f"Error getting recent form: {e}")
            return 0.5
    
    def _get_head_to_head_advantage(self, home_team_id: int, away_team_id: int, season: str) -> float:
        """Get head-to-head advantage (simplified)"""
        # For now, return 0 (no advantage)
        # In a full implementation, you'd look at historical matchups
        return 0.0
    
    def _get_team_defensive_stats(self, team_id: int, season: str) -> Optional[Dict]:
        """Get team defensive statistics"""
        try:
            # Get team dashboard for defensive stats
            dashboard = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
                team_id=team_id, season=season
            )
            df = dashboard.get_data_frames()[1]  # Opponent stats
            
            if len(df) > 0:
                return {
                    'PTS_ALLOWED': df['PTS'].iloc[0],
                    'FG_PCT_ALLOWED': df['FG_PCT'].iloc[0]
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting defensive stats: {e}")
            return None
    
    def _calculate_usage_rate(self, stats: Dict) -> float:
        """Calculate usage rate (simplified)"""
        try:
            # Simplified usage rate calculation
            fga = stats.get('FGA', 0)
            fta = stats.get('FTA', 0)
            tov = stats.get('TOV', 0)
            
            # Estimate team possessions (simplified)
            team_possessions = 100  # Average team possessions per game
            
            usage = (fga + 0.44 * fta + tov) / team_possessions
            return min(1.0, max(0.0, usage))
            
        except Exception as e:
            logger.error(f"Error calculating usage rate: {e}")
            return 0.2
    
    def _get_recent_player_points(self, player_id: int, season: str, games: int = 5) -> float:
        """Get recent player points average"""
        try:
            game_log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
            df = game_log.get_data_frames()[0]
            
            if len(df) < games:
                games = len(df)
            
            recent_games = df.head(games)
            return recent_games['PTS'].mean()
            
        except Exception as e:
            logger.error(f"Error getting recent player points: {e}")
            return 0.0
    
    def _get_pace_factor(self, stats: Dict) -> float:
        """Get pace factor (simplified)"""
        try:
            # Simplified pace calculation based on possessions
            fga = stats.get('FGA', 0)
            tov = stats.get('TOV', 0)
            oreb = stats.get('OREB', 0)
            fta = stats.get('FTA', 0)
            
            possessions = fga - oreb + tov + 0.4 * fta
            return min(2.0, max(0.5, possessions / 50))  # Normalize to reasonable range
            
        except Exception as e:
            logger.error(f"Error calculating pace factor: {e}")
            return 1.0
