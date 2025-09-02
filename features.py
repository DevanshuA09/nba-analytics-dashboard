"""
Advanced Feature Engineering for NBA Predictive Models
Creates comprehensive features from raw NBA data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NBAFeatureEngineer:
    """Advanced feature engineering for NBA predictive models"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.features_dir = self.data_dir / "features"
        self.features_dir.mkdir(exist_ok=True)
        
        # Feature configuration
        self.rolling_windows = [5, 10, 20]  # Rolling average windows
        self.team_features = [
            'PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
            'STL', 'BLK', 'TOV', 'OREB', 'DREB', 'FGA', 'FG3A', 'FTA'
        ]
        self.player_features = [
            'PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
            'STL', 'BLK', 'TOV', 'MIN', 'FGA', 'FG3A', 'FTA'
        ]
        
    def create_team_features(self, team_game_logs: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive team features"""
        logger.info("Creating team features...")
        
        features_list = []
        
        for team_id in team_game_logs['TEAM_ID'].unique():
            team_logs = team_game_logs[team_game_logs['TEAM_ID'] == team_id].copy()
            team_logs = team_logs.sort_values('GAME_DATE')
            
            # Create rolling features
            team_features = self._create_rolling_features(
                team_logs, self.team_features, 'team'
            )
            
            # Add advanced metrics
            team_features = self._add_advanced_team_metrics(team_features)
            
            # Add contextual features
            team_features = self._add_contextual_features(team_features)
            
            features_list.append(team_features)
        
        if features_list:
            combined_features = pd.concat(features_list, ignore_index=True)
            logger.info(f"Created team features for {len(combined_features)} games")
            return combined_features
        else:
            return pd.DataFrame()
    
    def create_player_features(self, player_game_logs: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive player features"""
        logger.info("Creating player features...")
        
        features_list = []
        
        for player_id in player_game_logs['PLAYER_ID'].unique():
            player_logs = player_game_logs[player_game_logs['PLAYER_ID'] == player_id].copy()
            player_logs = player_logs.sort_values('GAME_DATE')
            
            # Create rolling features
            player_features = self._create_rolling_features(
                player_logs, self.player_features, 'player'
            )
            
            # Add advanced metrics
            player_features = self._add_advanced_player_metrics(player_features)
            
            # Add contextual features
            player_features = self._add_contextual_features(player_features)
            
            features_list.append(player_features)
        
        if features_list:
            combined_features = pd.concat(features_list, ignore_index=True)
            logger.info(f"Created player features for {len(combined_features)} games")
            return combined_features
        else:
            return pd.DataFrame()
    
    def _create_rolling_features(self, df: pd.DataFrame, 
                                base_features: List[str], 
                                entity_type: str) -> pd.DataFrame:
        """Create rolling average features"""
        df = df.copy()
        
        # Ensure numeric columns
        for feature in base_features:
            if feature in df.columns:
                df[feature] = pd.to_numeric(df[feature], errors='coerce')
        
        # Create rolling averages for different windows
        for window in self.rolling_windows:
            for feature in base_features:
                if feature in df.columns:
                    # Rolling average
                    df[f'{feature}_ROLL_{window}'] = df[feature].rolling(
                        window=window, min_periods=1
                    ).mean()
                    
                    # Rolling standard deviation
                    df[f'{feature}_STD_{window}'] = df[feature].rolling(
                        window=window, min_periods=1
                    ).std()
                    
                    # Rolling trend (slope of linear regression)
                    df[f'{feature}_TREND_{window}'] = df[feature].rolling(
                        window=window, min_periods=2
                    ).apply(lambda x: self._calculate_trend(x), raw=False)
        
        # Create momentum features (recent vs. older performance)
        for feature in base_features:
            if feature in df.columns:
                # Recent 5 games vs. previous 10 games
                df[f'{feature}_MOMENTUM_5_10'] = (
                    df[feature].rolling(5, min_periods=1).mean() - 
                    df[feature].rolling(10, min_periods=1).mean().shift(5)
                )
                
                # Recent 10 games vs. previous 20 games
                df[f'{feature}_MOMENTUM_10_20'] = (
                    df[feature].rolling(10, min_periods=1).mean() - 
                    df[feature].rolling(20, min_periods=1).mean().shift(10)
                )
        
        return df
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate trend (slope) of a series"""
        if len(series) < 2:
            return 0.0
        
        x = np.arange(len(series))
        y = series.values
        
        # Remove NaN values
        mask = ~np.isnan(y)
        if np.sum(mask) < 2:
            return 0.0
        
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Calculate slope
        slope = np.polyfit(x_clean, y_clean, 1)[0]
        return slope
    
    def _add_advanced_team_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced team metrics"""
        df = df.copy()
        
        # Offensive Rating (points per 100 possessions)
        df['ORtg'] = df.apply(lambda row: self._calculate_offensive_rating(row), axis=1)
        
        # Defensive Rating (opponent points per 100 possessions)
        df['DRtg'] = df.apply(lambda row: self._calculate_defensive_rating(row), axis=1)
        
        # Net Rating
        df['NetRtg'] = df['ORtg'] - df['DRtg']
        
        # Pace (possessions per 48 minutes)
        df['Pace'] = df.apply(lambda row: self._calculate_pace(row), axis=1)
        
        # True Shooting Percentage
        df['TS_PCT'] = df.apply(lambda row: self._calculate_ts_pct(row), axis=1)
        
        # Effective Field Goal Percentage
        df['eFG_PCT'] = df.apply(lambda row: self._calculate_efg_pct(row), axis=1)
        
        # Turnover Rate
        df['TOV_RATE'] = df.apply(lambda row: self._calculate_tov_rate(row), axis=1)
        
        # Rebound Rate
        df['REB_RATE'] = df.apply(lambda row: self._calculate_reb_rate(row), axis=1)
        
        return df
    
    def _add_advanced_player_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced player metrics"""
        df = df.copy()
        
        # Usage Rate (simplified calculation)
        df['USG_RATE'] = df.apply(lambda row: self._calculate_usage_rate(row), axis=1)
        
        # Player Efficiency Rating (simplified)
        df['PER'] = df.apply(lambda row: self._calculate_per(row), axis=1)
        
        # True Shooting Percentage
        df['TS_PCT'] = df.apply(lambda row: self._calculate_ts_pct(row), axis=1)
        
        # Effective Field Goal Percentage
        df['eFG_PCT'] = df.apply(lambda row: self._calculate_efg_pct(row), axis=1)
        
        # Assist Rate
        df['AST_RATE'] = df.apply(lambda row: self._calculate_ast_rate(row), axis=1)
        
        # Turnover Rate
        df['TOV_RATE'] = df.apply(lambda row: self._calculate_tov_rate(row), axis=1)
        
        return df
    
    def _add_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add contextual features"""
        df = df.copy()
        
        # Home/Away
        df['IS_HOME'] = df['MATCHUP'].str.contains('vs.').astype(int)
        
        # Rest days (simplified - would need actual schedule data)
        df['REST_DAYS'] = self._calculate_rest_days(df)
        
        # Back-to-back games
        df['IS_B2B'] = (df['REST_DAYS'] == 0).astype(int)
        
        # Season progress (game number in season)
        df['GAME_NUMBER'] = df.groupby(['TEAM_ID', 'SEASON']).cumcount() + 1
        
        # Days since season start
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df['DAYS_SINCE_START'] = (df['GAME_DATE'] - df.groupby('SEASON')['GAME_DATE'].transform('min')).dt.days
        
        return df
    
    def _calculate_offensive_rating(self, row: pd.Series) -> float:
        """Calculate offensive rating"""
        try:
            fga = row.get('FGA', 0)
            fta = row.get('FTA', 0)
            tov = row.get('TOV', 0)
            oreb = row.get('OREB', 0)
            pts = row.get('PTS', 0)
            
            # Estimate possessions
            possessions = fga - oreb + tov + 0.4 * fta
            if possessions <= 0:
                return 100.0
            
            return (pts / possessions) * 100
        except:
            return 100.0
    
    def _calculate_defensive_rating(self, row: pd.Series) -> float:
        """Calculate defensive rating (simplified)"""
        # This would need opponent data for accurate calculation
        # For now, return a placeholder
        return 110.0
    
    def _calculate_pace(self, row: pd.Series) -> float:
        """Calculate pace (possessions per 48 minutes)"""
        try:
            fga = row.get('FGA', 0)
            fta = row.get('FTA', 0)
            tov = row.get('TOV', 0)
            oreb = row.get('OREB', 0)
            min_played = row.get('MIN', 48)
            
            possessions = fga - oreb + tov + 0.4 * fta
            if min_played <= 0:
                return 100.0
            
            return (possessions * 48) / min_played
        except:
            return 100.0
    
    def _calculate_ts_pct(self, row: pd.Series) -> float:
        """Calculate true shooting percentage"""
        try:
            pts = row.get('PTS', 0)
            fga = row.get('FGA', 0)
            fta = row.get('FTA', 0)
            
            denominator = 2 * (fga + 0.44 * fta)
            if denominator <= 0:
                return 0.0
            
            return pts / denominator
        except:
            return 0.0
    
    def _calculate_efg_pct(self, row: pd.Series) -> float:
        """Calculate effective field goal percentage"""
        try:
            fgm = row.get('FGM', 0)
            fg3m = row.get('FG3M', 0)
            fga = row.get('FGA', 0)
            
            if fga <= 0:
                return 0.0
            
            return (fgm + 0.5 * fg3m) / fga
        except:
            return 0.0
    
    def _calculate_tov_rate(self, row: pd.Series) -> float:
        """Calculate turnover rate"""
        try:
            tov = row.get('TOV', 0)
            fga = row.get('FGA', 0)
            fta = row.get('FTA', 0)
            
            possessions = fga + 0.44 * fta + tov
            if possessions <= 0:
                return 0.0
            
            return tov / possessions
        except:
            return 0.0
    
    def _calculate_reb_rate(self, row: pd.Series) -> float:
        """Calculate rebound rate"""
        try:
            reb = row.get('REB', 0)
            oreb = row.get('OREB', 0)
            dreb = row.get('DREB', 0)
            
            # Simplified calculation
            if reb > 0:
                return reb / (reb + 40)  # Assume 40 opponent rebounds
            return 0.0
        except:
            return 0.0
    
    def _calculate_usage_rate(self, row: pd.Series) -> float:
        """Calculate usage rate"""
        try:
            fga = row.get('FGA', 0)
            fta = row.get('FTA', 0)
            tov = row.get('TOV', 0)
            min_played = row.get('MIN', 0)
            
            if min_played <= 0:
                return 0.0
            
            # Simplified usage rate calculation
            usage = (fga + 0.44 * fta + tov) / min_played
            return min(1.0, max(0.0, usage))
        except:
            return 0.0
    
    def _calculate_per(self, row: pd.Series) -> float:
        """Calculate Player Efficiency Rating (simplified)"""
        try:
            pts = row.get('PTS', 0)
            reb = row.get('REB', 0)
            ast = row.get('AST', 0)
            stl = row.get('STL', 0)
            blk = row.get('BLK', 0)
            tov = row.get('TOV', 0)
            min_played = row.get('MIN', 0)
            
            if min_played <= 0:
                return 0.0
            
            # Simplified PER calculation
            per = (pts + reb + ast + stl + blk - tov) / min_played * 15
            return max(0, per)
        except:
            return 0.0
    
    def _calculate_ast_rate(self, row: pd.Series) -> float:
        """Calculate assist rate"""
        try:
            ast = row.get('AST', 0)
            fga = row.get('FGA', 0)
            fta = row.get('FTA', 0)
            tov = row.get('TOV', 0)
            
            possessions = fga + 0.44 * fta + tov
            if possessions <= 0:
                return 0.0
            
            return ast / possessions
        except:
            return 0.0
    
    def _calculate_rest_days(self, df: pd.DataFrame) -> pd.Series:
        """Calculate rest days between games"""
        df = df.copy()
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values(['TEAM_ID', 'GAME_DATE'])
        
        rest_days = []
        for team_id in df['TEAM_ID'].unique():
            team_games = df[df['TEAM_ID'] == team_id].copy()
            team_games = team_games.sort_values('GAME_DATE')
            
            team_rest = [0]  # First game of season
            for i in range(1, len(team_games)):
                days_diff = (team_games.iloc[i]['GAME_DATE'] - 
                           team_games.iloc[i-1]['GAME_DATE']).days
                team_rest.append(days_diff)
            
            rest_days.extend(team_rest)
        
        return pd.Series(rest_days, index=df.index)
    
    def create_matchup_features(self, team_features: pd.DataFrame) -> pd.DataFrame:
        """Create features for team matchups"""
        logger.info("Creating matchup features...")
        
        # This would create features comparing two teams
        # For now, return the team features as-is
        return team_features
    
    def save_features(self, features: pd.DataFrame, filename: str) -> None:
        """Save features to file"""
        file_path = self.features_dir / f"{filename}.parquet"
        features.to_parquet(file_path, index=False)
        logger.info(f"Saved features to {file_path}")
    
    def load_features(self, filename: str) -> pd.DataFrame:
        """Load features from file"""
        file_path = self.features_dir / f"{filename}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded features from {file_path}")
            return df
        else:
            logger.warning(f"Features file {file_path} not found")
            return pd.DataFrame()

def main():
    """Main function to create features"""
    logger.info("Starting feature engineering...")
    
    engineer = NBAFeatureEngineer()
    
    # Load data (this would be called after data_fetch.py)
    # For now, we'll create a placeholder
    logger.info("Feature engineering pipeline ready!")
    logger.info("Run data_fetch.py first to get raw data, then call engineer methods")

if __name__ == "__main__":
    main()
