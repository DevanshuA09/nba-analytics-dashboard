"""
NBA Data Acquisition Pipeline
Fetches real NBA data from nba_api for the last 3 seasons
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from typing import List, Dict, Any, Optional
import time
import requests
from pathlib import Path

# NBA API imports
from nba_api.stats.endpoints import (
    teamgamelog, playergamelog, scoreboard, 
    leaguedashteamstats, leaguedashplayerstats,
    teamdashboardbygeneralsplits, playerdashboardbygeneralsplits,
    commonteamroster, commonplayerinfo
)
from nba_api.stats.static import teams, players
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NBADataFetcher:
    """Fetches and processes NBA data from nba_api"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "features").mkdir(exist_ok=True)
        
        # Rate limiting
        self.request_delay = 0.6  # NBA API rate limit
        
    def fetch_seasons_data(self, seasons: List[str] = None) -> Dict[str, Any]:
        """
        Fetch data for multiple seasons
        
        Args:
            seasons: List of seasons in format "2021-22", "2022-23", etc.
        """
        if seasons is None:
            # Default to last 3 seasons
            current_year = datetime.now().year
            seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(current_year-3, current_year)]
        
        logger.info(f"Fetching data for seasons: {seasons}")
        
        all_data = {}
        
        for season in seasons:
            logger.info(f"Processing season {season}...")
            try:
                season_data = self.fetch_season_data(season)
                all_data[season] = season_data
                
                # Save season data
                self.save_season_data(season, season_data)
                
                # Rate limiting
                time.sleep(self.request_delay)
                
            except Exception as e:
                logger.error(f"Error fetching data for season {season}: {e}")
                continue
        
        return all_data
    
    def fetch_season_data(self, season: str) -> Dict[str, Any]:
        """Fetch all data for a single season"""
        logger.info(f"Fetching season {season} data...")
        
        season_data = {
            'team_game_logs': self.fetch_team_game_logs(season),
            'player_game_logs': self.fetch_player_game_logs(season),
            'team_stats': self.fetch_team_stats(season),
            'player_stats': self.fetch_player_stats(season),
            'team_rosters': self.fetch_team_rosters(season),
            'schedule': self.fetch_season_schedule(season)
        }
        
        return season_data
    
    def fetch_team_game_logs(self, season: str) -> pd.DataFrame:
        """Fetch team game logs for all teams in a season"""
        logger.info(f"Fetching team game logs for {season}...")
        
        all_game_logs = []
        team_dict = teams.get_teams()
        
        for team in team_dict:
            try:
                logger.info(f"Fetching game logs for {team['full_name']}...")
                
                game_log = teamgamelog.TeamGameLog(
                    team_id=team['id'], 
                    season=season
                )
                df = game_log.get_data_frames()[0]
                
                if not df.empty:
                    df['TEAM_ID'] = team['id']
                    df['TEAM_NAME'] = team['full_name']
                    df['SEASON'] = season
                    all_game_logs.append(df)
                
                time.sleep(self.request_delay)
                
            except Exception as e:
                logger.error(f"Error fetching game logs for {team['full_name']}: {e}")
                continue
        
        if all_game_logs:
            combined_df = pd.concat(all_game_logs, ignore_index=True)
            logger.info(f"Fetched {len(combined_df)} team game records for {season}")
            return combined_df
        else:
            logger.warning(f"No team game logs found for {season}")
            return pd.DataFrame()
    
    def fetch_player_game_logs(self, season: str) -> pd.DataFrame:
        """Fetch player game logs for all players in a season"""
        logger.info(f"Fetching player game logs for {season}...")
        
        all_game_logs = []
        
        # Get all players for the season
        try:
            player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed='PerGame'
            )
            players_df = player_stats.get_data_frames()[0]
            
            # Filter to players with significant minutes
            players_df = players_df[players_df['MIN'] >= 10]  # At least 10 minutes per game
            
            logger.info(f"Fetching game logs for {len(players_df)} players...")
            
            for _, player in players_df.iterrows():
                try:
                    player_id = player['PLAYER_ID']
                    player_name = player['PLAYER_NAME']
                    
                    game_log = playergamelog.PlayerGameLog(
                        player_id=player_id,
                        season=season
                    )
                    df = game_log.get_data_frames()[0]
                    
                    if not df.empty:
                        df['PLAYER_ID'] = player_id
                        df['PLAYER_NAME'] = player_name
                        df['SEASON'] = season
                        all_game_logs.append(df)
                    
                    time.sleep(self.request_delay)
                    
                except Exception as e:
                    logger.error(f"Error fetching game logs for {player_name}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error fetching player list for {season}: {e}")
            return pd.DataFrame()
        
        if all_game_logs:
            combined_df = pd.concat(all_game_logs, ignore_index=True)
            logger.info(f"Fetched {len(combined_df)} player game records for {season}")
            return combined_df
        else:
            logger.warning(f"No player game logs found for {season}")
            return pd.DataFrame()
    
    def fetch_team_stats(self, season: str) -> pd.DataFrame:
        """Fetch team statistics for a season"""
        logger.info(f"Fetching team stats for {season}...")
        
        try:
            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                per_mode_detailed='PerGame'
            )
            df = team_stats.get_data_frames()[0]
            df['SEASON'] = season
            
            logger.info(f"Fetched team stats for {len(df)} teams in {season}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching team stats for {season}: {e}")
            return pd.DataFrame()
    
    def fetch_player_stats(self, season: str) -> pd.DataFrame:
        """Fetch player statistics for a season"""
        logger.info(f"Fetching player stats for {season}...")
        
        try:
            player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed='PerGame'
            )
            df = player_stats.get_data_frames()[0]
            df['SEASON'] = season
            
            logger.info(f"Fetched player stats for {len(df)} players in {season}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching player stats for {season}: {e}")
            return pd.DataFrame()
    
    def fetch_team_rosters(self, season: str) -> pd.DataFrame:
        """Fetch team rosters for a season"""
        logger.info(f"Fetching team rosters for {season}...")
        
        all_rosters = []
        team_dict = teams.get_teams()
        
        for team in team_dict:
            try:
                roster = commonteamroster.CommonTeamRoster(
                    team_id=team['id'],
                    season=season
                )
                df = roster.get_data_frames()[0]
                
                if not df.empty:
                    df['TEAM_ID'] = team['id']
                    df['TEAM_NAME'] = team['full_name']
                    df['SEASON'] = season
                    all_rosters.append(df)
                
                time.sleep(self.request_delay)
                
            except Exception as e:
                logger.error(f"Error fetching roster for {team['full_name']}: {e}")
                continue
        
        if all_rosters:
            combined_df = pd.concat(all_rosters, ignore_index=True)
            logger.info(f"Fetched rosters for {len(combined_df)} players in {season}")
            return combined_df
        else:
            logger.warning(f"No rosters found for {season}")
            return pd.DataFrame()
    
    def fetch_season_schedule(self, season: str) -> pd.DataFrame:
        """Fetch season schedule"""
        logger.info(f"Fetching schedule for {season}...")
        
        # This is a simplified approach - in practice, you'd need to fetch
        # the full schedule from NBA API or use a different endpoint
        try:
            # For now, we'll create a placeholder that can be enhanced
            # with actual schedule data from NBA API
            schedule_data = []
            
            # Get all teams
            team_dict = teams.get_teams()
            team_ids = [team['id'] for team in team_dict]
            
            # Create a simple schedule structure
            # In a real implementation, you'd fetch the actual schedule
            for team_id in team_ids:
                try:
                    # This would be replaced with actual schedule fetching
                    # For now, we'll use team game logs to infer schedule
                    game_log = teamgamelog.TeamGameLog(
                        team_id=team_id,
                        season=season
                    )
                    df = game_log.get_data_frames()[0]
                    
                    if not df.empty:
                        schedule_data.append(df[['GAME_DATE', 'MATCHUP', 'WL', 'TEAM_ID']])
                    
                    time.sleep(self.request_delay)
                    
                except Exception as e:
                    logger.error(f"Error fetching schedule for team {team_id}: {e}")
                    continue
            
            if schedule_data:
                combined_df = pd.concat(schedule_data, ignore_index=True)
                combined_df['SEASON'] = season
                logger.info(f"Fetched schedule data for {season}")
                return combined_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching schedule for {season}: {e}")
            return pd.DataFrame()
    
    def save_season_data(self, season: str, season_data: Dict[str, Any]) -> None:
        """Save season data to files"""
        logger.info(f"Saving season {season} data...")
        
        season_dir = self.data_dir / "raw" / season
        season_dir.mkdir(exist_ok=True)
        
        for data_type, df in season_data.items():
            if not df.empty:
                file_path = season_dir / f"{data_type}.parquet"
                df.to_parquet(file_path, index=False)
                logger.info(f"Saved {data_type} to {file_path}")
    
    def load_season_data(self, season: str) -> Dict[str, pd.DataFrame]:
        """Load season data from files"""
        logger.info(f"Loading season {season} data...")
        
        season_dir = self.data_dir / "raw" / season
        season_data = {}
        
        if not season_dir.exists():
            logger.warning(f"No data found for season {season}")
            return season_data
        
        for file_path in season_dir.glob("*.parquet"):
            data_type = file_path.stem
            try:
                df = pd.read_parquet(file_path)
                season_data[data_type] = df
                logger.info(f"Loaded {data_type} with {len(df)} records")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return season_data
    
    def get_all_seasons_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load all available season data"""
        all_data = {}
        
        for season_dir in (self.data_dir / "raw").iterdir():
            if season_dir.is_dir():
                season = season_dir.name
                all_data[season] = self.load_season_data(season)
        
        return all_data

def main():
    """Main function to fetch NBA data"""
    logger.info("Starting NBA data acquisition...")
    
    fetcher = NBADataFetcher()
    
    # Fetch data for last 3 seasons
    seasons = ["2021-22", "2022-23", "2023-24"]
    
    try:
        all_data = fetcher.fetch_seasons_data(seasons)
        
        logger.info("Data acquisition completed successfully!")
        logger.info(f"Fetched data for {len(all_data)} seasons")
        
        # Print summary
        for season, season_data in all_data.items():
            logger.info(f"\nSeason {season} summary:")
            for data_type, df in season_data.items():
                if not df.empty:
                    logger.info(f"  {data_type}: {len(df)} records")
        
    except Exception as e:
        logger.error(f"Data acquisition failed: {e}")
        raise

if __name__ == "__main__":
    main()
