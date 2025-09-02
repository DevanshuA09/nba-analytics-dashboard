"""
Demo data generator for testing when NBA API is not available
"""

import random
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

class DemoDataGenerator:
    """Generate demo data for testing purposes"""
    
    def __init__(self):
        self.teams = [
            "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
            "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
            "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
            "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
            "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
            "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
            "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
            "Utah Jazz", "Washington Wizards"
        ]
        
        self.players = [
            "LeBron James", "Stephen Curry", "Kevin Durant", "Giannis Antetokounmpo",
            "Luka Doncic", "Jayson Tatum", "Joel Embiid", "Nikola Jokic",
            "Jimmy Butler", "Kawhi Leonard", "Paul George", "Damian Lillard",
            "Russell Westbrook", "Kyrie Irving", "James Harden", "Anthony Davis",
            "Devin Booker", "Donovan Mitchell", "Trae Young", "Ja Morant",
            "Zion Williamson", "Brandon Ingram", "Bradley Beal", "Kemba Walker",
            "Kyle Lowry", "Chris Paul", "John Wall", "Blake Griffin"
        ]
    
    def get_team_list(self) -> List[Dict[str, Any]]:
        """Get demo team list"""
        return [
            {
                "id": i + 1,
                "name": team,
                "abbreviation": team.split()[-1][:3].upper(),
                "city": " ".join(team.split()[:-1]),
                "conference": "Eastern" if i < 15 else "Western",
                "division": self._get_division(i)
            }
            for i, team in enumerate(self.teams)
        ]
    
    def get_player_list(self, team: str = None) -> List[Dict[str, Any]]:
        """Get demo player list"""
        if team:
            # Return 3-5 players for the specified team
            num_players = random.randint(3, 5)
            team_players = random.sample(self.players, num_players)
        else:
            team_players = self.players
        
        return [
            {
                "id": i + 1,
                "name": player,
                "first_name": player.split()[0],
                "last_name": player.split()[-1],
                "team": team or random.choice(self.teams)
            }
            for i, player in enumerate(team_players)
        ]
    
    def get_todays_games(self) -> List[Dict[str, Any]]:
        """Get demo today's games"""
        # Generate 2-4 random games
        num_games = random.randint(2, 4)
        games = []
        
        for i in range(num_games):
            home_team = random.choice(self.teams)
            away_team = random.choice([t for t in self.teams if t != home_team])
            
            # Generate realistic win probabilities
            home_prob = random.uniform(0.3, 0.7)
            away_prob = 1 - home_prob
            
            games.append({
                "game_id": f"demo_{i+1}",
                "home_team": home_team,
                "away_team": away_team,
                "home_team_id": self.teams.index(home_team) + 1,
                "away_team_id": self.teams.index(away_team) + 1,
                "game_time": f"{random.randint(7, 10)}:{random.choice(['00', '30'])} PM ET",
                "home_rest_days": random.randint(1, 3),
                "away_rest_days": random.randint(1, 3),
                "home_win_probability": round(home_prob, 3),
                "away_win_probability": round(away_prob, 3)
            })
        
        return games
    
    def _get_division(self, team_index: int) -> str:
        """Get division for team index"""
        if team_index < 5:
            return "Atlantic"
        elif team_index < 10:
            return "Central"
        elif team_index < 15:
            return "Southeast"
        elif team_index < 20:
            return "Northwest"
        elif team_index < 25:
            return "Pacific"
        else:
            return "Southwest"
    
    def generate_team_features(self) -> np.ndarray:
        """Generate demo team features"""
        # Generate 16 features (matching the feature extractor)
        features = np.random.normal(0, 1, 16)
        # Ensure no NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        return features
    
    def generate_player_features(self) -> np.ndarray:
        """Generate demo player features"""
        # Generate 14 features (matching the feature extractor)
        features = np.random.normal(0, 1, 14)
        # Ensure no NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        return features
