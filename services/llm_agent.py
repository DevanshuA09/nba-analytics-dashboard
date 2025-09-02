"""
LLM Agent for NBA Analytics Chatbot
Handles natural language queries and orchestrates NBA API tools
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import re
from services.nba_tools import NBATools, NBA_TOOLS
from services.cache import get_cache_service
from inference import predict_team_win, predict_player_points

logger = logging.getLogger(__name__)

class NBAAgent:
    """NBA Analytics AI Agent"""
    
    def __init__(self):
        self.nba_tools = NBATools(cache_service=get_cache_service())
        self.cache_service = get_cache_service()
        self.conversation_history = []
        
        # Tool mapping for easy access
        self.tool_functions = {
            name: tool["function"] for name, tool in NBA_TOOLS.items()
        }
        
        # Initialize tools with cache service
        nba_tools_instance = NBATools(cache_service=self.cache_service)
        for tool_name in self.tool_functions:
            if hasattr(self.tool_functions[tool_name], '__self__'):
                # If it's a bound method, we need to create a new instance
                method_name = tool_name.replace('get_', '')
                if hasattr(nba_tools_instance, method_name):
                    self.tool_functions[tool_name] = getattr(nba_tools_instance, method_name)
    
    def process_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process natural language query and return structured response
        
        Args:
            query: User's natural language query
            context: Optional context from previous queries
            
        Returns:
            Dictionary with response data and visualization info
        """
        try:
            # Add to conversation history
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "context": context
            })
            
            # Parse query and determine intent
            intent = self._parse_intent(query)
            
            # Execute appropriate action
            if intent["type"] == "player_stats":
                result = self._handle_player_query(intent, query)
            elif intent["type"] == "team_stats":
                result = self._handle_team_query(intent, query)
            elif intent["type"] == "prediction":
                result = self._handle_prediction_query(intent, query)
            elif intent["type"] == "comparison":
                result = self._handle_comparison_query(intent, query)
            elif intent["type"] == "schedule":
                result = self._handle_schedule_query(intent, query)
            elif intent["type"] == "standings":
                result = self._handle_standings_query(intent, query)
            else:
                result = self._handle_general_query(query)
            
            # Add to conversation history
            self.conversation_history[-1]["response"] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "type": "error",
                "message": f"Sorry, I encountered an error processing your query: {str(e)}",
                "data": None,
                "visualization": None
            }
    
    def _parse_intent(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query to determine intent
        
        Args:
            query: User's query
            
        Returns:
            Dictionary with parsed intent
        """
        query_lower = query.lower()
        
        # Player-related queries
        if any(word in query_lower for word in ["player", "scored", "points", "rebounds", "assists", "stats"]):
            player_name = self._extract_player_name(query)
            if player_name:
                return {
                    "type": "player_stats",
                    "player_name": player_name,
                    "query": query
                }
        
        # Team-related queries
        if any(word in query_lower for word in ["team", "record", "wins", "losses", "roster"]):
            team_name = self._extract_team_name(query)
            if team_name:
                return {
                    "type": "team_stats",
                    "team_name": team_name,
                    "query": query
                }
        
        # Prediction queries
        if any(word in query_lower for word in ["predict", "win", "lose", "chance", "probability", "forecast"]):
            return {
                "type": "prediction",
                "query": query
            }
        
        # Comparison queries
        if any(word in query_lower for word in ["compare", "vs", "versus", "against", "head to head"]):
            return {
                "type": "comparison",
                "query": query
            }
        
        # Schedule queries
        if any(word in query_lower for word in ["today", "schedule", "games", "when", "time"]):
            return {
                "type": "schedule",
                "query": query
            }
        
        # Standings queries
        if any(word in query_lower for word in ["standings", "rank", "position", "conference"]):
            return {
                "type": "standings",
                "query": query
            }
        
        return {
            "type": "general",
            "query": query
        }
    
    def _extract_player_name(self, query: str) -> Optional[str]:
        """Extract player name from query"""
        # Common NBA player names (simplified extraction)
        player_patterns = [
            r"([A-Z][a-z]+ [A-Z][a-z]+)",  # First Last
            r"([A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+)",  # First M. Last
        ]
        
        for pattern in player_patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_team_name(self, query: str) -> Optional[str]:
        """Extract team name from query"""
        # Common NBA team names
        team_names = [
            "Lakers", "Warriors", "Celtics", "Heat", "Bulls", "Knicks",
            "Nets", "76ers", "Raptors", "Bucks", "Pacers", "Cavaliers",
            "Pistons", "Magic", "Hawks", "Hornets", "Wizards", "Spurs",
            "Rockets", "Mavericks", "Grizzlies", "Pelicans", "Thunder",
            "Trail Blazers", "Jazz", "Nuggets", "Timberwolves", "Suns",
            "Kings", "Clippers"
        ]
        
        query_lower = query.lower()
        for team in team_names:
            if team.lower() in query_lower:
                return team
        
        return None
    
    def _handle_player_query(self, intent: Dict, query: str) -> Dict[str, Any]:
        """Handle player-related queries"""
        player_name = intent["player_name"]
        
        # Get player game log
        player_data = self.nba_tools.get_player_gamelog(player_name, last_n_games=10)
        
        if "error" in player_data:
            return {
                "type": "error",
                "message": player_data["error"],
                "data": None,
                "visualization": None
            }
        
        # Format response
        response = {
            "type": "player_stats",
            "message": f"Here are {player_name}'s recent statistics:",
            "data": player_data,
            "visualization": {
                "type": "table",
                "title": f"{player_name} - Recent Performance",
                "data": player_data["game_log"],
                "columns": ["GAME_DATE", "MATCHUP", "WL", "PTS", "REB", "AST", "MIN"]
            }
        }
        
        return response
    
    def _handle_team_query(self, intent: Dict, query: str) -> Dict[str, Any]:
        """Handle team-related queries"""
        team_name = intent["team_name"]
        
        # Get team stats and recent games
        team_stats = self.nba_tools.get_team_stats(team_name)
        team_games = self.nba_tools.get_team_gamelog(team_name, last_n_games=10)
        
        if "error" in team_stats:
            return {
                "type": "error",
                "message": team_stats["error"],
                "data": None,
                "visualization": None
            }
        
        # Combine data
        combined_data = {
            "team_stats": team_stats,
            "recent_games": team_games
        }
        
        response = {
            "type": "team_stats",
            "message": f"Here are the {team_name}'s statistics and recent performance:",
            "data": combined_data,
            "visualization": {
                "type": "mixed",
                "charts": [
                    {
                        "type": "bar",
                        "title": f"{team_name} - Recent Games",
                        "data": team_games.get("game_log", []),
                        "x": "GAME_DATE",
                        "y": "PTS"
                    }
                ]
            }
        }
        
        return response
    
    def _handle_prediction_query(self, intent: Dict, query: str) -> Dict[str, Any]:
        """Handle prediction queries"""
        query_lower = query.lower()
        
        # Try to extract teams for win prediction
        if "win" in query_lower or "lose" in query_lower:
            teams = self._extract_teams_from_query(query)
            if len(teams) >= 2:
                # Make team win prediction
                prediction_data = {
                    "home_team": teams[0],
                    "away_team": teams[1],
                    "home_rest_days": 1,
                    "away_rest_days": 1,
                    "season": "2023-24"
                }
                
                try:
                    prediction = predict_team_win(prediction_data)
                    return {
                        "type": "prediction",
                        "message": f"Win probability prediction for {teams[0]} vs {teams[1]}:",
                        "data": prediction,
                        "visualization": {
                            "type": "gauge",
                            "title": "Win Probability",
                            "value": prediction["win_probability"],
                            "max": 1.0
                        }
                    }
                except Exception as e:
                    return {
                        "type": "error",
                        "message": f"Could not make prediction: {str(e)}",
                        "data": None,
                        "visualization": None
                    }
        
        # Try to extract player for points prediction
        player_name = self._extract_player_name(query)
        if player_name:
            prediction_data = {
                "player_name": player_name,
                "opponent_team": "Unknown",
                "is_home": True,
                "season": "2023-24"
            }
            
            try:
                prediction = predict_player_points(prediction_data)
                return {
                    "type": "prediction",
                    "message": f"Points prediction for {player_name}:",
                    "data": prediction,
                    "visualization": {
                        "type": "bar",
                        "title": "Predicted Points",
                        "value": prediction["predicted_points"],
                        "confidence": prediction["confidence_interval"]
                    }
                }
            except Exception as e:
                return {
                    "type": "error",
                    "message": f"Could not make prediction: {str(e)}",
                    "data": None,
                    "visualization": None
                }
        
        return {
            "type": "error",
            "message": "I couldn't understand what you'd like me to predict. Please specify teams or players.",
            "data": None,
            "visualization": None
        }
    
    def _handle_comparison_query(self, intent: Dict, query: str) -> Dict[str, Any]:
        """Handle comparison queries"""
        # Extract entities to compare
        teams = self._extract_teams_from_query(query)
        players = [self._extract_player_name(query)]
        
        if len(teams) >= 2:
            # Team comparison
            h2h_data = self.nba_tools.get_head_to_head(teams[0], teams[1])
            
            return {
                "type": "comparison",
                "message": f"Head-to-head comparison between {teams[0]} and {teams[1]}:",
                "data": h2h_data,
                "visualization": {
                    "type": "comparison_table",
                    "title": f"{teams[0]} vs {teams[1]}",
                    "data": h2h_data
                }
            }
        
        return {
            "type": "error",
            "message": "I couldn't identify what you'd like to compare. Please specify teams or players.",
            "data": None,
            "visualization": None
        }
    
    def _handle_schedule_query(self, intent: Dict, query: str) -> Dict[str, Any]:
        """Handle schedule queries"""
        todays_games = self.nba_tools.get_todays_games()
        
        if "error" in todays_games:
            return {
                "type": "error",
                "message": todays_games["error"],
                "data": None,
                "visualization": None
            }
        
        return {
            "type": "schedule",
            "message": f"Here are today's NBA games:",
            "data": todays_games,
            "visualization": {
                "type": "schedule_table",
                "title": "Today's Games",
                "data": todays_games["games"]
            }
        }
    
    def _handle_standings_query(self, intent: Dict, query: str) -> Dict[str, Any]:
        """Handle standings queries"""
        # Extract conference if mentioned
        conference = None
        if "east" in query.lower():
            conference = "East"
        elif "west" in query.lower():
            conference = "West"
        
        standings = self.nba_tools.get_standings(conference=conference)
        
        if "error" in standings:
            return {
                "type": "error",
                "message": standings["error"],
                "data": None,
                "visualization": None
            }
        
        return {
            "type": "standings",
            "message": f"Here are the NBA standings:",
            "data": standings,
            "visualization": {
                "type": "standings_table",
                "title": f"{conference or 'NBA'} Standings",
                "data": standings["standings"]
            }
        }
    
    def _handle_general_query(self, query: str) -> Dict[str, Any]:
        """Handle general queries"""
        return {
            "type": "general",
            "message": "I can help you with NBA statistics, predictions, and analysis. Try asking about players, teams, or today's games!",
            "data": None,
            "visualization": None
        }
    
    def _extract_teams_from_query(self, query: str) -> List[str]:
        """Extract team names from query"""
        teams = []
        team_name = self._extract_team_name(query)
        if team_name:
            teams.append(team_name)
        
        # Try to find second team
        query_without_first = query.replace(team_name, "", 1) if team_name else query
        second_team = self._extract_team_name(query_without_first)
        if second_team:
            teams.append(second_team)
        
        return teams
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

# Global agent instance
nba_agent = None

def get_nba_agent() -> NBAAgent:
    """Get global NBA agent instance"""
    global nba_agent
    if nba_agent is None:
        nba_agent = NBAAgent()
    return nba_agent
