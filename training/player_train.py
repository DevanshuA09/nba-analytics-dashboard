"""
Player Points Forecast Model Training
Trains a ridge regression model to predict player points
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any, Tuple

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import sys
sys.path.append('..')
from services.features import FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlayerPointsModelTrainer:
    """Trainer for player points prediction model"""
    
    def __init__(self, season: str = "2023-24"):
        self.season = season
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from NBA player game logs
        
        Returns:
            X: Feature matrix
            y: Target values (points scored)
        """
        logger.info("Preparing training data...")
        
        # Get all NBA players
        player_dict = players.get_players()
        
        # Filter to active players (simplified - take first 100 players)
        active_players = player_dict[:100]  # In production, filter by active status
        
        features_list = []
        points_list = []
        
        # Collect data from each player's game log
        for player in active_players:
            try:
                player_id = player['id']
                player_name = player['full_name']
                
                logger.info(f"Processing player {player_name}...")
                
                # Get player game log
                game_log = playergamelog.PlayerGameLog(player_id=player_id, season=self.season)
                df = game_log.get_data_frames()[0]
                
                if df.empty or len(df) < 20:  # Need enough games for rolling stats
                    continue
                
                # Process each game (skip first 10 games to allow for rolling stats)
                for i in range(10, len(df)):
                    try:
                        game = df.iloc[i]
                        
                        # Get opponent team (simplified)
                        opponent_team = "Unknown Team"  # Would need to parse from matchup
                        is_home = True  # Simplified
                        
                        # Get features for this game
                        features = self.feature_extractor.extract_player_features(
                            player_name=player_name,
                            opponent_team=opponent_team,
                            is_home=is_home,
                            season=self.season
                        )
                        
                        if features is not None:
                            features_list.append(features)
                            points_list.append(game['PTS'])
                        
                    except Exception as e:
                        logger.warning(f"Error processing game for {player_name}: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Error processing player {player.get('full_name', 'Unknown')}: {e}")
                continue
        
        if not features_list:
            raise ValueError("No training data could be collected")
        
        X = np.array(features_list)
        y = np.array(points_list)
        
        logger.info(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Points distribution - Mean: {y.mean():.2f}, Std: {y.std():.2f}")
        
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the player points prediction model
        
        Args:
            X: Feature matrix
            y: Target values (points)
            
        Returns:
            Dictionary with training results
        """
        logger.info("Training player points prediction model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models and select best
        models = {
            'ridge_regression': Ridge(alpha=1.0, random_state=42),
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_score = float('inf')
        best_name = ""
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Cross-validation (using negative MSE)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
            mean_cv_score = -cv_scores.mean()  # Convert back to positive MSE
            
            logger.info(f"{name} CV MSE: {mean_cv_score:.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            if mean_cv_score < best_score:
                best_score = mean_cv_score
                best_model = model
                best_name = name
        
        # Train best model on full training set
        best_model.fit(X_train_scaled, y_train)
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test_scaled)
        test_mse = mean_squared_error(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Best model: {best_name}")
        logger.info(f"Test MSE: {test_mse:.3f}")
        logger.info(f"Test MAE: {test_mae:.3f}")
        logger.info(f"Test R²: {test_r2:.3f}")
        
        self.model = best_model
        
        return {
            'model_name': best_name,
            'cv_mse': best_score,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'feature_importance': self._get_feature_importance(best_model, best_name)
        }
    
    def _get_feature_importance(self, model, model_name: str) -> List[float]:
        """Get feature importance if available"""
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_.tolist()
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_).tolist()
        else:
            return []
    
    def save_model(self, model_path: str = "models/player_points.pkl") -> None:
        """Save the trained model and scaler"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and scaler
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'training_date': datetime.now().isoformat(),
            'season': self.season
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = "models/player_points.pkl") -> None:
        """Load a trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        
        logger.info(f"Model loaded from {model_path}")
    
    def predict(self, player_name: str, opponent_team: str, 
                is_home: bool = True) -> float:
        """
        Make a prediction for player points
        
        Args:
            player_name: Player name
            opponent_team: Opponent team name
            is_home: Whether player is at home
            
        Returns:
            Predicted points
        """
        if self.model is None:
            raise ValueError("No model loaded. Load or train a model first.")
        
        # Extract features
        features = self.feature_extractor.extract_player_features(
            player_name=player_name,
            opponent_team=opponent_team,
            is_home=is_home,
            season=self.season
        )
        
        if features is None:
            raise ValueError("Could not extract features for the specified player")
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        predicted_points = self.model.predict(features_scaled)[0]
        
        return max(0, predicted_points)  # Ensure non-negative points

def main():
    """Main training function"""
    logger.info("Starting player points prediction model training...")
    
    # Initialize trainer
    trainer = PlayerPointsModelTrainer(season="2023-24")
    
    try:
        # Prepare training data
        X, y = trainer.prepare_training_data()
        
        # Train model
        results = trainer.train_model(X, y)
        
        # Save model
        trainer.save_model()
        
        logger.info("Training completed successfully!")
        logger.info(f"Results: {results}")
        
        # Test prediction
        test_prediction = trainer.predict(
            player_name="LeBron James",
            opponent_team="Golden State Warriors"
        )
        logger.info(f"Test prediction - LeBron James vs Warriors: {test_prediction:.1f} points")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
