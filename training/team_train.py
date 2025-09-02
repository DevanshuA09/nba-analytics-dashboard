"""
Team Win Probability Model Training
Trains a logistic regression model to predict team win probability
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any, Tuple

from nba_api.stats.endpoints import teamgamelog, scoreboard
from nba_api.stats.static import teams
import sys
sys.path.append('..')
from services.features import FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TeamWinModelTrainer:
    """Trainer for team win probability model"""
    
    def __init__(self, season: str = "2023-24"):
        self.season = season
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from NBA game logs
        
        Returns:
            X: Feature matrix
            y: Target labels (1 for home team win, 0 for away team win)
        """
        logger.info("Preparing training data...")
        
        # Get all NBA teams
        team_dict = teams.get_teams()
        team_ids = [team['id'] for team in team_dict]
        
        features_list = []
        labels_list = []
        
        # Collect data from each team's game log
        for team_id in team_ids:
            try:
                logger.info(f"Processing team {team_id}...")
                
                # Get team game log
                game_log = teamgamelog.TeamGameLog(team_id=team_id, season=self.season)
                df = game_log.get_data_frames()[0]
                
                if df.empty:
                    continue
                
                # Process each game
                for _, game in df.iterrows():
                    try:
                        # Determine if this team was home or away
                        is_home = game['MATCHUP'].find('@') == -1
                        
                        if is_home:
                            # This team was home
                            home_team = teams.find_team_name_by_id(team_id)
                            away_team = game['MATCHUP'].split(' vs. ')[1] if ' vs. ' in game['MATCHUP'] else game['MATCHUP'].split(' @ ')[1]
                            
                            # Get features for this matchup
                            features = self.feature_extractor.extract_team_features(
                                home_team=home_team,
                                away_team=away_team,
                                home_rest_days=1,  # Simplified
                                away_rest_days=1,  # Simplified
                                season=self.season
                            )
                            
                            if features is not None:
                                features_list.append(features)
                                # Label: 1 if home team won, 0 if away team won
                                label = 1 if game['WL'] == 'W' else 0
                                labels_list.append(label)
                        
                    except Exception as e:
                        logger.warning(f"Error processing game: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Error processing team {team_id}: {e}")
                continue
        
        if not features_list:
            raise ValueError("No training data could be collected")
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        logger.info(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the team win probability model
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Dictionary with training results
        """
        logger.info("Training team win probability model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models and select best
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            mean_cv_score = cv_scores.mean()
            
            logger.info(f"{name} CV accuracy: {mean_cv_score:.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            if mean_cv_score > best_score:
                best_score = mean_cv_score
                best_model = model
                best_name = name
        
        # Train best model on full training set
        best_model.fit(X_train_scaled, y_train)
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Best model: {best_name}")
        logger.info(f"Test accuracy: {test_accuracy:.3f}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        self.model = best_model
        
        return {
            'model_name': best_name,
            'cv_accuracy': best_score,
            'test_accuracy': test_accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
    
    def save_model(self, model_path: str = "models/team_win.pkl") -> None:
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
    
    def load_model(self, model_path: str = "models/team_win.pkl") -> None:
        """Load a trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        
        logger.info(f"Model loaded from {model_path}")
    
    def predict(self, home_team: str, away_team: str, 
                home_rest_days: int = 1, away_rest_days: int = 1) -> float:
        """
        Make a prediction for a team matchup
        
        Args:
            home_team: Home team name
            away_team: Away team name
            home_rest_days: Home team rest days
            away_rest_days: Away team rest days
            
        Returns:
            Win probability for home team
        """
        if self.model is None:
            raise ValueError("No model loaded. Load or train a model first.")
        
        # Extract features
        features = self.feature_extractor.extract_team_features(
            home_team=home_team,
            away_team=away_team,
            home_rest_days=home_rest_days,
            away_rest_days=away_rest_days,
            season=self.season
        )
        
        if features is None:
            raise ValueError("Could not extract features for the specified teams")
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        win_probability = self.model.predict_proba(features_scaled)[0][1]
        
        return win_probability

def main():
    """Main training function"""
    logger.info("Starting team win probability model training...")
    
    # Initialize trainer
    trainer = TeamWinModelTrainer(season="2023-24")
    
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
            home_team="Golden State Warriors",
            away_team="Los Angeles Lakers"
        )
        logger.info(f"Test prediction - Warriors vs Lakers: {test_prediction:.3f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
