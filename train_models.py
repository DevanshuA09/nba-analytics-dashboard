#!/usr/bin/env python3
"""
Training script for NBA predictive models
Trains both team win probability and player points models
"""

import os
import sys
import logging
from datetime import datetime

# Add current directory to path
sys.path.append('.')

from training.team_train import TeamWinModelTrainer
from training.player_train import PlayerPointsModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_team_model():
    """Train the team win probability model"""
    logger.info("=" * 60)
    logger.info("TRAINING TEAM WIN PROBABILITY MODEL")
    logger.info("=" * 60)
    
    try:
        trainer = TeamWinModelTrainer(season="2023-24")
        
        # Prepare training data
        logger.info("Preparing training data...")
        X, y = trainer.prepare_training_data()
        
        if len(X) == 0:
            logger.warning("No training data available for team model")
            return False
        
        # Train model
        logger.info("Training model...")
        results = trainer.train_model(X, y)
        
        # Save model
        logger.info("Saving model...")
        trainer.save_model("models/team_win.pkl")
        
        logger.info("Team model training completed successfully!")
        logger.info(f"Results: {results}")
        return True
        
    except Exception as e:
        logger.error(f"Team model training failed: {e}")
        return False

def train_player_model():
    """Train the player points prediction model"""
    logger.info("=" * 60)
    logger.info("TRAINING PLAYER POINTS PREDICTION MODEL")
    logger.info("=" * 60)
    
    try:
        trainer = PlayerPointsModelTrainer(season="2023-24")
        
        # Prepare training data
        logger.info("Preparing training data...")
        X, y = trainer.prepare_training_data()
        
        if len(X) == 0:
            logger.warning("No training data available for player model")
            return False
        
        # Train model
        logger.info("Training model...")
        results = trainer.train_model(X, y)
        
        # Save model
        logger.info("Saving model...")
        trainer.save_model("models/player_points.pkl")
        
        logger.info("Player model training completed successfully!")
        logger.info(f"Results: {results}")
        return True
        
    except Exception as e:
        logger.error(f"Player model training failed: {e}")
        return False

def main():
    """Main training function"""
    logger.info("Starting NBA Analytics Model Training")
    logger.info(f"Training started at: {datetime.now()}")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Train models
    team_success = train_team_model()
    player_success = train_player_model()
    
    # Summary
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Team Win Model: {'SUCCESS' if team_success else 'FAILED'}")
    logger.info(f"Player Points Model: {'SUCCESS' if player_success else 'FAILED'}")
    
    if team_success and player_success:
        logger.info("All models trained successfully!")
        logger.info("You can now start the FastAPI server with: python app.py")
    else:
        logger.warning("Some models failed to train. Check the logs above.")
    
    logger.info(f"Training completed at: {datetime.now()}")

if __name__ == "__main__":
    main()
