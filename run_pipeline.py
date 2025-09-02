"""
Complete NBA Analytics Pipeline
Runs the full pipeline from data acquisition to model training
"""

import logging
from pathlib import Path
import sys
from datetime import datetime

# Import our modules
from data_fetch import NBADataFetcher
from features import NBAFeatureEngineer
from train_team import TeamWinModelTrainer
from train_player import PlayerPointsModelTrainer
from inference import NBAInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class NBAPipeline:
    """Complete NBA analytics pipeline"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_fetcher = NBADataFetcher(data_dir)
        self.feature_engineer = NBAFeatureEngineer(data_dir)
        self.team_trainer = TeamWinModelTrainer(data_dir)
        self.player_trainer = PlayerPointsModelTrainer(data_dir)
        
    def run_full_pipeline(self, seasons: list = None) -> bool:
        """Run the complete pipeline"""
        logger.info("Starting NBA Analytics Pipeline...")
        
        try:
            # Step 1: Data Acquisition
            logger.info("=" * 60)
            logger.info("STEP 1: DATA ACQUISITION")
            logger.info("=" * 60)
            
            if seasons is None:
                seasons = ["2021-22", "2022-23", "2023-24"]
            
            all_data = self.data_fetcher.fetch_seasons_data(seasons)
            
            if not all_data:
                logger.error("No data acquired. Pipeline failed.")
                return False
            
            # Step 2: Feature Engineering
            logger.info("=" * 60)
            logger.info("STEP 2: FEATURE ENGINEERING")
            logger.info("=" * 60)
            
            # Process team features
            team_features_list = []
            for season, season_data in all_data.items():
                if 'team_game_logs' in season_data and not season_data['team_game_logs'].empty:
                    logger.info(f"Processing team features for {season}...")
                    team_features = self.feature_engineer.create_team_features(
                        season_data['team_game_logs']
                    )
                    if not team_features.empty:
                        team_features_list.append(team_features)
            
            if team_features_list:
                combined_team_features = pd.concat(team_features_list, ignore_index=True)
                self.feature_engineer.save_features(combined_team_features, "team_features")
                logger.info(f"Saved team features: {len(combined_team_features)} records")
            else:
                logger.warning("No team features created")
            
            # Process player features
            player_features_list = []
            for season, season_data in all_data.items():
                if 'player_game_logs' in season_data and not season_data['player_game_logs'].empty:
                    logger.info(f"Processing player features for {season}...")
                    player_features = self.feature_engineer.create_player_features(
                        season_data['player_game_logs']
                    )
                    if not player_features.empty:
                        player_features_list.append(player_features)
            
            if player_features_list:
                combined_player_features = pd.concat(player_features_list, ignore_index=True)
                self.feature_engineer.save_features(combined_player_features, "player_features")
                logger.info(f"Saved player features: {len(combined_player_features)} records")
            else:
                logger.warning("No player features created")
            
            # Step 3: Model Training
            logger.info("=" * 60)
            logger.info("STEP 3: MODEL TRAINING")
            logger.info("=" * 60)
            
            # Train team model
            logger.info("Training team win probability model...")
            try:
                team_X, team_y = self.team_trainer.load_and_prepare_data()
                team_results = self.team_trainer.train_models(team_X, team_y)
                best_team_model_name, best_team_model = self.team_trainer.select_best_model(team_results)
                self.team_trainer.save_model(best_team_model_name, best_team_model['model'], 
                                           self.team_trainer.scaler, team_X.columns.tolist(), team_results)
                
                # Save team evaluation report
                team_report = self.team_trainer.generate_evaluation_report(team_results)
                with open("evaluation_team.md", 'w') as f:
                    f.write(team_report)
                
                logger.info("Team model training completed successfully!")
                
            except Exception as e:
                logger.error(f"Team model training failed: {e}")
                return False
            
            # Train player model
            logger.info("Training player points forecast model...")
            try:
                player_X, player_y = self.player_trainer.load_and_prepare_data()
                player_results = self.player_trainer.train_models(player_X, player_y)
                best_player_model_name, best_player_model = self.player_trainer.select_best_model(player_results)
                self.player_trainer.save_model(best_player_model_name, best_player_model['model'], 
                                             self.player_trainer.scaler, player_X.columns.tolist(), player_results)
                
                # Save player evaluation report
                player_report = self.player_trainer.generate_evaluation_report(player_results)
                with open("evaluation_player.md", 'w') as f:
                    f.write(player_report)
                
                logger.info("Player model training completed successfully!")
                
            except Exception as e:
                logger.error(f"Player model training failed: {e}")
                return False
            
            # Step 4: Model Validation
            logger.info("=" * 60)
            logger.info("STEP 4: MODEL VALIDATION")
            logger.info("=" * 60)
            
            # Test inference functions
            try:
                inference_engine = NBAInference()
                
                # Test team prediction
                team_input = {
                    'home_team': 'Golden State Warriors',
                    'away_team': 'Los Angeles Lakers',
                    'home_rest_days': 1,
                    'away_rest_days': 2,
                    'season': '2023-24'
                }
                
                team_result = inference_engine.predict_team_win(team_input)
                logger.info(f"Team prediction test: {team_result['win_probability']:.3f}")
                
                # Test player prediction
                player_input = {
                    'player_name': 'LeBron James',
                    'opponent_team': 'Golden State Warriors',
                    'is_home': True,
                    'season': '2023-24'
                }
                
                player_result = inference_engine.predict_player_points(player_input)
                logger.info(f"Player prediction test: {player_result['predicted_points']:.1f} points")
                
                logger.info("Model validation completed successfully!")
                
            except Exception as e:
                logger.error(f"Model validation failed: {e}")
                return False
            
            # Step 5: Generate Final Report
            logger.info("=" * 60)
            logger.info("STEP 5: GENERATING FINAL REPORT")
            logger.info("=" * 60)
            
            self._generate_final_report(team_results, player_results)
            
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False
    
    def _generate_final_report(self, team_results: dict, player_results: dict) -> None:
        """Generate final pipeline report"""
        report = f"""# NBA Analytics Pipeline - Final Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Pipeline Summary

The NBA Analytics Pipeline has been successfully completed with the following components:

### 1. Data Acquisition
- Fetched NBA data for multiple seasons using nba_api
- Collected team game logs, player game logs, and team statistics
- Implemented rate limiting and error handling

### 2. Feature Engineering
- Created comprehensive feature sets with rolling averages
- Added advanced metrics (ORtg, DRtg, Pace, TS%, eFG%, etc.)
- Implemented contextual features (home/away, rest days, momentum)

### 3. Model Training

#### Team Win Probability Model
- **Best Model**: {max(team_results.keys(), key=lambda x: team_results[x]['auc'])}
- **AUC**: {max(team_results[x]['auc'] for x in team_results):.4f}
- **Accuracy**: {max(team_results[x]['accuracy'] for x in team_results):.4f}

#### Player Points Forecast Model
- **Best Model**: {min(player_results.keys(), key=lambda x: player_results[x]['rmse'])}
- **RMSE**: {min(player_results[x]['rmse'] for x in player_results):.4f}
- **R²**: {max(player_results[x]['r2'] for x in player_results):.4f}

### 4. Model Validation
- Both models successfully loaded and tested
- Inference functions working correctly
- Ready for production deployment

### 5. Files Generated
- `models/team_win_final.pkl` - Trained team win model
- `models/player_pts_final.pkl` - Trained player points model
- `evaluation_team.md` - Team model evaluation report
- `evaluation_player.md` - Player model evaluation report
- `data/features/` - Processed feature datasets
- `pipeline.log` - Complete pipeline log

## Next Steps
1. Deploy models to production API
2. Set up real-time data updates
3. Monitor model performance
4. Retrain models periodically with new data

## Model Performance Summary

### Team Win Probability Models
"""
        
        for model_name, result in team_results.items():
            report += f"- **{model_name}**: AUC={result['auc']:.4f}, Accuracy={result['accuracy']:.4f}\n"
        
        report += "\n### Player Points Forecast Models\n"
        
        for model_name, result in player_results.items():
            report += f"- **{model_name}**: RMSE={result['rmse']:.4f}, R²={result['r2']:.4f}\n"
        
        report += "\n---\n*Pipeline completed successfully!*"
        
        # Save report
        with open("pipeline_report.md", 'w') as f:
            f.write(report)
        
        logger.info("Final report saved to pipeline_report.md")

def main():
    """Main pipeline function"""
    logger.info("Starting NBA Analytics Pipeline...")
    
    pipeline = NBAPipeline()
    
    # Run the complete pipeline
    success = pipeline.run_full_pipeline()
    
    if success:
        logger.info("Pipeline completed successfully!")
        logger.info("Models are ready for production deployment.")
    else:
        logger.error("Pipeline failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
