"""
Player Points Forecast Model Training with Multiple Algorithms
Compares Ridge Regression, RandomForest, XGBoost, and Neural Networks
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import joblib
import optuna
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlayerPointsModelTrainer:
    """Trainer for player points forecast models with hyperparameter optimization"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        
        # Model configurations
        self.models = {
            'ridge_regression': Ridge(random_state=42),
            'random_forest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'xgboost': xgb.XGBRegressor(random_state=42, n_jobs=-1),
            'lightgbm': lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
            'neural_network': MLPRegressor(random_state=42, max_iter=500)
        }
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare player data for training"""
        logger.info("Loading and preparing player data...")
        
        # Load player features
        features_file = self.data_dir / "features" / "player_features.parquet"
        if not features_file.exists():
            raise FileNotFoundError(f"Player features not found at {features_file}")
        
        df = pd.read_parquet(features_file)
        logger.info(f"Loaded {len(df)} player game records")
        
        # Prepare features and target
        X, y = self._prepare_features_and_target(df)
        
        return X, y
    
    def _prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variable"""
        logger.info("Preparing features and target...")
        
        # Use PTS as target variable
        df['TARGET'] = df['PTS']
        
        # Select features
        feature_columns = self._get_feature_columns(df)
        X = df[feature_columns].copy()
        y = df['TARGET'].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Remove rows with missing target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Prepared {X.shape[1]} features for {len(X)} samples")
        logger.info(f"Target statistics: mean={y.mean():.2f}, std={y.std():.2f}")
        
        return X, y
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns"""
        # Exclude non-feature columns
        exclude_columns = [
            'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'TARGET',
            'PLAYER_ID', 'PLAYER_NAME', 'SEASON', 'GAME_NUMBER', 'PTS'
        ]
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Filter to numeric columns only
        numeric_columns = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        
        return numeric_columns
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                                model_name: str) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        logger.info(f"Optimizing hyperparameters for {model_name}...")
        
        def objective(trial):
            if model_name == 'ridge_regression':
                params = {
                    'alpha': trial.suggest_float('alpha', 0.01, 100, log=True),
                    'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                    'normalize': trial.suggest_categorical('normalize', [True, False])
                }
                model = Ridge(random_state=42, **params)
                
            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
                model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
                
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
                }
                model = xgb.XGBRegressor(random_state=42, n_jobs=-1, **params)
                
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
                }
                model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, **params)
                
            elif model_name == 'neural_network':
                params = {
                    'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', 
                        [(50,), (100,), (50, 50), (100, 50), (100, 100)]),
                    'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                    'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True),
                    'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive'])
                }
                model = MLPRegressor(random_state=42, max_iter=500, **params)
            
            # Use time series split for validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
            
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        logger.info(f"Best parameters for {model_name}: {study.best_params}")
        logger.info(f"Best score: {study.best_value:.4f}")
        
        return study.best_params
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train all models and compare performance"""
        logger.info("Training and comparing models...")
        
        # Split data with time series consideration
        # Use last 20% for testing to simulate forward prediction
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for model_name, base_model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Optimize hyperparameters
                best_params = self.optimize_hyperparameters(X_train, y_train, model_name)
                
                # Train final model with best parameters
                if model_name == 'ridge_regression':
                    model = Ridge(random_state=42, **best_params)
                elif model_name == 'random_forest':
                    model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
                elif model_name == 'xgboost':
                    model = xgb.XGBRegressor(random_state=42, n_jobs=-1, **best_params)
                elif model_name == 'lightgbm':
                    model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, **best_params)
                elif model_name == 'neural_network':
                    model = MLPRegressor(random_state=42, max_iter=500, **best_params)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                results[model_name] = {
                    'model': model,
                    'params': best_params,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': rmse
                }
                
                logger.info(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        return results
    
    def select_best_model(self, results: Dict[str, Any]) -> Tuple[str, Any]:
        """Select the best model based on RMSE score"""
        best_model_name = min(results.keys(), key=lambda x: results[x]['rmse'])
        best_model = results[best_model_name]
        
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Best RMSE: {best_model['rmse']:.4f}")
        logger.info(f"Best MAE: {best_model['mae']:.4f}")
        logger.info(f"Best R²: {best_model['r2']:.4f}")
        
        return best_model_name, best_model
    
    def save_model(self, model_name: str, model: Any, scaler: StandardScaler, 
                   feature_columns: List[str], results: Dict[str, Any]) -> None:
        """Save the best model and metadata"""
        logger.info(f"Saving {model_name} model...")
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'model_name': model_name,
            'training_date': datetime.now().isoformat(),
            'results': results
        }
        
        model_path = self.models_dir / "player_pts_final.pkl"
        joblib.dump(model_data, model_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate evaluation report"""
        report = "# Player Points Forecast Model Evaluation Report\n\n"
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## Model Comparison\n\n"
        report += "| Model | RMSE | MAE | R² |\n"
        report += "|-------|------|-----|----|\n"
        
        for model_name, result in results.items():
            report += f"| {model_name} | {result['rmse']:.4f} | {result['mae']:.4f} | {result['r2']:.4f} |\n"
        
        # Find best model
        best_model_name = min(results.keys(), key=lambda x: results[x]['rmse'])
        best_result = results[best_model_name]
        
        report += f"\n## Best Model: {best_model_name}\n\n"
        report += f"- **RMSE**: {best_result['rmse']:.4f}\n"
        report += f"- **MAE**: {best_result['mae']:.4f}\n"
        report += f"- **R²**: {best_result['r2']:.4f}\n"
        report += f"- **Parameters**: {best_result['params']}\n\n"
        
        return report

def main():
    """Main training function"""
    logger.info("Starting player points forecast model training...")
    
    trainer = PlayerPointsModelTrainer()
    
    try:
        # Load and prepare data
        X, y = trainer.load_and_prepare_data()
        
        # Train models
        results = trainer.train_models(X, y)
        
        # Select best model
        best_model_name, best_model = trainer.select_best_model(results)
        
        # Save best model
        feature_columns = X.columns.tolist()
        trainer.save_model(best_model_name, best_model['model'], 
                          trainer.scaler, feature_columns, results)
        
        # Generate evaluation report
        report = trainer.generate_evaluation_report(results)
        
        # Save report
        report_path = Path("evaluation_player.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation report saved to {report_path}")
        logger.info("Player model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
