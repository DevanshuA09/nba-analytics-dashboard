"""
Team Win Probability Model Training with Multiple Algorithms
Compares Logistic Regression, RandomForest, XGBoost, and LightGBM
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report
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

class TeamWinModelTrainer:
    """Trainer for team win probability models with hyperparameter optimization"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        
        # Model configurations
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'xgboost': xgb.XGBClassifier(random_state=42, n_jobs=-1),
            'lightgbm': lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
        }
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare team data for training"""
        logger.info("Loading and preparing team data...")
        
        # Load team features
        features_file = self.data_dir / "features" / "team_features.parquet"
        if not features_file.exists():
            raise FileNotFoundError(f"Team features not found at {features_file}")
        
        df = pd.read_parquet(features_file)
        logger.info(f"Loaded {len(df)} team game records")
        
        # Prepare features and target
        X, y = self._prepare_features_and_target(df)
        
        return X, y
    
    def _prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variable"""
        logger.info("Preparing features and target...")
        
        # Create target variable (1 if team won, 0 if lost)
        df['TARGET'] = (df['WL'] == 'W').astype(int)
        
        # Select features
        feature_columns = self._get_feature_columns(df)
        X = df[feature_columns].copy()
        y = df['TARGET'].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        logger.info(f"Prepared {X.shape[1]} features for {len(X)} samples")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns"""
        # Exclude non-feature columns
        exclude_columns = [
            'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'TARGET',
            'TEAM_ID', 'TEAM_NAME', 'SEASON', 'GAME_NUMBER'
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
            if model_name == 'logistic_regression':
                params = {
                    'C': trial.suggest_float('C', 0.01, 100, log=True),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
                    'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                    'max_iter': trial.suggest_int('max_iter', 100, 2000)
                }
                model = LogisticRegression(random_state=42, **params)
                
            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
                model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
                
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
                model = xgb.XGBClassifier(random_state=42, n_jobs=-1, **params)
                
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
                model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **params)
            
            # Use time series split for validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc', n_jobs=-1)
            
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
                if model_name == 'logistic_regression':
                    model = LogisticRegression(random_state=42, **best_params)
                elif model_name == 'random_forest':
                    model = RandomForestClassifier(random_state=42, n_jobs=-1, **best_params)
                elif model_name == 'xgboost':
                    model = xgb.XGBClassifier(random_state=42, n_jobs=-1, **best_params)
                elif model_name == 'lightgbm':
                    model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **best_params)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba)
                logloss = log_loss(y_test, y_pred_proba)
                
                results[model_name] = {
                    'model': model,
                    'params': best_params,
                    'accuracy': accuracy,
                    'auc': auc,
                    'log_loss': logloss,
                    'classification_report': classification_report(y_test, y_pred)
                }
                
                logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, Log Loss: {logloss:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        return results
    
    def select_best_model(self, results: Dict[str, Any]) -> Tuple[str, Any]:
        """Select the best model based on AUC score"""
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
        best_model = results[best_model_name]
        
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Best AUC: {best_model['auc']:.4f}")
        logger.info(f"Best Accuracy: {best_model['accuracy']:.4f}")
        logger.info(f"Best Log Loss: {best_model['log_loss']:.4f}")
        
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
        
        model_path = self.models_dir / "team_win_final.pkl"
        joblib.dump(model_data, model_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate evaluation report"""
        report = "# Team Win Probability Model Evaluation Report\n\n"
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## Model Comparison\n\n"
        report += "| Model | Accuracy | AUC | Log Loss |\n"
        report += "|-------|----------|-----|----------|\n"
        
        for model_name, result in results.items():
            report += f"| {model_name} | {result['accuracy']:.4f} | {result['auc']:.4f} | {result['log_loss']:.4f} |\n"
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
        best_result = results[best_model_name]
        
        report += f"\n## Best Model: {best_model_name}\n\n"
        report += f"- **AUC**: {best_result['auc']:.4f}\n"
        report += f"- **Accuracy**: {best_result['accuracy']:.4f}\n"
        report += f"- **Log Loss**: {best_result['log_loss']:.4f}\n"
        report += f"- **Parameters**: {best_result['params']}\n\n"
        
        report += "## Classification Report\n\n"
        report += "```\n"
        report += best_result['classification_report']
        report += "```\n"
        
        return report

def main():
    """Main training function"""
    logger.info("Starting team win probability model training...")
    
    trainer = TeamWinModelTrainer()
    
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
        report_path = Path("evaluation_team.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation report saved to {report_path}")
        logger.info("Team model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
