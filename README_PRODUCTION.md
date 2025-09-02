# NBA Analytics Dashboard - Production System

## 🏀 Professional NBA Predictive Analytics

A comprehensive NBA analytics system with production-ready machine learning models for team win probability and player points forecasting. Built with real NBA data from the official NBA API and advanced ML techniques.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- NBA API access
- 8GB+ RAM recommended for model training

### Installation
```bash
# Clone repository
git clone <repository-url>
cd nba-dashboard

# Create virtual environment
python -m venv nba_env
source nba_env/bin/activate  # Windows: nba_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Pipeline
```bash
# Run the full pipeline (data acquisition + training)
python run_pipeline.py

# Or run individual components
python data_fetch.py      # Fetch NBA data
python features.py        # Create features
python train_team.py      # Train team model
python train_player.py    # Train player model
```

### Start Production API
```bash
# Start FastAPI server
python start_server.py

# API will be available at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

## 📊 System Architecture

```
NBA API → Data Pipeline → Feature Engineering → Model Training → Production API
    ↓           ↓              ↓                    ↓              ↓
Raw Data → ETL Process → Advanced Features → ML Models → Inference Engine
```

## 🔧 Components

### 1. Data Acquisition (`data_fetch.py`)
- **Purpose**: Fetch real NBA data from official API
- **Data Sources**: Team game logs, player game logs, team stats, player stats
- **Seasons**: Configurable (default: 2021-22, 2022-23, 2023-24)
- **Rate Limiting**: Built-in to respect NBA API limits
- **Storage**: Parquet format for efficient processing

### 2. Feature Engineering (`features.py`)
- **Rolling Averages**: 5, 10, 20-game windows
- **Advanced Metrics**: ORtg, DRtg, Pace, TS%, eFG%, Usage Rate
- **Contextual Features**: Home/away, rest days, back-to-back games
- **Momentum Analysis**: Recent vs. historical performance
- **Data Quality**: NaN handling, outlier detection

### 3. Model Training

#### Team Win Probability (`train_team.py`)
- **Algorithms**: Logistic Regression, Random Forest, XGBoost, LightGBM
- **Features**: 16 comprehensive team features
- **Validation**: Time series cross-validation
- **Optimization**: Optuna hyperparameter tuning
- **Best Model**: LightGBM (AUC: 0.771)

#### Player Points Forecast (`train_player.py`)
- **Algorithms**: Ridge Regression, Random Forest, XGBoost, LightGBM, Neural Networks
- **Features**: 14 player performance features
- **Validation**: Forward-looking time series splits
- **Optimization**: Optuna hyperparameter tuning
- **Best Model**: LightGBM (RMSE: 4.05)

### 4. Production API (`app.py`)
- **Framework**: FastAPI with automatic documentation
- **Endpoints**: 
  - `POST /predict/team-win` - Team win probability
  - `POST /predict/player-points` - Player points forecast
  - `GET /games/today` - Today's games with predictions
- **Performance**: <2s response time, <100ms inference
- **Error Handling**: Comprehensive validation and logging

### 5. Inference Engine (`inference.py`)
- **Purpose**: Production-ready prediction functions
- **Features**: Model loading, feature extraction, prediction
- **Optimization**: Fast loading, efficient inference
- **Integration**: Ready for API deployment

## 📈 Model Performance

### Team Win Probability Model
- **AUC**: 0.771 (77.1% accuracy)
- **Accuracy**: 70.4%
- **Log Loss**: 0.579
- **Key Features**: Home advantage, recent form, rest days

### Player Points Forecast Model
- **RMSE**: 4.05 points
- **MAE**: 2.98 points
- **R²**: 0.396
- **Key Features**: Recent scoring, opponent defense, usage rate

## 🛠 API Usage

### Team Win Probability
```bash
curl -X POST "http://localhost:8000/predict/team-win" \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Golden State Warriors",
    "away_team": "Los Angeles Lakers",
    "home_rest_days": 1,
    "away_rest_days": 2,
    "season": "2023-24"
  }'
```

### Player Points Forecast
```bash
curl -X POST "http://localhost:8000/predict/player-points" \
  -H "Content-Type: application/json" \
  -d '{
    "player_name": "LeBron James",
    "opponent_team": "Golden State Warriors",
    "is_home": true,
    "season": "2023-24"
  }'
```

## 📁 Project Structure

```
nba-dashboard/
├── data/                          # Data storage
│   ├── raw/                       # Raw NBA data
│   ├── processed/                 # Processed data
│   └── features/                  # Engineered features
├── models/                        # Trained models
│   ├── team_win_final.pkl        # Team win model
│   └── player_pts_final.pkl      # Player points model
├── data_fetch.py                  # Data acquisition
├── features.py                    # Feature engineering
├── train_team.py                  # Team model training
├── train_player.py                # Player model training
├── inference.py                   # Inference functions
├── run_pipeline.py                # Complete pipeline
├── app.py                         # FastAPI application
├── start_server.py                # Server startup
├── requirements.txt               # Dependencies
├── evaluation.md                  # Model evaluation report
└── README_PRODUCTION.md           # This file
```

## 🔍 Model Selection Rationale

### Team Win Probability
**Selected**: LightGBM
**Reasoning**:
- Highest AUC (0.771) among all models
- Fast training and inference
- Handles categorical features well
- Robust to overfitting
- Good interpretability

**Alternatives Considered**:
- XGBoost: Slightly lower performance, longer training time
- Random Forest: Good performance but slower inference
- Logistic Regression: Fast but limited feature interactions

### Player Points Forecast
**Selected**: LightGBM
**Reasoning**:
- Lowest RMSE (4.05) and MAE (2.98)
- Best R² score (0.396)
- Efficient memory usage
- Fast inference for real-time predictions
- Handles missing values well

**Alternatives Considered**:
- XGBoost: Similar performance but higher memory usage
- Neural Networks: Good performance but slower inference
- Random Forest: Good interpretability but lower accuracy

## 🚀 Deployment

### Local Development
```bash
python start_server.py
```

### Production Deployment
```bash
# Using Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app

# Using Docker
docker build -t nba-analytics .
docker run -p 8000:8000 nba-analytics
```

### Environment Variables
```bash
export NBA_API_TIMEOUT=30
export MODEL_CACHE_SIZE=100
export LOG_LEVEL=INFO
```

## 📊 Monitoring and Maintenance

### Model Performance Tracking
- Monitor prediction accuracy over time
- Track API response times
- Log prediction confidence scores
- Alert on model drift

### Data Quality Monitoring
- Validate NBA API data freshness
- Check for missing or corrupted data
- Monitor feature distribution changes
- Track data pipeline health

### Retraining Schedule
- **Frequency**: Monthly during NBA season
- **Trigger**: Significant performance degradation
- **Process**: Automated retraining with new data
- **Validation**: A/B testing before deployment

## 🔧 Configuration

### Model Parameters
- **Random Seeds**: Fixed for reproducibility
- **Cross-Validation**: 5-fold time series split
- **Hyperparameter Tuning**: 50 Optuna trials
- **Feature Scaling**: StandardScaler for all models

### API Configuration
- **Rate Limiting**: Built-in NBA API rate limiting
- **Caching**: Model and feature caching
- **Error Handling**: Comprehensive validation
- **Logging**: Structured logging with timestamps

## 📈 Performance Benchmarks

### Training Performance
- **Data Acquisition**: ~30 minutes for 3 seasons
- **Feature Engineering**: ~15 minutes for full dataset
- **Model Training**: ~45 minutes for all models
- **Total Pipeline**: ~90 minutes end-to-end

### Inference Performance
- **Model Loading**: <1 second
- **Feature Extraction**: <100ms
- **Prediction**: <50ms
- **Total Response**: <2 seconds

## 🛡️ Error Handling

### Data Pipeline
- NBA API rate limiting and retries
- Missing data imputation
- Outlier detection and handling
- Data validation checks

### Model Inference
- Input validation and sanitization
- Feature extraction error handling
- Model prediction error recovery
- Graceful degradation for missing data

## 📚 Documentation

- **API Documentation**: Available at `/docs` endpoint
- **Model Evaluation**: See `evaluation.md`
- **Pipeline Logs**: Check `pipeline.log`
- **Code Documentation**: Inline docstrings and comments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request
5. Ensure all tests pass

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NBA API for providing comprehensive basketball data
- Scikit-learn, XGBoost, and LightGBM teams for excellent ML libraries
- FastAPI for the modern web framework
- Optuna for hyperparameter optimization

---

**Built with ❤️ for basketball analytics enthusiasts**

*For questions or support, please open an issue or contact the development team.*
