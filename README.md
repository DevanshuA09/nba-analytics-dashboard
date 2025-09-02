
# 🏀 NBA Analytics Dashboard

A comprehensive NBA analytics dashboard with **predictive models** for team win probability and player points forecasting. Built with FastAPI backend, React frontend, and machine learning models trained on NBA data.



## 🌟 Features

### 🤖 Predictive Models
- **Team Win Probability Model** - Predicts home team win probability using rolling stats, rest days, and home advantage
- **Player Points Forecast Model** - Predicts player points using rolling performance, opponent defense, and usage rate
- **Real-time Predictions** - Fast API endpoints (<2s response time)
- **Confidence Intervals** - Uncertainty quantification for predictions

### 📊 Advanced Analytics
- **Rolling Statistics** - 10-game rolling averages for teams and players
- **Feature Engineering** - Home/away advantage, rest days, usage rate, pace
- **Model Evaluation** - Cross-validation, accuracy metrics, and performance monitoring
- **Scalable Architecture** - Modular design for easy model updates

### 🎨 Modern UI/UX
- **React Frontend** - Modern, responsive interface with Vite
- **Interactive Components** - Win probability bars, player forecast cards
- **Real-time Updates** - Live predictions for today's games
- **Mobile Responsive** - Works on desktop and mobile devices

### 🔧 Technical Features
- **FastAPI Backend** - High-performance API with automatic documentation
- **Machine Learning** - Scikit-learn models with joblib serialization
- **Caching** - Efficient data caching to minimize API calls
- **Error Handling** - Graceful fallbacks and demo data when NBA API unavailable

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Node.js 16+ (for React frontend)
- pip and npm package managers

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DevanshuA09/nba-analytics-dashboard.git
   cd nba-analytics-dashboard
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv nba_env
   source nba_env/bin/activate  # On Windows: nba_env\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install React dependencies**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

5. **Create demo models** (for testing without NBA API)
   ```bash
   python create_demo_models.py
   ```

6. **Start the backend server**
   ```bash
   python start_server.py
   ```

7. **Start the React frontend** (in a new terminal)
   ```bash
   cd frontend
   npm run dev
   ```

8. **Open your browser** and navigate to `http://localhost:3000`

## 📱 Usage

### API Endpoints

#### Team Win Probability
```bash
POST /predict/team-win
{
  "home_team": "Golden State Warriors",
  "away_team": "Los Angeles Lakers",
  "home_rest_days": 1,
  "away_rest_days": 2,
  "season": "2023-24"
}
```

#### Player Points Forecast
```bash
POST /predict/player-points
{
  "player_name": "LeBron James",
  "opponent_team": "Golden State Warriors",
  "is_home": true,
  "season": "2023-24"
}
```

### Frontend Features

#### 🏆 Team Win Probability Card
- **Matchup Predictions**: Select any two teams to get win probability
- **Today's Games**: View all games with predicted win probabilities
- **Confidence Levels**: See model confidence and prediction quality
- **Visual Indicators**: Color-coded probability bars

#### 🎯 Player Points Forecast Card
- **Player Selection**: Choose any NBA player
- **Opponent Analysis**: Select opponent team for context
- **Home/Away Toggle**: Account for venue advantage
- **Confidence Intervals**: Range of predicted points
- **Performance Levels**: Excellent/Good/Average/Below Average

### Model Training

#### Train Team Win Model
```bash
python training/team_train.py
```

#### Train Player Points Model
```bash
python training/player_train.py
```

#### Train Both Models
```bash
python train_models.py
```

## 🛠 Technical Architecture

### System Architecture
```
React Frontend (Port 3000) ←→ FastAPI Backend (Port 8000) ←→ NBA API
                                      ↓
                              Machine Learning Models
                                      ↓
                              Feature Engineering
                                      ↓
                              Data Processing & Caching
```

### Key Technologies
- **Backend**: FastAPI + Uvicorn for high-performance API
- **Frontend**: React + Vite for modern web interface
- **ML Models**: Scikit-learn (Logistic Regression, Ridge Regression)
- **Data Source**: NBA API with demo data fallback
- **Processing**: Pandas + NumPy for data manipulation
- **Serialization**: Joblib for model persistence

### Model Architecture

#### Team Win Probability Model
```python
# Features (15 total):
# - Rolling team stats (last 10 games)
# - Home/away advantage
# - Rest days advantage
# - Recent form (last 5 games)
# - Head-to-head record

# Model: Logistic Regression
# Evaluation: Cross-validation accuracy
# Output: Win probability (0-1)
```

#### Player Points Forecast Model
```python
# Features (12 total):
# - Rolling player stats (last 10 games)
# - Opponent defensive rating
# - Home/away advantage
# - Usage rate and pace
# - Recent performance trend

# Model: Ridge Regression
# Evaluation: RMSE, MAE, R²
# Output: Predicted points with confidence interval
```

## 📊 Data Sources

- **NBA API**: Official NBA statistics and player data
- **Team Information**: NBA.com official team colors and logos
- **Historical Data**: Multiple seasons of team and player statistics

## 🔧 Configuration

### Environment Variables
Create a `.env` file for configuration:
```
DEBUG=True
PORT=8050
NBA_API_TIMEOUT=30
CACHE_TIMEOUT=300
```

### Customization
- **Team Colors**: Update `NBA_TEAM_COLORS` dictionary
- **Metrics**: Modify calculation functions for custom analytics
- **Styling**: Adjust Bootstrap themes and custom CSS

## 🚀 Deployment

### Heroku Deployment
1. **Create Heroku app**
   ```bash
   heroku create your-nba-dashboard
   ```

2. **Deploy**
   ```bash
   git push heroku main
   ```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8050
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "app:server"]
```

## 📈 Performance Features

- **Efficient Data Caching**: Minimizes API calls
- **Lazy Loading**: Components load as needed
- **Optimized Calculations**: Vectorized operations with NumPy
- **Responsive Design**: Fast rendering across devices

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NBA API Community** for excellent documentation and support
- **Plotly Team** for powerful visualization tools
- **Dash Community** for framework development
- **NBA.com** for official team branding assets

## 📞 Contact

**Devansu Agarwal** - devanshuagarwal1714@gmail.com

**LinkedIn**: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)

**Project Link**: [https://github.com/yourusername/nba-analytics-dashboard](https://github.com/yourusername/nba-analytics-dashboard)

---

⭐ **Star this repository if you found it helpful!** ⭐

*Built with ❤️ for basketball analytics enthusiasts*



##  Features

### Advanced Analytics
- **Player Efficiency Rating (PER)** calculations
- **Offensive/Defensive Ratings** per 100 possessions  
- **True Shooting Percentage** and advanced shooting metrics
- **Pace and Net Rating** analysis
- **Team comparison radar charts**

### Professional UI/UX
- **Team-branded interface** with official NBA colors and logos
- **Responsive design** that works on desktop and mobile
- **Interactive visualizations** with hover details and filtering
- **Real-time data updates** from NBA API

### 📈 Visualizations
- Enhanced radar charts for team performance profiles
- Player efficiency bubble charts (size = minutes played)
- Advanced metrics bar charts and comparisons
- Interactive data tables with conditional formatting

##  Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DevanshuA09/nba-analytics-dashboard.git
   cd nba-analytics-dashboard
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv nba_env
   source nba_env/bin/activate  # On Windows: nba_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser** and navigate to `http://127.0.0.1:8050`

## 📱 Usage

### Dashboard Controls
- **Season Selection**: Choose from 2015-16 to 2022-23 seasons
- **Primary Team**: Select your main team for analysis
- **Comparison Team**: Choose a second team for head-to-head comparisons
- **Refresh Data**: Update with latest NBA statistics

### Key Sections

####  Team Performance Profile
Enhanced radar chart showing team strengths across:
- Offensive efficiency
- Defensive performance  
- Rebounding ability
- Playmaking skills
- Shooting accuracy
- Overall efficiency

#### ⚡ Player Efficiency Ratings
Interactive bubble chart displaying:
- Player Efficiency Rating (PER) calculations
- Minutes played (bubble size)
- Top performers highlighted

#### Team Comparisons
Side-by-side radar comparisons between selected teams across key metrics.

#### 📋 Advanced Roster Analysis
Comprehensive player statistics table featuring:
- Traditional stats (PPG, RPG, APG)
- Advanced metrics (PER, efficiency ratings)
- Sortable and filterable interface

## 🛠 Technical Architecture

### Data Pipeline
```
NBA API → Data Processing → Caching → Visualization → Interactive Dashboard
```

### Key Technologies
- **Frontend**: Dash + Plotly for interactive web components
- **Data Source**: NBA API for real-time statistics
- **Styling**: Bootstrap + custom team branding
- **Charts**: Plotly.js for professional visualizations
- **Processing**: Pandas + NumPy for data manipulation

### Advanced Metrics Calculations

#### Player Efficiency Rating (PER)
```python
def calculate_player_efficiency_rating(player_data):
    # Simplified PER calculation accounting for:
    # - Points, rebounds, assists, steals, blocks
    # - Turnovers, missed shots (negative impact)
    # - Normalized per minute played
```

#### Team Offensive Rating
```python
# Points scored per 100 possessions
possessions = FGA - OREB + TOV + 0.4 * FTA  
offensive_rating = (PTS / possessions) * 100
```

## 📊 Data Sources

- **NBA API**: Official NBA statistics and player data
- **Team Information**: NBA.com official team colors and logos
- **Historical Data**: Multiple seasons of team and player statistics

## 🔧 Configuration

### Environment Variables
Create a `.env` file for configuration:
```
DEBUG=True
PORT=8050
NBA_API_TIMEOUT=30
CACHE_TIMEOUT=300
```

### Customization
- **Team Colors**: Update `NBA_TEAM_COLORS` dictionary
- **Metrics**: Modify calculation functions for custom analytics
- **Styling**: Adjust Bootstrap themes and custom CSS

## 🚀 Deployment

### Heroku Deployment
1. **Create Heroku app**
   ```bash
   heroku create your-nba-dashboard
   ```

2. **Deploy**
   ```bash
   git push heroku main
   ```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8050
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "app:server"]
```

## 📈 Performance Features

- **Efficient Data Caching**: Minimizes API calls
- **Lazy Loading**: Components load as needed
- **Optimized Calculations**: Vectorized operations with NumPy
- **Responsive Design**: Fast rendering across devices

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **NBA API Community** for excellent documentation and support
- **Plotly Team** for powerful visualization tools
- **Dash Community** for framework development
- **NBA.com** for official team branding assets

## 📞 Contact

**Devansu Agarwal** - devanshuagarwal1714@gmail.com

**LinkedIn**: [[linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)](https://www.linkedin.com/in/devansua9/)

**Project Link**: [https://github.com/yourusername/nba-analytics-dashboard](https://github.com/DevanshuA09/nba-analytics-dashboard)

---

⭐ **Star this repository if you found it helpful!** ⭐

*Built with ❤️ for basketball analytics enthusiasts*
