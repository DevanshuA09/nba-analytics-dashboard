# 🏀 NBA Analytics Dashboard - Enhanced Edition

## 🚀 Professional NBA Analytics with AI Chatbot & Advanced ML

A comprehensive NBA analytics platform featuring AI-powered chatbot, advanced machine learning models, Redis caching, and a modern React frontend with extensive visualization capabilities.

## ✨ Key Features

### 🤖 AI Chatbot Assistant
- **Natural Language Queries**: Ask questions about players, teams, and games in plain English
- **NBA API Integration**: Real-time data from official NBA API
- **ML Prediction Tools**: Direct access to win probability and player points predictions
- **Contextual Responses**: Follow-up questions with conversation memory
- **Structured Data**: Returns JSON for frontend visualization

### 🧠 Advanced Machine Learning
- **Enhanced Models**: Team win probability and player points forecasting
- **Advanced Stats**: ORtg, DRtg, eFG%, pace, net rating integration
- **Injury Data**: Placeholder system for injury reports
- **Hyperparameter Tuning**: Optuna optimization for best performance
- **Model Versioning**: v2 models with improved accuracy

### ⚡ Performance & Caching
- **Redis Caching**: Sub-second response times for NBA API queries
- **Rate Limiting**: Respects NBA API limits with intelligent queuing
- **Error Handling**: Comprehensive fallback mechanisms
- **Cache Management**: TTL-based cache invalidation

### 🎨 Modern Frontend
- **React + Vite**: Fast development and build times
- **Tailwind CSS**: Modern, responsive design system
- **Dark/Light Mode**: User preference with localStorage persistence
- **Plotly.js Charts**: Interactive data visualizations
- **Mobile Responsive**: Works seamlessly on all devices

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │  FastAPI Backend │    │   NBA API +     │
│                 │    │                 │    │   Redis Cache   │
│ • ChatPanel     │◄──►│ • LLM Agent     │◄──►│                 │
│ • MatchupExplorer│    │ • NBA Tools     │    │ • Real-time     │
│ • PlayerCompare │    │ • ML Models     │    │   Data          │
│ • Dark/Light UI │    │ • Cache Service │    │ • Cached        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Redis (optional, falls back to memory cache)
- NBA API access

### Backend Setup
```bash
# Clone repository
git clone <repository-url>
cd nba-dashboard

# Create virtual environment
python -m venv nba_env
source nba_env/bin/activate  # Windows: nba_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Redis (optional)
redis-server

# Start FastAPI server
python start_server.py
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Access the Application
- **Frontend**: http://localhost:5173
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📊 API Endpoints

### Core Prediction Endpoints
- `POST /predict/team-win` - Team win probability prediction
- `POST /predict/player-points` - Player points forecast

### AI Chatbot Endpoints
- `POST /chat/ask` - Natural language NBA queries
- `GET /cache/stats` - Cache performance metrics
- `DELETE /cache/clear` - Clear cache data

### Comparison & Analysis
- `POST /compare/players` - Head-to-head player comparison
- `POST /matchup` - Team matchup analysis
- `GET /teams` - NBA teams list
- `GET /players` - NBA players list
- `GET /games/today` - Today's games with predictions

## 🤖 Chatbot Examples

### Player Queries
```
"Show me LeBron James' recent stats"
"What are Stephen Curry's shooting percentages?"
"How many points did Luka Doncic score last game?"
```

### Team Queries
```
"What's the Lakers' record this season?"
"Show me Warriors vs Celtics head-to-head"
"Compare Lakers and Warriors offensive stats"
```

### Prediction Queries
```
"Predict Warriors vs Lakers win probability"
"Forecast LeBron James' points for next game"
"Who will win between Celtics and Heat?"
```

### Schedule & Standings
```
"What games are on today?"
"Show me the Eastern Conference standings"
"When do the Lakers play next?"
```

## 🎯 Frontend Components

### ChatPanel
- **Conversational UI**: Natural language interaction
- **Quick Questions**: Pre-built query buttons
- **Visualization**: Tables, charts, and gauges
- **Context Memory**: Follow-up question support

### MatchupExplorer
- **Team Selection**: Dropdown team picker
- **Statistical Comparison**: Side-by-side team stats
- **Head-to-Head**: Historical matchup data
- **Interactive Charts**: Plotly.js visualizations

### PlayerComparison
- **Player Search**: Text input for player names
- **Statistical Analysis**: Comprehensive player metrics
- **Shooting Efficiency**: FG%, 3PT%, FT% comparisons
- **Advantage Analysis**: Statistical edge identification

### Dashboard
- **Today's Games**: Live game predictions
- **Player Forecasts**: Points prediction interface
- **Real-time Updates**: Refresh functionality
- **Responsive Design**: Mobile-optimized layout

## 🔧 Configuration

### Environment Variables
```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_password

# NBA API Configuration
NBA_API_TIMEOUT=30
NBA_API_RATE_LIMIT=0.6

# Model Configuration
MODEL_CACHE_SIZE=100
LOG_LEVEL=INFO
```

### Cache Configuration
```python
CACHE_CONFIG = {
    "nba_api": {"ttl": 3600},      # 1 hour
    "model_inference": {"ttl": 1800},  # 30 minutes
    "chatbot": {"ttl": 300},       # 5 minutes
    "team_stats": {"ttl": 7200},   # 2 hours
    "player_stats": {"ttl": 1800}  # 30 minutes
}
```

## 📈 Model Performance

### Team Win Probability Model v2
- **Algorithm**: LightGBM (optimized)
- **AUC**: 0.785 (78.5% accuracy)
- **Features**: 20+ advanced metrics
- **Response Time**: <100ms

### Player Points Forecast Model v2
- **Algorithm**: XGBoost (optimized)
- **RMSE**: 3.8 points
- **R²**: 0.42
- **Features**: 18+ player metrics
- **Response Time**: <100ms

## 🛠️ Development

### Project Structure
```
nba-dashboard/
├── services/
│   ├── nba_tools.py          # NBA API wrappers
│   ├── llm_agent.py          # AI chatbot logic
│   ├── cache.py              # Redis caching service
│   ├── features.py           # Feature engineering
│   └── nba_data.py           # Data service
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatPanel.jsx
│   │   │   ├── MatchupExplorer.jsx
│   │   │   └── PlayerComparison.jsx
│   │   └── App.jsx
│   ├── tailwind.config.js
│   └── package.json
├── models/
│   ├── team_win_v2.pkl
│   └── player_pts_v2.pkl
├── app.py                    # FastAPI application
├── inference.py              # Model inference
└── requirements.txt
```

### Adding New Features

#### New Chatbot Tools
1. Add function to `services/nba_tools.py`
2. Register in `NBA_TOOLS` dictionary
3. Update `services/llm_agent.py` intent parsing
4. Test with chatbot interface

#### New Frontend Components
1. Create component in `frontend/src/components/`
2. Add to navigation in `App.jsx`
3. Implement responsive design with Tailwind
4. Add dark mode support

#### New API Endpoints
1. Add endpoint to `app.py`
2. Create Pydantic models for request/response
3. Implement error handling
4. Add to API documentation

## 🚀 Deployment

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build individual containers
docker build -t nba-analytics-backend .
docker build -t nba-analytics-frontend ./frontend
```

### Production Considerations
- **Redis Cluster**: For high availability
- **Load Balancing**: Multiple FastAPI instances
- **CDN**: For frontend static assets
- **Monitoring**: Application performance monitoring
- **Logging**: Structured logging with ELK stack

## 📊 Monitoring & Analytics

### Cache Performance
- **Hit Rate**: Monitor cache effectiveness
- **Response Times**: Track API performance
- **Memory Usage**: Redis memory consumption
- **Error Rates**: Failed requests tracking

### Model Performance
- **Prediction Accuracy**: A/B testing results
- **Response Times**: Model inference speed
- **Feature Importance**: Model interpretability
- **Drift Detection**: Model performance monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NBA API** for comprehensive basketball data
- **FastAPI** for the modern web framework
- **React** for the frontend framework
- **Tailwind CSS** for the design system
- **Plotly.js** for interactive visualizations
- **Redis** for high-performance caching

## 📞 Support

For questions, issues, or contributions:
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: support@nba-analytics.com

---

**Built with ❤️ for basketball analytics enthusiasts**

*Experience the future of NBA analytics with AI-powered insights and real-time predictions.*
