#!/usr/bin/env python3
"""
Create demo models for testing when NBA API is not available
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def create_demo_team_model():
    """Create a demo team win probability model"""
    print("Creating demo team win model...")
    
    # Create a simple logistic regression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    scaler = StandardScaler()
    
    # Generate some dummy training data
    np.random.seed(42)
    X_dummy = np.random.normal(0, 1, (100, 16))  # 16 features
    y_dummy = np.random.randint(0, 2, 100)  # Binary labels
    
    # Fit the model
    X_scaled = scaler.fit_transform(X_dummy)
    model.fit(X_scaled, y_dummy)
    
    # Save the model
    model_data = {
        'model': model,
        'scaler': scaler,
        'training_date': datetime.now().isoformat(),
        'season': '2023-24'
    }
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_data, 'models/team_win.pkl')
    print("Demo team model saved to models/team_win.pkl")

def create_demo_player_model():
    """Create a demo player points prediction model"""
    print("Creating demo player points model...")
    
    # Create a simple ridge regression model
    model = Ridge(alpha=1.0, random_state=42)
    scaler = StandardScaler()
    
    # Generate some dummy training data
    np.random.seed(42)
    X_dummy = np.random.normal(0, 1, (200, 14))  # 14 features
    y_dummy = np.random.normal(15, 8, 200)  # Points (mean=15, std=8)
    y_dummy = np.maximum(0, y_dummy)  # Ensure non-negative
    
    # Fit the model
    X_scaled = scaler.fit_transform(X_dummy)
    model.fit(X_scaled, y_dummy)
    
    # Save the model
    model_data = {
        'model': model,
        'scaler': scaler,
        'training_date': datetime.now().isoformat(),
        'season': '2023-24'
    }
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_data, 'models/player_points.pkl')
    print("Demo player model saved to models/player_points.pkl")

def main():
    """Create demo models"""
    print("Creating demo models for testing...")
    
    create_demo_team_model()
    create_demo_player_model()
    
    print("\nDemo models created successfully!")
    print("You can now start the server with: python start_server.py")

if __name__ == "__main__":
    main()
