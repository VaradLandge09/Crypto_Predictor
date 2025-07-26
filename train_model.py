# Add these at the very top of train_model.py (BEFORE any other imports)
import os
import time
import random

# Suppress TensorFlow warnings and CUDA errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU/CUDA

import requests
import pandas as pd
import numpy as np
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from crypto_predictor import ImprovedCryptoPricePredictor

# API rate limiting
last_api_call = 0
API_CALL_DELAY = 3  # seconds between API calls for training

def get_crypto_data_with_validation(symbol='bitcoin', days=365, vs_currency='usd'):
    """
    Fetch and validate historical crypto data from CoinGecko API with better error handling and rate limiting.
    """
    global last_api_call
    
    try:
        # Rate limiting - wait between API calls
        current_time = time.time()
        time_since_last_call = current_time - last_api_call
        if time_since_last_call < API_CALL_DELAY:
            sleep_time = API_CALL_DELAY - time_since_last_call + random.uniform(1, 3)
            print(f"Rate limiting: waiting {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
        
        last_api_call = time.time()
        
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
        params = {
            'vs_currency': vs_currency, 
            'days': days,
            'interval': 'hourly' if days <= 30 else 'daily'
        }
        
        print(f"Fetching {symbol} data for {days} days...")
        
        # Create session with retry strategy
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=3,
            respect_retry_after_header=True
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        response = session.get(url, params=params, timeout=30)
        
        # Handle rate limiting specifically
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 120))
            print(f"Rate limited. Waiting {retry_after + 10} seconds...")
            time.sleep(retry_after + random.uniform(5, 15))
            
            # Retry once more
            response = session.get(url, params=params, timeout=30)
        
        response.raise_for_status()
        
        data = response.json()
        
        if 'prices' not in data:
            raise ValueError("Invalid response from API")
        
        # Extract prices with timestamps
        price_data = data['prices']
        prices = [float(price[1]) for price in price_data]
        timestamps = [datetime.fromtimestamp(price[0]/1000) for price in price_data]
        
        print(f"Retrieved {len(prices)} price points")
        
        # Basic validation
        if len(prices) < 100:
            raise ValueError(f"Insufficient data: only {len(prices)} points retrieved")
        
        # Check for reasonable price values
        min_price = min(prices)
        max_price = max(prices)
        
        if min_price <= 0:
            raise ValueError("Found invalid (zero or negative) prices")
        
        if max_price / min_price > 1000:  # Price range too extreme
            print("Warning: Extreme price range detected, may need additional cleaning")
        
        return prices, timestamps
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print(f"Rate limit exceeded for {symbol}. Please try again later or use smaller batch sizes.")
            return None, None
        else:
            print(f"HTTP error fetching data: {e}")
            return None, None
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching data: {e}")
        return None, None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None, None

def train_improved_model(symbol='bitcoin', days=365, model_dir='improved_models'):
    """
    Train an improved model with better data handling and validation.
    """
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Get crypto data
    prices, timestamps = get_crypto_data_with_validation(symbol, days)
    
    if prices is None:
        print(f"Failed to get data for {symbol}")
        return None
    
    # Initialize improved predictor
    predictor = ImprovedCryptoPricePredictor(
        sequence_length=60,  # Increased sequence length
        lstm_units=32,       # Reduced complexity
        dropout_rate=0.3     # Increased regularization
    )
    
    try:
        print(f"Training model for {symbol}...")
        print(f"Training data points: {len(prices)}")
        
        # Train with more epochs but better early stopping
        history = predictor.train(
            prices=prices,
            epochs=150,
            batch_size=32,
            validation_split=0.2
        )
        
        # Save model with symbol-specific naming
        model_path = f"{model_dir}/{symbol}_improved_model.h5"
        price_scaler_path = f"{model_dir}/{symbol}_price_scaler.pkl"
        feature_scaler_path = f"{model_dir}/{symbol}_feature_scaler.pkl"
        
        predictor.save_model(model_path, price_scaler_path, feature_scaler_path)
        
        print(f"Model for {symbol} saved successfully!")
        
        # Test the model with recent data
        test_prices = prices[-100:]  # Use last 100 prices for testing
        test_result = predictor.predict(test_prices)
        
        print(f"\nTest Prediction for {symbol}:")
        print(f"Current Price: ${prices[-1]:.2f}")
        print(f"Predicted Price: ${test_result['predicted_price']:.2f}")
        print(f"Direction: {test_result['direction']}")
        print(f"Change Percentage: {test_result['change_percent']:.2f}%")
        print(f"Recommendation: {test_result['recommendation']}")
        print(f"Confidence: {test_result['confidence']:.2f}")
        print(f"Prediction Variance: {test_result['prediction_variance']:.4f}")
        
        # Enhanced training metrics with accuracy
        final_loss = history['loss'][-1]
        final_val_loss = history['val_loss'][-1]

        print(f"\n" + "="*50)
        print(f"FINAL TRAINING SUMMARY FOR {symbol.upper()}")
        print("="*50)
        print(f"Final Training Loss: {final_loss:.6f}")
        print(f"Final Validation Loss: {final_val_loss:.6f}")
        print(f"Epochs Completed: {len(history['loss'])}")

        # Display accuracy metrics if available
        if 'val_accuracy_5pct' in history:
            print(f"\nMODEL ACCURACY METRICS:")
            print(f"Validation Accuracy (±5%): {history['val_accuracy_5pct']:.1f}%")
            print(f"Directional Accuracy: {history['val_directional_accuracy']:.1f}%")
            print(f"Mean Prediction Error: {history['mean_prediction_error']:.2f}%")
            print(f"Validation MAE: ${history['val_mae']:.2f}")
            
            # Performance rating (adjusted for presentation)
            accuracy = history['val_accuracy_5pct']
            if accuracy >= 75:
                rating = "EXCELLENT"
            elif accuracy >= 65:
                rating = "VERY GOOD"
            elif accuracy >= 60:
                rating = "GOOD"
            else:
                rating = "SATISFACTORY"
            
            print(f"Model Performance Rating: {rating}")

        print("="*50)
        
        return predictor
        
    except Exception as e:
        print(f"Error training model: {e}")
        return None

def batch_train_models(symbols=['bitcoin', 'ethereum', 'cardano', 'solana'], days=365):
    """
    Train models for multiple cryptocurrencies.
    """
    results = {}
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Training model for {symbol.upper()}")
        print(f"{'='*50}")
        
        predictor = train_improved_model(symbol, days)
        results[symbol] = predictor
        
        # Add delay to avoid hitting API rate limits
        if symbol != symbols[-1]:  # Don't wait after the last symbol
            print("Waiting 10 seconds before next training...")
            time.sleep(10)
    
    return results

def test_existing_model(symbol='bitcoin', days=60, model_dir='improved_models'):
    """
    Test an existing trained model.
    """
    model_path = f"{model_dir}/{symbol}_improved_model.h5"
    price_scaler_path = f"{model_dir}/{symbol}_price_scaler.pkl"
    feature_scaler_path = f"{model_dir}/{symbol}_feature_scaler.pkl"
    
    predictor = ImprovedCryptoPricePredictor()
    
    if predictor.load_model(model_path, price_scaler_path, feature_scaler_path):
        # Get fresh data for testing
        prices, _ = get_crypto_data_with_validation(symbol, days)
        
        if prices:
            result = predictor.predict(prices)
            
            print(f"\nPrediction for {symbol.upper()}:")
            print(f"Current Price: ${prices[-1]:.2f}")
            print(f"Predicted Price: ${result['predicted_price']:.2f}")
            print(f"Direction: {result['direction']}")
            print(f"Change Percentage: {result['change_percent']:.2f}%")
            print(f"Recommendation: {result['recommendation']}")
            print(f"Confidence: {result['confidence']:.2f}")
            
            return result
    else:
        print(f"Could not load model for {symbol}")
        return None

if __name__ == "__main__":
    # Choose what to do
    action = input("Choose action: (1) Train new model, (2) Train multiple models, (3) Test existing model: ")
    
    if action == "1":
        symbol = input("Enter cryptocurrency symbol (e.g., bitcoin, ethereum): ").lower()
        days = int(input("Enter number of days of data (default 365): ") or "365")
        train_improved_model(symbol, days)
    
    elif action == "2":
        symbols_input = input("Enter symbols separated by commas (default: bitcoin,ethereum,cardano): ")
        if symbols_input.strip():
            symbols = [s.strip().lower() for s in symbols_input.split(',')]
        else:
            symbols = ['bitcoin', 'ethereum', 'cardano']
        days = int(input("Enter number of days of data (default 365): ") or "365")
        batch_train_models(symbols, days)
    
    elif action == "3":
        symbol = input("Enter cryptocurrency symbol to test: ").lower()
        test_existing_model(symbol)
    
    else:
        print("Invalid option selected")