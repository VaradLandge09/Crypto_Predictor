import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os

from crypto_predictor import ImprovedCryptoPricePredictor

# Import the improved predictor
# from improved_crypto_predictor import ImprovedCryptoPricePredictor

def get_crypto_data_with_validation(symbol='bitcoin', days=365, vs_currency='usd'):
    """
    Fetch and validate historical crypto data from CoinGecko API with better error handling.
    """
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
        params = {
            'vs_currency': vs_currency, 
            'days': days,
            'interval': 'hourly' if days <= 30 else 'daily'
        }
        
        print(f"Fetching {symbol} data for {days} days...")
        response = requests.get(url, params=params, timeout=10)
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
        
        # Calculate training metrics
        final_loss = history['loss'][-1]
        final_val_loss = history['val_loss'][-1]
        
        print(f"\nTraining Metrics:")
        print(f"Final Training Loss: {final_loss:.6f}")
        print(f"Final Validation Loss: {final_val_loss:.6f}")
        print(f"Epochs Completed: {len(history['loss'])}")
        
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