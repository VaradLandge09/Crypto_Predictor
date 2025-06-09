# Add these imports at the very top of flask_api.py (BEFORE any other imports)
import os
import time
import random
import signal
from contextlib import contextmanager
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Suppress TensorFlow warnings and CUDA errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU/CUDA
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import logging
from datetime import datetime
import traceback
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from crypto_predictor import ImprovedCryptoPricePredictor

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global predictors cache
predictors_cache = {}
MODEL_DIR = 'improved_models'

# API rate limiting variables
last_api_call = 0
API_CALL_DELAY = 0.5  # Reduced delay
MAX_REQUEST_TIMEOUT = 8  # Shorter timeout for API calls
MAX_PREDICTION_TIMEOUT = 15  # Maximum time for entire prediction

# Thread pool for non-blocking operations
executor = ThreadPoolExecutor(max_workers=2)

import requests
import time
import random
from datetime import datetime, timedelta

import requests
import time
import random
from datetime import datetime, timedelta

def get_crypto_data_with_fallback(symbol='bitcoin', days=90):
    """
    Try multiple free APIs with fallback system.
    Increased default days to ensure we get enough data points.
    """
    
    # Try CoinGecko first (with rate limiting)
    data = try_coingecko(symbol, days)
    if data:
        return data
    
    # Fallback to Binance (no API key required for market data)
    print("CoinGecko failed, trying Binance...")
    data = try_binance(symbol, days)
    if data:
        return data
    
    # Fallback to CryptoCompare
    print("Binance failed, trying CryptoCompare...")
    data = try_cryptocompare(symbol, days)
    if data:
        return data
    
    print("All APIs failed")
    return None

def try_coingecko(symbol, days):
    """Try CoinGecko with improved rate limiting."""
    try:
        SYMBOL_MAP = {
            'bitcoin': 'bitcoin',
            'ethereum': 'ethereum',
            'dogecoin': 'dogecoin'
        }
        
        cg_symbol = SYMBOL_MAP.get(symbol.lower())
        if not cg_symbol:
            return None
        
        # Random delay to avoid rate limits
        time.sleep(random.uniform(2, 5))
        
        # Ensure we request enough days to get 80+ data points
        api_days = max(days, 90)  # Request at least 90 days
        
        url = f"https://api.coingecko.com/api/v3/coins/{cg_symbol}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': min(api_days, 365),
            'interval': 'daily'
        }
        
        headers = {
            'User-Agent': 'CryptoPredictionApp/1.0',
            'Accept': 'application/json'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 429:
            print("CoinGecko rate limited")
            return None
            
        response.raise_for_status()
        data = response.json()
        
        prices = [price[1] for price in data['prices']]
        
        # Ensure we have enough data points
        if len(prices) < 80:
            print(f"CoinGecko: Only got {len(prices)} prices, need at least 80")
            return None
            
        print(f"CoinGecko: Retrieved {len(prices)} prices for {symbol}")
        return prices
        
    except Exception as e:
        print(f"CoinGecko error: {e}")
        return None

def try_binance(symbol, days):
    """Try Binance public API (no auth required)."""
    try:
        SYMBOL_MAP = {
            'bitcoin': 'BTCUSDT',
            'ethereum': 'ETHUSDT',
            'dogecoin': 'DOGEUSDT'
        }
        
        binance_symbol = SYMBOL_MAP.get(symbol.lower())
        if not binance_symbol:
            return None
        
        url = "https://api.binance.com/api/v3/klines"
        
        # Request more days to ensure we get enough data
        api_days = max(days, 90)
        
        # Calculate start time (days ago)
        end_time = int(time.time() * 1000)
        start_time = end_time - (api_days * 24 * 60 * 60 * 1000)
        
        params = {
            'symbol': binance_symbol,
            'interval': '1d',
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000  # Binance allows up to 1000
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract closing prices (index 4 in kline data)
        prices = [float(kline[4]) for kline in data]
        
        # Ensure we have enough data points
        if len(prices) < 80:
            print(f"Binance: Only got {len(prices)} prices, need at least 80")
            return None
            
        print(f"Binance: Retrieved {len(prices)} prices for {symbol}")
        return prices
        
    except Exception as e:
        print(f"Binance error: {e}")
        return None

def try_cryptocompare(symbol, days):
    """Try CryptoCompare API (free tier)."""
    try:
        SYMBOL_MAP = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH',
            'dogecoin': 'DOGE'
        }
        
        cc_symbol = SYMBOL_MAP.get(symbol.lower())
        if not cc_symbol:
            return None
        
        # Request more days to ensure we get enough data
        api_days = max(days, 90)
        
        url = "https://min-api.cryptocompare.com/data/v2/histoday"
        params = {
            'fsym': cc_symbol,
            'tsym': 'USD',
            'limit': min(api_days, 2000),  # CryptoCompare allows up to 2000
            'toTs': int(time.time())
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data['Response'] != 'Success':
            return None
            
        # Extract closing prices
        prices = [day['close'] for day in data['Data']['Data']]
        
        # Ensure we have enough data points
        if len(prices) < 80:
            print(f"CryptoCompare: Only got {len(prices)} prices, need at least 80")
            return None
            
        print(f"CryptoCompare: Retrieved {len(prices)} prices for {symbol}")
        return prices
        
    except Exception as e:
        print(f"CryptoCompare error: {e}")
        return None

def load_predictor_fast(symbol):
    """
    Fast predictor loading with minimal validation.
    """
    if symbol in predictors_cache:
        return predictors_cache[symbol]
    
    model_path = f"{MODEL_DIR}/{symbol}_improved_model.h5"
    price_scaler_path = f"{MODEL_DIR}/{symbol}_price_scaler.pkl"
    feature_scaler_path = f"{MODEL_DIR}/{symbol}_feature_scaler.pkl"
    
    if not all(os.path.exists(path) for path in [model_path, price_scaler_path, feature_scaler_path]):
        logger.warning(f"Model files not found for {symbol}")
        return None
    
    try:
        print(f"Improved model loaded from {model_path}")
        predictor = ImprovedCryptoPricePredictor()
        if predictor.load_model(model_path, price_scaler_path, feature_scaler_path):
            predictors_cache[symbol] = predictor
            logger.info(f"Loaded model for {symbol}")
            return predictor
        else:
            logger.error(f"Failed to load model for {symbol}")
            return None
    except Exception as e:
        logger.error(f"Error loading model for {symbol}: {e}")
        return None

def predict_with_timeout(predictor, prices, timeout_seconds=MAX_PREDICTION_TIMEOUT):
    """
    Run prediction with strict timeout using threading.
    """
    result = [None]
    exception = [None]
    
    def prediction_task():
        try:
            result[0] = predictor.predict(prices)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=prediction_task)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        # Thread is still running, prediction timed out
        logger.error("Prediction timed out")
        return None, "Prediction timed out"
    
    if exception[0]:
        return None, str(exception[0])
    
    return result[0], None

@app.route('/health', methods=['GET'])
def health_check():
    """Ultra-fast health check."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'cached_predictors': list(predictors_cache.keys())
    })

@app.route('/predict/<symbol>', methods=['GET', 'POST'])
def predict_crypto(symbol):
    """
    Ultra-fast prediction endpoint with aggressive timeouts.
    """
    start_time = time.time()
    
    try:
        symbol = symbol.lower()
        
        # Quick validation
        if not symbol or not symbol.isalnum():
            return jsonify({
                'error': 'Invalid symbol',
                'message': 'Symbol must be alphanumeric'
            }), 400
        
        # Load predictor quickly
        predictor = load_predictor_fast(symbol)
        if not predictor:
            return jsonify({
                'error': 'Model not found',
                'message': f'No model for {symbol}'
            }), 404
        
        # Get price data
        if request.method == 'GET':
            days = min(max(request.args.get('days', 30, type=int), 30), 90)  # Smaller range
            
            prices = get_crypto_data_with_fallback(symbol, days)
            
            if not prices:
                return jsonify({
                    'error': 'Data fetch failed',
                    'message': f'Could not fetch data for {symbol}'
                }), 500
        
        else:  # POST request
            data = request.get_json()
            if not data or 'prices' not in data:
                return jsonify({
                    'error': 'Invalid request',
                    'message': 'POST must contain "prices"'
                }), 400
            
            try:
                prices = [float(p) for p in data['prices']]
                if len(prices) < 30:
                    return jsonify({
                        'error': 'Insufficient data',
                        'message': f'Need at least 30 prices, got {len(prices)}'
                    }), 400
            except (ValueError, TypeError):
                return jsonify({
                    'error': 'Invalid prices',
                    'message': 'All prices must be numbers'
                }), 400
        
        # Check time so far
        elapsed = time.time() - start_time
        if elapsed > 20:  # If we're already taking too long
            return jsonify({
                'error': 'Request timeout',
                'message': 'Request taking too long'
            }), 408
        
        # Make prediction with timeout
        remaining_time = max(5, 25 - elapsed)  # At least 5 seconds for prediction
        result, error = predict_with_timeout(predictor, prices, remaining_time)
        
        if result is None:
            return jsonify({
                'error': 'Prediction failed',
                'message': error or 'Unknown prediction error'
            }), 500
        
        # Add minimal metadata
        result.update({
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'current_price': prices[-1],
            'processing_time': round(time.time() - start_time, 2)
        })
        
        logger.info(f"Prediction for {symbol}: {result.get('predicted_price', 'N/A')} ({result.get('direction', 'N/A')})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in prediction for {symbol}: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e),
            'processing_time': round(time.time() - start_time, 2)
        }), 500

@app.route('/train/<symbol>', methods=['POST'])
def train_model_endpoint(symbol):
    """
    Fast model training endpoint.
    """
    try:
        symbol = symbol.lower()
        
        if not symbol or not symbol.isalnum():
            return jsonify({
                'error': 'Invalid symbol'
            }), 400
        
        data = request.get_json() or {}
        
        days = min(max(data.get('days', 90), 60), 180)  # Reduced range
        epochs = min(max(data.get('epochs', 30), 20), 50)  # Reduced epochs
        
        logger.info(f"Starting training for {symbol} with {days} days, {epochs} epochs")
        
        # Fetch training data
        prices = get_crypto_data_with_fallback(symbol, days)
        if not prices:
            return jsonify({
                'error': 'Data fetch failed'
            }), 500
        
        if len(prices) < 60:
            return jsonify({
                'error': 'Insufficient data',
                'message': f'Need at least 60 points, got {len(prices)}'
            }), 400
        
        # Train model
        predictor = ImprovedCryptoPricePredictor()
        history = predictor.train(prices, epochs=epochs)
        
        # Save model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = f"{MODEL_DIR}/{symbol}_improved_model.h5"
        price_scaler_path = f"{MODEL_DIR}/{symbol}_price_scaler.pkl"
        feature_scaler_path = f"{MODEL_DIR}/{symbol}_feature_scaler.pkl"
        
        predictor.save_model(model_path, price_scaler_path, feature_scaler_path)
        
        # Cache predictor
        predictors_cache[symbol] = predictor
        
        # Quick test
        test_result = predictor.predict(prices[-60:])  # Use less data for test
        
        result = {
            'message': f'Model for {symbol} trained successfully',
            'symbol': symbol,
            'training_data_points': len(prices),
            'epochs_completed': len(history.get('loss', [])),
            'test_prediction': test_result,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Training completed for {symbol}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Training error for {symbol}: {str(e)}")
        return jsonify({
            'error': 'Training failed',
            'message': str(e)
        }), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List available models quickly."""
    try:
        models = []
        if os.path.exists(MODEL_DIR):
            for file in os.listdir(MODEL_DIR):
                if file.endswith('_improved_model.h5'):
                    symbol = file.replace('_improved_model.h5', '')
                    models.append(symbol)
        
        return jsonify({
            'models': models,
            'cached': list(predictors_cache.keys()),
            'count': len(models)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal error'}), 500

@app.errorhandler(408)
def timeout_error(error):
    return jsonify({'error': 'Timeout'}), 408

if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info("Starting Fast Crypto Prediction API")
    app.run(host='0.0.0.0', port=5000, debug=False)