# Add these imports at the very top of flask_api.py (BEFORE any other imports)
import os
import time
import random

# Suppress TensorFlow warnings and CUDA errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU/CUDA

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
API_CALL_DELAY = 2  # seconds between API calls

def get_crypto_data_api_improved(symbol='bitcoin', days=60):
    """
    Improved crypto data fetching with rate limiting and retry logic.
    """
    global last_api_call
    
    try:
        # Rate limiting - wait between API calls
        current_time = time.time()
        time_since_last_call = current_time - last_api_call
        if time_since_last_call < API_CALL_DELAY:
            sleep_time = API_CALL_DELAY - time_since_last_call + random.uniform(0.5, 1.5)
            print(f"Rate limiting: waiting {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
        
        last_api_call = time.time()
        
        # Always fetch more data than requested to account for cleaning
        fetch_days = min(max(days * 1.5, 90), 365)  # Ensure not more than 365
        print(f"Fetch days calculated: {fetch_days}")
        
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
        params = {
            'vs_currency': 'usd', 
            'days': fetch_days,
            'interval': 'hourly' if fetch_days <= 30 else 'daily'
        }
        
        print(f"Fetching {symbol} data for {fetch_days} days (requested: {days})")
        
        # Create session with retry strategy
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=2,
            respect_retry_after_header=True
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Make the request with longer timeout
        response = session.get(url, params=params, timeout=30)
        
        # Handle rate limiting specifically
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after + random.uniform(1, 5))
            
            # Retry once more
            response = session.get(url, params=params, timeout=30)
        
        response.raise_for_status()
        data = response.json()
        
        if 'prices' not in data or len(data['prices']) == 0:
            raise ValueError("No price data received from API")
        
        prices = [float(price[1]) for price in data['prices']]
        
        print(f"Retrieved {len(prices)} price points")
        
        # Basic validation
        if len(prices) < 50:
            raise ValueError(f"Insufficient data from API: only {len(prices)} points")
        
        # If we have way more data than needed, trim to reasonable size for performance
        if len(prices) > days * 3:
            prices = prices[-(days * 2):]  # Keep last 2x requested days
            print(f"Trimmed to {len(prices)} most recent price points")
        
        return prices
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logger.error(f"Rate limit exceeded for {symbol}. Please try again later.")
            return None
        else:
            logger.error(f"HTTP error fetching {symbol} data: {e}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching {symbol} data: {e}")
        return None
    except ValueError as e:
        logger.error(f"Data validation error for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching {symbol} data: {e}")
        return None

def load_predictor(symbol):
    """
    Load predictor for a specific symbol with caching.
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

def validate_symbol(symbol):
    """Basic symbol validation"""
    if not symbol or not isinstance(symbol, str):
        return False
    # Basic validation - can be expanded with more symbols
    symbol = symbol.lower()
    return len(symbol) > 0 and symbol.isalnum()

def validate_prices(prices):
    """Validate price data"""
    try:
        prices = [float(p) for p in prices]
        if len(prices) < 90:
            return None, f"Need at least 90 price points, got {len(prices)}"
        if any(p <= 0 for p in prices):
            return None, "All prices must be positive numbers"
        return prices, None
    except (ValueError, TypeError):
        return None, "All prices must be valid numbers"

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint."""
    try:
        # Check available models
        available_models = []
        if os.path.exists(MODEL_DIR):
            for file in os.listdir(MODEL_DIR):
                if file.endswith('_improved_model.h5'):
                    symbol = file.replace('_improved_model.h5', '')
                    available_models.append(symbol)
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'available_models': available_models,
            'cached_predictors': list(predictors_cache.keys())
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict/<symbol>', methods=['GET', 'POST'])
def predict_crypto(symbol):
    """
    Improved prediction endpoint with better data handling.
    """
    try:
        symbol = symbol.lower()
        
        # Validate symbol
        if not validate_symbol(symbol):
            return jsonify({
                'error': 'Invalid symbol',
                'message': 'Symbol must be a valid alphanumeric string'
            }), 400
        
        # Load predictor
        predictor = load_predictor(symbol)
        if not predictor:
            return jsonify({
                'error': 'Model not found',
                'message': f'No trained model available for {symbol}. Please train the model first.'
            }), 404
        
        # Get price data
        if request.method == 'GET':
            days = request.args.get('days', 90, type=int)
            days = max(min(days, 365), 90)  # Ensure between 90 and 365
            
            prices = get_crypto_data_api_improved(symbol, days)
            
            if not prices:
                return jsonify({
                    'error': 'Data fetch failed',
                    'message': f'Could not fetch real-time data for {symbol}'
                }), 500
        
        else:  # POST request
            data = request.get_json()
            if not data or 'prices' not in data:
                return jsonify({
                    'error': 'Invalid request',
                    'message': 'POST request must contain "prices" field'
                }), 400
            
            prices, error_msg = validate_prices(data['prices'])
            if prices is None:
                return jsonify({
                    'error': 'Invalid price data',
                    'message': error_msg
                }), 400
        
        # Make prediction
        result = predictor.predict(prices)
        
        # Add metadata
        result.update({
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'current_price': prices[-1],
            'total_data_points': len(prices),
            'prediction_method': 'adaptive_lstm'
        })
        
        logger.info(f"Prediction for {symbol}: {result.get('predicted_price', 'N/A'):.2f} ({result.get('direction', 'N/A')})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in prediction for {symbol}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/train/<symbol>', methods=['POST'])
def train_model_endpoint(symbol):
    """
    Train a new model for a specific cryptocurrency.
    """
    try:
        symbol = symbol.lower()
        
        # Validate symbol
        if not validate_symbol(symbol):
            return jsonify({
                'error': 'Invalid symbol',
                'message': 'Symbol must be a valid alphanumeric string'
            }), 400
        
        data = request.get_json() or {}
        
        days = max(min(data.get('days', 365), 365), 200)  # Between 200 and 365
        epochs = max(min(data.get('epochs', 100), 200), 50)  # Between 50 and 200
        
        logger.info(f"Starting training for {symbol} with {days} days of data")
        
        # Fetch training data
        prices = get_crypto_data_api_improved(symbol, days)
        if not prices:
            return jsonify({
                'error': 'Data fetch failed',
                'message': f'Could not fetch training data for {symbol}'
            }), 500
        
        if len(prices) < 200:
            return jsonify({
                'error': 'Insufficient data',
                'message': f'Need at least 200 data points for training, got {len(prices)}'
            }), 400
        
        # Initialize and train predictor
        predictor = ImprovedCryptoPricePredictor()
        
        # Train the model
        history = predictor.train(prices, epochs=epochs)
        
        # Save the model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = f"{MODEL_DIR}/{symbol}_improved_model.h5"
        price_scaler_path = f"{MODEL_DIR}/{symbol}_price_scaler.pkl"
        feature_scaler_path = f"{MODEL_DIR}/{symbol}_feature_scaler.pkl"
        
        predictor.save_model(model_path, price_scaler_path, feature_scaler_path)
        
        # Cache the predictor
        predictors_cache[symbol] = predictor
        
        # Test the model
        test_result = predictor.predict(prices)
        
        result = {
            'message': f'Model for {symbol} trained successfully',
            'symbol': symbol,
            'training_data_points': len(prices),
            'epochs_completed': len(history.get('loss', [])),
            'final_loss': float(history['loss'][-1]) if history.get('loss') else None,
            'final_val_loss': float(history['val_loss'][-1]) if history.get('val_loss') else None,
            'test_prediction': test_result,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Model training completed for {symbol}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error training model for {symbol}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Training failed',
            'message': str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Make predictions for multiple cryptocurrencies.
    """
    try:
        data = request.get_json()
        if not data or 'symbols' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Request must contain "symbols" field'
            }), 400
        
        symbols = data['symbols']
        if not isinstance(symbols, list) or len(symbols) == 0:
            return jsonify({
                'error': 'Invalid symbols',
                'message': 'Symbols must be a non-empty list'
            }), 400
        
        # Limit batch size to prevent abuse
        if len(symbols) > 10:
            return jsonify({
                'error': 'Too many symbols',
                'message': 'Maximum 10 symbols allowed per batch request'
            }), 400
        
        days = max(min(data.get('days', 60), 365), 90)  # Between 90 and 365
        results = []
        
        for symbol in symbols:
            try:
                symbol = str(symbol).lower()
                
                if not validate_symbol(symbol):
                    results.append({
                        'symbol': symbol,
                        'error': 'Invalid symbol',
                        'message': 'Symbol must be a valid alphanumeric string'
                    })
                    continue
                
                predictor = load_predictor(symbol)
                
                if not predictor:
                    results.append({
                        'symbol': symbol,
                        'error': 'Model not found',
                        'message': f'No trained model available for {symbol}'
                    })
                    continue
                
                # Get real-time data
                prices = get_crypto_data_api_improved(symbol, days)
                if not prices:
                    results.append({
                        'symbol': symbol,
                        'error': 'Data fetch failed',
                        'message': f'Could not fetch data for {symbol}'
                    })
                    continue
                
                # Make prediction
                prediction = predictor.predict(prices)
                prediction['symbol'] = symbol
                prediction['current_price'] = prices[-1]
                results.append(prediction)
                
            except Exception as e:
                results.append({
                    'symbol': symbol,
                    'error': 'Prediction failed',
                    'message': str(e)
                })
        
        return jsonify({
            'predictions': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500

@app.route('/models', methods=['GET'])
def list_models():
    """
    List all available trained models.
    """
    try:
        available_models = []
        
        if os.path.exists(MODEL_DIR):
            for file in os.listdir(MODEL_DIR):
                if file.endswith('_improved_model.h5'):
                    symbol = file.replace('_improved_model.h5', '')
                    
                    # Get model file stats
                    model_path = os.path.join(MODEL_DIR, file)
                    stat = os.stat(model_path)
                    
                    model_info = {
                        'symbol': symbol,
                        'model_file': file,
                        'size_mb': round(stat.st_size / (1024 * 1024), 2),
                        'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'is_loaded': symbol in predictors_cache
                    }
                    
                    available_models.append(model_info)
        
        return jsonify({
            'available_models': available_models,
            'total_models': len(available_models),
            'cached_models': len(predictors_cache)
        })
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return jsonify({
            'error': 'Failed to list models',
            'message': str(e)
        }), 500

@app.route('/model/<symbol>/info', methods=['GET'])
def model_info(symbol):
    """
    Get detailed information about a specific model.
    """
    try:
        symbol = symbol.lower()
        
        if not validate_symbol(symbol):
            return jsonify({
                'error': 'Invalid symbol',
                'message': 'Symbol must be a valid alphanumeric string'
            }), 400
        
        predictor = load_predictor(symbol)
        
        if not predictor:
            return jsonify({
                'error': 'Model not found',
                'message': f'No trained model available for {symbol}'
            }), 404
        
        # Get model file info
        model_path = f"{MODEL_DIR}/{symbol}_improved_model.h5"
        
        if not os.path.exists(model_path):
            return jsonify({
                'error': 'Model file not found',
                'message': f'Model file for {symbol} does not exist'
            }), 404
        
        stat = os.stat(model_path)
        
        model_info = {
            'symbol': symbol,
            'sequence_length': getattr(predictor, 'sequence_length', 'N/A'),
            'lstm_units': getattr(predictor, 'lstm_units', 'N/A'),
            'dropout_rate': getattr(predictor, 'dropout_rate', 'N/A'),
            'is_trained': getattr(predictor, 'is_trained', False),
            'model_size_mb': round(stat.st_size / (1024 * 1024), 2),
            'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'is_cached': symbol in predictors_cache
        }
        
        return jsonify(model_info)
        
    except Exception as e:
        logger.error(f"Error getting model info for {symbol}: {str(e)}")
        return jsonify({
            'error': 'Failed to get model info',
            'message': str(e)
        }), 500

@app.route('/model/<symbol>/test', methods=['GET'])
def test_model(symbol):
    """
    Test a model with recent data and return performance metrics.
    """
    try:
        symbol = symbol.lower()
        
        if not validate_symbol(symbol):
            return jsonify({
                'error': 'Invalid symbol',
                'message': 'Symbol must be a valid alphanumeric string'
            }), 400
        
        predictor = load_predictor(symbol)
        
        if not predictor:
            return jsonify({
                'error': 'Model not found',
                'message': f'No trained model available for {symbol}'
            }), 404
        
        # Get test data
        days = max(min(request.args.get('days', 30, type=int), 90), 30)  # Between 30 and 90
        prices = get_crypto_data_api_improved(symbol, days)
        
        if not prices:
            return jsonify({
                'error': 'Data fetch failed',
                'message': f'Could not fetch test data for {symbol}'
            }), 500
        
        # Make prediction
        result = predictor.predict(prices)
        
        # Calculate additional test metrics
        if len(prices) >= 10:
            price_volatility = np.std(prices[-10:]) / np.mean(prices[-10:])
            price_trend = (prices[-1] - prices[-10]) / prices[-10] * 100
        else:
            price_volatility = 0
            price_trend = 0
        
        test_result = {
            'symbol': symbol,
            'prediction': result,
            'test_data_points': len(prices),
            'current_price': prices[-1],
            'recent_volatility': round(price_volatility, 4),
            'recent_trend_percent': round(price_trend, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(test_result)
        
    except Exception as e:
        logger.error(f"Error testing model for {symbol}: {str(e)}")
        return jsonify({
            'error': 'Model test failed',
            'message': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        'error': 'Bad request',
        'message': 'The request was invalid'
    }), 400

if __name__ == '__main__':
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    logger.info("Starting Improved Crypto Prediction API")
    logger.info(f"Model directory: {MODEL_DIR}")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)