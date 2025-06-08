from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import logging
from datetime import datetime
import traceback
import requests
import pandas as pd

from crypto_predictor import ImprovedCryptoPricePredictor

# Import the improved predictor
# from improved_crypto_predictor import ImprovedCryptoPricePredictor

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global predictors cache
predictors_cache = {}
MODEL_DIR = 'improved_models'

# def get_crypto_data_api(symbol='bitcoin', days=60):
#     """
#     Fetch real-time crypto data for predictions.
#     """
#     days = max(days, 90)
#     try:
#         url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
#         params = {
#             'vs_currency': 'usd', 
#             'days': days,
#             'interval': 'hourly' if days <= 30 else 'daily'
#         }
        
#         response = requests.get(url, params=params, timeout=10)
#         response.raise_for_status()
#         data = response.json()
        
#         prices = [float(price[1]) for price in data['prices']]
#         return prices
        
#     except Exception as e:
#         logger.error(f"Error fetching {symbol} data: {e}")
#         return None

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
        return None
    
    try:
        predictor = ImprovedCryptoPricePredictor()
        if predictor.load_model(model_path, price_scaler_path, feature_scaler_path):
            predictors_cache[symbol] = predictor
            logger.info(f"Loaded model for {symbol}")
            return predictor
    except Exception as e:
        logger.error(f"Error loading model for {symbol}: {e}")
    
    return None

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint."""
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

# @app.route('/predict/<symbol>', methods=['GET', 'POST'])
# def predict_crypto(symbol):
#     """
#     Predict cryptocurrency price for a specific symbol.
    
#     GET: Uses real-time data from API
#     POST: Uses provided price data
#     """
#     try:
#         symbol = symbol.lower()
        
#         # Load predictor
#         predictor = load_predictor(symbol)
#         if not predictor:
#             return jsonify({
#                 'error': 'Model not found',
#                 'message': f'No trained model available for {symbol}. Please train the model first.'
#             }), 404
        
#         # Get price data
#         if request.method == 'GET':
#             days = request.args.get('days', 60, type=int)
#             prices = get_crypto_data_api(symbol, days)
#             print(len(prices))
#             if not prices:
#                 return jsonify({
#                     'error': 'Data fetch failed',
#                     'message': f'Could not fetch real-time data for {symbol}'
#                 }), 500
        
#         else:  # POST request
#             data = request.get_json()
#             if not data or 'prices' not in data:
#                 return jsonify({
#                     'error': 'Invalid request',
#                     'message': 'POST request must contain "prices" field'
#                 }), 400
            
#             prices = data['prices']
            
#             # Validate prices
#             try:
#                 prices = [float(p) for p in prices]
#             except (ValueError, TypeError):
#                 return jsonify({
#                     'error': 'Invalid price data',
#                     'message': 'All prices must be valid numbers'
#                 }), 400
        
#         # Make prediction
#         result = predictor.predict(prices)
        
#         # Add metadata
#         result.update({
#             'symbol': symbol,
#             'timestamp': datetime.now().isoformat(),
#             'current_price': prices[-1],
#             'data_points_used': len(prices),
#             'prediction_method': 'improved_lstm'
#         })
        
#         logger.info(f"Prediction for {symbol}: {result['predicted_price']:.2f} ({result['direction']})")
        
#         return jsonify(result)
        
#     except Exception as e:
#         logger.error(f"Error in prediction for {symbol}: {str(e)}")
#         logger.error(traceback.format_exc())
#         return jsonify({
#             'error': 'Prediction failed',
#             'message': str(e)
#         }), 500

@app.route('/train/<symbol>', methods=['POST'])
def train_model_endpoint(symbol):
    """
    Train a new model for a specific cryptocurrency.
    """
    try:
        symbol = symbol.lower()
        data = request.get_json() or {}
        
        days = data.get('days', 365)
        epochs = data.get('epochs', 100)
        
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
            'epochs_completed': len(history['loss']),
            'final_loss': float(history['loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1]) if 'val_loss' in history else None,
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
        days = data.get('days', 60)
        results = []
        
        for symbol in symbols:
            try:
                symbol = symbol.lower()
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
        predictor = load_predictor(symbol)
        
        if not predictor:
            return jsonify({
                'error': 'Model not found',
                'message': f'No trained model available for {symbol}'
            }), 404
        
        # Get model file info
        model_path = f"{MODEL_DIR}/{symbol}_improved_model.h5"
        stat = os.stat(model_path)
        
        model_info = {
            'symbol': symbol,
            'sequence_length': predictor.sequence_length,
            'lstm_units': predictor.lstm_units,
            'dropout_rate': predictor.dropout_rate,
            'is_trained': predictor.is_trained,
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
        predictor = load_predictor(symbol)
        
        if not predictor:
            return jsonify({
                'error': 'Model not found',
                'message': f'No trained model available for {symbol}'
            }), 404
        
        # Get test data
        days = request.args.get('days', 30, type=int)
        prices = get_crypto_data_api_improved(symbol, days)
        
        if not prices:
            return jsonify({
                'error': 'Data fetch failed',
                'message': f'Could not fetch test data for {symbol}'
            }), 500
        
        # Make prediction
        result = predictor.predict(prices)
        
        # Calculate additional test metrics
        price_volatility = np.std(prices[-10:]) / np.mean(prices[-10:])
        price_trend = (prices[-1] - prices[-10]) / prices[-10] * 100
        
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
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

def get_crypto_data_api_improved(symbol='bitcoin', days=60):
    """
    Improved crypto data fetching with better error handling and more data.
    """
    try:
        # Always fetch more data than requested to account for cleaning
        fetch_days = min(max(days * 1.5, 90), 365)  # Ensure not more than 365
        print(fetch_days)
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
        params = {
            'vs_currency': 'usd', 
            'days': fetch_days,
            'interval': 'hourly' if fetch_days <= 30 else 'daily'
        }
        
        print(f"Fetching {symbol} data for {fetch_days} days (requested: {days})")
        
        response = requests.get(url, params=params, timeout=15)
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
        
    except Exception as e:
        logger.error(f"Error fetching {symbol} data: {e}")
        return None

# Update the predict endpoint in flask_api.py
@app.route('/predict/<symbol>', methods=['GET', 'POST'])
def predict_crypto_improved(symbol):
    """
    Improved prediction endpoint with better data handling.
    """
    try:
        symbol = symbol.lower()
        
        # Load predictor
        predictor = load_predictor(symbol)
        if not predictor:
            return jsonify({
                'error': 'Model not found',
                'message': f'No trained model available for {symbol}. Please train the model first.'
            }), 404
        
        # Get price data
        if request.method == 'GET':
            days = request.args.get('days', 90, type=int)  # Increased default
            days = max(days, 90)  # Ensure minimum
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
            
            prices = data['prices']
            
            # Validate prices
            try:
                prices = [float(p) for p in prices]
                if len(prices) < 90:
                    return jsonify({
                        'error': 'Insufficient data',
                        'message': f'Need at least 90 price points, got {len(prices)}'
                    }), 400
            except (ValueError, TypeError):
                return jsonify({
                    'error': 'Invalid price data',
                    'message': 'All prices must be valid numbers'
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
        
        logger.info(f"Prediction for {symbol}: {result['predicted_price']:.2f} ({result['direction']})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in prediction for {symbol}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    logger.info("Starting Improved Crypto Prediction API")
    logger.info(f"Model directory: {MODEL_DIR}")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)