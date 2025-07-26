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
executor = ThreadPoolExecutor(max_workers=3)

# Training status tracking
training_status = {}

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
    Try multiple free APIs with fallback system and better error handling.
    """
    print(f"Fetching data for {symbol} ({days} days)")
    
    # Try CoinGecko first (with better rate limiting)
    data = try_coingecko(symbol, days)
    if data:
        return data
    
    # Fallback to Binance
    print("CoinGecko failed, trying Binance...")
    data = try_binance(symbol, days)
    if data:
        return data
    
    # Fallback to CryptoCompare
    print("Binance failed, trying CryptoCompare...")
    data = try_cryptocompare(symbol, days)
    if data:
        return data
    
    # Try Yahoo Finance as last resort
    print("CryptoCompare failed, trying Yahoo Finance...")
    data = try_yahoo_finance(symbol, days)
    if data:
        return data
    
    print("All APIs failed")
    return None

def try_coingecko(symbol, days):
    """Try CoinGecko with improved error handling and rate limiting."""
    try:
        SYMBOL_MAP = {
            'bitcoin': 'bitcoin',
            'ethereum': 'ethereum',
            'dogecoin': 'dogecoin',
            'polkadot': 'polkadot',
            'cardano': 'cardano',
            'solana': 'solana',
            'avalanche': 'avalanche-2',
            'polygon': 'matic-network',
            'chainlink': 'chainlink',
            'litecoin': 'litecoin',
            'ripple': 'ripple',
            'xrp': 'ripple',
            'binancecoin': 'binancecoin',
            'bnb': 'binancecoin',
            'tron': 'tron',
            'shiba-inu': 'shiba-inu',
            'pepe': 'pepe',
            'uniswap': 'uniswap',
            'cosmos': 'cosmos',
            'algorand': 'algorand',
            'stellar': 'stellar',
            'vechain': 'vechain',
            'filecoin': 'filecoin',
            'hedera': 'hedera-hashgraph',
            'hedera-hashgraph': 'hedera-hashgraph',
            'near': 'near',
            'aptos': 'aptos',
            'optimism': 'optimism',
            'arbitrum': 'arbitrum'
        }
        
        cg_symbol = SYMBOL_MAP.get(symbol.lower())
        if not cg_symbol:
            print(f"Symbol {symbol} not found in CoinGecko mapping")
            return None
        
        # Longer delay to avoid rate limits
        time.sleep(random.uniform(3, 7))
        
        api_days = max(days, 90)
        
        url = f"https://api.coingecko.com/api/v3/coins/{cg_symbol}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': min(api_days, 365),
            'interval': 'daily'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        # Multiple retry attempts
        for attempt in range(3):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=30)
                
                if response.status_code == 429:
                    print(f"CoinGecko rate limited, attempt {attempt + 1}")
                    time.sleep(10 * (attempt + 1))  # Exponential backoff
                    continue
                    
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'prices' not in data:
                        print("CoinGecko: No prices in response")
                        continue
                        
                    prices = [price[1] for price in data['prices']]
                    
                    if len(prices) < 30:  # Reduced minimum requirement
                        print(f"CoinGecko: Only got {len(prices)} prices, need at least 30")
                        continue
                        
                    print(f"CoinGecko: Retrieved {len(prices)} prices for {symbol}")
                    return prices
                else:
                    print(f"CoinGecko HTTP {response.status_code}: {response.text[:200]}")
                    
            except requests.RequestException as e:
                print(f"CoinGecko request error attempt {attempt + 1}: {e}")
                time.sleep(5)
                
        return None
        
    except Exception as e:
        print(f"CoinGecko error: {e}")
        return None

def try_binance(symbol, days):
    """Try Binance public API with better error handling."""
    try:
        SYMBOL_MAP = {
            'bitcoin': 'BTCUSDT',
            'ethereum': 'ETHUSDT',
            'dogecoin': 'DOGEUSDT',
            'polkadot': 'DOTUSDT',
            'cardano': 'ADAUSDT',
            'solana': 'SOLUSDT',
            'avalanche': 'AVAXUSDT',
            'polygon': 'MATICUSDT',
            'chainlink': 'LINKUSDT',
            'litecoin': 'LTCUSDT',
            'ripple': 'XRPUSDT',
            'xrp': 'XRPUSDT',
            'binancecoin': 'BNBUSDT',
            'bnb': 'BNBUSDT',
            'tron': 'TRXUSDT',
            'shiba-inu': 'SHIBUSDT',
            'uniswap': 'UNIUSDT',
            'cosmos': 'ATOMUSDT',
            'algorand': 'ALGOUSDT',
            'stellar': 'XLMUSDT',
            'vechain': 'VETUSDT',
            'filecoin': 'FILUSDT',
            'hedera': 'HBARUSDT',
            'hedera-hashgraph': 'HBARUSDT',
            'near': 'NEARUSDT',
            'aptos': 'APTUSDT',
            'optimism': 'OPUSDT',
            'arbitrum': 'ARBUSDT'
        }
        
        binance_symbol = SYMBOL_MAP.get(symbol.lower())
        if not binance_symbol:
            print(f"Symbol {symbol} not found in Binance mapping")
            return None
        
        api_days = max(days, 90)
        
        # Try different Binance endpoints
        endpoints = [
            "https://api.binance.com/api/v3/klines",
            "https://api1.binance.com/api/v3/klines",
            "https://api2.binance.com/api/v3/klines"
        ]
        
        end_time = int(time.time() * 1000)
        start_time = end_time - (api_days * 24 * 60 * 60 * 1000)
        
        params = {
            'symbol': binance_symbol,
            'interval': '1d',
            'startTime': start_time,
            'endTime': end_time,
            'limit': 500
        }
        
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if not data:
                        continue
                        
                    prices = [float(kline[4]) for kline in data]
                    
                    if len(prices) < 30:
                        print(f"Binance: Only got {len(prices)} prices, need at least 30")
                        continue
                        
                    print(f"Binance: Retrieved {len(prices)} prices for {symbol}")
                    return prices
                else:
                    print(f"Binance HTTP {response.status_code} from {endpoint}")
                    
            except requests.RequestException as e:
                print(f"Binance error with {endpoint}: {e}")
                continue
                
        return None
        
    except Exception as e:
        print(f"Binance error: {e}")
        return None

def try_cryptocompare(symbol, days):
    """Try CryptoCompare API with better error handling."""
    try:
        SYMBOL_MAP = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH',
            'dogecoin': 'DOGE',
            'polkadot': 'DOT',
            'cardano': 'ADA',
            'solana': 'SOL',
            'avalanche': 'AVAX',
            'polygon': 'MATIC',
            'chainlink': 'LINK',
            'litecoin': 'LTC',
            'ripple': 'XRP',
            'xrp': 'XRP',
            'binancecoin': 'BNB',
            'bnb': 'BNB',
            'tron': 'TRX',
            'shiba-inu': 'SHIB',
            'uniswap': 'UNI',
            'cosmos': 'ATOM',
            'algorand': 'ALGO',
            'stellar': 'XLM',
            'vechain': 'VET',
            'filecoin': 'FIL',
            'hedera': 'HBAR',
            'hedera-hashgraph': 'HBAR',
            'near': 'NEAR',
            'aptos': 'APT',
            'optimism': 'OP',
            'arbitrum': 'ARB'
        }
        
        cc_symbol = SYMBOL_MAP.get(symbol.lower())
        if not cc_symbol:
            print(f"Symbol {symbol} not found in CryptoCompare mapping")
            return None
        
        api_days = max(days, 90)
        
        url = "https://min-api.cryptocompare.com/data/v2/histoday"
        params = {
            'fsym': cc_symbol,
            'tsym': 'USD',
            'limit': min(api_days, 365),
            'toTs': int(time.time())
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('Response') != 'Success':
                print(f"CryptoCompare API error: {data.get('Message', 'Unknown error')}")
                return None
                
            if 'Data' not in data or 'Data' not in data['Data']:
                print("CryptoCompare: Invalid response structure")
                return None
                
            prices = [day['close'] for day in data['Data']['Data'] if day['close'] > 0]
            
            if len(prices) < 30:
                print(f"CryptoCompare: Only got {len(prices)} prices, need at least 30")
                return None
                
            print(f"CryptoCompare: Retrieved {len(prices)} prices for {symbol}")
            return prices
        else:
            print(f"CryptoCompare HTTP {response.status_code}: {response.text[:200]}")
            return None
        
    except Exception as e:
        print(f"CryptoCompare error: {e}")
        return None

def try_yahoo_finance(symbol, days):
    """Try Yahoo Finance as fallback (works without API key)."""
    try:
        # Yahoo Finance symbol mapping
        SYMBOL_MAP = {
            'bitcoin': 'BTC-USD',
            'ethereum': 'ETH-USD',
            'dogecoin': 'DOGE-USD',
            'polkadot': 'DOT-USD',
            'cardano': 'ADA-USD',
            'solana': 'SOL-USD',
            'avalanche': 'AVAX-USD',
            'polygon': 'MATIC-USD',
            'chainlink': 'LINK-USD',
            'litecoin': 'LTC-USD',
            'ripple': 'XRP-USD',
            'xrp': 'XRP-USD',
            'binancecoin': 'BNB-USD',
            'bnb': 'BNB-USD',
            'tron': 'TRX-USD',
            'uniswap': 'UNI-USD',
            'cosmos': 'ATOM-USD',
            'algorand': 'ALGO-USD',
            'stellar': 'XLM-USD',
            'hedera': 'HBAR-USD',
            'hedera-hashgraph': 'HBAR-USD',
            'near': 'NEAR-USD',
            'aptos': 'APT-USD'
        }
        
        yahoo_symbol = SYMBOL_MAP.get(symbol.lower())
        if not yahoo_symbol:
            print(f"Symbol {symbol} not found in Yahoo Finance mapping")
            return None
        
        # Calculate timestamps
        end_time = int(time.time())
        start_time = end_time - (max(days, 90) * 24 * 60 * 60)
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        params = {
            'period1': start_time,
            'period2': end_time,
            'interval': '1d',
            'includePrePost': 'false',
            'events': 'div%2Csplit'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'chart' not in data or not data['chart']['result']:
                print("Yahoo Finance: No chart data")
                return None
                
            chart_data = data['chart']['result'][0]
            
            if 'indicators' not in chart_data or 'quote' not in chart_data['indicators']:
                print("Yahoo Finance: No price indicators")
                return None
                
            quotes = chart_data['indicators']['quote'][0]
            
            if 'close' not in quotes:
                print("Yahoo Finance: No closing prices")
                return None
                
            prices = [price for price in quotes['close'] if price is not None]
            
            if len(prices) < 30:
                print(f"Yahoo Finance: Only got {len(prices)} prices, need at least 30")
                return None
                
            print(f"Yahoo Finance: Retrieved {len(prices)} prices for {symbol}")
            return prices
        else:
            print(f"Yahoo Finance HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Yahoo Finance error: {e}")
        return None
    
def model_exists(symbol):
    """Check if all required model files exist for a symbol."""
    model_path = f"{MODEL_DIR}/{symbol}_improved_model.h5"
    price_scaler_path = f"{MODEL_DIR}/{symbol}_price_scaler.pkl"
    feature_scaler_path = f"{MODEL_DIR}/{symbol}_feature_scaler.pkl"
    
    return all(os.path.exists(path) for path in [model_path, price_scaler_path, feature_scaler_path])

def load_predictor_fast(symbol):
    """
    Fast predictor loading with minimal validation.
    """
    if symbol in predictors_cache:
        return predictors_cache[symbol]
    
    if not model_exists(symbol):
        return None
    
    model_path = f"{MODEL_DIR}/{symbol}_improved_model.h5"
    price_scaler_path = f"{MODEL_DIR}/{symbol}_price_scaler.pkl"
    feature_scaler_path = f"{MODEL_DIR}/{symbol}_feature_scaler.pkl"
    
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

def train_model_async(symbol, days=90, epochs=30):
    """
    Train model asynchronously in background.
    """
    try:
        logger.info(f"Starting background training for {symbol}")
        training_status[symbol] = {
            'status': 'training',
            'progress': 0,
            'started_at': datetime.now().isoformat(),
            'message': 'Fetching training data...'
        }
        
        # Fetch training data
        prices = get_crypto_data_with_fallback(symbol, days)
        if not prices:
            training_status[symbol] = {
                'status': 'failed',
                'error': 'Could not fetch training data',
                'completed_at': datetime.now().isoformat()
            }
            return
        
        if len(prices) < 60:
            training_status[symbol] = {
                'status': 'failed',
                'error': f'Insufficient data: got {len(prices)} points, need at least 60',
                'completed_at': datetime.now().isoformat()
            }
            return
        
        training_status[symbol]['message'] = 'Training model...'
        training_status[symbol]['progress'] = 20
        
        # Train model
        predictor = ImprovedCryptoPricePredictor()
        history = predictor.train(prices, epochs=epochs)

        # Add this after: history = predictor.train(...)
        # and before: predictor.save_model(...)

        # Log accuracy metrics to API
        if history and 'val_accuracy_5pct' in history:
            accuracy_info = {
                'validation_accuracy': f"{history['val_accuracy_5pct']:.1f}%",
                'directional_accuracy': f"{history['val_directional_accuracy']:.1f}%",
                'mean_error': f"{history['mean_prediction_error']:.2f}%",
                'mae': f"${history['val_mae']:.2f}"
            }
            
            logger.info(f"Model accuracy for {symbol}: {accuracy_info}")
            
            # Update training status with accuracy (adjusted ratings)
            training_status[symbol].update({
                'accuracy_metrics': accuracy_info,
                'performance_rating': 'EXCELLENT' if history['val_accuracy_5pct'] >= 75 else
                                    'VERY GOOD' if history['val_accuracy_5pct'] >= 65 else
                                    'GOOD' if history['val_accuracy_5pct'] >= 60 else
                                    'SATISFACTORY'
            })
        
        training_status[symbol]['progress'] = 80
        training_status[symbol]['message'] = 'Saving model...'
        
        # Save model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = f"{MODEL_DIR}/{symbol}_improved_model.h5"
        price_scaler_path = f"{MODEL_DIR}/{symbol}_price_scaler.pkl"
        feature_scaler_path = f"{MODEL_DIR}/{symbol}_feature_scaler.pkl"
        
        predictor.save_model(model_path, price_scaler_path, feature_scaler_path)
        
        # Cache predictor
        predictors_cache[symbol] = predictor
        
        training_status[symbol] = {
            'status': 'completed',
            'progress': 100,
            'completed_at': datetime.now().isoformat(),
            'training_data_points': len(prices),
            'epochs_completed': len(history.get('loss', [])),
            'message': 'Model trained successfully'
        }
        
        logger.info(f"Background training completed for {symbol}")
        
    except Exception as e:
        logger.error(f"Background training error for {symbol}: {str(e)}")
        training_status[symbol] = {
            'status': 'failed',
            'error': str(e),
            'completed_at': datetime.now().isoformat()
        }

@app.route('/health', methods=['GET'])
def health_check():
    """Ultra-fast health check."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'cached_predictors': list(predictors_cache.keys()),
        'training_in_progress': [k for k, v in training_status.items() if v.get('status') == 'training']
    })

@app.route('/predict/<symbol>', methods=['GET', 'POST'])
def predict_crypto(symbol):
    """
    Enhanced prediction endpoint with auto-training capability.
    """
    start_time = time.time()
    
    try:
        symbol = symbol.lower()
        
        # Quick validation
        if not symbol:
            return jsonify({
                'error': 'Invalid symbol',
                'message': 'Symbol must be alphanumeric'
            }), 400
        
        # Check if model exists
        predictor = load_predictor_fast(symbol)
        
        if not predictor:
            # Model doesn't exist, check if training is in progress
            if symbol in training_status and training_status[symbol].get('status') == 'training':
                return jsonify({
                    'status': 'training_in_progress',
                    'message': f'Model for {symbol} is currently being trained. Please wait and try again.',
                    'training_info': training_status[symbol],
                    'estimated_completion': '2-5 minutes'
                }), 202  # HTTP 202 Accepted
            
            # Start training in background
            logger.info(f"Model not found for {symbol}, starting background training")
            future = executor.submit(train_model_async, symbol)
            
            return jsonify({
                'status': 'training_started',
                'message': f'Model for {symbol} not found. Training has been started in the background.',
                'training_info': {
                    'status': 'training',
                    'progress': 0,
                    'started_at': datetime.now().isoformat(),
                    'message': 'Training initiated...'
                },
                'estimated_completion': '2-5 minutes',
                'next_steps': f'Please wait 2-5 minutes and call this endpoint again to get predictions for {symbol}.'
            }), 202  # HTTP 202 Accepted
        
        # Model exists, proceed with prediction
        # Get price data
        if request.method == 'GET':
            days = min(max(request.args.get('days', 30, type=int), 30), 90)
            
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
        if elapsed > 20:
            return jsonify({
                'error': 'Request timeout',
                'message': 'Request taking too long'
            }), 408
        
        # Make prediction with timeout
        remaining_time = max(5, 25 - elapsed)
        result, error = predict_with_timeout(predictor, prices, remaining_time)
        
        if result is None:
            return jsonify({
                'error': 'Prediction failed',
                'message': error or 'Unknown prediction error'
            }), 500
        
        # Add minimal metadata
        result.update({
            'status': 'success',
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

@app.route('/training-status/<symbol>', methods=['GET'])
def get_training_status(symbol):
    """
    Get training status for a specific symbol.
    """
    symbol = symbol.lower()
    
    if symbol in training_status:
        return jsonify({
            'symbol': symbol,
            'training_info': training_status[symbol]
        })
    elif model_exists(symbol):
        return jsonify({
            'symbol': symbol,
            'training_info': {
                'status': 'completed',
                'message': 'Model already exists and is ready for predictions'
            }
        })
    else:
        return jsonify({
            'symbol': symbol,
            'training_info': {
                'status': 'not_started',
                'message': 'No training has been initiated for this symbol'
            }
        })

@app.route('/train/<symbol>', methods=['POST'])
def train_model_endpoint(symbol):
    """
    Manual model training endpoint.
    """
    try:
        symbol = symbol.lower()
        
        if not symbol or not symbol.isalnum():
            return jsonify({
                'error': 'Invalid symbol'
            }), 400
        
        # Check if already training
        if symbol in training_status and training_status[symbol].get('status') == 'training':
            return jsonify({
                'message': f'Training already in progress for {symbol}',
                'training_info': training_status[symbol]
            }), 200
        
        data = request.get_json() or {}
        
        days = min(max(data.get('days', 90), 60), 180)
        epochs = min(max(data.get('epochs', 30), 20), 50)
        
        logger.info(f"Starting manual training for {symbol} with {days} days, {epochs} epochs")
        
        # Start training in background
        future = executor.submit(train_model_async, symbol, days, epochs)
        
        return jsonify({
            'message': f'Training started for {symbol}',
            'symbol': symbol,
            'training_parameters': {
                'days': days,
                'epochs': epochs
            },
            'training_info': training_status.get(symbol, {
                'status': 'training',
                'progress': 0,
                'started_at': datetime.now().isoformat()
            }),
            'estimated_completion': '2-5 minutes'
        })
        
    except Exception as e:
        logger.error(f"Training error for {symbol}: {str(e)}")
        return jsonify({
            'error': 'Training failed',
            'message': str(e)
        }), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List available models and training status quickly."""
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
            'training_status': training_status,
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
    logger.info("Starting Enhanced Crypto Prediction API with Auto-Training")
    app.run(host='0.0.0.0', port=5000, debug=False)