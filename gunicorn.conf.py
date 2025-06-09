# gunicorn.conf.py - Optimized for crypto prediction API

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes
workers = 1  # Single worker to avoid memory issues
worker_class = "sync"
worker_connections = 100
max_requests = 500  # Restart worker after 500 requests to prevent memory leaks
max_requests_jitter = 50
timeout = 45  # Increased from 30 to 45 seconds
keepalive = 2

# Memory management
preload_app = False  # Don't preload to save memory
max_worker_memory = 400  # MB - restart if worker uses more

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "crypto_predictor_api"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Performance tuning
worker_tmp_dir = "/dev/shm"  # Use shared memory for better performance
tmp_upload_dir = None

# Graceful handling
graceful_timeout = 60
preload_app = False

def when_ready(server):
    server.log.info("Crypto Predictor API is ready to serve requests")

def worker_int(worker):
    worker.log.info("Worker received INT or QUIT signal")

def pre_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)
    
def worker_abort(worker):
    worker.log.info("Worker received SIGABRT signal")