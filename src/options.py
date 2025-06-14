import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'garbage-classification')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'garbage-classification')