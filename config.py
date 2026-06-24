# config.py
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = 'AIzaSyBBdhceI2Qg9YxjsV62dcoLsTZ3GyeqInI'

DATA_ROOT = "./data/FB15k_237"

MODEL_NAME = "gemini-3-flash-preview"
TEMPERATURE = 1
TIMEOUT = 200
MAX_RETRIES = 0

NUM_SAMPLES = 10
SLEEP_SEC = 0.5
MAX_EVIDENCE_TRIPLES = None