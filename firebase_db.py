"""
Firebase Firestore Database Initialization Module

This module initializes the Firebase Admin SDK and provides a Firestore client
for database operations throughout the application.
"""

import os
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the path to service account key from environment
service_account_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'serviceAccountKey.json')

# Initialize Firebase Admin SDK
try:
    cred = credentials.Certificate(service_account_path)
    firebase_admin.initialize_app(cred)
    print(f"✓ Firebase initialized successfully using {service_account_path}")
except Exception as e:
    print(f"✗ Firebase initialization failed: {e}")
    raise

# Initialize Firestore client
db = firestore.client()

print("✓ Firestore client ready")
