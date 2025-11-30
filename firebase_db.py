"""
Firebase Database Initialization

Handles Firebase Admin SDK initialization for both local development
and production (Render) environments.
"""

import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

def initialize_firebase():
    """
    Initialize Firebase Admin SDK. 
    
    Priority:
    1.  FIREBASE_CREDENTIALS env var (JSON string) - for production
    2. firebase-credentials.json file - for local development
    """
    
    # Check if already initialized
    if firebase_admin._apps:
        print("✅ Firebase already initialized")
        return firestore. client()
    
    try:
        # Option 1: Environment variable with JSON string (Production - Render)
        firebase_creds_str = os.getenv("FIREBASE_CREDENTIALS")
        
        if firebase_creds_str:
            print("🔥 Initializing Firebase from FIREBASE_CREDENTIALS environment variable...")
            # Parse the JSON string
            firebase_creds_dict = json.loads(firebase_creds_str)
            cred = credentials.Certificate(firebase_creds_dict)
            firebase_admin.initialize_app(cred)
            print("✅ Firebase initialized successfully (Production)")
        
        # Option 2: Local JSON file (Local development)
        elif os.path.exists("firebase-credentials.json"):
            print("🔥 Initializing Firebase from local firebase-credentials.json file...")
            cred = credentials.Certificate("firebase-credentials.json")
            firebase_admin.initialize_app(cred)
            print("✅ Firebase initialized successfully (Local)")
        
        else:
            print("❌ ERROR: No Firebase credentials found!")
            print("   Set FIREBASE_CREDENTIALS env var or add firebase-credentials.json")
            raise ValueError("Firebase credentials not configured")
        
        return firestore.client()
        
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON in FIREBASE_CREDENTIALS: {e}")
        raise
    except Exception as e:
        print(f"❌ ERROR: Firebase initialization failed: {e}")
        raise

# Initialize and export the db client
db = initialize_firebase()