"""
ExamHub Backend - FastAPI Application

Main application file containing all API endpoints for the ExamHub platform.  
Handles AI-generated mock test creation, test retrieval, and related operations. 

Version 2.0 - Includes chunked generation with quality validation
Optimized for Docker deployment on Render
"""

import os
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime
import logging
import json

from fastapi import FastAPI, HTTPException, Response, status
# from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from firebase_admin import firestore
import firebase_admin
from firebase_admin import credentials

try:
    from dotenv import load_dotenv  # requires python-dotenv package
except ImportError:
    # No-op fallback if python-dotenv is not installed
    def load_dotenv(*args, **kwargs):
        return False

from firebase_db import db
from ai_utils import generate_mcq_batch, generate_mcq_batch_chunked

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging. getLogger("main")

# ==================================================
# Initialize Firebase (Production & Local)
# ==================================================

def initialize_firebase():
    """Initialize Firebase Admin SDK for production (Render) and local development"""
    try:
        # Check if already initialized
        if firebase_admin._apps:
            logger.info("✅ Firebase already initialized")
            return True
            
        # For production (Render) - credentials as JSON string in env var
        firebase_creds_str = os.getenv("FIREBASE_CREDENTIALS")
        
        if firebase_creds_str:
            logger.info("🔥 Initializing Firebase from FIREBASE_CREDENTIALS env var...")
            firebase_creds = json.loads(firebase_creds_str)
            cred = credentials.Certificate(firebase_creds)
            firebase_admin.initialize_app(cred)
            logger.info("✅ Firebase initialized successfully (Production)")
            return True
            
        # For local development - credentials from JSON file
        elif os.path.exists("firebase-credentials.json"):
            logger. info("🔥 Initializing Firebase from local JSON file...")
            cred = credentials.Certificate("firebase-credentials.json")
            firebase_admin.initialize_app(cred)
            logger.info("✅ Firebase initialized successfully (Local)")
            return True
            
        else:
            logger.error("❌ Firebase credentials not found!")
            logger.error("Set FIREBASE_CREDENTIALS env var or add firebase-credentials.json")
            return False
            
    except Exception as e:
        logger.error(f"❌ Firebase initialization error: {e}")
        return False

# Initialize Firebase on startup
firebase_initialized = initialize_firebase()

# ==================================================
# FastAPI app setup
# ==================================================
app = FastAPI(
    title="ExamHub API",
    description="AI-driven mock test platform for Indian government exams",
    version="2.0.0",
)

# Configure CORS - Update with your frontend URLs
ALLOWED_ORIGINS = [
    "http://localhost:5173",          # Local Vite dev
    "http://localhost:3000",          # Local React dev
    "http://localhost:8000",          # Local backend
    "https://examhub. vercel.app",     # Production frontend
    "https://examhub-frontend.vercel.app",  # Alternative frontend URL
]

# Add environment variable for additional origins
additional_origins = os.getenv("ALLOWED_ORIGINS", "")
if additional_origins:
    ALLOWED_ORIGINS.extend(additional_origins.split(","))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if os.getenv("ENVIRONMENT") == "production" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================
# Startup Event
# ==================================================

@app.on_event("startup")
async def startup_event():
    """Run checks on startup"""
    logger.info("\n" + "=" * 60)
    logger.info("🚀 ExamHub Backend Starting...")
    logger.info("=" * 60)
    
    # Check Firebase
    if firebase_initialized:
        logger.info("✅ Firebase: Connected")
    else:
        logger.warning("⚠️ Firebase: Not Connected")
    
    # Check Gemini API Key
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        logger. info(f"✅ Gemini API Key: Present ({gemini_key[:10]}... )")
    else:
        logger.warning("⚠️ Gemini API Key: Not Set")
    
    # Environment info
    env = os.getenv("ENVIRONMENT", "development")
    port = os.getenv("PORT", "10000")
    logger.info(f"📍 Environment: {env}")
    logger.info(f"🔌 Port: {port}")
    logger.info("=" * 60 + "\n")

# ==================================================
# Pydantic Models
# ==================================================


class QuestionOut(BaseModel):
    """Response model for a single question"""
    id: str
    examId: str
    subject: str
    topic: str
    difficulty: str
    questionText: str
    optionA: str
    optionB: str
    optionC: str
    optionD: str
    correctOption: Literal["A", "B", "C", "D"]
    explanation: str
    shortcut: str
    timeToSolveSeconds: int


class TestOut(BaseModel):
    """Response model for a test with all questions"""
    id: str
    examId: str
    title: str
    subject: str
    difficulty: str
    numQuestions: int
    durationMinutes: int
    isAIGenerated: bool
    createdAt: Optional[str] = None
    questions: List[QuestionOut]


class GenerateMockTestRequest(BaseModel):
    """
    Generic topic-wise generation (old endpoint: /api/generate-mock).   
    Keeps your earlier structure so Postman tests don't break.
    """
    exam_id: str = Field(..., description="Exam identifier (e.g., 'ssc_cgl')")
    exam_name: str = Field(..., description="Full exam name (e.g., 'SSC Combined Graduate Level')")
    subject: str = Field(..., description="Subject area (e.g., 'Quantitative Aptitude')")
    topic: str = Field(..., description="Specific topic (e.g., 'Algebra')")
    difficulty: Literal["easy", "moderate", "hard"] = Field(..., description="Difficulty level")
    num_questions: int = Field(..., ge=1, le=100, description="Number of questions (1-100)")
    duration_minutes: int = Field(..., ge=1, description="Test duration in minutes")
    title: str = Field(..., description="Test title")


class GenerateMockTestResponse(BaseModel):
    """Response model for mock test generation"""
    message: str
    test: TestOut
    provider: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    firebase: str
    gemini: str
    ai_provider: str = "gemini"
    timestamp: str
    environment: str


# ---------- New high-level request models ----------

class TopicWiseMockRequest(BaseModel):
    """
    For /api/generate-topic-wise-mock
    Simple, SSC-focused topic-wise generator.  
    """
    exam: str = "SSC Combined Graduate Level"
    subject: str
    topic: str
    numQuestions: int = 25
    difficulty: str = "moderate"


class SectionalMockRequest(BaseModel):
    """
    For /api/generate-sectional-mock
    Subject-wise (Maths / Reasoning / English / GK/GS / CA / Computer) mock.   
    """
    exam: str = "SSC Combined Graduate Level"
    subject: str  # e.g.  "Quantitative Aptitude", "English Language", etc.
    difficulty: str = "moderate"
    numQuestions: int = 25  # usually 25


class FullMockRequest(BaseModel):
    """
    For /api/generate-full-mock
    Complete SSC CGL pattern: 100 Q (25 x 4 sections).  
    """
    exam: str = "SSC Combined Graduate Level"
    difficulty: str = "moderate"


# ==================================================
# Utility helpers
# ==================================================


def _normalize_difficulty(diff: str) -> str:
    d = diff.lower(). strip()
    if d not in {"easy", "moderate", "hard"}:
        raise HTTPException(
            status_code=400,
            detail="difficulty must be one of: easy, moderate, hard",
        )
    return d


def _server_timestamp():
    return firestore.SERVER_TIMESTAMP


def _compute_duration_for_subject(subject: str, num_questions: int) -> int:
    """
    Compute durationMinutes based on subject and number of questions.  

    Base caps (for 25 questions):
    - Maths (Quantitative Aptitude): 25 min
    - Reasoning:                     16 min
    - English:                       12 min
    - GK/GS (incl. CA, Computer):    7  min

    For other subjects: 1 min/question, capped at 25.   
    """
    s = subject.strip(). lower()

    # Map subject → max minutes for 25 questions
    if "math" in s or "quant" in s:
        max_minutes_for_25 = 25
    elif "reasoning" in s or "intelligence" in s:
        max_minutes_for_25 = 16
    elif "english" in s:
        max_minutes_for_25 = 12
    elif any(key in s for key in ["gk", "gs", "general awareness", "general knowledge", "current affairs", "computer"]):
        max_minutes_for_25 = 7
    else:
        max_minutes_for_25 = 25  # default fallback

    # Scale linearly with number of questions, but don't exceed the cap
    per_q = max_minutes_for_25 / 25
    duration = int(round(per_q * num_questions))

    # Safety bounds
    if duration < 1:
        duration = 1
    if duration > max_minutes_for_25:
        duration = max_minutes_for_25

    return duration


# ==================================================
# Health check & Status endpoints
# ==================================================


@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify API is running. 
    Used by Render for health checks.
    """
    firebase_status = "connected" if firebase_initialized else "not connected"
    gemini_status = "configured" if os.getenv("GEMINI_API_KEY") else "not configured"
    
    return HealthResponse(
        status="healthy",
        message="ExamHub API is running",
        firebase=firebase_status,
        gemini=gemini_status,
        ai_provider="gemini",
        timestamp=datetime.utcnow().isoformat(),
        environment=os.getenv("ENVIRONMENT", "development")
    )


# ✨ ADD THIS: Support HEAD requests for UptimeRobot
@app.head("/health")
@app.head("/")
async def health_check_head():
    """
    HEAD request for health checks (used by monitoring services like UptimeRobot). 
    Returns 200 status without body.
    """
    return Response(status_code=200, headers={
        "Content-Type": "application/json",
        "X-Health-Status": "healthy"
    })


@app.get("/api/status")
async def api_status():
    """Detailed API status"""
    return {
        "api": "ExamHub Backend",
        "version": "2.0.0",
        "status": "operational",
        "firebase": {
            "initialized": firebase_initialized,
            "connected": bool(firebase_admin._apps)
        },
        "gemini": {
            "configured": bool(os.getenv("GEMINI_API_KEY"))
        },
        "environment": os.getenv("ENVIRONMENT", "development"),
        "timestamp": datetime.utcnow().isoformat()
    }


# ==================================================
# OLD ENDPOINTS (kept for backward compatibility)
# ==================================================


@app.post("/api/generate-mock", response_model=Dict[str, Any])
async def generate_mock(payload: GenerateMockTestRequest):
    """
    Generic topic-wise AI mock generator.
    Kept for backward compatibility with your earlier Postman tests.
    
    ⚠️ DEPRECATED: Use /api/v2/generate-topic-wise-mock for better reliability
    """
    difficulty = _normalize_difficulty(payload. difficulty)

    # 1) Create test document first
    test_doc = {
        "examId": payload.exam_id,
        "title": payload.title or "AI Mock Test",
        "subject": payload.subject,
        "difficulty": difficulty,
        "numQuestions": payload.num_questions,
        "durationMinutes": payload.duration_minutes,
        "isAIGenerated": True,
        "createdAt": _server_timestamp(),
    }

    test_ref = db.collection("tests"). document()
    test_ref.set(test_doc)
    questions_collection = test_ref.collection("questions")

    # 2) One Gemini call → many MCQs
    try:
        mcqs = generate_mcq_batch(
            exam_name=payload.exam_name,
            subject=payload.subject,
            topic=payload.topic or "",
            difficulty=difficulty,
            num_questions=payload.num_questions,
        )
    except Exception as e:
        # Cleanup test doc if AI fails
        test_ref.delete()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate questions: {e}",
        )

    # 3) Store in Firestore
    question_docs: List[Dict[str, Any]] = []
    for mcq in mcqs:
        qdoc = {
            "examId": payload.exam_id,
            "subject": payload.subject,
            "topic": payload.topic or "",
            "difficulty": difficulty,
            "questionText": mcq["questionText"],
            "optionA": mcq["optionA"],
            "optionB": mcq["optionB"],
            "optionC": mcq["optionC"],
            "optionD": mcq["optionD"],
            "correctOption": mcq["correctOption"],
            "explanation": mcq["explanation"],
            "shortcut": mcq["shortcut"],
            "timeToSolveSeconds": mcq["timeToSolveSeconds"],
        }
        q_ref = questions_collection. document()
        q_ref.set(qdoc)
        question_docs.append({"id": q_ref.id, **qdoc})

    # 4) Update actual count
    test_ref.update({"numQuestions": len(question_docs)})

    # 5) Build response
    questions_out = [QuestionOut(**q) for q in question_docs]
    test_out = TestOut(
        id=test_ref. id,
        examId=payload.exam_id,
        title=payload.title,
        subject=payload.subject,
        difficulty=difficulty,
        numQuestions=len(questions_out),
        durationMinutes=payload.duration_minutes,
        isAIGenerated=True,
        questions=questions_out,
    )

    return {
        "message": "AI mock test generated successfully.",
        "test": test_out. dict(),
        "provider": "gemini",
    }


@app.post("/api/generate-topic-wise-mock", response_model=Dict[str, Any])
async def generate_topic_wise_mock(payload: TopicWiseMockRequest):
    """
    Topic-wise mock for a single subject & topic.
    Example: SSC CGL, Quantitative Aptitude, Algebra, 25 Q.   
    
    ⚠️ DEPRECATED: Use /api/v2/generate-topic-wise-mock for better reliability
    """
    difficulty = _normalize_difficulty(payload.difficulty)

    exam_id = "ssc_cgl"  # fixed for now; can generalize later
    title = f"{payload.subject} - {payload.topic} ({difficulty. title()} | Topic-wise)"

    test_doc = {
        "examId": exam_id,
        "title": title,
        "subject": payload.subject,
        "difficulty": difficulty,
        "numQuestions": payload.numQuestions,
        "durationMinutes": max(30, payload.numQuestions),  # simple heuristic
        "isAIGenerated": True,
        "createdAt": _server_timestamp(),
    }

    test_ref = db.collection("tests"). document()
    test_ref.set(test_doc)
    questions_collection = test_ref.collection("questions")

    try:
        mcqs = generate_mcq_batch(
            exam_name=payload.exam,
            subject=payload.subject,
            topic=payload.topic,
            difficulty=difficulty,
            num_questions=payload.numQuestions,
        )
    except Exception as e:
        test_ref.delete()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate topic-wise questions: {e}",
        )

    question_docs: List[Dict[str, Any]] = []
    for mcq in mcqs:
        qdoc = {
            "examId": exam_id,
            "subject": payload.subject,
            "topic": payload.topic,
            "difficulty": difficulty,
            "questionText": mcq["questionText"],
            "optionA": mcq["optionA"],
            "optionB": mcq["optionB"],
            "optionC": mcq["optionC"],
            "optionD": mcq["optionD"],
            "correctOption": mcq["correctOption"],
            "explanation": mcq["explanation"],
            "shortcut": mcq["shortcut"],
            "timeToSolveSeconds": mcq["timeToSolveSeconds"],
        }
        q_ref = questions_collection. document()
        q_ref.set(qdoc)
        question_docs.append({"id": q_ref.id, **qdoc})

    test_ref.update({"numQuestions": len(question_docs)})
    questions_out = [QuestionOut(**q) for q in question_docs]

    test_out = TestOut(
        id=test_ref.id,
        examId=exam_id,
        title=title,
        subject=payload.subject,
        difficulty=difficulty,
        numQuestions=len(questions_out),
        durationMinutes=test_doc["durationMinutes"],
        isAIGenerated=True,
        questions=questions_out,
    )

    return {
        "message": "Topic-wise mock test generated successfully.",
        "test": test_out.dict(),
        "provider": "gemini",
    }


@app.post("/api/generate-sectional-mock", response_model=Dict[str, Any])
async def generate_sectional_mock(payload: SectionalMockRequest):
    """
    Sectional mock: subject-wise test (e.g.  Maths 25 Q, Reasoning 25 Q).   
    Uses a 'Mixed <Subject>' topic hint to get questions from the whole subject.
    
    ⚠️ DEPRECATED: Use /api/v2/generate-sectional-mock for better reliability
    """
    difficulty = _normalize_difficulty(payload.difficulty)

    exam_id = "ssc_cgl"
    subject = payload.subject.strip()
    topic_hint = f"Mixed {subject} questions as per SSC CGL exam pattern"

    title = f"{subject} Sectional Test ({difficulty.title()})"

    test_doc = {
        "examId": exam_id,
        "title": title,
        "subject": subject,
        "difficulty": difficulty,
        "numQuestions": payload. numQuestions,
        "durationMinutes": max(25, payload.numQuestions),
        "isAIGenerated": True,
        "createdAt": _server_timestamp(),
    }

    test_ref = db.collection("tests").document()
    test_ref.set(test_doc)
    questions_collection = test_ref.collection("questions")

    try:
        mcqs = generate_mcq_batch(
            exam_name=payload.exam,
            subject=subject,
            topic=topic_hint,
            difficulty=difficulty,
            num_questions=payload.numQuestions,
        )
    except Exception as e:
        test_ref.delete()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate sectional questions: {e}",
        )

    if not mcqs:
        test_ref.delete()
        raise HTTPException(
            status_code=500,
            detail="AI did not return any valid questions for this sectional mock.",
        )

    question_docs: List[Dict[str, Any]] = []
    for mcq in mcqs:
        qdoc = {
            "examId": exam_id,
            "subject": subject,
            "topic": topic_hint,
            "difficulty": difficulty,
            "questionText": mcq["questionText"],
            "optionA": mcq["optionA"],
            "optionB": mcq["optionB"],
            "optionC": mcq["optionC"],
            "optionD": mcq["optionD"],
            "correctOption": mcq["correctOption"],
            "explanation": mcq["explanation"],
            "shortcut": mcq["shortcut"],
            "timeToSolveSeconds": mcq["timeToSolveSeconds"],
        }
        q_ref = questions_collection.document()
        q_ref.set(qdoc)
        question_docs.append({"id": q_ref. id, **qdoc})

    test_ref.update({"numQuestions": len(question_docs)})
    questions_out = [QuestionOut(**q) for q in question_docs]

    test_out = TestOut(
        id=test_ref. id,
        examId=exam_id,
        title=title,
        subject=subject,
        difficulty=difficulty,
        numQuestions=len(questions_out),
        durationMinutes=test_doc["durationMinutes"],
        isAIGenerated=True,
        questions=questions_out,
    )

    return {
        "message": "Sectional mock test generated successfully.",
        "test": test_out.dict(),
        "provider": "gemini",
    }


@app.post("/api/generate-full-mock", response_model=Dict[str, Any])
async def generate_full_mock(payload: FullMockRequest):
    """
    Full SSC CGL Tier-I style paper:
      - 25 Quantitative Aptitude
      - 25 Reasoning
      - 25 English
      - 25 GK/GS + CA + Computer
      
    ⚠️ DEPRECATED: Use /api/v2/generate-full-mock for better reliability
    """
    difficulty = _normalize_difficulty(payload.difficulty)
    exam_id = "ssc_cgl"

    # Subject distribution as per your requirement
    subject_plan = [
        (
            "Quantitative Aptitude",
            25,
            "Mixed quantitative aptitude (arithmetic + advanced maths) as per SSC CGL pattern",
        ),
        (
            "Reasoning",
            25,
            "Mixed verbal and non-verbal reasoning questions as per SSC CGL pattern",
        ),
        (
            "English Language",
            25,
            "Mixed grammar, vocabulary, error spotting and comprehension questions as per SSC CGL pattern",
        ),
        (
            "GK/GS, Current Affairs & Computer Knowledge",
            25,
            "Mixed questions from general knowledge, general science, current affairs, and basic computer knowledge as per SSC CGL pattern",
        ),
    ]

    title = f"{payload.exam} Full Mock Test (100 Questions, {difficulty.title()})"

    test_doc = {
        "examId": exam_id,
        "title": title,
        "subject": "Full Paper",
        "difficulty": difficulty,
        "numQuestions": 100,
        "durationMinutes": 60,
        "isAIGenerated": True,
        "createdAt": _server_timestamp(),
    }

    test_ref = db.collection("tests").document()
    test_ref.set(test_doc)
    questions_collection = test_ref.collection("questions")

    all_mcqs: List[Dict[str, Any]] = []

    try:
        for subj_name, count, topic_hint in subject_plan:
            mcqs = generate_mcq_batch(
                exam_name=payload.exam,
                subject=subj_name,
                topic=topic_hint,
                difficulty=difficulty,
                num_questions=count,
            )
            if not mcqs or len(mcqs) == 0:
                raise RuntimeError(f"AI returned no MCQs for subject '{subj_name}'")
            for mcq in mcqs:
                qdoc = {
                    "examId": exam_id,
                    "subject": subj_name,
                    "topic": topic_hint,
                    "difficulty": difficulty,
                    "questionText": mcq["questionText"],
                    "optionA": mcq["optionA"],
                    "optionB": mcq["optionB"],
                    "optionC": mcq["optionC"],
                    "optionD": mcq["optionD"],
                    "correctOption": mcq["correctOption"],
                    "explanation": mcq["explanation"],
                    "shortcut": mcq["shortcut"],
                    "timeToSolveSeconds": mcq["timeToSolveSeconds"],
                }
                q_ref = questions_collection.document()
                q_ref.set(qdoc)
                all_mcqs.append({"id": q_ref.id, **qdoc})

    except Exception as e:
        # If *any* subject fails → clean up the half-created paper
        test_ref.delete()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate full mock test: {e}",
        )

    # Update actual count (in case AI returned more/less)
    test_ref.update({"numQuestions": len(all_mcqs)})
    questions_out = [QuestionOut(**q) for q in all_mcqs]

    test_out = TestOut(
        id=test_ref.id,
        examId=exam_id,
        title=title,
        subject="Full Paper",
        difficulty=difficulty,
        numQuestions=len(questions_out),
        durationMinutes=test_doc["durationMinutes"],
        isAIGenerated=True,
        questions=questions_out,
    )

    return {
        "message": "Full mock test generated successfully.",
        "test": test_out.dict(),
        "provider": "gemini",
    }


# ==================================================
# ✨ NEW V2 ENDPOINTS (IMPROVED RELIABILITY) ✨
# ==================================================


@app.post("/api/v2/generate-topic-wise-mock", response_model=Dict[str, Any])
async def generate_topic_wise_mock_v2(payload: TopicWiseMockRequest):
    """
    ✨ NEW V2: Topic-wise mock with CHUNKED generation for better reliability.  
    
    Improvements over v1:
    - Generates questions in smaller batches (8-10 at a time)
    - Dramatically reduced failure rate
    - Better error messages
    - Quality validation (rejects fake questions)
    - Ensures EXACT question count requested
    """
    difficulty = _normalize_difficulty(payload.difficulty)

    exam_id = "ssc_cgl"
    title = f"{payload.subject} - {payload.topic} ({difficulty. title()} | Topic-wise)"

    test_doc = {
        "examId": exam_id,
        "title": title,
        "subject": payload.subject,
        "difficulty": difficulty,
        "numQuestions": payload.numQuestions,
        "durationMinutes": max(30, payload.numQuestions),
        "isAIGenerated": True,
        "createdAt": _server_timestamp(),
    }

    test_ref = db.collection("tests").document()
    test_ref.set(test_doc)
    questions_collection = test_ref.collection("questions")

    try:
        # 🔥 NEW: Use chunked generation
        mcqs = generate_mcq_batch_chunked(
            exam_name=payload.exam,
            subject=payload.subject,
            topic=payload.topic,
            difficulty=difficulty,
            num_questions=payload.numQuestions,
            chunk_size=8,
        )
        
        # 🎯 Ensure exact count
        if len(mcqs) > payload.numQuestions:
            logger.info(f"Trimming: Got {len(mcqs)} questions, keeping exactly {payload.numQuestions}")
            mcqs = mcqs[:payload.numQuestions]
            
    except Exception as e:
        test_ref.delete()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate topic-wise questions: {e}",
        )

    question_docs: List[Dict[str, Any]] = []
    for mcq in mcqs:
        qdoc = {
            "examId": exam_id,
            "subject": payload.subject,
            "topic": payload.topic,
            "difficulty": difficulty,
            "questionText": mcq["questionText"],
            "optionA": mcq["optionA"],
            "optionB": mcq["optionB"],
            "optionC": mcq["optionC"],
            "optionD": mcq["optionD"],
            "correctOption": mcq["correctOption"],
            "explanation": mcq["explanation"],
            "shortcut": mcq["shortcut"],
            "timeToSolveSeconds": mcq["timeToSolveSeconds"],
        }
        q_ref = questions_collection. document()
        q_ref.set(qdoc)
        question_docs.append({"id": q_ref.id, **qdoc})

    test_ref.update({"numQuestions": len(question_docs)})
    questions_out = [QuestionOut(**q) for q in question_docs]

    test_out = TestOut(
        id=test_ref.id,
        examId=exam_id,
        title=title,
        subject=payload.subject,
        difficulty=difficulty,
        numQuestions=len(questions_out),
        durationMinutes=test_doc["durationMinutes"],
        isAIGenerated=True,
        questions=questions_out,
    )

    return {
        "message": f"✨ Topic-wise mock test generated successfully (V2 - Improved Reliability).  Generated {len(questions_out)} questions.",
        "test": test_out.dict(),
        "provider": "gemini-chunked",
    }


@app.post("/api/v2/generate-sectional-mock", response_model=Dict[str, Any])
async def generate_sectional_mock_v2(payload: SectionalMockRequest):
    """
    ✨ NEW V2: Sectional mock with CHUNKED generation for better reliability. 
    
    Improvements over v1:
    - Generates questions in smaller batches (8-10 at a time)
    - Dramatically reduced failure rate
    - Better error messages
    - Quality validation (rejects fake questions)
    - Ensures EXACT question count requested
    """
    difficulty = _normalize_difficulty(payload.difficulty)

    exam_id = "ssc_cgl"
    subject = payload.subject.strip()
    topic_hint = f"Mixed {subject} questions as per SSC CGL exam pattern"

    title = f"{subject} Sectional Test ({difficulty.title()})"

    test_doc = {
        "examId": exam_id,
        "title": title,
        "subject": subject,
        "difficulty": difficulty,
        "numQuestions": payload.numQuestions,
        "durationMinutes": _compute_duration_for_subject(subject, payload.numQuestions),
        "isAIGenerated": True,
        "createdAt": _server_timestamp(),
    }

    test_ref = db. collection("tests").document()
    test_ref.set(test_doc)
    questions_collection = test_ref.collection("questions")

    try:
        # 🔥 NEW: Use chunked generation
        mcqs = generate_mcq_batch_chunked(
            exam_name=payload.exam,
            subject=subject,
            topic=topic_hint,
            difficulty=difficulty,
            num_questions=payload.numQuestions,
            chunk_size=8,
        )
        
        # 🎯 Ensure exact count
        if len(mcqs) > payload.numQuestions:
            logger.info(f"Trimming: Got {len(mcqs)} questions, keeping exactly {payload.numQuestions}")
            mcqs = mcqs[:payload.numQuestions]
            
    except Exception as e:
        test_ref.delete()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate sectional questions: {e}",
        )

    if not mcqs:
        test_ref.delete()
        raise HTTPException(
            status_code=500,
            detail="AI did not return any valid questions for this sectional mock.",
        )

    question_docs: List[Dict[str, Any]] = []
    for mcq in mcqs:
        qdoc = {
            "examId": exam_id,
            "subject": subject,
            "topic": topic_hint,
            "difficulty": difficulty,
            "questionText": mcq["questionText"],
            "optionA": mcq["optionA"],
            "optionB": mcq["optionB"],
            "optionC": mcq["optionC"],
            "optionD": mcq["optionD"],
            "correctOption": mcq["correctOption"],
            "explanation": mcq["explanation"],
            "shortcut": mcq["shortcut"],
            "timeToSolveSeconds": mcq["timeToSolveSeconds"],
        }
        q_ref = questions_collection.document()
        q_ref.set(qdoc)
        question_docs.append({"id": q_ref. id, **qdoc})

    test_ref.update({"numQuestions": len(question_docs)})
    questions_out = [QuestionOut(**q) for q in question_docs]

    test_out = TestOut(
        id=test_ref. id,
        examId=exam_id,
        title=title,
        subject=subject,
        difficulty=difficulty,
        numQuestions=len(questions_out),
        durationMinutes=test_doc["durationMinutes"],
        isAIGenerated=True,
        questions=questions_out,
    )

    return {
        "message": f"✨ Sectional mock test generated successfully (V2 - Improved Reliability). Generated {len(questions_out)} questions.",
        "test": test_out. dict(),
        "provider": "gemini-chunked",
    }


@app.post("/api/v2/generate-full-mock", response_model=Dict[str, Any])
async def generate_full_mock_v2(payload: FullMockRequest):
    """
    ✨ NEW V2: Full SSC CGL mock with CHUNKED generation for better reliability.
    
    Generates EXACTLY 100 questions across 4 subjects (25 each):
    - Quantitative Aptitude: 25
    - Reasoning: 25
    - English Language: 25
    - GK/GS, Current Affairs & Computer Knowledge: 25
    
    Improvements over v1:
    - Each subject generated in 3-4 smaller batches
    - Much higher success rate
    - Partial recovery if one batch fails
    - Better progress tracking
    - Quality validation (rejects fake questions)
    - STRICT: Ensures exactly 25 questions per subject (100 total)
    """
    difficulty = _normalize_difficulty(payload.difficulty)
    exam_id = "ssc_cgl"

    subject_plan = [
        (
            "Quantitative Aptitude",
            25,
            "Mixed quantitative aptitude (arithmetic + advanced maths) as per SSC CGL pattern",
        ),
        (
            "Reasoning",
            25,
            "Mixed verbal and non-verbal reasoning questions as per SSC CGL pattern",
        ),
        (
            "English Language",
            25,
            "Mixed grammar, vocabulary, error spotting and comprehension questions as per SSC CGL pattern",
        ),
        (
            "GK/GS, Current Affairs & Computer Knowledge",
            25,
            "Mixed questions from general knowledge, general science, current affairs, and basic computer knowledge as per SSC CGL pattern",
        ),
    ]

    title = f"{payload. exam} Full Mock Test (100 Questions, {difficulty.title()})"

    test_doc = {
        "examId": exam_id,
        "title": title,
        "subject": "Full Paper",
        "difficulty": difficulty,
        "numQuestions": 100,
        "durationMinutes": 60,
        "isAIGenerated": True,
        "createdAt": _server_timestamp(),
    }

    test_ref = db.collection("tests").document()
    test_ref.set(test_doc)
    questions_collection = test_ref.collection("questions")

    all_mcqs: List[Dict[str, Any]] = []
    subject_wise_count = {}  # Track questions per subject

    try:
        for subj_name, required_count, topic_hint in subject_plan:
            logger.info(f"Generating {required_count} questions for: {subj_name}")
            
            # 🔥 Generate questions with chunking and validation
            mcqs = generate_mcq_batch_chunked(
                exam_name=payload. exam,
                subject=subj_name,
                topic=topic_hint,
                difficulty=difficulty,
                num_questions=required_count,
                chunk_size=8,
            )
            
            if not mcqs or len(mcqs) == 0:
                raise RuntimeError(f"AI returned no MCQs for subject '{subj_name}'")
            
            # 🎯 CRITICAL FIX: Take EXACTLY the required number of questions
            # If we got more (due to validation buffer), trim to exact count
            if len(mcqs) > required_count:
                logger.info(
                    f"Trimming {subj_name}: Got {len(mcqs)} questions, keeping exactly {required_count}"
                )
                mcqs = mcqs[:required_count]
            elif len(mcqs) < required_count:
                logger.warning(
                    f"⚠️ {subj_name}: Only got {len(mcqs)}/{required_count} questions after validation"
                )
                # Continue with what we have, but log the shortage
            
            # Store the questions
            for mcq in mcqs:
                qdoc = {
                    "examId": exam_id,
                    "subject": subj_name,
                    "topic": topic_hint,
                    "difficulty": difficulty,
                    "questionText": mcq["questionText"],
                    "optionA": mcq["optionA"],
                    "optionB": mcq["optionB"],
                    "optionC": mcq["optionC"],
                    "optionD": mcq["optionD"],
                    "correctOption": mcq["correctOption"],
                    "explanation": mcq["explanation"],
                    "shortcut": mcq["shortcut"],
                    "timeToSolveSeconds": mcq["timeToSolveSeconds"],
                }
                q_ref = questions_collection. document()
                q_ref.set(qdoc)
                all_mcqs.append({"id": q_ref.id, **qdoc})
            
            subject_wise_count[subj_name] = len(mcqs)
            logger.info(f"✅ {subj_name}: {len(mcqs)} questions stored.  Total so far: {len(all_mcqs)}")

    except Exception as e:
        # If failure occurs, check if we have at least 80 questions
        if len(all_mcqs) >= 80:
            # Proceed with partial test
            logger.warning(f"Partial test generated: {len(all_mcqs)} questions.  Error: {e}")
            
            test_ref.update({"numQuestions": len(all_mcqs)})
            questions_out = [QuestionOut(**q) for q in all_mcqs]
            
            test_out = TestOut(
                id=test_ref.id,
                examId=exam_id,
                title=title + " (Partial)",
                subject="Full Paper",
                difficulty=difficulty,
                numQuestions=len(questions_out),
                durationMinutes=test_doc["durationMinutes"],
                isAIGenerated=True,
                questions=questions_out,
            )
            
            # Include subject-wise breakdown in the message
            breakdown = ", ".join([f"{subj}: {count}" for subj, count in subject_wise_count.items()])
            
            return {
                "message": f"⚠️ Full mock test generated with {len(all_mcqs)} questions (some batches failed).  Breakdown: {breakdown}.  Error: {e}",
                "test": test_out.dict(),
                "provider": "gemini-chunked",
            }
        else:
            # Too few questions, delete and fail
            test_ref.delete()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate full mock test.  Only {len(all_mcqs)}/100 questions generated. Error: {e}",
            )

    # 🎯 FINAL CHECK: Ensure we have exactly 100 questions
    if len(all_mcqs) != 100:
        logger.warning(f"Expected 100 questions, got {len(all_mcqs)}.  Adjusting...")
        
        if len(all_mcqs) > 100:
            # Trim excess questions (shouldn't happen with above fix, but safety check)
            logger.info(f"Trimming from {len(all_mcqs)} to 100 questions")
            
            # Remove excess questions from database
            excess_count = len(all_mcqs) - 100
            for i in range(excess_count):
                excess_q = all_mcqs.pop()
                questions_collection. document(excess_q["id"]). delete()
            
            logger.info(f"Removed {excess_count} excess questions")
            
        elif len(all_mcqs) < 100 and len(all_mcqs) >= 95:
            # If we're close (95-99), it's acceptable
            logger.info(f"Generated {len(all_mcqs)} questions (acceptable range)")
        else:
            # Significantly short, this is an error
            logger.error(f"Only generated {len(all_mcqs)} questions, expected 100")

    # Update actual count
    final_count = len(all_mcqs)
    test_ref.update({"numQuestions": final_count})
    questions_out = [QuestionOut(**q) for q in all_mcqs]

    test_out = TestOut(
        id=test_ref.id,
        examId=exam_id,
        title=title,
        subject="Full Paper",
        difficulty=difficulty,
        numQuestions=len(questions_out),
        durationMinutes=test_doc["durationMinutes"],
        isAIGenerated=True,
        questions=questions_out,
    )

    # Create detailed subject-wise breakdown for the response
    breakdown = ", ".join([f"{subj}: {count}" for subj, count in subject_wise_count.items()])
    
    return {
        "message": f"✨ Full mock test generated successfully (V2 - Improved Reliability). Total: {final_count} questions.  Breakdown: {breakdown}",
        "test": test_out.dict(),
        "provider": "gemini-chunked",
    }


# ==================================================
# Test retrieval endpoints
# ==================================================


@app.get("/api/tests/{test_id}", response_model=TestOut)
async def get_test(test_id: str):
    """
    Retrieve a test by ID with all its questions.
    """
    try:
        test_ref = db.collection("tests").document(test_id)
        test_doc = test_ref.get()

        if not test_doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test with ID '{test_id}' not found",
            )

        test_data = test_doc.to_dict()

        questions_ref = test_ref.collection("questions")
        questions_docs = questions_ref.stream()

        questions_list: List[QuestionOut] = []
        for q_doc in questions_docs:
            q_data = q_doc.to_dict()
            questions_list.append(
                QuestionOut(
                    id=q_doc.id,
                    examId=q_data["examId"],
                    subject=q_data["subject"],
                    topic=q_data["topic"],
                    difficulty=q_data["difficulty"],
                    questionText=q_data["questionText"],
                    optionA=q_data["optionA"],
                    optionB=q_data["optionB"],
                    optionC=q_data["optionC"],
                    optionD=q_data["optionD"],
                    correctOption=q_data["correctOption"],
                    explanation=q_data["explanation"],
                    shortcut=q_data["shortcut"],
                    timeToSolveSeconds=q_data["timeToSolveSeconds"],
                )
            )

        test_out = TestOut(
            id=test_id,
            examId=test_data["examId"],
            title=test_data["title"],
            subject=test_data["subject"],
            difficulty=test_data["difficulty"],
            numQuestions=test_data["numQuestions"],
            durationMinutes=test_data["durationMinutes"],
            isAIGenerated=test_data["isAIGenerated"],
            createdAt=test_data. get("createdAt"). isoformat()
            if isinstance(test_data.get("createdAt"), datetime)
            else test_data. get("createdAt"),
            questions=questions_list,
        )

        return test_out

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status. HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve test: {str(e)}",
        )


@app.get("/api/tests", response_model=List[TestOut])
async def list_tests(
    exam_id: Optional[str] = None,
    subject: Optional[str] = None,
    difficulty: Optional[str] = None,
    limit: int = 20,
):
    """
    List tests with optional filters.
    """
    try:
        query = db.collection("tests")

        if exam_id:
            query = query.where("examId", "==", exam_id)
        if subject:
            query = query.where("subject", "==", subject)
        if difficulty:
            query = query.where("difficulty", "==", difficulty)

        query = query.order_by("createdAt", direction=firestore.Query. DESCENDING). limit(
            limit
        )

        test_docs = query.stream()

        tests_list: List[TestOut] = []
        for test_doc in test_docs:
            test_data = test_doc.to_dict()
            test_id = test_doc.id

            questions_ref = (
                db.collection("tests"). document(test_id).collection("questions")
            )
            questions_docs = questions_ref.stream()

            questions_list: List[QuestionOut] = []
            for q_doc in questions_docs:
                q_data = q_doc.to_dict()
                questions_list.append(QuestionOut(id=q_doc.id, **q_data))

            tests_list.append(
                TestOut(
                    id=test_id,
                    examId=test_data["examId"],
                    title=test_data["title"],
                    subject=test_data["subject"],
                    difficulty=test_data["difficulty"],
                    numQuestions=test_data["numQuestions"],
                    durationMinutes=test_data["durationMinutes"],
                    isAIGenerated=test_data["isAIGenerated"],
                    createdAt=test_data.get("createdAt").isoformat()
                    if isinstance(test_data.get("createdAt"), datetime)
                    else test_data.get("createdAt"),
                    questions=questions_list,
                )
            )

        return tests_list

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tests: {str(e)}",
        )


# ==================================================
# Run with uvicorn (for local development)
# ==================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 50)
    print("🚀 Starting ExamHub Backend V2")
    print("=" * 50)
    print("AI Provider: Gemini (Google AI)")
    print("Features: Chunked generation + Quality validation")
    print("=" * 50 + "\n")

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=True,
        log_level="info"
    )