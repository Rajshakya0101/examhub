# ExamHub Backend

> **AI-generated mock tests — never run out of fresh practice**

ExamHub is an AI-driven mock test platform for Indian government exams (SSC CGL, CHSL, MTS, CPO). This repository contains the **backend API** built with FastAPI, Firestore, and Google Gemini AI.

---

## 🎯 Features

- **AI-Powered Question Generation**: Generate unique MCQs using Google Gemini (1.5-pro)
- **Flexible Mock Tests**: Customize by exam, subject, topic, difficulty, and question count
- **Cloud Database**: All tests and questions stored in Google Firestore
- **REST API**: Clean FastAPI endpoints with auto-generated docs
- **High Quality**: Gemini 1.5 Pro for accurate, exam-relevant questions

---

## 🏭️ Architecture

```
Backend (FastAPI + Uvicorn)
    ↓
Google Gemini AI
    ↓
Google Firestore (Database)
```

### Project Structure

```
examhub-backend/
├── .env                      # Environment configuration (not in git)
├── .env.example              # Example environment file
├── .gitignore               # Git ignore rules
├── requirements.txt          # Python dependencies
├── serviceAccountKey.json    # Firebase credentials (not in git)
├── firebase_db.py           # Firestore initialization
├── ai_utils.py              # AI integration & question generation
├── main.py                  # FastAPI app & routes
└── README.md                # This file
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- **Firebase Project** with Firestore enabled
- **Gemini API Key** (from Google AI Studio)

### Step 1: Clone & Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd examhub-backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Firebase Setup

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Create a new project or use existing one
3. Enable **Firestore Database**
4. Navigate to **Project Settings → Service Accounts**
5. Click **Generate New Private Key**
6. Save the JSON file as `serviceAccountKey.json` in the project root

### Step 3: Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your credentials
```

**`.env` file contents:**

```env
# Firebase Configuration
GOOGLE_APPLICATION_CREDENTIALS=serviceAccountKey.json

# Gemini API Key (get from https://makersuite.google.com/app/apikey)
GEMINI_API_KEY=your_gemini_api_key_here
```

### Step 4: Run the Server

```bash
# Start the development server
python main.py
```

The API will be available at:
- **Base URL**: `http://localhost:8000`
- **API Docs**: `http://localhost:8000/docs` (Swagger UI)
- **ReDoc**: `http://localhost:8000/redoc`

---

## 📡 API Endpoints

### 1. Health Check

```http
GET /
```

**Response:**
```json
{
  "status": "healthy",
  "message": "ExamHub API is running",
  "ai_provider": "gemini",
  "timestamp": "2025-11-25T10:30:00.000Z"
}
```

---

### 2. Generate Mock Test

```http
POST /api/generate-mock
Content-Type: application/json
```

**Request Body:**
```json
{
  "exam_id": "ssc_cgl",
  "exam_name": "SSC Combined Graduate Level",
  "subject": "Quantitative Aptitude",
  "topic": "Algebra",
  "difficulty": "moderate",
  "num_questions": 10,
  "duration_minutes": 30,
  "title": "SSC CGL Quant Algebra Mock 1"
}
```

**Response:**
```json
{
  "message": "AI mock test generated successfully.",
  "test": {
    "id": "abc123xyz",
    "examId": "ssc_cgl",
    "title": "SSC CGL Quant Algebra Mock 1",
    "subject": "Quantitative Aptitude",
    "difficulty": "moderate",
    "numQuestions": 10,
    "durationMinutes": 30,
    "isAIGenerated": true,
    "createdAt": "2025-11-25T10:30:00.000Z",
    "questions": [
      {
        "id": "q1",
        "examId": "ssc_cgl",
        "subject": "Quantitative Aptitude",
        "topic": "Algebra",
        "difficulty": "moderate",
        "questionText": "If x + y = 10 and xy = 21, find x² + y²",
        "optionA": "58",
        "optionB": "68",
        "optionC": "78",
        "optionD": "88",
        "correctOption": "A",
        "explanation": "We know (x+y)² = x² + y² + 2xy...",
        "shortcut": "Use (x+y)² - 2xy formula",
        "timeToSolveSeconds": 60
      }
      // ... more questions
    ]
  },
  "provider": "gemini"
}
```

**Parameters:**
- `exam_id`: Exam identifier (e.g., `ssc_cgl`, `ssc_chsl`)
- `exam_name`: Full exam name
- `subject`: Subject area (e.g., "Quantitative Aptitude", "General Awareness")
- `topic`: Specific topic within subject
- `difficulty`: One of `"easy"`, `"moderate"`, `"hard"`
- `num_questions`: Number of questions (1-100)
- `duration_minutes`: Test duration in minutes
- `title`: Display title for the test

---

### 3. Get Test by ID

```http
GET /api/tests/{test_id}
```

**Response:**
```json
{
  "id": "abc123xyz",
  "examId": "ssc_cgl",
  "title": "SSC CGL Quant Algebra Mock 1",
  "subject": "Quantitative Aptitude",
  "difficulty": "moderate",
  "numQuestions": 10,
  "durationMinutes": 30,
  "isAIGenerated": true,
  "createdAt": "2025-11-25T10:30:00.000Z",
  "questions": [...]
}
```

---

### 4. List Tests (with filters)

```http
GET /api/tests?exam_id=ssc_cgl&subject=Quantitative%20Aptitude&difficulty=moderate&limit=10
```

**Query Parameters:**
- `exam_id` (optional): Filter by exam
- `subject` (optional): Filter by subject
- `difficulty` (optional): Filter by difficulty
- `limit` (optional): Max results (default: 20)

**Response:**
```json
[
  {
    "id": "test1",
    "examId": "ssc_cgl",
    "title": "Mock Test 1",
    // ... full test object with questions
  },
  // ... more tests
]
```

---

## 🗄️ Firestore Structure

```
tests (collection)
├── {testId} (document)
│   ├── examId: "ssc_cgl"
│   ├── title: "SSC CGL Mock 1"
│   ├── subject: "Quantitative Aptitude"
│   ├── difficulty: "moderate"
│   ├── numQuestions: 10
│   ├── durationMinutes: 30
│   ├── isAIGenerated: true
│   ├── createdAt: "2025-11-25T10:30:00.000Z"
│   └── questions (subcollection)
│       ├── {questionId} (document)
│       │   ├── examId: "ssc_cgl"
│       │   ├── subject: "Quantitative Aptitude"
│       │   ├── topic: "Algebra"
│       │   ├── difficulty: "moderate"
│       │   ├── questionText: "..."
│       │   ├── optionA: "..."
│       │   ├── optionB: "..."
│       │   ├── optionC: "..."
│       │   ├── optionD: "..."
│       │   ├── correctOption: "A"
│       │   ├── explanation: "..."
│       │   ├── shortcut: "..."
│       │   └── timeToSolveSeconds: 60
│       └── ...
└── ...
```

---

## 🧪 Testing

### Manual Testing with cURL

```bash
# Health check
curl http://localhost:8000/

# Generate mock test
curl -X POST http://localhost:8000/api/generate-mock \
  -H "Content-Type: application/json" \
  -d '{
    "exam_id": "ssc_cgl",
    "exam_name": "SSC Combined Graduate Level",
    "subject": "Quantitative Aptitude",
    "topic": "Algebra",
    "difficulty": "moderate",
    "num_questions": 5,
    "duration_minutes": 15,
    "title": "Test Mock 1"
  }'

# Get test by ID
curl http://localhost:8000/api/tests/{test_id}

# List tests
curl http://localhost:8000/api/tests?exam_id=ssc_cgl&limit=5
```

### Test AI Utils Standalone

```bash
python ai_utils.py
```

---

## 🔧 Configuration

### Adjusting Gemini Model

Edit `ai_utils.py`:

```python
# Change Gemini model (line ~174)
model = genai.GenerativeModel('gemini-1.5-pro')  # or 'gemini-1.5-flash' for faster/cheaper
```

Restart the server for changes to take effect.

---

## 🐛 Troubleshooting

### Firebase Initialization Error

**Error:** `Firebase initialization failed`

**Solution:**
- Verify `serviceAccountKey.json` exists in project root
- Check Firebase project has Firestore enabled
- Ensure service account has proper permissions

### Gemini API Error

**Error:** `Gemini API key not configured`

**Solution:**
- Verify `GEMINI_API_KEY` in `.env` file
- Get your API key from https://makersuite.google.com/app/apikey
- Ensure API key is active and has quota available

### Question Generation Failures

**Error:** Questions fail to generate or validation errors

**Solution:**
- Check Gemini API status
- Review prompt in `ai_utils.py` → `build_question_prompt()`
- Increase `max_retries_per_question` in `main.py`
- Check API quota limits (free tier has limits)

---

## 🚢 Deployment

### Deploy to Render / Railway

1. Push code to GitHub
2. Create new Web Service
3. Set environment variables:
   - `GOOGLE_APPLICATION_CREDENTIALS` → Upload `serviceAccountKey.json`
   - `GEMINI_API_KEY` → Your Gemini API key

4. Set start command:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

### Deploy to Generic VM

```bash
# Install dependencies
pip install -r requirements.txt

# Run with production server
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 📝 Future Enhancements

- [ ] User authentication (Firebase Auth)
- [ ] Test attempts & scoring
- [ ] Performance analytics
- [ ] Leaderboard & rankings
- [ ] Adaptive test generation based on weak topics
- [ ] Batch question generation
- [ ] Caching layer for improved performance
- [ ] Rate limiting
- [ ] Admin dashboard

---

## 📄 License

MIT License - feel free to use for your projects.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Contact

For questions or support, reach out to the ExamHub team.

---

**Built with ❤️ for Indian exam aspirants**
