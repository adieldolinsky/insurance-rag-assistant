# יודוקוליס — סוכן ביטוח בריאות חכם  
## Yodoculis — AI Health Insurance Agent

> **Mid-course project · Adiel Dolinsky**  
> An end-to-end AI application demonstrating Retrieval-Augmented Generation (RAG) on real-world Hebrew insurance documents.

---

## 1. Project Goal

Yodoculis is an intelligent health-insurance advisory agent built as a mid-course project. The project demonstrates a complete, production-ready AI application that combines document ingestion, cloud-native AWS services, and a conversational UI.

**What the system does:**
- Accepts PDF files of Israeli health-insurance policies (e.g., Clalit Platinum, Maccabi Gold, Menorah Ambulatori).
- Converts each PDF to structured Markdown using an OCR-capable pipeline (Docling).
- Indexes the Markdown into an **AWS Bedrock Knowledge Base** backed by a managed vector store (FAISS-compatible).
- Exposes a Hebrew-language chat interface where users ask natural-language questions about their specific policy coverage, co-payments, and exclusions.
- The Bedrock Agent performs **RAG** — it retrieves the most relevant policy passages and generates grounded, cited answers via Claude (Anthropic) — without hallucinating data that is not present in the uploaded document.

The result is a secure, scalable advisor that can instantly answer "Is procedure X covered?", "What is my deductible?", or "Compare my policy with Maccabi Gold" from the user's own uploaded documents.

---

## 2. System Architecture

```
Browser (HTML / JS / Tailwind)
        │
        │  HTTP (REST)
        ▼
┌─────────────────────────────────────────┐
│         Flask API  (Gunicorn WSGI)      │
│  ┌──────────────┐  ┌────────────────┐  │
│  │ /upload      │  │ /chat          │  │
│  │ POST         │  │ POST           │  │
│  └──────┬───────┘  └───────┬────────┘  │
│         │                  │            │
│   Background            Bedrock         │
│   Thread                Agent           │
│   (Docling)             Runtime         │
└─────────┬────────────────┬─────────────┘
          │                │
          ▼                ▼
   ┌─────────────┐  ┌──────────────────────┐
   │  Amazon S3  │  │  AWS Bedrock Agent   │
   │  (Markdown  │  │  ┌────────────────┐  │
   │   storage)  │  │  │ Knowledge Base │  │
   └──────┬──────┘  │  │ (Vector Store) │  │
          │         │  └────────────────┘  │
          │         │  ┌────────────────┐  │
          └────────►│  │ Claude Model   │  │
    S3 event +      │  │ (Anthropic)    │  │
    Lambda ETL       │  └────────────────┘  │
                    └──────────────────────┘
```

### Components

| Component | Technology | Purpose |
|---|---|---|
| **Frontend** | HTML5, Tailwind CSS, Vanilla JS | RTL Hebrew chat UI with drag-and-drop PDF upload and real-time status polling |
| **Backend API** | Python 3.10, Flask, Gunicorn | REST endpoints for file upload (`/upload`), job status (`/status/<id>`), and chat (`/chat`) |
| **PDF → Markdown** | Docling (OCR + layout engine) | Converts scanned or digital PDFs to clean Markdown, preserving tables and structure |
| **Object Storage** | Amazon S3 | Stores processed Markdown files; triggers the ETL Lambda via S3 event |
| **ETL Pipeline** | AWS Lambda (Python) | Parses Markdown, extracts structured coverage data, and writes a CSV to the user's session prefix |
| **RAG Engine** | AWS Bedrock Knowledge Base | Semantic vector search over indexed policy Markdown; powered by a managed FAISS-compatible vector store |
| **LLM** | Anthropic Claude (via Bedrock Agent) | Generates grounded, policy-specific answers from retrieved passages |
| **Infrastructure** | Docker, Docker Compose | Single-container deployment; runs identically on a developer laptop and an EC2 instance |

---

## 3. Live Demo — Cloud Deployment

The application is packaged as a Docker container and is currently running live on an **AWS EC2** instance.

**Live URL: [http://18.207.191.24:5000](http://18.207.191.24:5000)**

> All frontend API calls use relative paths (`/chat`, `/upload`, `/status/<id>`, `/static/…`) so the application works identically on localhost and in the cloud without any code changes.

---

## 4. Installation & Running

### Prerequisites

| Requirement | Notes |
|---|---|
| Docker ≥ 20 + Docker Compose | The only runtime dependency on the host |
| AWS account | With Bedrock, S3, and Lambda permissions |
| Bedrock Knowledge Base | Pre-created in your AWS region; data source connected to the S3 bucket |
| Bedrock Agent | Pre-created with the Knowledge Base attached; alias deployed |

### Step 1 — Clone the repository

```bash
git clone <repo-url>
cd insuranceRag
```

### Step 2 — Configure environment variables

Copy the example file and fill in your real values:

```bash
cp .env.example .env
```

Open `.env` and set the following required variables:

```ini
# AWS credentials (for local Docker — on EC2 use an IAM instance profile instead)
AWS_ACCESS_KEY_ID=<your-access-key>
AWS_SECRET_ACCESS_KEY=<your-secret-key>
AWS_REGION=us-east-1

# S3 bucket that stores processed Markdown and user session data
S3_BUCKET_NAME=<your-s3-bucket-name>

# Bedrock Knowledge Base
KNOWLEDGE_BASE_ID=<your-kb-id>
KNOWLEDGE_BASE_DATA_SOURCE_ID=<your-data-source-id>

# Bedrock Agent for RAG chat
BEDROCK_AGENT_ID=<your-agent-id>
BEDROCK_AGENT_ALIAS_ID=<your-alias-id>

# Bedrock model ID (example: Claude Sonnet)
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0

# Lambda ETL function names (as deployed in AWS)
ADMIN_LAMBDA_NAME=Admin_Policy_ETL_Lambda
USER_LAMBDA_NAME=Lambda3_User_ETL
```

> **Security note:** The `.env` file is git-ignored. Never commit it. On EC2, set `AWS_USE_ENV_CREDENTIALS=0` and attach an IAM instance profile with least-privilege permissions instead of using static keys.

### Step 3 — Build and start the container

```bash
docker compose up --build
```

The first build takes 10–20 minutes (downloads PyTorch + Docling OCR models). Subsequent builds use Docker's layer cache and complete in seconds.

### Step 4 — Open the application

Navigate to **[http://localhost:5000](http://localhost:5000)** in your browser.

### Step 5 — Upload a policy and start chatting

1. Click **"העלאת פוליסה (PDF)"** in the left sidebar.
2. Upload any Israeli health-insurance PDF.
3. Wait for the sidebar to show **"סרוק במערכת"** (the processing pipeline is complete).
4. Ask questions in the chat — e.g., "מה ההשתתפות העצמית שלי בניתוח?"

---

## 5. Useful Commands

```bash
# Start (detached)
docker compose up -d

# View live logs
docker compose logs -f

# Stop and remove the container
docker compose down

# Rebuild after a code change (uses cache for unchanged layers)
docker compose up --build -d

# Check container health
docker inspect insurance-rag-api --format='{{.State.Health.Status}}'

# Open a shell inside the running container
docker exec -it insurance-rag-api sh
```

---

## 6. Project Structure

```
insuranceRag/
├── backend/
│   ├── app.py                  # Flask entry point, all REST routes
│   ├── Dockerfile              # Production image definition
│   ├── requirements.txt        # Python dependencies
│   ├── services/
│   │   ├── aws_clients.py      # Cached boto3 client pool
│   │   ├── bedrock_service.py  # Bedrock Agent RAG chat
│   │   └── upload_service.py   # Docling PDF ingestion + S3 + Lambda trigger
│   ├── static/
│   │   └── png/                # Insurance company logos + UI assets
│   └── templates/
│       └── index.html          # Single-page chat UI
├── lambdas/
│   ├── admin_master_etl_md/    # Lambda: process admin policies → master CSV
│   └── user_session_etl_md/    # Lambda: process user upload → session CSV
├── schemas/
│   └── finanical_schema.json   # Bedrock Agent Action Group OpenAPI spec
├── deploy/
│   └── iam-policy.example.json # Least-privilege IAM policy for EC2
├── config.py                   # Centralised configuration (reads .env)
├── docker-compose.yml          # Local / EC2 deployment definition
├── .env.example                # Environment variable template
└── .dockerignore
```

---

## 7. IAM Permissions

The application requires the following AWS permissions. An example policy is provided in `deploy/iam-policy.example.json`.

| Service | Actions required |
|---|---|
| S3 | `GetObject`, `PutObject`, `ListBucket` on the policy bucket |
| Bedrock | `bedrock:InvokeModel`, `bedrock-agent:StartIngestionJob` |
| Bedrock Agent Runtime | `bedrock-agent-runtime:InvokeAgent`, `bedrock-agent-runtime:Retrieve` |
| Lambda | `lambda:InvokeFunction` on both ETL functions |
| STS | `sts:GetCallerIdentity` (startup credentials check) |

---

## 8. Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | Local only | Static AWS key (omit on EC2 with IAM role) |
| `AWS_SECRET_ACCESS_KEY` | Local only | Static AWS secret |
| `AWS_REGION` | ✅ | AWS region for all services |
| `S3_BUCKET_NAME` | ✅ | S3 bucket for Markdown and session data |
| `KNOWLEDGE_BASE_ID` | ✅ | Bedrock Knowledge Base ID |
| `KNOWLEDGE_BASE_DATA_SOURCE_ID` | ✅ | Bedrock KB data source ID |
| `BEDROCK_MODEL_ID` | ✅ | Model ID for Bedrock inference |
| `BEDROCK_AGENT_ID` | ✅ | Bedrock Agent ID |
| `BEDROCK_AGENT_ALIAS_ID` | ✅ | Bedrock Agent alias ID |
| `ADMIN_LAMBDA_NAME` | ✅ | Name of the admin ETL Lambda |
| `USER_LAMBDA_NAME` | ✅ | Name of the user-session ETL Lambda |
| `AWS_USE_ENV_CREDENTIALS` | ✅ | `1` = use `.env` keys, `0` = use IAM role |
| `UPLOAD_FOLDER` | ✅ | Container path for temporary uploads |
| `OUTPUT_FOLDER` | ✅ | Container path for temporary Docling output |
| `PORT` | | HTTP server port (default: `5000`) |
| `LOG_LEVEL` | | Logging verbosity (default: `INFO`) |

---


