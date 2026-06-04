# Insurance RAG — AWS EC2 Deployment Guide

Flask web application for uploading Israeli health-insurance policy PDFs, ingesting them into an **Amazon Bedrock Knowledge Base** (via **S3**), and answering comparison questions in Hebrew using **RAG**.

This document is the primary guide for deploying the app on **AWS EC2 with Docker**. Authentication uses the **EC2 instance profile** (IAM role)—not hardcoded access keys.

---

## Architecture (production)

```
Browser → EC2 (Docker :5000) → Gunicorn/Flask
                              ├─ S3 (policy .md + .metadata.json)
                              ├─ Bedrock Runtime (classify + chat)
                              ├─ Bedrock Agent Runtime (retrieve)
                              └─ Bedrock Agent (start ingestion jobs)
```

**Operational constraints**

| Topic | Behavior |
| --- | --- |
| Credentials | Default AWS credential chain; attach IAM role to EC2 |
| Upload job status | In-memory (`JOB_STATUS`); use **one** Gunicorn worker |
| Docling / PDF | CPU- and memory-heavy; size EC2 accordingly |
| Ingestion | One Bedrock ingestion job per Knowledge Base at a time (retries in `upload_service`) |
| Health | `GET /health` (liveness), `GET /ready` (STS + S3) |

---

## AWS prerequisites

### 1. Existing resources (MVP defaults)

| Resource | Default ID / name | Env override |
| --- | --- | --- |
| Region | `us-east-1` | `AWS_REGION` |
| S3 bucket | `insurance-private-mvp` | `S3_BUCKET_NAME` |
| Knowledge Base | `NXYJDUMTAJ` | `KNOWLEDGE_BASE_ID` |
| Chat / classify model | `us.anthropic.claude-opus-4-6-v1` | `BEDROCK_MODEL_ID` |

Replace values in `.env` for your account. Set `REQUIRE_AWS_ENV=1` in production so missing env vars fail at startup.

### 2. Bedrock console setup

1. **Model access** — Enable the foundation model (or inference profile) matching `BEDROCK_MODEL_ID` in **Amazon Bedrock → Model access**.
2. **Knowledge Base** — S3 data source pointing at the same bucket; sync/ingestion enabled.
3. **Metadata filtering** — Map filterable fields on the data source:
   - `insurance_company` (required for retrieval filters)
   - `tier`
   - `source_filename`  
   Without `insurance_company` as filterable, company-scoped retrieval will not work.

### 3. IAM policy (EC2 instance role)

Create an IAM role and attach a policy like `deploy/iam-policy.example.json` (replace `YOUR_BUCKET_NAME`, `YOUR_ACCOUNT_ID`, `YOUR_KB_ID`, region as needed).

Minimum actions:

| Service | Actions |
| --- | --- |
| **S3** | `ListBucket`, `GetObject`, `PutObject` on the policy bucket |
| **Bedrock Runtime** | `InvokeModel` (and stream if used) for your model ARN / inference profile |
| **Bedrock Agent Runtime** | `Retrieve` on the Knowledge Base ARN |
| **Bedrock Agent** | `StartIngestionJob`, `ListDataSources`, `GetIngestionJob` on the KB |
| **STS** | `GetCallerIdentity` (used by `/ready`; usually allowed by default) |

Trust relationship for EC2:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "ec2.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

Attach the role as an **Instance profile** when launching the EC2 instance.

### 4. Recommended EC2 instance specs

| Workload | Instance type | vCPU | RAM | Notes |
| --- | --- | --- | --- | --- |
| **Minimum** | `t3.large` | 2 | 8 GiB | Light traffic; one upload at a time |
| **Recommended** | `t3.xlarge` or `m6i.xlarge` | 4 | 16 GiB | Docling + Bedrock concurrent threads |
| **Heavy PDFs** | `m6i.2xlarge` | 8 | 32 GiB | Large policies, frequent uploads |

Additional:

- **EBS**: 30–50 GiB gp3 (Docker image ~4–8 GiB + Docling cache + `tmp_uploads` / `tmp_outputs`)
- **Security group**: inbound TCP **5000** (or 80/443 if behind ALB/nginx)
- **Outbound**: HTTPS to AWS APIs and Hugging Face (Docling model download on first run)

### 5. Network

- Instance must reach **Bedrock**, **S3**, and **STS** endpoints in `AWS_REGION`.
- Optional: VPC interface endpoints for private subnets (no public internet).

---

## Environment variables

Copy `.env.example` to `.env` on the EC2 host (`chmod 600`). **Do not** commit `.env`.

### Required (when `REQUIRE_AWS_ENV=1`)

| Variable | Description |
| --- | --- |
| `AWS_REGION` | e.g. `us-east-1` |
| `S3_BUCKET_NAME` | Policy documents bucket |
| `KNOWLEDGE_BASE_ID` | Bedrock Knowledge Base ID |
| `BEDROCK_MODEL_ID` | Inference profile or model ID for `converse` |

### AWS client tuning

| Variable | Default | Description |
| --- | --- | --- |
| `BOTO_CONNECT_TIMEOUT` | `10` | Seconds |
| `BOTO_READ_TIMEOUT` | `300` | Seconds (large RAG responses) |
| `BOTO_MAX_RETRY_ATTEMPTS` | `10` | Adaptive retries for throttling |

### Knowledge Base / ingestion

| Variable | Default | Description |
| --- | --- | --- |
| `KNOWLEDGE_BASE_DATA_SOURCE_ID` | *(empty)* | Auto-resolved if unset |
| `INGESTION_MAX_RETRIES` | `15` | Conflict backoff when another job runs |
| `INGESTION_RETRY_DELAY_SECONDS` | `10` | Base delay (exponential cap 120s) |

### RAG retrieval (see `config.py`; max 100 per Bedrock call)

| Variable | Default |
| --- | --- |
| `RAG_SEARCH_TYPE` | `HYBRID` |
| `RAG_NARROW_RESULTS` | `50` |
| `RAG_BROAD_RESULTS` | `100` |
| `RAG_RESULTS_PER_COMPANY` | `50` |
| `RAG_RESULTS_PER_COMPANY_BROAD` | `60` |
| `RAG_MAX_TOTAL_RESULTS` | `130` |
| `RAG_MAX_COMPARE_COMPANIES` | `4` |
| `POLICY_REGISTRY_TTL_SECONDS` | `30` |

### Application / paths

| Variable | Default | Description |
| --- | --- | --- |
| `PORT` | `5000` | Container listen port |
| `BIND_HOST` | `0.0.0.0` | Dev server bind |
| `UPLOAD_FOLDER` | `./tmp_uploads` | PDF staging (mount EBS in Docker) |
| `OUTPUT_FOLDER` | `./tmp_outputs` | Markdown + metadata sidecars |
| `PAGES_PER_CHUNK` | `5` | Docling split size |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, … |
| `REQUIRE_AWS_ENV` | `0` | Set `1` in Docker/production |

### Do **not** set in production (use instance profile)

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

---

## Deploy on EC2 — step by step

### Step 1: Launch EC2

1. AMI: **Amazon Linux 2023** or **Ubuntu 22.04**
2. Instance: **t3.xlarge** (recommended)
3. Attach IAM **instance profile** with the policy above
4. Security group: allow inbound **5000** from your IP or ALB
5. Optional: attach extra EBS volume mounted at `/data`

### Step 2: Install Docker on the instance

**Amazon Linux 2023:**

```bash
sudo dnf update -y
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
# Log out and back in so group membership applies
```

**Ubuntu 22.04:**

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
sudo usermod -aG docker ubuntu
```

### Step 3: Get application files onto EC2

**Option A — Git clone**

```bash
cd ~
git clone <YOUR_REPO_URL> insuranceRag
cd insuranceRag
```

**Option B — Copy from your machine**

```bash
scp -i your-key.pem -r insuranceRag ec2-user@<EC2_PUBLIC_IP>:~/
```

### Step 4: Configure environment

```bash
cd ~/insuranceRag
cp .env.example .env
nano .env   # set S3_BUCKET_NAME, KNOWLEDGE_BASE_ID, BEDROCK_MODEL_ID, AWS_REGION
chmod 600 .env
mkdir -p /data/tmp_uploads /data/tmp_outputs
```

### Step 5: Build the Docker image

```bash
cd ~/insuranceRag
docker build -t insurance-rag:latest .
```

First build may take **10–20 minutes** (Docling / ML dependencies).

### Step 6: Run the container

```bash
docker run -d \
  --name insurance-rag \
  --restart unless-stopped \
  -p 5000:5000 \
  --env-file .env \
  -v /data/tmp_uploads:/app/tmp_uploads \
  -v /data/tmp_outputs:/app/tmp_outputs \
  insurance-rag:latest
```

Verify:

```bash
docker logs -f insurance-rag
curl -s http://127.0.0.1:5000/health
curl -s http://127.0.0.1:5000/ready
```

Open in browser: `http://<EC2_PUBLIC_IP>:5000`

### Step 7: Pull from a registry (optional)

On your build machine:

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
docker tag insurance-rag:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/insurance-rag:latest
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/insurance-rag:latest
```

On EC2:

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
docker pull <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/insurance-rag:latest
docker run -d --name insurance-rag --restart unless-stopped -p 5000:5000 --env-file .env \
  -v /data/tmp_uploads:/app/tmp_uploads -v /data/tmp_outputs:/app/tmp_outputs \
  <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/insurance-rag:latest
```

### Container maintenance

```bash
docker stop insurance-rag
docker rm insurance-rag
docker build -t insurance-rag:latest . && docker run ...   # redeploy
docker exec insurance-rag curl -s http://127.0.0.1:5000/ready
```

---

## Local development

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

export AWS_REGION=us-east-1
export S3_BUCKET_NAME=insurance-private-mvp
export KNOWLEDGE_BASE_ID=NXYJDUMTAJ
export BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0
# Use aws configure or temporary env keys for local creds only

python app.py
```

Open http://127.0.0.1:5000

---

## Project structure

```
app.py                    # Flask routes, health/ready probes
config.py                 # Env-based settings, validation
services/
  aws_clients.py          # Shared boto3 session (IAM role / default chain)
  bedrock_service.py      # RAG retrieval + generation
  upload_service.py       # PDF → MD, S3, ingestion
  policy_registry.py      # Policy list from S3 metadata sidecars
templates/index.html      # Hebrew RTL UI
Dockerfile
.env.example
deploy/iam-policy.example.json
```

---

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `/ready` 503, no credentials | Instance profile attached? `curl` from inside container: `aws sts get-caller-identity` |
| `/ready` 503, S3 | Bucket name, region, IAM `s3:ListBucket` / `HeadBucket` |
| Chat 500, AccessDenied | Model access in Bedrock; `bedrock:InvokeModel` on model ARN |
| Retrieval empty / wrong company | KB metadata field `insurance_company` filterable; sidecars uploaded |
| Upload stuck / conflict | Another ingestion job running; wait or check Bedrock console |
| 504 / timeout | Increase `BOTO_READ_TIMEOUT`; use larger instance; shorten question |
| Status `unknown` after upload | Do not scale Gunicorn beyond **1 worker** (in-memory job map) |

---

## Security checklist

- [ ] IAM role with least privilege (no `*` on all resources unless required)
- [ ] No `AWS_ACCESS_KEY_ID` in Docker image or `.env` on shared hosts
- [ ] Security group restricts port 5000 to trusted IPs or ALB only
- [ ] HTTPS termination at ALB or nginx reverse proxy for production
- [ ] S3 bucket blocks public access
- [ ] `.env` file mode `600`

---

## License / support

Internal MVP deployment. Adjust ARNs, bucket names, and model IDs per environment before production cutover.
