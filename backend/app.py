"""
Flask entry point — ingestion orchestration + Bedrock Agent chat.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import uuid

from dotenv import load_dotenv

load_dotenv()

# Ensure project root (config.py) is importable when running from backend/
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from config import BUCKET_NAME, UPLOAD_FOLDER, setup_logging, validate_aws_config_or_exit
from services import bedrock_service, upload_service
from services.aws_clients import check_aws_connectivity, check_s3_bucket

setup_logging()
validate_aws_config_or_exit()
logger = logging.getLogger(__name__)

_creds_ok, _creds_detail = check_aws_connectivity()
if not _creds_ok:
    logger.error("AWS credentials check failed at startup: %s", _creds_detail)
else:
    logger.info("AWS credentials OK: %s", _creds_detail)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/ready")
def ready():
    ok, detail = check_aws_connectivity()
    if not ok:
        return jsonify({"status": "not_ready", "error": detail}), 503

    bucket_ok, bucket_detail = check_s3_bucket(BUCKET_NAME)
    if not bucket_ok:
        return jsonify({"status": "not_ready", "error": bucket_detail}), 503

    return jsonify({"status": "ready", "identity": detail, "bucket": BUCKET_NAME}), 200


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "לא נשלח קובץ."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "לא נבחר קובץ."}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "יש להעלות קובץ PDF בלבד."}), 400

    original_filename = secure_filename(file.filename)
    job_id = uuid.uuid4().hex
    upload_type = request.form.get("upload_type", "user")
    session_id = (request.form.get("session_id") or "").strip() or uuid.uuid4().hex

    saved_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{original_filename}")
    file.save(saved_path)
    logger.info("Received upload '%s' (job %s, type=%s).", original_filename, job_id, upload_type)

    upload_service.JOB_STATUS[job_id] = {
        "state": "processing",
        "message": "מעלה ומעבד מסמך...",
        "filename": original_filename,
    }

    thread = threading.Thread(
        target=upload_service.process_document,
        args=(job_id, saved_path, original_filename),
        kwargs={"upload_type": upload_type, "session_id": session_id},
        daemon=True,
    )
    thread.start()

    return (
        jsonify(
            {
                "job_id": job_id,
                "session_id": session_id,
                "filename": original_filename,
                "message": "הקובץ התקבל ומעובד ברקע.",
            }
        ),
        202,
    )


@app.route("/status/<job_id>")
def status(job_id: str):
    job = upload_service.JOB_STATUS.get(job_id)
    if not job:
        return jsonify({"state": "unknown", "message": "מזהה משימה לא נמצא."}), 404
    return jsonify(job)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "השאלה ריקה."}), 400

    session_id = (data.get("session_id") or "").strip() or uuid.uuid4().hex

    try:
        answer = bedrock_service.retrieve_and_generate(query, session_id)
        return jsonify({"answer": answer, "session_id": session_id})
    except RuntimeError as exc:
        message = str(exc)
        logger.error("Chat failed: %s", message)
        return jsonify({"error": message}), 500
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat failed: %s", exc)
        return jsonify({"error": "אירעה שגיאה בעת ניתוח השאלה. נסו שוב."}), 500


if __name__ == "__main__":
    from config import APP_BIND_HOST, APP_PORT

    app.run(host=APP_BIND_HOST, port=APP_PORT, debug=False)
