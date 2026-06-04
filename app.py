"""
Flask entry point for the Insurance RAG web application.

Routes:
  GET  /              -> the Hebrew RTL chat + upload UI
  GET  /policies      -> policies currently in the Knowledge Base (sidebar source)
  POST /upload        -> save the PDF, start a background thread, return 202 Accepted
  GET  /status/<id>   -> poll background ingestion progress
  POST /chat          -> RAG retrieval + generation answer
"""
from __future__ import annotations

import logging
import os
import threading
import uuid

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER, setup_logging
from services import bedrock_service, policy_registry, upload_service

setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB cap for large policies.


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/policies")
def policies():
    """Return the policies currently tracked in the Knowledge Base."""
    try:
        force = request.args.get("refresh") == "1"
        items = [p.to_dict() for p in policy_registry.list_policies(force_refresh=force)]
        return jsonify({"policies": items})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to list policies: %s", exc)
        return jsonify({"policies": [], "error": "לא ניתן לטעון את רשימת הפוליסות."}), 500


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

    saved_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{original_filename}")
    file.save(saved_path)
    logger.info("Received upload '%s' (job %s).", original_filename, job_id)

    upload_service.JOB_STATUS[job_id] = {
        "state": "processing",
        "message": "מעלה ומעבד מסמך...",
        "filename": original_filename,
    }

    thread = threading.Thread(
        target=upload_service.process_document,
        args=(job_id, saved_path, original_filename),
        daemon=True,
    )
    thread.start()

    # Return immediately so the browser does not time out on long Docling jobs.
    return (
        jsonify(
            {
                "job_id": job_id,
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

    try:
        answer = bedrock_service.retrieve_and_generate(query)
        return jsonify({"answer": answer})
    except RuntimeError as exc:
        message = str(exc)
        logger.error("Chat failed: %s", message)
        status = 504 if "זמן רב מדי" in message or "timed out" in message.lower() else 500
        return jsonify({"error": message}), status
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat failed: %s", exc)
        return jsonify({"error": "אירעה שגיאה בעת ניתוח השאלה. נסו שוב."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
