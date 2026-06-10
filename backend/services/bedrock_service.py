"""
Bedrock Agent chat: passes session_id via sessionAttributes for Action Group Lambdas.
"""
from __future__ import annotations

import logging

from config import BEDROCK_AGENT_ALIAS_ID, BEDROCK_AGENT_ID
from services.aws_clients import bedrock_agent_runtime_client

logger = logging.getLogger(__name__)


def retrieve_and_generate(query: str, session_id: str) -> str:
    """Invoke the Bedrock Agent with session_id in sessionAttributes."""
    try:
        response = bedrock_agent_runtime_client().invoke_agent(
            agentId=BEDROCK_AGENT_ID,
            agentAliasId=BEDROCK_AGENT_ALIAS_ID,
            sessionId=session_id,
            inputText=query,
            sessionState={
                "sessionAttributes": {
                    "session_id": session_id,
                }
            },
        )

        output_text = ""
        for event in response.get("completion", []):
            if "chunk" in event:
                output_text += event["chunk"]["bytes"].decode("utf-8")

        return output_text.strip() or "לא התקבלה תשובה מהסוכן."

    except Exception as exc:  # noqa: BLE001
        logger.error("Bedrock Agent invocation failed: %s", exc)
        raise RuntimeError(
            "מצטער, חלה שגיאה בתקשורת עם סוכן הביטוח. נסה שנית."
        ) from exc
