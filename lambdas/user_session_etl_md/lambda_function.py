import os
import json
import boto3
import pandas as pd
from botocore.exceptions import ClientError

# --- AWS Clients ---
# Initialize globally to maintain connection pools
s3_client = boto3.client('s3')
bedrock_runtime = boto3.client('bedrock-runtime')

MODEL_ID = "us.anthropic.claude-sonnet-4-6"

def lambda_handler(event, context):
    """
    Asynchronous ETL Pipeline (MARKDOWN DRIVEN): 
    Extracts private discounts and exclusions from a user's pre-processed 
    Markdown policy, structures it via Claude 3.5, and saves a CSV to S3.
    """
    try:
        # 1. Parse Async Payload from Flask (Now strictly expecting Markdown)
        session_id = event['session_id']
        bucket = event['s3_bucket']
        md_key = event['s3_md_key'] # ARCHITECTURAL FIX: Reading MD, not PDF!
        company = event.get('company', 'לא ידוע')
        policy_name = event.get('policy_name', 'לא ידוע')
        shaban_tier = event.get('shaban_tier') # Can be None
        
        print(f"Starting ETL for Session: {session_id} | Company: {company} | Policy: {policy_name} | Shaban: {shaban_tier}")
        
        # 2. Download Markdown text (Microseconds latency, zero PDF overhead)
        response = s3_client.get_object(Bucket=bucket, Key=md_key)
        raw_text = response['Body'].read().decode('utf-8')
        
        # 3. Contextual Anchoring for the Prompt
        shaban_context = f"The user belongs to the specific Shaban tier: {shaban_tier}." if shaban_tier else "The user does not have or mention a supplementary Shaban tier."
        
        # 4. Deterministic Extraction Prompt for Claude 3.5
        prompt = f"""
        You are a strict data engineering AI for Israeli health insurance policies.
        Analyze the following MARKDOWN text from a private insurance policy document.
        
        KNOWN METADATA ANCHORS (Use as absolute facts):
        - Insurance Company: {company}
        - Policy/Coverage Name: {policy_name}
        - Shaban Context: {shaban_context}
        
        Extract the specific coverage riders (נספחים) this specific user has purchased.
        For each rider, extract exactly these fields:
- "appendix_code": The numeric code or ID of the rider (e.g., '986'). If none, output 'UNKNOWN'.
- "rider_name": The exact name of the cover.
- "discount_percent": Any percentage discount applied. Output as a float (e.g., 15.0). If no percentage is mentioned, output 0.0.
- "fixed_personal_price": If the document explicitly states a final fixed price the user pays in NIS instead of a percentage (e.g., "150 NIS", "250 NIS"), output it as a float. If none, output 0.0.
- "exclusions": Any specific personal medical exclusions. Output an empty string "" if none.

        CRITICAL: Return ONLY a raw JSON array of objects. No markdown formatting.
        Example format: [{{"appendix_code": "986", "rider_name": "ניתוחים", "discount_percent": 15.0, "exclusions": ""}}]
        
        Markdown Text:
        {raw_text}
        """
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1500,
            "messages": [{"role": "user", "content": prompt}]
        })
        
        # 5. Invoke Bedrock LLM
        bedrock_resp = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        
        resp_body = json.loads(bedrock_resp.get("body").read())
        extracted_json_str = resp_body["content"][0]["text"]
        
        # 6. Schema Normalization (Anti-Hallucination Guard)
        if extracted_json_str.startswith("```json"):
            extracted_json_str = extracted_json_str.replace("```json\n", "").replace("```", "")
            
        data = json.loads(extracted_json_str)
        df = pd.DataFrame(data)
        
        # Enforce column schema
        expected_cols = ["appendix_code", "rider_name", "discount_percent", "exclusions","fixed_personal_price"]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None
                
        # 7. Save output to CSV and Push to Session Path in S3
        csv_path = f"/tmp/{session_id}_discount.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        output_key = event.get(
            "output_csv_key",
            f"user_sessions/{session_id}/personal_rates.csv",
        )
        s3_client.upload_file(csv_path, bucket, output_key)
        
        if os.path.exists(csv_path): os.remove(csv_path)
        
        print(f"ETL Complete. Session CSV saved to {output_key}")
        return {"statusCode": 200, "body": "Session ETL successful"}
        
    except Exception as e:
        print(f"Architect Error - ETL Failure: {e}")
        return {"statusCode": 500, "body": str(e)}