import os
import json
import boto3
import pandas as pd
from botocore.exceptions import ClientError

# --- AWS Clients ---
s3_client = boto3.client('s3')
bedrock_runtime = boto3.client('bedrock-runtime')

MODEL_ID = "us.anthropic.claude-sonnet-4-6"
MASTER_CSV_KEY = "global_policies/master_policy.csv"

def lambda_handler(event, context):
    """
    Enterprise Admin ETL Pipeline (MARKDOWN DRIVEN): 
    Reads pre-processed Docling Markdown from S3.
    Extracts multi-track pricing and safely upserts to the Global Master CSV.
    """
    try:
        # 1. Parse Payload from Flask Admin Route (Strictly Markdown)
        bucket = event['s3_bucket']
        md_key = event['s3_md_key'] 
        company = event.get('company', 'לא ידוע')
        policy_name = event.get('policy_name', 'לא ידוע')
        shaban_tier = event.get('shaban_tier')
        
        print(f"Admin ETL Started: Company: {company} | Policy: {policy_name} | Shaban: {shaban_tier}")
        
        # 2. Download Markdown text (Microseconds latency)
        response = s3_client.get_object(Bucket=bucket, Key=md_key)
        raw_text = response['Body'].read().decode('utf-8')
        
        # 3. Contextual Anchoring
        shaban_context = f"Applies to Shaban tier: {shaban_tier}" if shaban_tier else "No specific Shaban tier."
        
        # 4. Advanced Deterministic Master Extraction Prompt
        prompt = f"""
        You are a strict data engineering AI for Israeli health insurance policies.
        Analyze the following MARKDOWN text from a PUBLIC insurance policy document.
        
        KNOWN METADATA ANCHORS (Use as absolute facts):
        - Insurance Company: {company}
        - Policy/Coverage Name: {policy_name}
        - Shaban Context: {shaban_context}
        
        Extract ALL medical treatments, coverage limits, and waiting periods.
        
        For each coverage, extract exactly these fields:
        - "appendix_code": The numeric code of the rider if mentioned (e.g., '986'). If not found, output 'UNKNOWN'.
        - "treatment_name": Main category of the treatment (e.g., 'התייעצות רופא מומחה', 'התפתחות הילד').
        - "covered_treatments": Comma-separated list of all sub-treatments. If none, output main name.
        - "coverage_limit": CRITICAL - You must capture ALL financial tracks. Combine them into one detailed string. Example: "רופא בהסכם: 150 שח | תור מהיר: 250 שח | החזר פרטי: 80% עד 750 שח | עד 4 בשנה".
        - "waiting_period": The waiting period required before claiming (e.g., '90 ימים', '12 חודשים'). If none, output '0'.

        CRITICAL: Return ONLY a raw JSON array of objects. No markdown, no prefixes.
        
        Markdown Text:
        {raw_text}
        """
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096, 
            "messages": [{"role": "user", "content": prompt}]
        })
        
        # 5. Invoke Bedrock LLM
        bedrock_resp = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        
        # 6. ARCHITECTURAL FIX: Extract safely with array index 
        resp_body = json.loads(bedrock_resp.get("body").read())
        extracted_json_str = resp_body["content"][0]["text"] 
        
        # 7. Schema Normalization
        if extracted_json_str.startswith("```json"):
            extracted_json_str = extracted_json_str.replace("```json\n", "").replace("```", "")
            
        data = json.loads(extracted_json_str)
        
        # Guardrail: If Claude found no matching data for our anchors, abort safely
        if not data:
            print("Guardrail Activated: No matching policy data found in text. Aborting CSV append.")
            return {"statusCode": 200, "body": "No data extracted (Empty JSON)."}

        new_df = pd.DataFrame(data)
        
        # 8. Inject Foreign Keys
        new_df['company'] = company
        new_df['policy_name'] = policy_name
        new_df['shaban_tier'] = shaban_tier
        
        master_csv_path = "/tmp/master_policy.csv"
        
        # 9. Append-Only Data Lake Pattern (Upsert logic)
        try:
            s3_client.download_file(bucket, MASTER_CSV_KEY, master_csv_path)
            master_df = pd.read_csv(master_csv_path)
            
            combined_df = pd.concat([master_df, new_df], ignore_index=True)
            
            # Upsert by keeping the latest upload for the same treatment
            combined_df.drop_duplicates(
                subset=['company', 'policy_name', 'shaban_tier', 'treatment_name'], 
                keep='last', 
                inplace=True
            )
            print("Successfully merged with existing Master CSV.")
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                print("Master CSV not found. Creating a new foundational Master CSV.")
                combined_df = new_df
            else:
                raise e
                
        # 10. Save and Upload
        combined_df.to_csv(master_csv_path, index=False, encoding='utf-8-sig')
        s3_client.upload_file(master_csv_path, bucket, MASTER_CSV_KEY)
        
        if os.path.exists(master_csv_path): os.remove(master_csv_path)
        
        print("Admin ETL Complete. Master DB updated.")
        return {"statusCode": 200, "body": "Admin ETL successful"}
        
    except Exception as e:
        print(f"Architect Error - Admin ETL Failure: {e}")
        return {"statusCode": 500, "body": str(e)}