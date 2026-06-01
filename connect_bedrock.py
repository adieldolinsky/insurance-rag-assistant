

import boto3
from botocore.exceptions import ClientError


REGION = "us-east-1"
MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
BUCKET_NAME = "insurance-private-mvp"
KNOWLEDGE_BASE_ID = "NXYJDUMTAJ"


def retrieve_and_generate(query: str) -> str:
    agent_client = boto3.client("bedrock-agent-runtime", region_name=REGION)
    bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)

    # 1. שליפה ממסד הנתונים בעזרת ה-Agent
    print("\n=== DEBUG START ===")
    print(f"Sending Query: {query}")
    print(f"Target Knowledge Base: {KNOWLEDGE_BASE_ID}")

    try:
        retrieval_response = agent_client.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            retrievalQuery={'text': query},
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': 40,
                    'overrideSearchType': 'HYBRID'
                }
            }
        )

        print(f"Raw AWS Response: {retrieval_response}")

        contexts = []
        for result in retrieval_response.get('retrievalResults', []):
            text = result['content']['text']


            s3_uri = result['location']['s3Location']['uri']


            file_name = s3_uri.split('/')[-1].replace('.md', '').replace('.pdf', '')


            labeled_text = f"--- מקור: {file_name} ---\n{text}\n"
            contexts.append(labeled_text)

        retrieved_context = "\n".join(contexts)

        if not retrieved_context:
            return ("There is no information on the subject in the appendices,\n"
                    " you are not insured or an appendix is missing.")

    except ClientError as e:
        raise RuntimeError(f"Retrieval failed: {e.response.get('Error', {}).get('Message')}") from e

    print("The system is analyzing the policy.")

    system_prompt = """
        You are an expert, highly deterministic insurance analyst for Israeli policies.
        Your task is to extract and compare medical coverage details with ZERO hallucinations.

        The retrieved text contains snippets from different insurance policies. Each snippet is strictly labeled at the top with '--- מקור: [FileName] ---'.

        CRITICAL RULES TO PREVENT CROSS-CONTAMINATION:
        1. Group your findings STRICTLY by the Source name (מקור).
        2. NEVER mix numbers, rules, or conditions from one Source with another.
        3. For each source, explicitly state the deductible (השתתפות עצמית), the maximum refund (תקרת החזר/אחוזים), and conditions.
        4. If the user asks to compare (השוואה), present a clear, structured breakdown contrasting what each company offers.
        5. If a specific data point is missing for a particular source, write 'לא מפורט בפוליסה' for that specific source. Do NOT infer it from the other source.

        Format the output cleanly using Markdown headers, bold text for numbers, and bullet points.
        """
    user_prompt = f"Retrieved policy text:\n{retrieved_context}\n\nQuestion:\n{query}"
    try:
        response = bedrock_client.converse(
            modelId = MODEL_ID,
            system = [{"text": system_prompt}],
            messages = [{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig = {
                "maxTokens": 4096,
                "temperature": 0.0
            },
        )
        return response["output"]["message"]["content"][0]["text"]

    except ClientError as e:
        raise RuntimeError(f"Generation failed: {e}")
    


if __name__ == "__main__":
    print( "=== מערכת RAG לניתוח פוליסות ===")
    print(" if you want finish press exit\n")
    while True:
        user_query = input(">: ")
        if user_query.strip().lower() in ["exit", "quit", "q"]:
            print("goodbye")
            break
        if not user_query.strip():
            continue
        try:
            out = retrieve_and_generate(user_query)
            print(out)
            print("-"*50)
        except Exception as e:
            print(f"Error: {e}")
            print("-" * 50)
