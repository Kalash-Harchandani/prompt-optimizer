import json
import os

from dotenv import load_dotenv
from google import genai
from pathlib import Path


load_dotenv()


class Extractor:

    def __init__(self, model_name: str):

        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )

        self.model_name = model_name

    def extract(
        self,
        pdf_path: Path,
        schema: dict,
        prompt: str
    ) -> dict:

        uploaded_file = self.client.files.upload(
            file=str(pdf_path)
        )

        full_prompt = f"""
You are an expert structured document extraction system.

Extract information from the PDF and return ONLY valid JSON.

Follow this schema exactly:

{json.dumps(schema, indent=2)}

Instructions:
- Return valid JSON only
- Do not use markdown
- Do not explain anything
- Preserve nested structure
- Use null for missing values
- Do not omit fields

Additional instructions:

{prompt}
"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                uploaded_file,
                full_prompt
            ]
        )

        raw_text = response.text.strip()

        cleaned_text = raw_text

        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text.replace("```json", "", 1)

        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]

        cleaned_text = cleaned_text.strip()

        try:
            parsed = json.loads(cleaned_text)
            return parsed

        except json.JSONDecodeError:

            print("Failed to parse JSON response")

            return {
                "error": "invalid_json",
                "raw_response": raw_text
            }