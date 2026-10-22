import json
import os
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError
import pdfplumber

client = genai.Client()


def process_single_pdf(pdf_path):
    """Extracts raw text from a PDF and sends it to Gemini with automated retry logic."""
    raw_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                raw_text += extracted + "\n"

    if not raw_text.strip():
        raise ValueError("PDF contains no extractable text.")

    prompt = f"""
    You are an automated document parsing assistant. 
    Analyze the text below from a single document. 
    
    1. Extract metadata fields into JSON format.
    2. Remove all non-essential web UI elements, navigation menus, header print artifacts, and footer copyright boilerplate.
    3. Return the main content formatted as clean, continuous body text.

    DOCUMENT TEXT:
    {raw_text}
    """

    max_retries = 5
    delay = 5  # Increased base delay slightly for quota stability

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "metadata": {
                                "type": "OBJECT",
                                "properties": {
                                    "document": {"type": "STRING"},
                                    "date": {"type": "STRING"},
                                    "organisation": {"type": "STRING"},
                                    "document_type": {"type": "STRING"},
                                    "url": {"type": "STRING"},
                                    "page": {"type": "INTEGER"},
                                },
                            },
                            "cleaned_body_text": {"type": "STRING"},
                        },
                    },
                ),
            )
            return json.loads(response.text)

        except APIError as e:
            # Handle rate limits (429) or temporary server unavailability (503)
            if e.code in (429, 503) and attempt < max_retries - 1:
                print(
                    f"   API Limit/Busy ({e.code}). Sleeping {delay}s before retry... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)
                delay *= 2  # Exponential backoff: 5s, 10s, 20s, 40s
            else:
                raise e


def batch_clean_folder(input_folder, output_folder):
    abs_input = os.path.abspath(input_folder)
    abs_output = os.path.abspath(output_folder)

    print(f"Looking for PDFs in: {abs_input}")
    print(f"Saving JSONs to:     {abs_output}\n")

    if not os.path.exists(abs_input):
        print(f"ERROR: Input directory '{abs_input}' does not exist!")
        return

    os.makedirs(abs_output, exist_ok=True)

    all_files = os.listdir(abs_input)
    pdf_files = [f for f in all_files if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in '{abs_input}'.")
        return

    print(f"Found {len(pdf_files)} PDF(s) in folder...\n")

    for filename in pdf_files:
        full_pdf_path = os.path.join(abs_input, filename)
        json_filename = os.path.splitext(filename)[0] + ".json"
        output_json_path = os.path.join(abs_output, json_filename)

        # ---------------------------------------------------------------
        # 1. SKIP LOGIC: Protect existing JSONs with manual edits/URLs
        # ---------------------------------------------------------------
        if os.path.exists(output_json_path):
            print(
                f"Skipping {filename}: JSON already exists in output directory."
            )
            continue

        print(f"Processing new file: {filename}...")

        try:
            new_data = process_single_pdf(full_pdf_path)

            # ---------------------------------------------------------------
            # 2. MERGE LOGIC: Safely preserve any pre-existing custom metadata
            # ---------------------------------------------------------------
            if os.path.exists(output_json_path):
                try:
                    with open(output_json_path, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)

                    existing_metadata = existing_data.get("metadata", {})
                    new_metadata = new_data.get("metadata", {})

                    # Preserve manual fields like 'url' if already present
                    merged_metadata = {**new_metadata, **existing_metadata}
                    new_data["metadata"] = merged_metadata
                except Exception as merge_err:
                    print(
                        f"   Warning: Could not merge existing metadata ({merge_err}). Overwriting cleanly."
                    )

            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(new_data, f, indent=4, ensure_ascii=False)

            print(f"   Saved JSON to: {output_json_path}")

            # Pace out requests to stay below RPM rate limits
            time.sleep(3.0)

        except Exception as e:
            print(f"   Failed to process {filename}. Error: {e}")

    print("\nProcessing complete!")


INPUT_DIRECTORY = "input_pdfs"
OUTPUT_DIRECTORY = "output_json"

batch_clean_folder(INPUT_DIRECTORY, OUTPUT_DIRECTORY)