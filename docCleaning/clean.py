import json
import os
import time
from google import genai
from google.genai import types
import pdfplumber

# Initialize Gemini Client (grabs GEMINI_API_KEY from terminal environment)
client = genai.Client()


def process_single_pdf(pdf_path):
    """Opens a single PDF file, extracts text, and cleans it via Gemini API."""
    raw_text = ""

    # Open PDF and extract text
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                raw_text += extracted + "\n"

    prompt = f"""
    You are an automated document parsing assistant. 
    Analyze the text below from a single document. 
    
    1. Extract metadata fields into JSON format.
    2. Remove all non-essential web UI elements, navigation menus, header print artifacts (dates, URLs, page counts like "1/3"), and footer copyright boilerplate.
    3. Return the main content formatted as clean, continuous body text.

    DOCUMENT TEXT:
    {raw_text}
    """

    # Updated to gemini-3.6-flash as requested by the API error
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


def batch_clean_folder(input_folder, output_folder):
    """Finds ALL PDFs in input_folder and saves a JSON for each into output_folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_files = os.listdir(input_folder)
    pdf_files = [f for f in all_files if f.lower().endswith(".pdf")]

    print(f"Found {len(pdf_files)} PDF(s) to process...\n")

    for filename in pdf_files:
        full_pdf_path = os.path.join(input_folder, filename)
        print(f"Processing: {filename}...")

        try:
            # Send to Gemini
            data = process_single_pdf(full_pdf_path)

            # Name JSON file after PDF file
            json_filename = os.path.splitext(filename)[0] + ".json"
            output_json_path = os.path.join(output_folder, json_filename)

            # Save clean output
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

            print(f"   Saved JSON to: {output_json_path}")

        except Exception as e:
            print(f"   Failed to process {filename}. Error: {e}")

        # Slight delay to prevent rate limiting when processing hundreds of docs
        time.sleep(1)

    print("\nProcessing complete!")


# Run execution
INPUT_DIRECTORY = "input_pdfs"
OUTPUT_DIRECTORY = "output_json"

batch_clean_folder(INPUT_DIRECTORY, OUTPUT_DIRECTORY)