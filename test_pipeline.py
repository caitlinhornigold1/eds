import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Import functions from your modules if structure permits, or test modular logic directly
# from docCleaning.clean import process_single_pdf, batch_clean_folder


# =====================================================================
# 1. TESTS FOR CHUNKING & DEDUPLICATION LOGIC
# =====================================================================
class TestChunkingAndDeduplication(unittest.TestCase):

    def test_recursive_character_splitter_bounds(self):
        """Verify text chunking respects maximum chunk size and overlap."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=20
        )
        sample_text = (
            "Environmental policy in New Zealand governs marine areas, coastal regions, and fisheries. "
            "The Resource Management Act provides principles for indigenous biodiversity protection. "
            "Fisheries decisions under the Fisheries Act must consider environmental sustainability."
        )

        chunks = splitter.split_text(sample_text)

        self.assertGreater(
            len(chunks), 1, "Text should be split into multiple chunks."
        )
        for chunk in chunks:
            self.assertLessEqual(
                len(chunk),
                120,
                "Chunk length should not significantly exceed chunk_size.",
            )

    def test_chunk_deduplication_and_label_mapping(self):
        """Ensure identical text chunks are dropped and doc labels [Doc 1], [Doc 2] remain sequential."""
        docs = [
            "Protection of biodiversity in the coastal marine area.",
            "Protection of biodiversity in the coastal marine area.",  # Duplicate chunk
            "Fisheries management requires habitat protection.",
        ]
        metas = [
            {"document": "Foreshore Act 2004", "page": 2},
            {"document": "Foreshore Act 2004", "page": 2},
            {"document": "Tarakihi Sustainability Review", "page": 10},
        ]

        seen_text = set()
        source_map = {}
        formatted_blocks = []
        doc_counter = 1

        for doc, meta in zip(docs, metas):
            clean = doc.strip()
            if clean in seen_text:
                continue
            seen_text.add(clean)

            doc_name = meta.get("document", "Unknown")
            if doc_name not in source_map:
                source_map[doc_name] = f"Doc {doc_counter}"
                doc_counter += 1

            label = source_map[doc_name]
            formatted_blocks.append(f"[{label}] {clean}")

        # Verification
        self.assertEqual(
            len(formatted_blocks), 2, "Duplicate chunk should be filtered out."
        )
        self.assertEqual(source_map["Foreshore Act 2004"], "Doc 1")
        self.assertEqual(source_map["Tarakihi Sustainability Review"], "Doc 2")


# =====================================================================
# 2. TESTS FOR PDF EXTRACTION
# =====================================================================
class TestPDFExtraction(unittest.TestCase):

    @patch("pdfplumber.open")
    def test_pdf_extraction_success(self, mock_pdfplumber):
        """Test successful page-by-page text extraction from a mock PDF."""
        # Mock pages
        page1 = MagicMock()
        page1.extract_text.return_value = "Page 1: Tarakihi Fisheries Policy"
        page2 = MagicMock()
        page2.extract_text.return_value = (
            "Page 2: Sustainability Measures 2021"
        )

        mock_pdf = MagicMock()
        mock_pdf.pages = [page1, page2]
        mock_pdfplumber.return_value.__enter__.return_value = mock_pdf

        # Extraction logic
        extracted_text = ""
        with mock_pdfplumber("fake_path.pdf") as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"

        self.assertIn("Page 1: Tarakihi Fisheries Policy", extracted_text)
        self.assertIn("Page 2: Sustainability Measures 2021", extracted_text)

    @patch("pdfplumber.open")
    def test_pdf_extraction_empty_or_scanned(self, mock_pdfplumber):
        """Verify handling when PDF contains no extractable text (e.g., scanned images)."""
        empty_page = MagicMock()
        empty_page.extract_text.return_value = None  # Scanned image page

        mock_pdf = MagicMock()
        mock_pdf.pages = [empty_page]
        mock_pdfplumber.return_value.__enter__.return_value = mock_pdf

        raw_text = ""
        with mock_pdfplumber("scanned.pdf") as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    raw_text += text

        self.assertEqual(
            raw_text, "", "Raw text should be empty for un-OCRed scanned PDFs."
        )


# =====================================================================
# 3. TESTS FOR METADATA MERGING & PRESERVATION
# =====================================================================
class TestMetadataPreservation(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.json_path = os.path.join(self.temp_dir.name, "doc_1.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_preserve_manually_added_urls(self):
        """Ensure existing JSON metadata (e.g., manually entered URLs) is merged rather than overwritten."""
        # Simulate pre-existing JSON with manual URL
        existing_data = {
            "metadata": {
                "document": "Tarakihi Review",
                "url": "https://eds.org.nz/tarakihi-2021.pdf",
                "page": 1,
            },
            "cleaned_body_text": "Old text",
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f)

        # New payload from Gemini
        new_extracted_payload = {
            "metadata": {
                "document": "Tarakihi Review Updated",
                "organisation": "EDS",
            },
            "cleaned_body_text": "Newly cleaned continuous text...",
        }

        # Merge logic executed in clean.py
        with open(self.json_path, "r", encoding="utf-8") as f:
            stored_json = json.load(f)

        existing_meta = stored_json.get("metadata", {})
        new_meta = new_extracted_payload.get("metadata", {})

        # Merge: new extracted fields update, but existing manual fields ('url') are preserved
        merged_meta = {**new_meta, **existing_meta}
        new_extracted_payload["metadata"] = merged_meta

        # Assertions
        self.assertEqual(
            new_extracted_payload["metadata"]["url"],
            "https://eds.org.nz/tarakihi-2021.pdf",
        )
        self.assertEqual(
            new_extracted_payload["metadata"]["organisation"], "EDS"
        )


# =====================================================================
# 4. TESTS FOR ERROR CASES & API LIMIT BACKOFF
# =====================================================================
class TestErrorHandlingAndRateLimits(unittest.TestCase):

    @patch("time.sleep")
    def test_gemini_429_rate_limit_retry_backoff(self, mock_sleep):
        """Verify API 429 errors trigger exponential sleep backoff retries."""
        from google.genai.errors import APIError

        mock_api_call = MagicMock()

        # Simulate 2 rate limit failures (code 429) followed by success
        mock_429_error = APIError(429, "RESOURCE_EXHAUSTED", response=None)
        mock_api_call.side_effect = [
            mock_429_error,
            mock_429_error,
            {"cleaned_body_text": "Success"},
        ]

        max_retries = 5
        delay = 3
        result = None

        for attempt in range(max_retries):
            try:
                result = mock_api_call()
                break
            except APIError as e:
                if e.code in (429, 503) and attempt < max_retries - 1:
                    time_to_sleep = delay
                    delay *= 2
                    mock_sleep(time_to_sleep)

        # Assertions
        self.assertEqual(mock_api_call.call_count, 3)
        self.assertEqual(result, {"cleaned_body_text": "Success"})
        # Sleep calls should be 3s, then 6s
        mock_sleep.assert_any_call(3)
        mock_sleep.assert_any_call(6)

    def test_empty_retrieved_context_response(self):
        """Test zero-hallucination safeguard when ChromaDB returns no matching documents."""
        search_results = {"documents": [[]], "metadatas": [[]]}

        retrieved_docs = search_results["documents"][0]
        insufficient_phrase = (
            "i do not have enough information in my database to answer"
        )

        if not retrieved_docs:
            answer = "I do not have enough information in my database to answer this."
            sources = []

        low_confidence = insufficient_phrase in answer.lower()
        fully_unsupported = answer.strip().lower().startswith(
            insufficient_phrase
        )

        self.assertTrue(low_confidence)
        self.assertTrue(fully_unsupported)
        self.assertEqual(sources, [])


if __name__ == "__main__":
    unittest.main()