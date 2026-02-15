"""
PDF Extractor Tool for smolagents
Extracts and parses PDF files to markdown format using mineru backend
"""

import os
from pathlib import Path
from typing import Optional

from smolagents import Tool
from loguru import logger

from tools.pdf.pdf_utils import parse_doc
from rag_processor.embeddings import OpenAIEmbedding
from rag_processor.retrievers import SimpleVectorRetriever


class PDFExtractorTool(Tool):
    """
    Tool for extracting and parsing PDF files to markdown format.

    This tool takes a PDF file path as input and returns the first 2000 characters
    of the parsed markdown content. It uses the mineru backend for PDF parsing
    with support for formulas, tables, and OCR.
    """

    name = "pdf_extractor"
    description = (
        "Extract and parse PDF files to markdown format. Takes a PDF name as input "
        "and returns the first 2000 characters of the parsed markdown content. "
        "Supports complex PDFs with formulas, tables, and images. Uses cached results if available."
    )
    inputs = {
        "pdf_name": {
            "type": "string",
            "description": "The name of the PDF file (without .pdf extension) to be parsed",
        },
    }
    output_type = "string"

    def __init__(
        self,
        pdf_base_dir: Path | str = None,
        output_base_dir: Path | str = None,
    ):
        """Initialize the PDF extractor tool"""
        super().__init__()
        # Fixed paths for PDF processing
        if pdf_base_dir is None:
            self.pdf_base_dir = Path(
                "/data/wdy/AgenticPaperQA/paper-prepare/sampled_papers"
            )
        else:
            self.pdf_base_dir = Path(pdf_base_dir)
        if output_base_dir is None:
            self.output_base_dir = Path(
                "/data/wdy/AgenticPaperQA/paper-prepare/sampled_jsons"
            )
        else:
            self.output_base_dir = Path(output_base_dir)

    def setup(self):
        """Setup method called before first use"""
        logger.info("🔧 Setting up PDF extractor tool...")
        self.is_initialized = True

    def _validate_pdf_path(self, pdf_path: str) -> tuple[Path, Optional[str]]:
        """
        Validate PDF file path and return Path object

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (Path object, error message or None)
        """
        path = Path(pdf_path)

        if not path.exists():
            return None, f"PDF file not found: {pdf_path}"

        if not path.is_file():
            return None, f"Path is not a file: {pdf_path}"

        if path.suffix.lower() != ".pdf":
            return None, f"File is not a PDF: {pdf_path}"

        return path, None

    def _extract_markdown_content(
        self, output_dir: Path, pdf_name: str
    ) -> tuple[str, Optional[str]]:
        """
        Extract markdown content from parsed PDF output

        Args:
            output_dir: Directory containing parsed output
            pdf_name: Name of the PDF file (without extension)

        Returns:
            Tuple of (markdown content, error message or None)
        """
        # Use the fixed expected path: pdf_name/auto/pdf_name.md
        md_file_path = output_dir / pdf_name / "auto" / f"{pdf_name}.md"

        if not md_file_path.exists():
            return None, f"Markdown file not found at expected location: {md_file_path}"

        logger.info(f"📄 Reading markdown file: {md_file_path}")

        # Read the markdown content
        try:
            with open(md_file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content, None
        except Exception as e:
            error_msg = f"Failed to read markdown file: {e}"
            logger.error(f"❌ {error_msg}")
            return None, error_msg

    def forward(self, pdf_name: str) -> str:
        """
        Extract and parse PDF file to markdown format

        Args:
            pdf_name: Name of the PDF file (without .pdf extension)

        Returns:
            First 2000 characters of the parsed markdown content
        """
        # Fixed values for language and backend
        language = "en"
        backend = "pipeline"

        try:
            logger.info(f"🚀 Starting PDF extraction for: {pdf_name}")

            # Build paths
            pdf_file_path = self.pdf_base_dir / f"{pdf_name}.pdf"
            output_dir = self.output_base_dir

            # Expected markdown file path based on the structure: pdf_name/auto/pdf_name.md
            expected_md_path = output_dir / pdf_name / "auto" / f"{pdf_name}.md"

            # Check if already parsed (cache check)
            if expected_md_path.exists():
                logger.info(f"✅ Found cached markdown file: {expected_md_path}")
                try:
                    with open(expected_md_path, "r", encoding="utf-8") as f:
                        markdown_content = f.read()
                except Exception as e:
                    return f"Error: Failed to read cached markdown file: {e}"
            else:
                # Validate input PDF exists
                validated_pdf_path, validation_error = self._validate_pdf_path(
                    str(pdf_file_path)
                )
                if validation_error:
                    return f"Error: {validation_error}"

                logger.info(f"📁 Using output directory: {output_dir}")
                logger.info(f"⚙️ Using language: {language}, backend: {backend}")

                # Parse PDF using pdf_utils
                try:
                    parse_doc(
                        path_list=[validated_pdf_path],
                        output_dir=str(output_dir),
                        lang=language,
                        backend=backend,
                        method="auto",
                    )
                except Exception as e:
                    return f"Error: PDF parsing failed: {e}"

                # Extract markdown content
                markdown_content, extraction_error = self._extract_markdown_content(
                    output_dir, pdf_name
                )
                if extraction_error:
                    return f"Error: {extraction_error}"

            # Return first 2000 characters
            if len(markdown_content) > 2000:
                result = (
                    markdown_content[:2000]
                    + "\n\n... (content truncated to 2000 characters)"
                )
                logger.info(f"✂️ Content truncated to 2000 characters")
            else:
                result = markdown_content
                logger.info(f"📏 Content length: {len(markdown_content)} characters")

            logger.info("✅ PDF extraction completed successfully")
            return result

        except Exception as e:
            error_msg = f"❌ PDF extraction failed: {str(e)}"
            logger.error(error_msg)
            return f"Error: {str(e)}"

        finally:
            # No cleanup needed as we use persistent output directory
            pass


class PDFExtractorRAGTool(Tool):
    """
    Tool for extracting PDF content with query-based retrieval.

    This tool parses a PDF and then searches for content relevant to a specific query,
    returning the most relevant sections up to 2000 characters.
    """

    name = "pdf_extractor_with_query"
    description = (
        "Extract and parse PDF files, then retrieve content relevant to a specific query. "
        "Takes a PDF name and a search query as input, returns the most relevant "
        "sections of the parsed content up to 2000 characters."
    )
    inputs = {
        "pdf_name": {
            "type": "string",
            "description": "The name of the PDF file (without .pdf extension) to be parsed",
        },
        "query": {
            "type": "string",
            "description": "Search query to find relevant content in the PDF",
        },
    }
    output_type = "string"

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        pdf_base_dir: Path | str = None,
        output_base_dir: Path | str = None,
    ):
        """Initialize the PDF extractor RAG tool"""
        super().__init__()
        # Fixed paths for PDF processing (same as PDFExtractorTool)
        if pdf_base_dir is None:
            self.pdf_base_dir = Path(
                "/data/wdy/AgenticPaperQA/paper-prepare/sampled_papers"
            )
        else:
            self.pdf_base_dir = Path(pdf_base_dir)
        if output_base_dir is None:
            self.output_base_dir = Path(
                "/data/wdy/AgenticPaperQA/paper-prepare/sampled_jsons"
            )
        else:
            self.output_base_dir = Path(output_base_dir)

        # Initialize vector retriever for content search
        self.retriever = SimpleVectorRetriever(
            embedding_model=OpenAIEmbedding(),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            path=None,  # Don't use persistent storage for PDF content
        )

    def setup(self):
        """Setup method called before first use"""
        logger.info("🔧 Setting up PDF extractor RAG tool...")
        self.is_initialized = True

    def _search_content(self, content: str, query: str) -> str:
        """
        Vector-based content search using SimpleVectorRetriever

        Args:
            content: Full markdown content
            query: Search query

        Returns:
            Most relevant content sections up to 2000 characters
        """
        try:
            # Use vector retriever to find most relevant content
            relevant_chunks = self.retriever.retrieve(query=query, contents=[content])

            if not relevant_chunks:
                # If no relevant chunks found, return beginning of content
                return content[:2000] + "\n\n... (no specific matches found for query)"

            # Combine the most relevant chunks
            result_parts = []
            total_length = 0

            for chunk in relevant_chunks:
                chunk_text = (
                    chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                )
                if (
                    total_length + len(chunk_text) > 1800
                ):  # Leave room for truncation message
                    break
                result_parts.append(chunk_text)
                total_length += len(chunk_text)

            result = "\n\n---\n\n".join(result_parts)
            if len(result) > 2000:
                result = result[:2000] + "\n\n... (content truncated)"

            return result

        except Exception as e:
            logger.warning(
                f"⚠️ Vector search failed, falling back to simple search: {e}"
            )
            # Fallback to returning beginning of content
            return (
                content[:2000]
                + "\n\n... (vector search failed, showing beginning of content)"
            )

    def forward(self, pdf_name: str, query: str) -> str:
        """
        Extract PDF and retrieve query-relevant content

        Args:
            pdf_name: Name of the PDF file (without .pdf extension)
            query: Search query for relevant content

        Returns:
            Query-relevant content up to 2000 characters
        """
        try:
            logger.info(f"🔍 Extracting PDF with query: '{query}'")

            # Build paths
            pdf_file_path = self.pdf_base_dir / f"{pdf_name}.pdf"
            output_dir = self.output_base_dir

            # Expected markdown file path based on the structure: pdf_name/auto/pdf_name.md
            expected_md_path = output_dir / pdf_name / "auto" / f"{pdf_name}.md"

            # Check if already parsed (cache check)
            if expected_md_path.exists():
                logger.info(f"✅ Found cached markdown file: {expected_md_path}")
                try:
                    with open(expected_md_path, "r", encoding="utf-8") as f:
                        full_content = f.read()
                except Exception as e:
                    return f"Error: Failed to read cached markdown file: {e}"
            else:
                # Validate input PDF exists
                if not pdf_file_path.exists():
                    return f"Error: PDF file not found: {pdf_file_path}"

                if not pdf_file_path.is_file():
                    return f"Error: Path is not a file: {pdf_file_path}"

                if pdf_file_path.suffix.lower() != ".pdf":
                    return f"Error: File is not a PDF: {pdf_file_path}"

                logger.info(f"📁 Using output directory: {output_dir}")
                logger.info(f"⚙️ Using language: en, backend: pipeline")

                # Parse PDF using pdf_utils
                try:
                    parse_doc(
                        path_list=[pdf_file_path],
                        output_dir=str(output_dir),
                        lang="en",
                        backend="pipeline",
                        method="auto",
                    )
                except Exception as e:
                    return f"Error: PDF parsing failed: {e}"

                # Read the parsed markdown content
                if not expected_md_path.exists():
                    return f"Error: Markdown file not found at expected location: {expected_md_path}"

                try:
                    with open(expected_md_path, "r", encoding="utf-8") as f:
                        full_content = f.read()
                except Exception as e:
                    return f"Error: Failed to read markdown file: {e}"

            # Search for query-relevant content
            relevant_content = self._search_content(full_content, query)

            logger.info("✅ PDF extraction with query completed successfully")
            return f"Query: '{query}'\n\nRelevant content:\n\n{relevant_content}"

        except Exception as e:
            error_msg = f"❌ PDF extraction with query failed: {str(e)}"
            logger.error(error_msg)
            return f"Error: {str(e)}"


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    # Test 1: Non-existent PDF (graceful error)
    tool = PDFExtractorTool()
    result = tool.forward("nonexistent_pdf_12345")
    assert "Error:" in result or "not found" in result.lower()
    print("✅ PDFExtractorTool: graceful error for missing PDF")

    # Test 2: With custom temp dirs (no PDF - expect error)
    with tempfile.TemporaryDirectory() as tmp:
        pdf_dir = Path(tmp) / "pdfs"
        out_dir = Path(tmp) / "out"
        pdf_dir.mkdir()
        out_dir.mkdir()
        tool2 = PDFExtractorTool(pdf_base_dir=pdf_dir, output_base_dir=out_dir)
        result2 = tool2.forward("missing")
        assert "Error:" in result2 or "not found" in result2.lower()
        print("✅ PDFExtractorTool: custom paths work, missing PDF handled")

    # Test 3: If default path has a real PDF, run full extraction
    tool3 = PDFExtractorTool()
    if tool3.pdf_base_dir.exists():
        pdfs = list(tool3.pdf_base_dir.glob("*.pdf"))
        if pdfs:
            name = pdfs[0].stem
            r = tool3.forward(name)
            if "Error:" not in r:
                print(f"✅ PDFExtractorTool: extracted {name} (len={len(r)})")
    print("✅ pdf_extractor tests passed")
