"""Document loading and Egyptian Civil Code PDF processing.

The generic text-loading helpers remain available for compatibility. Legal PDFs
are processed directly here: the PDF's two-column table is read into paired
English/Arabic rows, article headings are detected, and article-level records
are emitted for ProcessingService.
"""

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import pymupdf
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

from models import ProcessingEnum
from utils.logger import get_logger

logger = get_logger(__name__)

_DIGIT_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)
_ARABIC_NORMALIZATION = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ؤ": "و",
        "ئ": "ي",
        "ى": "ي",
    }
)
_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_EN_ARTICLE_RE = re.compile(
    r"^\s*(?:ARTICLE|RTICLE|ARTICL[E3])\s*"
    r"(?:NO\.?\s*)?[:#-]?\s*(\d+)\b",
    re.IGNORECASE,
)
_BOOK_RE = re.compile(r"^BOOK\s+(?:[IVXLCDM]+|\d+|[A-Z])\b\s*(.*)$", re.IGNORECASE)
_CHAPTER_RE = re.compile(
    r"^CHAPTER\s+(?:[IVXLCDM]+|\d+|[A-Z])\b\s*(.*)$", re.IGNORECASE
)
_SECTION_RE = re.compile(
    r"^SECTION\s+(?:[IVXLCDM]+|\d+|[A-Z])\b\s*(.*)$", re.IGNORECASE
)
_TOPIC_RE = re.compile(r"^(\d{1,3})\s*[-.)]\s*(.+)$")
_ARABIC_ARTICLE_RE = re.compile(r"^(?:المادة|مادة|ماده)(\d+)")
_ARABIC_REVERSED_ARTICLE_RE = re.compile(
    r"^(?:(\d+)(?:ةداملا|ةدامل|ةدام)|(?:ةداملا|ةدامل|ةدام)(\d+))"
)
_REPEALED_RANGES = ((54, 80), (389, 417))

# The supplied Egyptian Civil Code PDF has no Arabic body for Article 1022.
# Keep its original English text; never fabricate an Arabic translation.
_KNOWN_ARABIC_MISSING_ARTICLES = {1022}
_MAX_CHUNK_CHARS = 4000


@dataclass
class Document:
    page_content: str
    metadata: dict


@dataclass(frozen=True)
class _PDFRow:
    page_number: int
    y_position: float
    text_en: str
    text_ar: str
    is_bold_en: bool = False


class DocumentProcessor:
    """Load documents and extract bilingual legal articles from table-based PDFs."""

    def __init__(
        self,
        project_path: str,
        chunk_size: int = 100,
        overlap_size: int = 20,
    ):
        self.project_path = project_path
        self._validate_chunk_config(chunk_size=chunk_size, overlap_size=overlap_size)
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        logger.info(
            "Document processor initialized: chunk_size=%s | overlap=%s",
            self.chunk_size,
            self.overlap_size,
        )

    @staticmethod
    def _validate_chunk_config(chunk_size: int, overlap_size: int) -> None:
        if not isinstance(chunk_size, int):
            raise TypeError("chunk_size must be an integer")
        if not isinstance(overlap_size, int):
            raise TypeError("overlap_size must be an integer")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap_size < 0:
            raise ValueError("overlap_size cannot be negative")
        if overlap_size >= chunk_size:
            raise ValueError("overlap_size must be smaller than chunk_size")

    @staticmethod
    def get_file_extension(file_id: str) -> str:
        return os.path.splitext(file_id)[-1].lower()

    def get_file_loader(self, file_id: str):
        file_ext = self.get_file_extension(file_id=file_id)
        file_path = os.path.join(self.project_path, file_id)

        if not os.path.exists(file_path):
            logger.error("File not found: file_id=%s | path=%s", file_id, file_path)
            return None
        if file_ext == ProcessingEnum.TXT.value:
            return TextLoader(file_path, encoding="utf-8")
        if file_ext == ProcessingEnum.PDF.value:
            return file_path

        logger.error("Unsupported file extension: %s", file_ext)
        return None

    def get_file_content(self, file_id: str) -> list[Document] | None:
        loader = self.get_file_loader(file_id=file_id)
        if loader is None:
            return None

        try:
            file_ext = self.get_file_extension(file_id=file_id)
            if file_ext == ProcessingEnum.TXT.value:
                documents = loader.load()
                return [
                    Document(page_content=doc.page_content, metadata=doc.metadata)
                    for doc in documents
                    if doc.page_content.strip()
                ]

            if file_ext == ProcessingEnum.PDF.value:
                pdf = pymupdf.open(loader)
                documents = []
                try:
                    for page_number, page in enumerate(pdf):
                        text = page.get_text("text").strip()
                        if not text:
                            continue
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={"source": file_id, "page": page_number + 1},
                            )
                        )
                finally:
                    pdf.close()

                logger.info(
                    "Loaded PDF successfully: file_id=%s | pages=%s",
                    file_id,
                    len(documents),
                )
                return documents
        except Exception:
            logger.exception("Failed to load file: file_id=%s", file_id)
            return None

        return None

    def process_file_content(self, file_content: list[Document], file_id: str):
        """Legacy character-based processing, retained for existing callers."""
        logger.info(
            "Processing file: file_id=%s | chunk_size=%s characters | overlap=%s characters",
            file_id,
            self.chunk_size,
            self.overlap_size,
        )
        if not file_content:
            logger.warning("No content found for file_id=%s", file_id)
            return []

        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap_size,
            separator=" ",
            length_function=len,
        )
        chunks = text_splitter.create_documents(
            [record.page_content for record in file_content],
            metadatas=[record.metadata for record in file_content],
        )
        for index, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "file_id": file_id,
                    "chunk_index": index,
                    "chunk_size": len(chunk.page_content),
                }
            )
        self._log_chunk_statistics(chunks=chunks, file_id=file_id)
        return chunks

    @staticmethod
    def _log_chunk_statistics(chunks: list, file_id: str) -> None:
        if not chunks:
            logger.warning("No chunks generated for file_id=%s", file_id)
            return
        lengths = [len(chunk.page_content) for chunk in chunks]
        logger.info(
            "Chunk statistics | file_id=%s | chunks=%s | min=%s characters | "
            "max=%s characters | mean=%.2f characters | median=%.2f characters",
            file_id,
            len(chunks),
            min(lengths),
            max(lengths),
            mean(lengths),
            median(lengths),
        )

    # ---------------------------------------------------------------------
    # Legal PDF processing
    # ---------------------------------------------------------------------

    @staticmethod
    def _clean_text(value: Any, *, arabic: bool = False) -> str:
        if value is None:
            return ""
        text = unicodedata.normalize("NFKC", str(value))
        text = text.replace("\x00", "").replace("\u00ad", "").replace("\ufeff", "")
        text = text.replace("\r", "\n")
        if arabic:
            text = _ARABIC_DIACRITICS.sub("", text)
            text = text.replace("ـ", "")
            text = text.translate(_ARABIC_NORMALIZATION)
        lines = [
            re.sub(r"[ \t\u00a0]+", " ", line).strip() for line in text.split("\n")
        ]
        return "\n".join(line for line in lines if line)

    @classmethod
    def _heading_text(cls, value: str) -> str:
        return re.sub(r"\s+", " ", cls._clean_text(value)).strip()

    @classmethod
    def _arabic_compact(cls, value: str) -> str:
        return re.sub(r"\s+", "", cls._clean_text(value, arabic=True)).translate(
            _DIGIT_TRANSLATION
        )

    @staticmethod
    def _script_side(value: str) -> str:
        arabic_letters = len(re.findall(r"[\u0600-\u06ff]", value))
        latin_letters = len(re.findall(r"[A-Za-z]", value))
        if arabic_letters > latin_letters:
            return "ar"
        return "en"

    @staticmethod
    def _normalize_number_markers(value: str) -> str:
        """Fix common malformed numeric paragraph markers only."""
        value = value or ""

        patterns = (
            # (1 ( -> (1)
            re.compile(r"(?<![0-9٠-٩۰-۹])\(\s*([0-9٠-٩۰-۹]{1,4})\s*\("),
            # )1 ( -> (1)
            re.compile(r"(?<![0-9٠-٩۰-۹])\)\s*([0-9٠-٩۰-۹]{1,4})\s*\("),
            # )1 ) -> (1)
            re.compile(r"(?<![0-9٠-٩۰-۹])\)\s*([0-9٠-٩۰-۹]{1,4})\s*\)"),
            # ]1 [ -> (1)
            re.compile(r"(?<![0-9٠-٩۰-۹])\]\s*([0-9٠-٩۰-۹]{1,4})\s*\["),
        )

        def replace_marker(match):
            number = match.group(1).translate(_DIGIT_TRANSLATION)
            return f"({number})"

        for pattern in patterns:
            value = pattern.sub(replace_marker, value)

        # Ensure a space after a normalized marker when Arabic or English
        # text immediately follows it, e.g. "(2)ومع" -> "(2) ومع".
        value = re.sub(r"\)(?=[\u0600-\u06FFA-Za-z])", ") ", value)

        return value

    @staticmethod
    def _normalize_english_linebreaks(value: str) -> str:
        """Join obvious split words and convert layout line breaks to spaces."""
        value = value or ""

        # Repair one-letter fragments only when they form a known two-letter word.
        two_letter_words = {
            "of",
            "to",
            "in",
            "an",
            "as",
            "at",
            "be",
            "by",
            "do",
            "if",
            "is",
            "it",
            "no",
            "on",
            "or",
            "so",
            "up",
            "us",
            "we",
            "he",
            "my",
            "me",
            "am",
        }

        single_letter_break = re.compile(
            r"(?<![A-Za-z])([A-Za-z])\n([A-Za-z])(?![A-Za-z])"
        )

        def join_single_letter_fragment(match):
            combined = match.group(1) + match.group(2)
            if combined.casefold() in two_letter_words:
                return combined
            return match.group(0)

        value = single_letter_break.sub(join_single_letter_fragment, value)

        # Join two-letter prefixes such as "pa" in "pa\nrties", but do not
        # join complete words such as "cost\nof" or "own\ncost".
        complete_words = {
            "am",
            "an",
            "as",
            "at",
            "be",
            "by",
            "do",
            "go",
            "he",
            "if",
            "in",
            "is",
            "it",
            "me",
            "my",
            "no",
            "of",
            "on",
            "or",
            "to",
            "up",
            "us",
            "we",
        }

        split_prefix = re.compile(
            r"(?<![A-Za-z])([A-Za-z]{2})\n([a-z]{2,12})"
            r"(?=\s|[.,;:!?)]|$)"
        )

        def join_split_prefix(match):
            prefix = match.group(1)
            if prefix.casefold() in complete_words:
                return match.group(0)
            return prefix + match.group(2)

        value = split_prefix.sub(join_split_prefix, value)

        # Remaining newlines are PDF layout wraps between words.
        value = re.sub(r"[ \t]*\n[ \t]*", " ", value)
        value = re.sub(r"[ \t]{2,}", " ", value)

        # Repair the confirmed missing space in the supplied PDF extraction.
        value = re.sub(r"\bfor(?=carrying\b)", "for ", value, flags=re.IGNORECASE)

        # Repair a confirmed missing-space artifact in the source PDF extraction.
        value = re.sub(
            r"\bfor(?=carrying\b)",
            "for ",
            value,
            flags=re.IGNORECASE,
        )

        return value.strip()

    @classmethod
    def _extract_page_rows(cls, page, page_number: int) -> list[_PDFRow]:
        """Extract positioned bilingual row text and retain English bold styling."""
        page_midpoint = float(page.rect.width) / 2
        page_width = float(page.rect.width)
        page_lines: list[dict] = []

        try:
            page_dict = page.get_text("dict")
        except Exception:  # noqa: BLE001
            page_dict = {"blocks": []}

        # Unlike table.extract(), positioned text spans retain font information
        # and, for this PDF, provide the Arabic text in readable logical order.
        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                raw_text = "".join(str(span.get("text", "")) for span in spans)
                value = cls._clean_text(raw_text)

                if not value:
                    continue

                bbox = line.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue

                page_lines.append(
                    {
                        "bbox": tuple(float(v) for v in bbox[:4]),
                        "text": value,
                        "bold": any(
                            "bold" in str(span.get("font", "")).lower()
                            or bool(int(span.get("flags", 0)) & 16)
                            for span in spans
                        ),
                    }
                )

        page_lines.sort(
            key=lambda item: (
                round(item["bbox"][1], 1),
                (
                    -item["bbox"][2]
                    if cls._script_side(item["text"]) == "ar"
                    else item["bbox"][0]
                ),
            )
        )

        try:
            finder = page.find_tables(strategy="lines")
            tables = getattr(finder, "tables", []) or []
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Table geometry unavailable on page %s: %s",
                page_number,
                type(exc).__name__,
            )
            tables = []

        found_rows: list[_PDFRow] = []

        for table in tables:
            try:
                extracted_rows = table.extract() or []
                table_rows = getattr(table, "rows", []) or []
            except Exception:
                logger.exception(
                    "Could not read table geometry on page %s", page_number
                )
                continue

            for row_index, cell_texts in enumerate(extracted_rows):
                geometry = (
                    getattr(table_rows[row_index], "cells", [])
                    if row_index < len(table_rows)
                    else []
                )

                english_parts: list[str] = []
                arabic_parts: list[str] = []
                y_positions: list[float] = []
                row_bold_en = False

                for cell_index, cell_text in enumerate(cell_texts or []):
                    bbox = geometry[cell_index] if cell_index < len(geometry) else None

                    raw_cell = cls._clean_text(cell_text)
                    if not raw_cell and not bbox:
                        continue

                    cell_lines: list[dict] = []
                    side = cls._script_side(raw_cell) if raw_cell else "en"

                    if bbox and len(bbox) >= 4:
                        x0, y0, x1, y1 = map(float, bbox[:4])
                        cell_width = abs(x1 - x0)
                        center_x = (x0 + x1) / 2
                        (y0 + y1) / 2

                        y_positions.append(y0)

                        full_width_cell = cell_width > page_width * 0.70
                        central_cell = (
                            abs(center_x - page_midpoint) < page_width * 0.035
                        )

                        # Full-width/central cells can contain one language per
                        # positioned line, so classify those lines individually.
                        for item in page_lines:
                            bx0, by0, bx1, by1 = item["bbox"]
                            line_x = (bx0 + bx1) / 2
                            line_y = (by0 + by1) / 2

                            if not (y0 - 1.5 <= line_y <= y1 + 1.5):
                                continue

                            if (
                                not full_width_cell
                                and not central_cell
                                and not (x0 - 1.5 <= line_x <= x1 + 1.5)
                            ):
                                continue

                            cell_lines.append(item)

                        if not full_width_cell and not central_cell:
                            side = "en" if center_x < page_midpoint else "ar"

                    # Positioned spans fix reversed Arabic extraction. If a span
                    # cannot be matched to a cell, retain the original cell text
                    # rather than silently dropping potentially important content.
                    if cell_lines:
                        cell_lines.sort(
                            key=lambda item: (
                                round(item["bbox"][1], 1),
                                (
                                    -item["bbox"][2]
                                    if cls._script_side(item["text"]) == "ar"
                                    else item["bbox"][0]
                                ),
                            )
                        )

                        if bbox and (
                            abs(float(bbox[2]) - float(bbox[0])) > page_width * 0.70
                            or abs(
                                (float(bbox[0]) + float(bbox[2])) / 2 - page_midpoint
                            )
                            < page_width * 0.035
                        ):
                            for item in cell_lines:
                                line_side = cls._script_side(item["text"])
                                if line_side == "ar":
                                    arabic_parts.append(item["text"])
                                else:
                                    english_parts.append(item["text"])
                                    row_bold_en = row_bold_en or item["bold"]
                        else:
                            value = "\n".join(
                                item["text"] for item in cell_lines
                            ).strip()

                            if side == "ar":
                                arabic_parts.append(value)
                            else:
                                english_parts.append(value)
                                row_bold_en = row_bold_en or any(
                                    item["bold"] for item in cell_lines
                                )
                    elif raw_cell:
                        if side == "ar":
                            arabic_parts.append(raw_cell)
                        else:
                            english_parts.append(raw_cell)

                text_en = cls._clean_text("\n".join(english_parts))
                text_ar = cls._clean_text("\n".join(arabic_parts), arabic=True)

                if text_en or text_ar:
                    found_rows.append(
                        _PDFRow(
                            page_number=page_number,
                            y_position=(
                                min(y_positions) if y_positions else float(row_index)
                            ),
                            text_en=text_en,
                            text_ar=text_ar,
                            is_bold_en=row_bold_en,
                        )
                    )

        if found_rows:
            return sorted(
                found_rows,
                key=lambda row: row.y_position,
            )

        # Fallback for pages where the PDF has no usable ruled table geometry.
        # Preserve each positioned text line and its style signal.
        for item in page_lines:
            value = item["text"]
            side = cls._script_side(value)
            bbox = item["bbox"]

            found_rows.append(
                _PDFRow(
                    page_number=page_number,
                    y_position=bbox[1],
                    text_en=value if side == "en" else "",
                    text_ar=value if side == "ar" else "",
                    is_bold_en=item["bold"] and side == "en",
                )
            )

        return sorted(
            found_rows,
            key=lambda row: row.y_position,
        )

    @classmethod
    def _find_article_number(cls, text_en: str, text_ar: str) -> int | None:
        """Find an article label at the beginning of either language column."""
        english = cls._heading_text(text_en).translate(_DIGIT_TRANSLATION)
        match = _EN_ARTICLE_RE.match(english)

        if match:
            return int(match.group(1))

        arabic = cls._arabic_compact(text_ar)

        match = _ARABIC_ARTICLE_RE.match(arabic)
        if match:
            return int(match.group(1))

        # Some PDF text layers return the Arabic label in reversed order.
        match = _ARABIC_REVERSED_ARTICLE_RE.match(arabic)
        if match:
            number = match.group(1) or match.group(2)
            return int(number) if number else None

        return None

    @classmethod
    def _strip_article_prefix(cls, text: str, *, arabic: bool = False) -> str:
        """Remove a leading article label while preserving the provision text."""
        if not text:
            return ""

        if not arabic:
            cleaned = cls._heading_text(text).translate(_DIGIT_TRANSLATION)
            return re.sub(
                r"^(?:ARTICLE|RTICLE|ARTICL[E3])\s*"
                r"(?:NO\.?\s*)?[:#-]?\s*\d+\s*[:.)-]?\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip()

        value = cls._clean_text(text, arabic=True).translate(_DIGIT_TRANSLATION)

        # The PDF sometimes separates the number and the word مادة across
        # lines, with extra parentheses between them.
        label = r"(?:(?:ا\s*ل\s*)?م\s*ا\s*د\s*ة|م\s*ا\s*د\s*ه)"

        number_first = r"[\(\[]?\s*\d+\s*[\)\]]?[\s\(\[]*" + label
        label_first = label + r"[\s\(\[]*\d+"

        prefix = re.compile(
            r"^\s*(?:" + number_first + r"|" + label_first + r")\s*[\)\]\(\s:.-]*"
        )

        return prefix.sub("", value, count=1).strip()

    @classmethod
    def _detect_heading(
        cls,
        text_en: str,
        text_ar: str,
        *,
        article_is_open: bool,
        is_bold_en: bool = False,
    ) -> tuple[str, str, bool] | None:
        """Return (level, English heading, waiting_for_continuation)."""
        english = cls._heading_text(text_en)
        upper = english.upper()
        arabic_compact = cls._arabic_compact(text_ar)

        if ("GENERAL PROVISIONS" in upper and len(english) <= 180) or (
            "الاحكامالعامة" in arabic_compact or "احكامعامة" in arabic_compact
        ):
            return "general", "General Provisions", False

        for level, pattern in (
            ("book", _BOOK_RE),
            ("chapter", _CHAPTER_RE),
            ("section", _SECTION_RE),
        ):
            match = pattern.match(english)
            if match:
                remainder = (match.group(1) or "").strip(" -:.;\n\t")
                if not remainder:
                    return level, english.strip(), True
                return level, english.strip(), False

        # Only use numbered headings as topics when their PDF typography marks
        # them as headings. This avoids classifying numbered article paragraphs.
        topic_match = _TOPIC_RE.match(english)
        if topic_match and is_bold_en:
            return "topic", english.strip(), False

        # The legal PDF formats ordinary topic headings in bold even where no
        # explicit "Topic" or numeric prefix is printed, for example:
        # "The Acquisition of Movables by Prescription".
        if (
            is_bold_en
            and english
            and len(english) <= 220
            and not _EN_ARTICLE_RE.match(english)
            and not re.match(
                r"^(?:BOOK|CHAPTER|SECTION)\b",
                english,
                re.IGNORECASE,
            )
            and len(english.split()) >= 2
        ):
            return "topic", english.strip(), False

        return None

    @classmethod
    def _is_repeal_notice(cls, text_en: str, text_ar: str) -> bool:
        """Recognize repeal notices without dropping articles that mention repeal."""
        english = re.sub(r"\s+", " ", text_en or "").strip()
        english = english.translate(_DIGIT_TRANSLATION)

        # Require an explicit article-number range associated with a repeal notice.
        # A provision discussing how laws may be repealed is not itself a notice.
        english_notice = bool(
            re.search(
                r"\barticles?\s+\d+\s*(?:-|–|—|to)\s*\d+\b"
                r".{0,120}\brepealed\b",
                english,
                flags=re.IGNORECASE,
            )
            or re.search(
                r"\brepealed\b.{0,80}"
                r"\barticles?\s+\d+\s*(?:-|–|—|to)\s*\d+\b",
                english,
                flags=re.IGNORECASE,
            )
        )

        # Arabic text may be visually reversed by the PDF extractor. Normalize
        # digits and inspect the notice terms without relying on digit order.
        arabic = cls._arabic_compact(text_ar or "")
        arabic = arabic.translate(_DIGIT_TRANSLATION)

        has_arabic_repeal_term = any(
            word in arabic for word in ("ملغاة", "ملغيت", "الغيت")
        )
        has_multiple_numbers = len(re.findall(r"\d+", arabic)) >= 2

        arabic_notice = (
            "المواد" in arabic and has_arabic_repeal_term and has_multiple_numbers
        )

        return english_notice or arabic_notice

    @staticmethod
    def _repealed_by_number(article_number: int) -> bool:
        return any(start <= article_number <= end for start, end in _REPEALED_RANGES)

    @classmethod
    def _parse_article_rows(cls, rows: list[_PDFRow]) -> list[dict]:
        """Parse bilingual article bodies and retain English hierarchy titles."""
        context: dict[str, str | None] = {
            "book": "General Provisions",
            "chapter": "General Provisions",
            "section": None,
            "topic": None,
        }

        articles: list[dict] = []
        current: dict | None = None
        pending_heading: str | None = None

        def flush_current() -> None:
            nonlocal current

            if current is None:
                return

            paragraphs = current.pop("_paragraphs", [])

            # Clean only obvious English line-wrap artifacts and numeric markers.
            # Arabic reading order and general Arabic text extraction remain unchanged.
            for paragraph in paragraphs:
                paragraph["en"] = cls._normalize_english_linebreaks(
                    cls._normalize_number_markers(paragraph.get("en", ""))
                )
                paragraph["ar"] = cls._normalize_number_markers(paragraph.get("ar", ""))

            current["text_en"] = cls._normalize_english_linebreaks(
                cls._normalize_number_markers(
                    "\n".join(
                        item["en"] for item in paragraphs if item.get("en")
                    ).strip()
                )
            )

            current["text_ar"] = cls._normalize_number_markers(
                "\n".join(item["ar"] for item in paragraphs if item.get("ar")).strip()
            )

            # Preserve normalized paired rows for long-article splitting.
            current["_paragraph_pairs"] = paragraphs
            articles.append(current)
            current = None

        for row in rows:
            english = cls._clean_text(row.text_en)
            arabic = cls._clean_text(row.text_ar, arabic=True)

            if not english and not arabic:
                continue

            if cls._is_repeal_notice(english, arabic):
                flush_current()
                pending_heading = None
                continue

            heading = cls._detect_heading(
                english,
                arabic,
                article_is_open=current is not None,
                is_bold_en=row.is_bold_en and not bool(pending_heading),
            )
            article_number = cls._find_article_number(english, arabic)

            if heading and article_number is None:
                flush_current()

                level, title, waiting = heading
                pending_heading = level if waiting else None

                if level == "general":
                    context = {
                        "book": "General Provisions",
                        "chapter": "General Provisions",
                        "section": None,
                        "topic": None,
                    }

                elif level == "book":
                    context["book"] = title
                    context["chapter"] = None
                    context["section"] = None
                    context["topic"] = None

                elif level == "chapter":
                    context["chapter"] = title
                    context["section"] = None
                    context["topic"] = None

                elif level == "section":
                    context["section"] = title
                    context["topic"] = None

                elif level == "topic":
                    context["topic"] = title

                continue

            # A heading identifier such as "CHAPTER I" may be followed by
            # its English title on the next row. Ignore Arabic-only title rows
            # because the hierarchy fields intentionally store English only.
            if pending_heading and article_number is None:
                candidate = cls._heading_text(english)

                if not candidate:
                    continue

                if len(candidate) <= 180:
                    old_value = context.get(pending_heading)

                    if old_value and candidate.casefold() not in old_value.casefold():
                        context[pending_heading] = f"{old_value} - {candidate}"
                    elif not old_value:
                        context[pending_heading] = candidate

                    pending_heading = None
                    continue

                pending_heading = None

            if article_number is not None:
                if current is not None and current["article_number"] == article_number:
                    # A repeated label at a page boundary continues the article.
                    body_en = cls._strip_article_prefix(english)
                    body_ar = cls._strip_article_prefix(arabic, arabic=True)

                    if body_en or body_ar:
                        current["_paragraphs"].append({"en": body_en, "ar": body_ar})

                    continue

                flush_current()

                current = {
                    "article_number": article_number,
                    "book": context.get("book"),
                    "chapter": context.get("chapter"),
                    "section": context.get("section"),
                    "topic": context.get("topic"),
                    "source_page": row.page_number,
                    "citation": f"Egyptian Civil Code, Article {article_number}",
                    "is_repealed": cls._repealed_by_number(article_number),
                    "_paragraphs": [],
                }

                body_en = cls._strip_article_prefix(english)
                body_ar = cls._strip_article_prefix(arabic, arabic=True)

                if body_en or body_ar:
                    current["_paragraphs"].append({"en": body_en, "ar": body_ar})

                continue

            if current is not None:
                current["_paragraphs"].append({"en": english, "ar": arabic})

        flush_current()
        return articles

    @classmethod
    def _insert_repealed_articles(cls, articles: list[dict]) -> list[dict]:
        """Keep known repealed article numbers as explicit legal records."""
        by_number: dict[int, dict] = {}

        for article in articles:
            number = article["article_number"]

            if number not in by_number:
                by_number[number] = article
                continue

            existing = by_number[number]
            new_pairs = article.get("_paragraph_pairs", [])

            if new_pairs:
                existing_pairs = existing.setdefault("_paragraph_pairs", [])
                existing_pairs.extend(new_pairs)

                existing["text_en"] = "\n".join(
                    item["en"] for item in existing_pairs if item.get("en")
                ).strip()

                existing["text_ar"] = "\n".join(
                    item["ar"] for item in existing_pairs if item.get("ar")
                ).strip()

            else:
                for language_key in ("text_en", "text_ar"):
                    old_value = existing.get(language_key, "")
                    new_value = article.get(language_key, "")

                    if new_value and new_value not in old_value:
                        existing[language_key] = "\n".join(
                            part for part in (old_value, new_value) if part
                        )

            if cls._repealed_by_number(number):
                existing["is_repealed"] = True

        for start, end in _REPEALED_RANGES:
            for number in range(start, end + 1):
                if number in by_number:
                    by_number[number]["is_repealed"] = True
                    continue

                prior_numbers = [
                    existing for existing in by_number if existing < number
                ]
                prior = by_number[max(prior_numbers)] if prior_numbers else {}

                by_number[number] = {
                    "article_number": number,
                    "book": prior.get("book", "General Provisions"),
                    "chapter": prior.get("chapter", "General Provisions"),
                    "section": prior.get("section"),
                    "topic": prior.get("topic"),
                    "text_en": "",
                    "text_ar": "",
                    "is_repealed": True,
                    "source_page": None,
                    "citation": f"Egyptian Civil Code, Article {number}",
                }

        result = [by_number[number] for number in sorted(by_number)]

        # Apply normalization to the final records after duplicate merging.
        for article in result:
            article["text_en"] = cls._normalize_english_linebreaks(
                cls._normalize_number_markers(article.get("text_en", ""))
            )
            article["text_ar"] = cls._normalize_number_markers(
                article.get("text_ar", "")
            )

        return result

    @classmethod
    def _split_long_articles(cls, articles: list[dict]) -> list[dict]:
        """Split oversized articles at row boundaries without losing citations."""
        chunks: list[dict] = []

        def render(paragraphs: list[dict], language: str) -> str:
            return "\n".join(
                item[language] for item in paragraphs if item.get(language)
            ).strip()

        for article in articles:
            text_en = (article.get("text_en") or "").strip()
            text_ar = (article.get("text_ar") or "").strip()

            # Synthetic repeal records do not need body chunking.
            if article.get("is_repealed"):
                record = dict(article)
                record.pop("_paragraph_pairs", None)
                chunks.append(record)
                continue

            if len(text_en) <= _MAX_CHUNK_CHARS and len(text_ar) <= _MAX_CHUNK_CHARS:
                record = dict(article)
                record.pop("_paragraph_pairs", None)
                chunks.append(record)
                continue

            paragraphs = article.get("_paragraph_pairs", [])
            if not paragraphs:
                raise ValueError(
                    "Cannot safely split long article without paired PDF rows: "
                    f"article={article['article_number']}"
                )

            groups: list[list[dict]] = []
            current_group: list[dict] = []

            for paragraph in paragraphs:
                paragraph_en = paragraph.get("en", "")
                paragraph_ar = paragraph.get("ar", "")

                if (
                    len(paragraph_en) > _MAX_CHUNK_CHARS
                    or len(paragraph_ar) > _MAX_CHUNK_CHARS
                ):
                    raise ValueError(
                        "A single extracted legal row exceeds the maximum "
                        f"chunk length: article={article['article_number']} "
                        f"| max={_MAX_CHUNK_CHARS}"
                    )

                candidate = current_group + [paragraph]
                candidate_en = render(candidate, "en")
                candidate_ar = render(candidate, "ar")

                exceeds_limit = (
                    len(candidate_en) > _MAX_CHUNK_CHARS
                    or len(candidate_ar) > _MAX_CHUNK_CHARS
                )

                if current_group and exceeds_limit:
                    groups.append(current_group)
                    current_group = [paragraph]
                else:
                    current_group = candidate

            if current_group:
                groups.append(current_group)

            total_parts = len(groups)

            for part_number, group in enumerate(groups, start=1):
                record = dict(article)
                record.pop("_paragraph_pairs", None)

                record["text_en"] = render(group, "en")
                record["text_ar"] = render(group, "ar")
                record["article_part"] = part_number
                record["article_parts"] = total_parts

                chunks.append(record)

        return chunks

    @classmethod
    def _validate_articles(cls, articles: list[dict]) -> None:
        """Validate article numbering and both language bodies before indexing."""
        if not articles:
            raise ValueError(
                "No articles were extracted from the PDF. "
                "Check its table geometry and text layer."
            )

        numbers = [article["article_number"] for article in articles]
        unique_numbers = sorted(set(numbers))

        if len(numbers) != len(unique_numbers):
            raise ValueError("Duplicate article numbers remain after PDF extraction")

        number_set = set(unique_numbers)
        gaps = [
            number
            for number in range(unique_numbers[0], unique_numbers[-1] + 1)
            if number not in number_set
        ]

        if gaps:
            preview = ", ".join(str(number) for number in gaps[:30])
            raise ValueError(f"Unexplained article-number gaps detected: {preview}")

        known_arabic_omissions = [
            article["article_number"]
            for article in articles
            if article.get("article_number") in _KNOWN_ARABIC_MISSING_ARTICLES
            and not article.get("is_repealed")
            and not (article.get("text_ar") or "").strip()
        ]

        missing_arabic = [
            article["article_number"]
            for article in articles
            if not article.get("is_repealed")
            and article["article_number"] not in _KNOWN_ARABIC_MISSING_ARTICLES
            and not (article.get("text_ar") or "").strip()
        ]

        if missing_arabic:
            preview = ", ".join(str(number) for number in missing_arabic[:20])
            raise ValueError(
                "Arabic text is empty for active articles. "
                f"Check PDF extraction: {preview}"
            )

        if known_arabic_omissions:
            logger.warning(
                "Source PDF has no Arabic text for known article(s): %s. "
                "Preserving the English text without generating Arabic text.",
                ", ".join(map(str, known_arabic_omissions)),
            )

        missing_english = [
            article["article_number"]
            for article in articles
            if not article.get("is_repealed") and not article.get("text_en", "").strip()
        ]

        if missing_english:
            preview = ", ".join(str(number) for number in missing_english[:20])
            raise ValueError(
                "English text is empty for active articles. "
                f"Check PDF extraction: {preview}"
            )

    @staticmethod
    def _remove_arabic_paragraph_numbering(
        articles: list[dict],
    ) -> list[dict]:
        """Remove Arabic subparagraph markers without changing article references."""
        import re

        marker_patterns = [
            re.compile(r"(?<!\w)\(\s*[0-9٠-٩۰-۹]{1,2}\s*\)(?!\w)"),
            re.compile(r"(?<!\w)\)\s*[0-9٠-٩۰-۹]{1,2}\s*\((?!\w)"),
            re.compile(r"(?<!\S)[0-9٠-٩۰-۹]{1,2}\s*\(\s*(?=\S)"),
        ]

        reference_context = re.compile(
            r"(?:المادة|مادة|المواد|المادتان|المادتين|"
            r"البند|الفقرة|فقرة)\s*"
            r"(?:رقم\s*)?"
            r"(?:[0-9٠-٩۰-۹]+\s*)?$"
        )

        for article in articles:
            text = article.get("text_ar")

            if not isinstance(text, str) or not text:
                continue

            for pattern in marker_patterns:
                original_text = text

                def replace_marker(match, source_text=original_text):
                    prefix = source_text[max(0, match.start() - 60) : match.start()]

                    # Preserve legal references such as "المادة (2)".
                    if reference_context.search(prefix):
                        return match.group(0)

                    return " "

                text = pattern.sub(replace_marker, text)

            # Join line-wrapped Arabic text into one continuous paragraph.
            article["text_ar"] = " ".join(text.split())

        return articles

    def extract_legal_articles(self, file_path: str | Path) -> list[dict]:
        """Extract bilingual article records directly from the supplied PDF."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Legal PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Legal article extraction requires a PDF file: {path.name}"
            )

        rows: list[_PDFRow] = []
        pdf = pymupdf.open(path)
        try:
            for page_number, page in enumerate(pdf, start=1):
                rows.extend(self._extract_page_rows(page, page_number))
        finally:
            pdf.close()

        rows.sort(key=lambda row: (row.page_number, row.y_position))
        logger.info(
            "Bilingual PDF rows extracted: file=%s | pages=%s | rows=%s",
            path.name,
            len({row.page_number for row in rows}),
            len(rows),
        )

        articles = self._parse_article_rows(rows)
        articles = self._remove_arabic_paragraph_numbering(articles)
        articles = self._insert_repealed_articles(articles)
        self._validate_articles(articles)
        chunks = self._split_long_articles(articles)

        logger.info(
            "Legal PDF processing completed: file=%s | articles=%s | article_chunks=%s",
            path.name,
            len(articles),
            len(chunks),
        )
        return chunks
