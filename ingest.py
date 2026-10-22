import argparse
from pathlib import Path
from typing import Iterator

import fitz

from app.knowledge import embed, ensure_index


ROOT = Path(__file__).resolve().parent
DEFAULT_PDF = ROOT / "data" / "Medical_book.pdf"
EMBED_BATCH_SIZE = 64
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> Iterator[str]:
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("Chunk size must be positive and overlap must be smaller than the chunk size.")

    text = " ".join(text.split())
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end == len(text):
            break
        start = max(end - overlap, start + 1)


def ingest_pdf(pdf_path: Path, batch_size: int = EMBED_BATCH_SIZE) -> int:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    records: list[tuple[str, int, str]] = []
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            for chunk_number, text in enumerate(chunk_text(page.get_text()), start=1):
                records.append((f"page-{page_number}-chunk-{chunk_number}", page_number, text))

    if not records:
        raise ValueError(f"No readable text found in PDF: {pdf_path}")

    index = ensure_index()
    for offset in range(0, len(records), batch_size):
        batch = records[offset:offset + batch_size]
        vectors = embed([record[2] for record in batch])
        index.upsert(
            vectors=[
                {
                    "id": record_id,
                    "values": vector,
                    "metadata": {"page": page_number, "text": text},
                }
                for (record_id, page_number, text), vector in zip(batch, vectors, strict=True)
            ]
        )
        print(f"Uploaded {min(offset + len(batch), len(records))}/{len(records)} chunks")

    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed a PDF and upload its chunks to Pinecone.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="PDF path (default: bundled medical book)")
    parser.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE, help="Embedding batch size")
    args = parser.parse_args()

    count = ingest_pdf(args.pdf.resolve(), args.batch_size)
    print(f"Successfully ingested {count} chunks from {args.pdf}")


if __name__ == "__main__":
    main()