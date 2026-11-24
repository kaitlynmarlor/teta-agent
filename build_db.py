"""Build the TETA RAG database.

Pipeline:
  - Extract text from the guideline PDFs, page by page
  - Split into SEMANTIC chunks (cut where meaning shifts, not every N chars)
  - Record each chunk's page AND line span, so citations are checkable
  - Embed with a BIOMEDICAL model and store in ChromaDB

THREE CHOICES WORTH KNOWING ABOUT
---------------------------------
1. **Biomedical embeddings.** ``MedEmbed-large`` is fine-tuned from
   BGE-large on medical text. Same 1024 dimensions, so it is a drop-in
   replacement, but it has actually seen "thymoma", "Masaoka-Koga" and
   "paraneoplastic" during training rather than inferring them from
   general web text.

2. **Semantic chunking.** See ``teta_agent/chunking.py``. Fixed 1000-char
   cuts were severing recommendations from their exceptions.

3. **A NEW directory.** Embeddings from different models are not
   comparable, so this writes to ``vectorstore_v2/`` and leaves the
   original ``vectorstore_data/`` untouched as a fallback.

Run:  python build_db.py
"""

from pathlib import Path
from pypdf import PdfReader
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from teta_agent.chunking import semantic_chunks
from teta_agent.pages import PageMap

# ── Configuration ──────────────────────────────────────────────

GUIDELINES_DIR = Path("data/guidelines")

#: New store. The previous general-domain index stays in
#: ``vectorstore_data/`` so a demo can fall back to it.
VECTORSTORE_DIR = Path("vectorstore_v2")

#: Biomedical embedding model (1024-dim, fine-tuned from BGE-large).
EMBED_MODEL = "abhinand/MedEmbed-large-v0.1"

COLLECTION = "teta_guidelines"

# Metadata for each paper — every chunk will carry this info
# so when we retrieve a result, we know where it came from.
PAPER_INFO = {
    "Marx_2015_WHO_4th_ed_Classification.pdf": {
        "title": "The 2015 WHO Classification of Tumors of the Thymus",
        "author": "Marx et al.",
        "year": "2015",
        "source": "WHO 4th Edition Summary",
    },
    "Marx_2022_WHO_5th_ed_Review.pdf": {
        "title": "The 2021 WHO Classification of Tumors of the Thymus and Mediastinum",
        "author": "Marx et al.",
        "year": "2022",
        "source": "WHO 5th Edition Review",
    },
    "Marx_2014_ITMIG_WHO_Consensus.pdf": {
        "title": "ITMIG Consensus on the WHO Histological Classification of Thymomas",
        "author": "Marx et al.",
        "year": "2014",
        "source": "ITMIG Consensus Statement",
    },
    "2015-ESMO-Guidelines.pdf": {
        "title": "Thymic epithelial tumours: ESMO Clinical Practice Guidelines",
        "author": "Girard et al.",
        "year": "2015",
        "source": "ESMO Clinical Practice Guidelines",
    },
}


# ── Step 3 logic: Extract text ─────────────────────────────────

def extract_text(pdf_path):
    """Pull all text from a PDF, page by page."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"text": text, "page": i + 1})
    return pages


# ── Step 4 logic: Chunk the text ───────────────────────────────

def chunk_text(pages, paper_meta, encode, page_map=None):
    """Split extracted pages into semantic chunks with metadata.

    Thin wrapper over ``teta_agent.chunking.semantic_chunks`` -- the
    interesting logic lives there. Kept as a named function here because
    the build script reads as a pipeline and this is one of its stages.

    Args:
        pages: ``[{"text": ..., "page": physical_page}, ...]``.
        paper_meta: Per-document metadata from ``PAPER_INFO``.
        encode: Sentence-embedding callable, used to find topic boundaries.
        page_map: Optional ``PageMap`` so each chunk also records the
            *printed* page number -- the one a clinician would cite.
    """
    return semantic_chunks(pages, paper_meta, encode, page_map=page_map)


# ── Step 5+6 logic: Embed and store ───────────────────────────

def main():
    print("=" * 60)
    print("TETA RAG Database Builder")
    print("=" * 60)

    # Set up ChromaDB — this is where everything gets stored
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    # Set up the embedding function — ChromaDB will use this
    # automatically whenever we add documents or search
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    print("   (first run downloads ~1.3 GB)")
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL,
        device="cpu",
    )

    # The same model is used directly for sentence-level boundary
    # detection. Reusing one instance matters: loading MedEmbed costs ~30 s
    # and two copies would not fit comfortably in memory alongside torch.
    from sentence_transformers import SentenceTransformer

    sentence_model = SentenceTransformer(EMBED_MODEL, device="cpu")

    def encode(texts):
        return sentence_model.encode(
            texts, batch_size=32, show_progress_bar=False,
            convert_to_numpy=True,
        )

    # Create a fresh collection (delete old one if re-running)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    print("Database ready.\n")

    # Process each PDF
    pdf_files = sorted(GUIDELINES_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {GUIDELINES_DIR}/")
        return

    all_chunks = []
    for pdf_path in pdf_files:
        filename = pdf_path.name
        meta = PAPER_INFO.get(filename, {"title": filename, "source": "Unknown"})

        print(f"📄 {meta['source']}")
        pages = extract_text(pdf_path)
        print(f"   Extracted {len(pages)} pages of text")

        # Work out how physical PDF pages map to printed journal pages.
        page_map = PageMap.from_pages(pages)
        print(f"   📑 {page_map.describe()}")

        print("   Finding semantic boundaries...")
        chunks = chunk_text(pages, meta, encode, page_map=page_map)

        sizes = [len(c["text"]) for c in chunks] or [0]
        print(f"   Created {len(chunks)} chunks "
              f"(min {min(sizes)}, median {sorted(sizes)[len(sizes)//2]}, "
              f"max {max(sizes)} chars)")

        for chunk in chunks:
            chunk_id = f"{filename}::{chunk['metadata']['chunk_index']:04d}"
            # The citation viewer has to reopen the original PDF to draw
            # highlights. ``source`` is a human-readable display name, not
            # a path, so record the actual filename too rather than making
            # the viewer parse it back out of the chunk id.
            chunk["metadata"]["source_file"] = filename
            all_chunks.append((chunk_id, chunk))

    # Add everything to ChromaDB
    # ChromaDB handles the embedding automatically — it calls the
    # embedding function on each document behind the scenes
    print(f"\n📦 Embedding and storing {len(all_chunks)} chunks...")
    print("   (This takes a minute — embedding each chunk)")

    BATCH = 50
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i : i + BATCH]
        collection.add(
            ids=[item[0] for item in batch],
            documents=[item[1]["text"] for item in batch],
            metadatas=[item[1]["metadata"] for item in batch],
        )
        done = min(i + BATCH, len(all_chunks))
        print(f"   Stored {done}/{len(all_chunks)} chunks")

    print(f"\n✅ Database built! {collection.count()} chunks stored.")
    print(f"   Location: {VECTORSTORE_DIR}/")

    # ── Test it! ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Let's test it — searching for 'thymoma B1 diagnostic criteria'")
    print("=" * 60)

    results = collection.query(
        query_texts=["What are the diagnostic criteria for thymoma B1?"],
        n_results=3,
    )

    for i, (doc, meta, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        score = 1 - distance  # convert distance to similarity
        print(f"\n--- Result {i + 1} (similarity: {score:.3f}) ---")
        print(f"Source: {meta.get('source', '?')}, "
              f"page {meta.get('page', '?')} "
              f"(printed {meta.get('printed_page') or 'n/a'}), "
              f"lines {meta.get('line_start')}-{meta.get('line_end')}")
        # Show first 300 characters of the chunk
        preview = doc[:300].replace("\n", " ")
        print(f"Text: {preview}...")

    # Line provenance is what makes the clickable viewer possible, so a
    # build that silently lost it should be loud rather than subtle.
    sample = collection.get(limit=500, include=["metadatas"])
    missing = [
        m for m in sample["metadatas"] if not m.get("line_start")
    ]
    print(f"\n📐 Line provenance: "
          f"{len(sample['metadatas']) - len(missing)}/{len(sample['metadatas'])}"
          f" chunks carry a line span")
    if missing:
        print(f"   ⚠️  {len(missing)} chunks are missing it")


if __name__ == "__main__":
    main()