"""Query the TETA RAG database interactively.

Ask clinical questions about thymic epithelial tumors and see
which guideline passages the database retrieves.

Run:  python query_db.py
"""

from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

VECTORSTORE_DIR = Path("vectorstore_data")


def get_collection():
    """Connect to the existing ChromaDB and return the guidelines collection."""
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-large-en-v1.5",
        device="cpu",
    )
    return client.get_collection(
        name="teta_guidelines",
        embedding_function=embed_fn,
    )


def search(collection, query, n_results=5):
    """Search the database and print results."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    print(f"\n🔍 Query: \"{query}\"")
    print(f"   Found {len(results['documents'][0])} results\n")

    for i, (doc, meta, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        score = 1 - distance
        source = meta.get("source", "?")
        page = meta.get("page", "?")
        year = meta.get("year", "?")
        author = meta.get("author", "?")

        print(f"--- Result {i + 1} (similarity: {score:.3f}) ---")
        print(f"📄 {source} ({author}, {year}) — page {page}")
        print()
        # Show the chunk text, cleaning up whitespace
        text = doc.replace("\n", " ").strip()
        # Show o1000 characters
        if len(text) > 1000:
            print(f"   {text[:1000]}...")
        else:
            print(f"   {text}")
        print()


def main():
    print("=" * 60)
    print("TETA RAG — Interactive Query")
    print("=" * 60)
    print()
    print("Ask questions about thymic epithelial tumors.")
    print("The database will find the most relevant guideline passages.")
    print()
    print("Commands:")
    print("  quit       — exit")
    print("  top N      — change number of results (default 5)")
    print()

    print("Loading database...")
    collection = get_collection()
    print(f"Ready! {collection.count()} chunks available.\n")

    n_results = 5

    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() == "quit":
            print("Goodbye!")
            break

        if query.lower().startswith("top "):
            try:
                n_results = int(query.split()[1])
                print(f"   Now showing top {n_results} results.\n")
            except (IndexError, ValueError):
                print("   Usage: top 3\n")
            continue

        search(collection, query, n_results=n_results)


if __name__ == "__main__":
    main()