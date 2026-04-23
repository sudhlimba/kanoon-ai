"""
document_downloader.py
-----------------------
Automatically downloads the 4 Indian legal documents from official / reliable sources.
Run this ONCE before starting the app:  python document_downloader.py

The PDFs are saved to the  documents/  folder.
"""

import os
import requests

DOCUMENTS_DIR = "documents"

# ── Document sources ──────────────────────────────────────────────────────────
# These are reliable public-domain / government sources for each act.
# If any URL stops working in the future, replace it with another source.
DOCUMENTS = [
    {
        "name"    : "Indian Constitution",
        "filename": "indian_constitution.pdf",
        "url"     : "https://legislative.gov.in/sites/default/files/COI...pdf",
        # FALLBACK: Download manually from https://legislative.gov.in/constitution-of-india/
        "manual_url": "https://legislative.gov.in/constitution-of-india/",
    },
    {
        "name"    : "RTI Act 2005",
        "filename": "rti_act_2005.pdf",
        "url"     : "https://rti.gov.in/rti-act.pdf",
        "manual_url": "https://rti.gov.in/",
    },
    {
        "name"    : "IT Act 2000",
        "filename": "it_act_2000.pdf",
        "url"     : "https://legislative.gov.in/sites/default/files/A2000-21.pdf",
        "manual_url": "https://legislative.gov.in/",
    },
    {
        "name"    : "Consumer Protection Act 2019",
        "filename": "consumer_protection_act_2019.pdf",
        "url"     : "https://consumeraffairs.nic.in/sites/default/files/CP%20Act%202019.pdf",
        "manual_url": "https://consumeraffairs.nic.in/",
    },
]


def download_documents():
    """Download all legal documents to the documents/ folder."""
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)

    print("\n📥 Downloading Indian Legal Documents...\n")
    print("=" * 55)

    for doc in DOCUMENTS:
        filepath = os.path.join(DOCUMENTS_DIR, doc["filename"])

        # Skip if already downloaded
        if os.path.exists(filepath) and os.path.getsize(filepath) > 10_000:
            print(f"  ✅ Already exists: {doc['name']}")
            continue

        print(f"  ⬇️  Downloading: {doc['name']} ...")
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
            response = requests.get(doc["url"], headers=headers, timeout=30)
            response.raise_for_status()

            with open(filepath, "wb") as f:
                f.write(response.content)

            size_kb = os.path.getsize(filepath) // 1024
            print(f"     ✅ Saved ({size_kb} KB): {filepath}")

        except Exception as e:
            print(f"     ❌ Auto-download failed: {e}")
            print(f"     👉 Please download manually from: {doc['manual_url']}")
            print(f"     👉 Save it as: {filepath}")

    print("\n" + "=" * 55)
    print("Done! Now run:  streamlit run app.py\n")


def check_documents_exist() -> dict:
    """
    Check which documents are present.
    Returns dict: { filename: bool (exists or not) }
    """
    status = {}
    for doc in DOCUMENTS:
        filepath = os.path.join(DOCUMENTS_DIR, doc["filename"])
        exists   = os.path.exists(filepath) and os.path.getsize(filepath) > 10_000
        status[doc["filename"]] = {
            "name"   : doc["name"],
            "exists" : exists,
            "path"   : filepath,
        }
    return status


def get_available_document_paths() -> list:
    """Returns list of paths for documents that actually exist on disk."""
    status = check_documents_exist()
    return [
        info["path"]
        for info in status.values()
        if info["exists"]
    ]


if __name__ == "__main__":
    download_documents()
