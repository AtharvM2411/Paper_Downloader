import os
import sqlite3
import requests
import arxiv
import hashlib
from pathlib import Path
import re
DB_NAME = "paper_cache.db"


# -----------------------------
# Database
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS papers(
        uid TEXT PRIMARY KEY,
        title TEXT,
        year INTEGER
    )
    """)

    conn.commit()
    return conn


def seen(conn, uid):
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM papers WHERE uid=?", (uid,))
    return cur.fetchone() is not None


def add(conn, uid, title, year):
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO papers VALUES (?,?,?)", (uid, title, year))
    conn.commit()


# -----------------------------
# Utilities
# -----------------------------
def uid_from_title(title):
    return hashlib.md5(title.lower().encode()).hexdigest()


def create_year_folder(base, year):
    p = Path(base) / str(year)
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_pdf(url, folder, title):

    safe = "".join(c for c in title if c.isalnum() or c in " _-")[:120]
    file_path = folder / f"{safe}.pdf"

    try:
        r = requests.get(url, stream=True, timeout=30)

        if r.status_code != 200:
            return False

        with open(file_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

        return True

    except Exception:
        return False


# -----------------------------
# Keyword Parsing
# -----------------------------
def load_keywords(file):
    kws = []
    with open(file) as f:
        for line in f:
            line = line.strip()
            if line:
                kws.append(line)
    return kws

def relevance_score(paper, tokens):

    title = (paper.get("title") or "").lower()
    abstract = (paper.get("abstract") or "").lower()

    score = 0

    for t in tokens:

        if t in title:
            score += 4

        if t in abstract:
            score += 2

    return score

def build_query(keywords):
    queries = []
    for kw in keywords:
        queries.append(f'ti:"{kw}" OR abs:"{kw}"')
    return " OR ".join(queries)

def build_token_set(keywords):

    tokens = set()

    for kw in keywords:

        words = re.findall(r"[a-zA-Z]+", kw.lower())

        for w in words:
            if len(w) > 2:   # ignore small words
                tokens.add(w)

    return tokens

# -----------------------------
# Source: arXiv
# -----------------------------
def search_arxiv(query):

    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=150,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []

    for r in client.results(search):

        papers.append({
            "id": r.get_short_id(),
            "title": r.title,
            "year": r.published.year,
            "pdf": r.pdf_url,
            "source": "arxiv"
        })

    return papers


# -----------------------------
# Source: Semantic Scholar
# -----------------------------
def search_semantic(query):

    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    params = {
        "query": query,
        "limit": 50,
        "fields": "title,year,openAccessPdf"
    }

    try:
        r = requests.get(url, params=params, timeout=30)

        if r.status_code != 200:
            print("SemanticScholar API error:", r.status_code)
            return []

        data_json = r.json()

        if "data" not in data_json:
            print("SemanticScholar response missing 'data'")
            print(data_json)
            return []

        data = data_json["data"]

    except Exception as e:
        print("SemanticScholar request failed:", e)
        return []

    papers = []

    for p in data:

        if not p.get("openAccessPdf"):
            continue

        papers.append({
            "id": p["title"],
            "title": p["title"],
            "year": p.get("year"),
            "pdf": p["openAccessPdf"]["url"],
            "source": "semantic"
        })

    return papers


# -----------------------------
# CORE API
# -----------------------------
def search_core(query):

    url = "https://api.core.ac.uk/v3/search/works"

    params = {
        "q": query,
        "limit": 50
    }

    r = requests.get(url, params=params)

    if r.status_code != 200:
        return []

    data = r.json()["results"]

    papers = []

    for p in data:

        if not p.get("downloadUrl"):
            continue

        papers.append({
            "id": str(p["id"]),
            "title": p["title"],
            "year": p.get("yearPublished"),
            "pdf": p["downloadUrl"],
            "source": "core"
        })

    return papers


# -----------------------------
# Unified Pipeline
# -----------------------------
def gather_papers(query):

    papers = []

    papers += search_arxiv(query)
    papers += search_semantic(query)
    papers += search_core(query)

    return papers


def run_pipeline(keyword_file, base_path, start_year, end_year):

    conn = init_db()

    kws = load_keywords(keyword_file)
    query = build_query(kws)

    print("Query:", query)

    papers = gather_papers(query)

    downloaded = 0
    tokens = build_token_set(kws)
    for p in papers:
        p["score"] = relevance_score(p, tokens)

    papers.sort(key=lambda x: x["score"], reverse=True)
    for p in papers:
        score = relevance_score(p, kws)
        if score < 2:
            continue

        if not p["year"]:
            continue

        if p["year"] < start_year or p["year"] > end_year:
            continue
        
        uid = uid_from_title(p["title"])

        if seen(conn, uid):
            print("Duplicate:", p["title"])
            continue

        folder = create_year_folder(base_path, p["year"])

        ok = download_pdf(p["pdf"], folder, p["title"])

        if ok:
            add(conn, uid, p["title"], p["year"])
            downloaded += 1
            print("Downloaded:", p["title"])

        if downloaded >= 200:
            break

    print("\nDownloaded", downloaded, "papers")

    conn.close()


# -----------------------------
# CLI
# -----------------------------
def main():

    keyword_file = input("Keyword file: ")
    folder = input("Download folder: ")

    start_year = int(input("Start year: "))
    end_year = int(input("End year: "))

    run_pipeline(keyword_file, folder, start_year, end_year)

##D:\__Atharv__\Self_Learning\Efficient Neural Network Training and Inference through Dynamic Model Adaptation\Papers
if __name__ == "__main__":
    main()
    