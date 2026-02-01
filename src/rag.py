
from collections.abc import Sequence
from datetime import datetime, timezone
import logging
import os
import re
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_community.vectorstores import Neo4jVector
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_tavily import TavilyExtract, TavilySearch

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://[^\s)>\]\"']+")


def extract_urls(text: str) -> list[str]:
    """Extract and normalize URLs from arbitrary text."""
    if not text:
        return []
    urls = _URL_RE.findall(text)
    cleaned = []
    for url in urls:
        cleaned.append(url.rstrip(").,;!?\"'"))
    # Preserve order while deduping
    return list(dict.fromkeys(u for u in cleaned if u))


def _neo4j_env() -> dict[str, str] | None:
    url = os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL")
    username = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    if not url or not username or not password:
        return None
    return {"url": url, "username": username, "password": password}


async def load_docs_pipeline(
    docs: Sequence[Document],
    *,
    splitter: TextSplitter,
    embedding: Embeddings,
    neo4j_url: str,
    neo4j_username: str,
    neo4j_password: str,
    index_name: str,
    node_label: str,
) -> Neo4jVector:
    """Chunk documents and write them to Neo4j using a Neo4jVector store.

    Neo4jVector.afrom_documents is the async constructor when you already have
    LangChain Document objects and want the method to handle inserts for you.
    """

    chunks = await splitter.atransform_documents(list(docs))
    return await Neo4jVector.afrom_documents(
        list(chunks),
        embedding=embedding,
        url=neo4j_url,
        username=neo4j_username,
        password=neo4j_password,
        index_name=index_name,
        node_label=node_label,
    )


async def ingest_relevant_websites(
    query: str,
    *,
    tavily_search: TavilySearch,
    tavily_extract: TavilyExtract,
    embedding: Embeddings,
    splitter: TextSplitter | None = None,
    neo4j_url: str | None = None,
    neo4j_username: str | None = None,
    neo4j_password: str | None = None,
    index_name: str = "web",
    node_label: str = "WebPage",
    seed_urls: Sequence[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    max_urls: int = 10,
) -> Neo4jVector | None:
    """Search + extract relevant webpages and store them in Neo4j.

    Uses Tavily Search/Extract to fetch web content, chunks it with a text splitter,
    and persists the chunks in a Neo4jVector index.
    """
    if not query and not seed_urls:
        return None

    urls: list[str] = []
    if seed_urls:
        urls.extend([u for u in seed_urls if u])

    search_payload = {"query": query} if query else None
    if search_payload:
        try:
            search_results = await tavily_search.ainvoke(search_payload)
            for result in search_results.get("results", []):
                url = result.get("url")
                if url:
                    urls.append(url)
        except Exception as exc:  # best-effort ingest
            logger.debug("Tavily search failed: %s", exc)

    urls = list(dict.fromkeys(u for u in urls if u))
    if not urls:
        return None
    if max_urls > 0:
        urls = urls[:max_urls]

    try:
        extract_results = await tavily_extract.ainvoke({"urls": urls})
    except Exception as exc:  # best-effort ingest
        logger.debug("Tavily extract failed: %s", exc)
        return None

    docs: list[Document] = []
    retrieved_at = datetime.now(timezone.utc).isoformat()
    base_metadata = {"retrieved_at": retrieved_at}
    if extra_metadata:
        base_metadata.update(extra_metadata)

    for result in extract_results.get("results", []):
        content = result.get("content") or result.get("raw_content") or ""
        if not content:
            continue
        metadata = dict(base_metadata)
        url = result.get("url") or result.get("source") or ""
        title = result.get("title") or result.get("name") or ""
        if url:
            metadata["source"] = url
        if title:
            metadata["title"] = title
        if query:
            metadata["query"] = query
        docs.append(Document(page_content=str(content), metadata=metadata))

    if not docs:
        return None

    splitter = splitter or RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    if not neo4j_url or not neo4j_username or not neo4j_password:
        env = _neo4j_env()
        if not env:
            return None
        neo4j_url = env["url"]
        neo4j_username = env["username"]
        neo4j_password = env["password"]

    return await load_docs_pipeline(
        docs,
        splitter=splitter,
        embedding=embedding,
        neo4j_url=neo4j_url,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
        index_name=index_name,
        node_label=node_label,
    )
