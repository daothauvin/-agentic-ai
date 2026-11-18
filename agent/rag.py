"""RAG sémantique sur la documentation PyPI des packages.

Pipeline :
    1. ``build_document``  : métadonnées PyPI JSON -> texte Markdown (README).
    2. ``chunk_markdown``  : découpage par sections Markdown (titres conservés).
    3. ``BgeM3Embedder``   : embeddings denses bge-m3 (servis par Ollama).
    4. ``FaissIndex``      : index FAISS cosinus (IndexFlatIP sur vecteurs normalisés),
                             persisté sur disque par package/version.

Chaque vecteur de l'index est une *entrée* rattachée à un chunk (son
« propriétaire »). Le score d'un chunk est le max des scores de ses entrées.
En mode **multi-niveau** (``multi_level=True``, défaut), chaque phrase / ligne
de code est indexée en plus de sa section : une phrase trouvée renvoie sa
section entière (« small-to-big »).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
import requests

from .logger import get_logger, preview

logger = get_logger("rag")

EMBED_MODEL = "bge-m3"
# À incrémenter quand le document indexé change : invalide les index en cache.
# v4 : section « Installation with pip » (commande pip de la page PyPI).
INDEX_FORMAT = 4



# ==============================================================
# 1. Document
# ==============================================================

def build_document(pypi_json: dict) -> tuple[str, str]:
    """Construit (métadonnées Markdown, README) à partir de la réponse JSON PyPI.

    Les deux parties sont découpées séparément pour que les titres du README
    ne s'imbriquent pas sous la section des métadonnées.

    La commande ``pip install <nom>`` affichée en tête de la page PyPI n'est pas
    dans le README (yt-dlp, pillow, openpyxl n'en donnent aucune) : elle est
    ajoutée comme section « Installation with pip », si la version
    publie des fichiers installables (wheel / sdist).

    Args:
        pypi_json: Réponse JSON de l'API PyPI (``/pypi/<nom>/json``).

    Returns:
        Tuple (document des métadonnées, README) en Markdown.
    """

    info = pypi_json.get("info", {}) or {}
    name = info.get("name") or "unknown"
    lines = [
        f"# {name} {info.get('version') or ''} - Package metadata".strip(),
        "",
        info.get("summary") or "",
        "",
        f"- PyPI name: {name}",
        f"- Requires Python: {info.get('requires_python') or 'unspecified'}",
    ]
    requires = info.get("requires_dist") or []
    if requires:
        lines.append("- Dependencies: " + ", ".join(requires[:30]))
    for label, url in (info.get("project_urls") or {}).items():
        lines.append(f"- {label}: {url}")
    files = pypi_json.get("urls") or []
    if files:
        lines += [
            "",
            # Titre de niveau 1 : section autonome (chemin « Installation with pip »,
            # sans le préfixe « Package metadata » qui dilue l'embedding).
            "# Installation with pip",
            "",
            # Formulation proche de la requête d'installation : le moindre bruit
            # (ex. « bdist_wheel, sdist available ») fait perdre ~0,025 de score.
            f"To install the {name} package from PyPI, run the pip install command:",
            "",
            "```",
            f"pip install {name}",
            "```",
        ]
    else:
        logger.warning("Aucun fichier publié sur PyPI pour %s : pas de commande pip ajoutée", name)
    return "\n".join(lines), (info.get("description") or "").replace("\r\n", "\n")


# ==============================================================
# 2. Chunking
# ==============================================================

@dataclass
class Chunk:
    """Unité renvoyée au LLM : une section (ou partie de section) Markdown.

    Attributes:
        id: Position du chunk dans le document.
        section: Chemin des titres, ex. ``"Usage > CLI"``.
        text: Contenu du chunk.
    """

    id: int
    section: str
    text: str

    def render(self) -> str:
        """Texte du chunk précédé de son titre de section, pour le contexte LLM."""

        return f"[Section: '{self.section}']\n{self.text}"


@dataclass
class Entry:
    """Texte embarqué dans l'index FAISS, rattaché à un chunk.

    Attributes:
        owner: Id du chunk renvoyé quand cette entrée est trouvée.
        kind: ``"chunk"`` ou ``"sentence"``.
        text: Texte réellement embarqué.
    """

    owner: int
    kind: str
    text: str


_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_SETEXT = re.compile(r"^(=+|-+)\s*$")


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """Découpe en (chemin_de_titres, contenu) sans couper les blocs de code."""

    sections: list[tuple[str, list[str]]] = []
    path: list[tuple[int, str]] = []
    current: list[str] = []
    in_code = False
    lines = markdown.split("\n")

    def flush():
        title = " > ".join(t for _, t in path) or "Introduction"
        if any(l.strip() for l in current):
            sections.append((title, current.copy()))
        current.clear()

    for i, line in enumerate(lines):
        if line.lstrip().startswith(("```", "~~~")):
            in_code = not in_code
            current.append(line)
            continue
        level, title = None, None
        if not in_code:
            m = _HEADING.match(line)
            if m:
                level, title = len(m.group(1)), m.group(2)
            elif (
                _SETEXT.match(line) and current and current[-1].strip()
                and not current[-1].lstrip().startswith(("-", "*", "|"))
                and len(line.strip()) >= 3
            ):
                # Titre "setext" (reStructuredText / Markdown souligné).
                title = current.pop().strip()
                level = 1 if line.strip()[0] == "=" else 2
        if title is not None:
            flush()
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, title))
        else:
            current.append(line)
    flush()
    return [(t, "\n".join(body).strip()) for t, body in sections]


def _split_long(text: str, max_chars: int) -> list[str]:
    """Découpe un texte trop long par paragraphes (les blocs de code restent entiers si possible)."""

    if len(text) <= max_chars:
        return [text]
    blocks, buf, in_code = [], [], False
    for line in text.split("\n"):
        if line.lstrip().startswith(("```", "~~~")):
            in_code = not in_code
        buf.append(line)
        if not in_code and not line.strip():
            blocks.append("\n".join(buf))
            buf = []
    if buf:
        blocks.append("\n".join(buf))

    parts, cur = [], ""
    for b in blocks:
        if cur and len(cur) + len(b) > max_chars:
            parts.append(cur.strip())
            cur = ""
        while len(b) > 2 * max_chars:  # bloc géant : coupe brute
            parts.append(b[:max_chars].strip())
            b = b[max_chars:]
        cur += b + "\n"
    if cur.strip():
        parts.append(cur.strip())
    return [p for p in parts if p]


def chunk_markdown(*documents: str, max_chars: int = 1500) -> list[Chunk]:
    """Découpe un ou plusieurs documents Markdown (indépendants) en chunks.

    Args:
        *documents: Documents Markdown, découpés indépendamment.
        max_chars: Taille maximale (en caractères) visée pour un chunk.

    Returns:
        Liste ordonnée des chunks (ids consécutifs).
    """

    chunks: list[Chunk] = []
    for doc_no, doc in enumerate(documents):
        sections = _split_sections(doc)
        logger.debug(
            "Chunking doc %d : %d car., %d sections", doc_no, len(doc), len(sections)
        )
        for section, body in sections:
            parts = _split_long(body, max_chars)
            if len(parts) > 1:
                logger.debug(
                    "  section '%s' (%d car.) re-découpée en %d parties",
                    section, len(body), len(parts),
                )
            for part in parts:
                chunks.append(Chunk(id=len(chunks), section=section, text=part))
    log_chunks(chunks, max_chars)
    return chunks


def log_chunks(chunks: list[Chunk], max_chars: int | None = None) -> None:
    """Logue les statistiques de taille et le détail de chaque chunk.

    Args:
        chunks: Chunks à décrire.
        max_chars: Taille maximale visée, pour signaler les dépassements.
    """

    if not chunks:
        logger.warning("Chunking : aucun chunk produit (document vide ?)")
        return
    sizes = np.array([len(c.text) for c in chunks])
    logger.info(
        "Chunking : %d chunks | taille min=%d moy=%d médiane=%d max=%d car.",
        len(chunks), sizes.min(), int(sizes.mean()), int(np.median(sizes)), sizes.max(),
    )
    for c in chunks:
        flag = ""
        if max_chars and len(c.text) > max_chars:
            flag = "  ⚠ > max_chars"
        elif len(c.text) < 50:
            flag = "  ⚠ très court"
        logger.debug(
            "  chunk #%d [%s] %d car.%s : %s",
            c.id, c.section, len(c.text), flag, preview(c.text, 200),
        )


_SENTENCE_END = re.compile(r"(?<=[.!?:])\s+(?=[A-Z0-9`*\"'(\[])")
_BULLET = re.compile(r"^\s*([-*+]|\d+[.)])\s+")


def split_sentences(text: str, max_chars: int = 400) -> list[str]:
    """Découpe un chunk en phrases (prose) et en lignes (blocs de code).

    Les lignes de code sont gardées une par une : une ligne ``pip install x``
    ou ``x.make(...)`` est un signal très précis qu'une section entière dilue.

    Args:
        text: Contenu d'un chunk.
        max_chars: Longueur maximale d'une phrase (au-delà : tronquée).

    Returns:
        Phrases / lignes non triviales, dans l'ordre du texte.
    """

    units: list[str] = []
    paragraph: list[str] = []
    in_code = False

    def flush_paragraph() -> None:
        prose = " ".join(l.strip() for l in paragraph if l.strip())
        paragraph.clear()
        units.extend(s for s in _SENTENCE_END.split(prose) if len(s.strip()) >= 15)

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            flush_paragraph()
            in_code = not in_code
            continue
        # Bloc de code (clôturé, indenté ou ligne de console) : une ligne = une unité.
        if in_code or line.startswith(("    ", "\t")) or stripped.startswith((">>>", "$ ")):
            flush_paragraph()
            if len(stripped) >= 6:
                units.append(stripped)
        elif not stripped or _BULLET.match(line) or stripped.startswith("|"):
            flush_paragraph()
            if stripped:
                paragraph.append(stripped)
        else:
            paragraph.append(stripped)
    flush_paragraph()
    return [u.strip()[:max_chars] for u in units]


# ==============================================================
# 3. Embeddings bge-m3
# ==============================================================

class BgeM3Embedder:
    """Embeddings denses bge-m3 via l'API /api/embed d'Ollama."""

    def __init__(self, ollama_url: str, model: str = EMBED_MODEL, batch_size: int = 16):
        self.url = ollama_url.rstrip("/") + "/api/embed"
        self.model = model
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        """Calcule les embeddings normalisés L2 d'une liste de textes.

        Args:
            texts: Textes à encoder.

        Returns:
            Matrice float32 (len(texts), dim), lignes de norme 1.
        """

        vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = requests.post(
                self.url,
                json={"model": self.model, "input": batch},
                timeout=300,
            )
            response.raise_for_status()
            got = response.json()["embeddings"]
            logger.debug(
                "Embedding batch %d-%d : %d textes -> %d vecteurs",
                i, i + len(batch) - 1, len(batch), len(got),
            )
            vectors.extend(got)
        arr = np.asarray(vectors, dtype="float32")
        if arr.size == 0:
            logger.warning("Embedding : aucun vecteur renvoyé pour %d textes", len(texts))
            return arr.reshape(0, 0)
        norms = np.linalg.norm(arr, axis=1)
        logger.debug(
            "Embedding : %d vecteurs dim=%d (modèle %s), norme brute min=%.3f max=%.3f",
            arr.shape[0], arr.shape[1], self.model, norms.min(), norms.max(),
        )
        if np.any(norms == 0) or not np.all(np.isfinite(arr)):
            logger.warning("Embedding : vecteurs nuls ou non finis détectés !")
        faiss.normalize_L2(arr)  # produit scalaire = cosinus
        return arr


# ==============================================================
# 4. Index FAISS
# ==============================================================

class FaissIndex:
    """Index FAISS cosinus d'entrées rattachées à des chunks, persistable sur disque.

    Attributes:
        index: Index FAISS (un vecteur par entrée).
        chunks: Chunks renvoyables (sections).
        entries: Entrées embarquées ; ``entries[i]`` correspond au vecteur ``i``.
        meta: Métadonnées du package (``name``, ``summary``).
    """

    def __init__(
        self,
        index: faiss.Index,
        chunks: list[Chunk],
        entries: list[Entry] | None = None,
        meta: dict | None = None,
    ):
        self.index = index
        self.chunks = chunks
        # Index ancien format : un vecteur par chunk.
        self.entries = entries or [Entry(c.id, "chunk", c.text) for c in chunks]
        self.meta = meta or {}

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        embedder: BgeM3Embedder,
        entries: list[Entry] | None = None,
        meta: dict | None = None,
    ) -> "FaissIndex":
        """Embarque les entrées (par défaut : un vecteur par chunk) et construit l'index.

        Args:
            chunks: Chunks renvoyables.
            embedder: Modèle d'embedding.
            entries: Entrées à indexer ; défaut = titre de section + texte de chaque chunk.
            meta: Métadonnées du package.

        Returns:
            Index construit.
        """

        if entries is None:
            # Le titre de section avec le texte : meilleur signal sémantique.
            entries = [Entry(c.id, "chunk", f"{c.section}\n{c.text}") for c in chunks]
        vectors = embedder.embed([e.text for e in entries])
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        kinds: dict[str, int] = {}
        for e in entries:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        logger.info(
            "Index FAISS : %d vecteurs dim=%d pour %d chunks (%s)",
            index.ntotal, index.d, len(chunks),
            ", ".join(f"{k}={v}" for k, v in kinds.items()),
        )
        chunk_rows = [i for i, e in enumerate(entries) if e.kind == "chunk"]
        if len(chunk_rows) > 1:
            # Similarité moyenne entre chunks : proche de 1 => embeddings peu
            # discriminants (chunks trop similaires ou mal découpés).
            v = vectors[chunk_rows]
            sim = v @ v.T
            off = sim[~np.eye(len(chunk_rows), dtype=bool)]
            logger.info(
                "Similarité inter-chunks moy=%.3f min=%.3f max=%.3f",
                off.mean(), off.min(), off.max(),
            )
        return cls(index, chunks, entries, meta)

    def save(self, directory: Path) -> None:
        """Écrit l'index, les chunks, les entrées et les métadonnées dans ``directory``."""

        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        for name, payload in (
            ("chunks.json", [asdict(c) for c in self.chunks]),
            ("entries.json", [asdict(e) for e in self.entries]),
            ("meta.json", self.meta),
        ):
            (directory / name).write_text(
                json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
            )

    @classmethod
    def load(cls, directory: Path) -> "FaissIndex | None":
        """Recharge un index ; ``None`` s'il est absent ou incohérent."""

        idx, meta = directory / "index.faiss", directory / "chunks.json"
        if not (idx.exists() and meta.exists()):
            return None
        chunks = [Chunk(**c) for c in json.loads(meta.read_text(encoding="utf-8"))]
        entries_file = directory / "entries.json"
        entries = (
            [Entry(**e) for e in json.loads(entries_file.read_text(encoding="utf-8"))]
            if entries_file.exists() else None
        )
        meta_file = directory / "meta.json"
        package_meta = (
            json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        )
        index = faiss.read_index(str(idx))
        expected = len(entries) if entries is not None else len(chunks)
        if index.ntotal != expected:
            logger.warning(
                "Index %s incohérent (%d vecteurs, %d entrées) : reconstruction",
                directory, index.ntotal, expected,
            )
            return None
        return cls(index, chunks, entries, package_meta)

    def search(self, query_vectors: np.ndarray, k: int) -> list[list[tuple[Chunk, float]]]:
        """Top-k chunks par requête ; le score d'un chunk est le max de ses entrées.

        Args:
            query_vectors: Vecteurs normalisés des requêtes.
            k: Nombre de chunks distincts par requête.

        Returns:
            Pour chaque requête, liste (chunk, score) triée par score décroissant.
        """

        if self.index.ntotal == 0:
            return [[] for _ in range(len(query_vectors))]
        # Recherche exhaustive : quelques milliers de vecteurs au plus, et il faut
        # tous les scores pour agréger correctement par chunk.
        scores, ids = self.index.search(query_vectors, self.index.ntotal)
        results = []
        for row_ids, row_scores in zip(ids, scores):
            best: dict[int, float] = {}
            for i, s in zip(row_ids, row_scores):
                if i < 0:
                    continue
                owner = self.entries[i].owner
                if s > best.get(owner, -2.0):
                    best[owner] = float(s)
            ranked = sorted(best.items(), key=lambda kv: -kv[1])[:k]
            results.append([(self.chunks[cid], s) for cid, s in ranked])
        return results

    def best_entry(self, query_vector: np.ndarray, chunk_id: int) -> Entry | None:
        """Entrée du chunk la plus proche de la requête (pour les logs)."""

        rows = [i for i, e in enumerate(self.entries) if e.owner == chunk_id]
        if not rows:
            return None
        vectors = np.vstack([self.index.reconstruct(i) for i in rows])
        return self.entries[rows[int(np.argmax(vectors @ query_vector))]]


# ==============================================================
# Façade
# ==============================================================

class PackageRAG:
    """Indexe la doc d'un package et retrouve les passages pertinents.

    Args:
        ollama_url: URL du serveur Ollama (embeddings bge-m3).
        cache_dir: Répertoire de cache des index (un sous-dossier par package/version).
        max_chars: Taille maximale d'un chunk.
        multi_level: Indexe aussi chaque phrase / ligne de code de chaque section.

    Attributes:
        last_retrieval: Détail du dernier ``retrieve`` (requêtes, rangs, scores,
            phrase ayant fait remonter le chunk), utilisé par les benchs / Phoenix.
    """

    def __init__(
        self,
        ollama_url: str,
        cache_dir: Path,
        max_chars: int = 1500,
        multi_level: bool = True,
    ):
        self.embedder = BgeM3Embedder(ollama_url)
        self.cache_dir = cache_dir
        self.max_chars = max_chars
        self.multi_level = multi_level
        self.last_retrieval: dict = {}
        logger.info("PackageRAG : multi_level=%s, max_chars=%d", multi_level, max_chars)

    # ---------------------------------------------------------- indexation

    def index_package(self, pypi_json: dict) -> FaissIndex:
        """Construit (ou recharge du cache) l'index de la doc d'un package.

        Args:
            pypi_json: Réponse JSON de l'API PyPI.

        Returns:
            Index FAISS du package.
        """

        info = pypi_json.get("info", {}) or {}
        key = re.sub(r"[^\w.-]+", "_", f"{info.get('name')}-{info.get('version')}")
        # L'index dépend du mode : cache séparé pour le multi-niveau.
        directory = self.cache_dir / (
            f"{key}-v{INDEX_FORMAT}" + ("-ml" if self.multi_level else "")
        )

        cached = FaissIndex.load(directory)
        if cached is not None:
            logger.info(
                "Index FAISS en cache: %s (%d chunks, %d entrées)",
                directory, len(cached.chunks), len(cached.entries),
            )
            log_chunks(cached.chunks, self.max_chars)
            return cached

        metadata, readme = build_document(pypi_json)
        logger.info(
            "Document %s : métadonnées %d car., README %d car.",
            key, len(metadata), len(readme),
        )
        if len(readme) < 200:
            logger.warning("README très court ou absent pour %s", key)
        chunks = chunk_markdown(metadata, readme, max_chars=self.max_chars)
        meta = {"name": info.get("name") or "", "summary": info.get("summary") or ""}
        index = FaissIndex.build(chunks, self.embedder, self._entries(chunks), meta)
        index.save(directory)
        logger.info("Index FAISS construit: %s (%d chunks)", directory, len(chunks))
        return index

    def _entries(self, chunks: list[Chunk]) -> list[Entry]:
        """Entrées à indexer : chaque chunk, plus ses phrases en multi-niveau."""

        entries = [Entry(c.id, "chunk", f"{c.section}\n{c.text}") for c in chunks]
        if self.multi_level:
            for c in chunks:
                for sentence in split_sentences(c.text):
                    entries.append(Entry(c.id, "sentence", f"{c.section}\n{sentence}"))
            logger.info(
                "Multi-niveau : %d phrases/lignes indexées en plus de %d chunks",
                len(entries) - len(chunks), len(chunks),
            )
        return entries

    # ---------------------------------------------------------- retrieval

    def retrieve(
        self, index: FaissIndex, queries: list[str], k: int = 4, purpose: str = ""
    ) -> list[Chunk]:
        """Top-k par requête, fusionnés (dédupliqués) et remis dans l'ordre du document.

        Args:
            index: Index FAISS du package.
            queries: Requêtes en langage naturel (une recherche top-k chacune).
            k: Nombre de chunks retournés par requête.
            purpose: Libellé du retrieval dans les logs (ex. "installation").

        Returns:
            Chunks retenus, dédupliqués, dans l'ordre du document.
        """

        label = purpose or "retrieval"
        logger.info(
            "===== RETRIEVAL %s : %d requête(s), top-%d, index de %d chunks / %d entrées =====",
            label.upper(), len(queries), k, len(index.chunks), len(index.entries),
        )
        query_vectors = self.embedder.embed(queries)
        results = index.search(query_vectors, k)
        best: dict[int, float] = {}
        details = []
        for q_no, (query, qv, row) in enumerate(zip(queries, query_vectors, results), 1):
            logger.info("REQUÊTE %d/%d : %r", q_no, len(queries), query)
            hits = []
            for rank, (chunk, score) in enumerate(row, 1):
                logger.info(
                    "  %d. score=%.3f  chunk #%d  section='%s' (%d car.) : %s",
                    rank, score, chunk.id, chunk.section, len(chunk.text),
                    preview(chunk.text, 120),
                )
                via = None
                if len(index.entries) > len(index.chunks):
                    entry = index.best_entry(qv, chunk.id)
                    if entry is not None and entry.kind != "chunk":
                        via = entry.text
                        logger.debug("     via %s : %s", entry.kind, preview(via, 160))
                hits.append({"rank": rank, "score": round(score, 4), "chunk": chunk.id,
                             "section": chunk.section, "via_sentence": via})
                best[chunk.id] = max(best.get(chunk.id, -1.0), score)
            if row and row[0][1] < 0.4:
                logger.warning("  meilleur score faible (%.3f) : retrieval peu fiable", row[0][1])
            details.append({"query": query, "hits": hits})
        selected = sorted((index.chunks[i] for i in best), key=lambda c: c.id)
        logger.info(
            "SECTIONS RETENUES (%s) : %d chunks sur %d", label, len(selected), len(index.chunks)
        )
        for c in selected:
            logger.info("  ✔ chunk #%d  score=%.3f  section='%s'", c.id, best[c.id], c.section)
        for c in index.chunks:
            if c.id not in best:
                logger.debug("  ✘ chunk #%d non retenu  section='%s'", c.id, c.section)
        context = self.to_context(selected)
        logger.debug(
            "CONTEXTE ENVOYÉ AU LLM (%s, %d car.) :\n%s\n===== FIN CONTEXTE %s =====",
            label, len(context), context, label.upper(),
        )
        self.last_retrieval = {"purpose": label, "k": k, "queries": details,
                               "selected": [c.id for c in selected]}
        return selected

    @staticmethod
    def to_context(chunks: list[Chunk]) -> str:
        """Concatène les chunks retenus en un contexte pour le LLM."""

        return "\n\n---\n\n".join(c.render() for c in chunks)
