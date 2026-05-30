# Mathematical Foundations
Formal definitions for every algorithm in Relay

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| $d$ | A document |
| $d_i$ | The $i$-th document in an ordered set |
| $D_e$ | Set of all documents in epoch $e$ |
| $e$ | A semantic epoch |
| $T$ | A tenant identifier |
| $q$ | A query text string |
| $t$ | A point in time (ISO 8601 string, compared lexicographically) |
| $k$ | Top-K retrieval parameter |
| $\vec{v}$ | A dense embedding vector |
| $H(x)$ | SHA-256 hash function |
| $\circ$ | Byte concatenation |

## 2. Hashing Functions

### 2.1 Content Hash

$$
\mathrm{ContentHash}(d) = H\bigl(\mathrm{UTF8}(d.text)\bigr)
$$

Deterministic over byte-level content. Two documents with identical text always produce the same content hash.

### 2.2 Embedding Hash

For a dense vector $\vec{v} = (v_1, v_2, \dots, v_n)$ where $n = 384$ (semantic) or $n = 1024$ (video):

$$
\mathrm{EmbedHash}(\vec{v}) = H\!\left(\mathrm{IEEE754}(v_1) \circ \mathrm{IEEE754}(v_2) \circ \cdots \circ \mathrm{IEEE754}(v_n)\right)
$$

where $\mathrm{IEEE754}(v_i)$ is the big-endian 64-bit double-precision representation (8 bytes per float, network byte order via `struct.pack("!d", v)`).

This is deterministic across platforms because IEEE 754 double-precision has a canonical binary form, and big-endian byte order is explicit.

### 2.3 Video Embedding Hash

$$
\mathrm{VideoEmbedHash}(\vec{v}) = \mathrm{EmbedHash}(\vec{v})
$$

Same algorithm applied to the 1024-d Marengo 3.0 fused-modality vector. Kept as a separate named function for clarity; the underlying computation is identical.

## 3. Dense Embedding

Model: `all-MiniLM-L6-v2` (sentence-transformers), producing $\vec{v} \in \mathbb{R}^{384}$.

$$
\vec{v} = \frac{\mathrm{MiniLM}(q)}{\lVert \mathrm{MiniLM}(q) \rVert_2}
$$

All vectors are L2-normalized at encoding time (`normalize_embeddings=True`), so cosine similarity reduces to a dot product:

$$
\cos(\vec{a}, \vec{b}) = \vec{a} \cdot \vec{b} \quad \text{when } \lVert\vec{a}\rVert = \lVert\vec{b}\rVert = 1
$$

Results are cached with `@lru_cache(maxsize=1024)` keyed on the raw text string.

## 4. Sparse Embedding (BM25)

Model: `Qdrant/bm25` via fastembed. Produces a sparse vector $\vec{s} = \{(i_j, w_j)\}$ where $i_j$ is a vocabulary token index and $w_j$ is the BM25 weight.

BM25 term weight for term $\tau$ in document $d$:

$$
w(\tau, d) = \mathrm{IDF}(\tau) \cdot \frac{f(\tau, d) \cdot (k_1 + 1)}{f(\tau, d) + k_1 \cdot \left(1 - b + b \cdot \dfrac{|d|}{\mathrm{avgdl}}\right)}
$$

where:
- $f(\tau, d)$ = term frequency of $\tau$ in $d$
- $|d|$ = document length in tokens
- $\mathrm{avgdl}$ = average document length across corpus
- $k_1 = 1.2$, $b = 0.75$ (standard BM25 parameters)

Inverse document frequency:

$$
\mathrm{IDF}(\tau) = \ln\!\left(\frac{N - n(\tau) + 0.5}{n(\tau) + 0.5} + 1\right)
$$

where $N$ = total docs and $n(\tau)$ = docs containing $\tau$.

The sparse vector is stored as Qdrant `SparseVector(indices, values)`.

## 5. Multimodal Video Embedding

Model: TwelveLabs Marengo 3.0 (Embed v2 API), producing $\vec{v}_{vid} \in \mathbb{R}^{1024}$.

The Marengo model fuses multiple modalities (visual frames, audio waveform, transcription text) into a single dense vector:

$$
\vec{v}_{vid} = \mathrm{Marengo}_{fused}(\text{visual}, \text{audio})
$$

For cross-modal query, text is embedded into the same 1024-d space:

$$
\vec{v}_{qry} = \mathrm{Marengo}_{text}(q)
$$

Both vectors live in the same latent space, so cosine similarity between a text query and a video embedding is meaningful:

$$
\mathrm{sim}(q, vid) = \cos\!\left(\vec{v}_{qry},\; \vec{v}_{vid}\right)
$$

## 6. Cosine Similarity

Used in semantic diff to measure embedding drift between epoch versions of the same document:

$$
\cos(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{\lVert\vec{a}\rVert_2 \;\cdot\; \lVert\vec{b}\rVert_2} = \frac{\displaystyle\sum_{i=1}^{n} a_i \, b_i}{\sqrt{\displaystyle\sum_{i=1}^{n} a_i^2} \;\cdot\; \sqrt{\displaystyle\sum_{i=1}^{n} b_i^2}}
$$

Returns a value in $[-1, 1]$. For L2-normalized vectors this simplifies to $\vec{a} \cdot \vec{b}$.

## 7. Reciprocal Rank Fusion (RRF)

Combines ranked lists from multiple retrieval sources (semantic + sparse, or semantic + video) into a single fused ranking.

Given $R$ ranked lists, the RRF score for document $d$ is:

$$
\mathrm{RRF}(d) = \sum_{r=1}^{R} \frac{1}{k + \mathrm{rank}_r(d)}
$$

where:
- $k = 60$ (Qdrant default constant, dampens high-rank dominance)
- $\mathrm{rank}_r(d)$ = position of $d$ in the $r$-th ranked list (1-indexed)
- If $d$ is absent from list $r$, it contributes $0$

**Hybrid** (dense + sparse): $R = 2$, list 1 = cosine-ranked semantic ANN, list 2 = BM25-ranked sparse.

**Multimodal** (semantic + video): $R = 2$, list 1 = cosine-ranked semantic ANN, list 2 = cosine-ranked Marengo video ANN.

Example with $k=60$:

| Document | Semantic rank | Sparse rank | RRF Score |
|----------|:---:|:---:|:---:|
| doc A | 1 | 2 | $1/61 + 1/62 = 0.0326$ |
| doc B | 3 | 1 | $1/63 + 1/61 = 0.0323$ |
| doc C | 2 | -- | $1/62 + 0 = 0.0161$ |

Documents are sorted by descending RRF score.

## 8. Merkle Tree

### 8.1 Leaf Computation

Each document $d$ produces a Merkle leaf. The leaf payload is the concatenation of all commitment-relevant fields, joined by `||`, encoded as UTF-8, then SHA-256 hashed:

$$
\ell(d) = H\bigl(d.id \;\|\; \mathrm{CH} \;\|\; \mathrm{EH} \;\|\; d.model \;\|\; d.from \;\|\; d.to \;\|\; \mathrm{sorted}(d.sup) \;[\|\; \mathrm{VH}]\bigr)
$$

where:
- $\mathrm{CH}$ = content hash
- $\mathrm{EH}$ = embedding hash (semantic)
- $\mathrm{VH}$ = video embedding hash (only appended when present; otherwise omitted for backward compatibility)
- $d.sup$ = list of superseded doc IDs, sorted lexicographically for determinism

### 8.2 Topological Ordering

Documents form a supersession DAG $G = (V, E)$:
- $V = \{d.id \mid d \in D_e\}$
- $E = \{(d.id, p) \mid p \in d.supersedes\}$

Leaves are ordered by topological sort of $G$ (via `graphlib.TopologicalSorter`), so parents appear before children. This ensures the Merkle root encodes lineage structure, not arbitrary order.

If $G$ contains a cycle, a `CycleError` is raised, preventing corrupt supersession chains.

### 8.3 Binary Tree Construction

Given ordered leaves $[\ell_1, \ell_2, \dots, \ell_n]$:

$$
\mathrm{HashPair}(L, R) = H(L \circ R)
$$

Build the tree bottom-up. Let $\mathrm{Lv}$ denote a level in the tree:

$$
\mathrm{Lv}_0 = [\ell_1, \ell_2, \dots, \ell_n]
$$

$$
\mathrm{Lv}_{j+1}[i] = \mathrm{HashPair}\!\left(\mathrm{Lv}_j[2i],\; \mathrm{Lv}_j[2i+1]\right)
$$

If $|\mathrm{Lv}_j|$ is odd, the last element is duplicated:

$$
\mathrm{Lv}_j[2i+1] = \mathrm{Lv}_j[2i] \quad \text{when } 2i+1 \geq |\mathrm{Lv}_j|
$$

Recursion terminates when $|\mathrm{Lv}_j| = 1$. The single remaining hash is the **Merkle root**.

$$
\mathrm{root} = \mathrm{Lv}_{\lceil \log_2 n \rceil}[0]
$$

Special case: empty tree produces $H(\varepsilon)$ where $\varepsilon$ is the empty byte string.

### 8.4 Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Leaf computation | $O(1)$ per doc | $O(1)$ |
| Tree construction | $O(n)$ | $O(n)$ |
| Root recomputation from stored leaves | $O(n)$ | $O(n)$ |
| Topological sort | $O(V + E)$ | $O(V + E)$ |

## 9. Epoch ID Resolution

Epoch IDs are strictly sequential and never deleted:

$$
\mathrm{NextEpoch}(T) = \mathrm{count}\bigl(\{e \mid e.tenant = T\}\bigr) + 1
$$

$$
\mathrm{CurrentEpoch}(T) = \mathrm{count}\bigl(\{e \mid e.tenant = T\}\bigr)
$$

Uses Qdrant `client.count()` with a payload filter, which is $O(1)$ with payload indexing.

## 10. Temporal Validity Filter

For a time-travel query at time $t$, a document $d$ is **temporally valid** iff all three conditions hold:

$$
d.from \leq t
$$

$$
d.to = \text{null} \;\;\lor\;\; d.to > t
$$

$$
d.supersededBy = \text{null}
$$

This is a **Python-side post-filter** applied after Qdrant returns ANN results. The epoch ID serves as a partition key (Qdrant-level filter); temporal correctness is enforced in application code.

The post-filter iterates Qdrant results in score order and stops after collecting $k$ valid documents. If many results are filtered out, fewer than $k$ documents are returned.

## 11. Semantic Drift Classification

For two epochs $e_1, e_2$ and the set of changed documents:

$$
C = \{d \in D_{e_1} \cap D_{e_2} \mid d.contentHash \text{ or } d.embedHash \text{ changed}\}
$$

Drift score per changed document:

$$
\delta(d) = 1 - \cos\!\left(\vec{v}_{e_1}(d),\; \vec{v}_{e_2}(d)\right)
$$

Average drift:

$$
\bar{\delta} = \frac{1}{|C|} \sum_{d \in C} \delta(d)
$$

Classification:

$$
\mathrm{DriftLevel} = \begin{cases}
\text{NONE} & |C| = 0,\; |\text{added}| = 0,\; |\text{removed}| = 0 \\
\text{LOW} & \bar{\delta} \leq 0.1 \\
\text{MEDIUM} & 0.1 < \bar{\delta} \leq 0.3 \\
\text{HIGH} & \bar{\delta} > 0.3 \\
\text{STRUCTURAL} & |C| = 0,\; (|\text{added}| > 0 \;\lor\; |\text{removed}| > 0)
\end{cases}
$$

## 12. Epoch Diff Set Operations

Given document ID sets $F = \{d.id \mid d \in D_{e_1}\}$ and $G = \{d.id \mid d \in D_{e_2}\}$:

$$
\text{Added} = G \setminus F
$$

$$
\text{Removed} = F \setminus G
$$

$$
\text{Common} = F \cap G
$$

$$
\text{Changed} = \{d \in \text{Common} \mid d.contentHash_{e_1} \neq d.contentHash_{e_2} \;\lor\; d.embedHash_{e_1} \neq d.embedHash_{e_2}\}
$$

## 13. Merkle Verification

Given a retrieval log $L$ referencing epoch $e$ with retrieved documents $R = \{d_1, \dots, d_m\}$:

**Verification predicate:**

$$
\text{VERIFIED} \iff \mathrm{rootMatch} \;\wedge\; \mathrm{docsPresent} \;\wedge\; \mathrm{tenantMatch}
$$

where:

$$
\mathrm{rootMatch} = \bigl(\mathrm{MerkleRoot}(e.leafHashes) = e.merkleRoot\bigr)
$$

$$
\mathrm{docsPresent} = \forall\, d_i \in R : d_i.id \in e.docIds
$$

$$
\mathrm{tenantMatch} = \bigl(L.tenantId = T\bigr)
$$

**Fast path** (epoch has stored leaf hashes): decode hex leaves, recompute root. $O(n)$ where $n$ = number of leaves.

**Scroll fallback** (legacy epochs without leaves): scroll all epoch documents from Qdrant, rebuild DAG, toposort, recompute all leaves, then build tree. $O(n \cdot s)$ where $s$ is the scroll cost.

## 14. Supersession DAG

A supersession is a directed edge from a new document to the old documents it replaces.

$$
G = (V, E)
$$

$$
V = \{d.id\}
$$

$$
E = \{(d_{new}, d_{old}) \mid d_{old} \in d_{new}.supersedes\}
$$

$G$ must be a **DAG** (directed acyclic graph). If a cycle is detected during topological sort, a `ValueError` is raised.

When superseding, the original epoch is **never mutated**. Instead, copies of all involved documents are created in a new epoch with updated metadata:

Old document copy:

$$
d_{old}' = d_{old} \cup \{supersededBy: d_{new}.id,\;\; validTo: \text{today}\}
$$

New document copy:

$$
d_{new}' = d_{new} \cup \{supersedes: [d_{old_1}.id, \dots, d_{old_m}.id]\}
$$

## 15. LRU Embedding Cache

Both `embed()` and `sparse_embed()` are wrapped with `@functools.lru_cache(maxsize=1024)`:

Dense cache type:

$$
\mathrm{cache}: \text{str} \to \mathbb{R}^{384}
$$

Sparse cache type:

$$
\mathrm{cache}: \text{str} \to (\mathbb{Z}^{\ast}, \mathbb{R}^{\ast})
$$

Cache key is the raw text string. On hit, model inference is skipped entirely. The cache uses a **least recently used** eviction policy with a fixed capacity of 1024 entries.

## 16. Named Vector Storage Layout

Each Qdrant point in `relay_documents` stores up to three named vectors:

$$
\mathrm{vectors} = \begin{cases}
\{semantic : \vec{v}_s \in \mathbb{R}^{384}\} & \text{always} \\
\{sparse : (\vec{i}, \vec{w})\} & \text{if collection has sparse} \\
\{video : \vec{v}_m \in \mathbb{R}^{1024}\} & \text{if video vector provided}
\end{cases}
$$

Collection capability is detected at runtime via `collection_has_sparse()` and `collection_has_video()`, each cached in a module-level dict keyed by collection name.

## 17. Query Hash

Each retrieval is logged with a deterministic query identifier:

$$
\mathrm{QueryHash}(q) = H\!\left(\mathrm{UTF8}(q)\right)
$$

This enables replay detection (same query text always maps to the same hash) and audit trail correlation.
