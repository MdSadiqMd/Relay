# Mathematical Foundations

Formal definitions for every algorithm in relay.

---

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| $d$ | A document |
| $d_i$ | The $i$-th document in an ordered set |
| $\mathcal{D}_e$ | Set of all documents in epoch $e$ |
| $e$ | A semantic epoch |
| $T$ | A tenant identifier |
| $q$ | A query text string |
| $t$ | A point in time (ISO 8601 string, compared lexicographically) |
| $k$ | Top-K retrieval parameter |
| $\vec{v}$ | A dense embedding vector |
| $\lVert \vec{v} \rVert$ | L2 norm of $\vec{v}$ |
| $\text{H}(x)$ | SHA-256 hash function: $\{0,1\}^* \to \{0,1\}^{256}$ |
| $\Vert$ | Concatenation (the `\|\|` delimiter in serialized payloads) |

---

## 2. Hashing Functions

### 2.1 Content Hash

$$
\text{content\_hash}(d) = \text{H}\bigl(\text{UTF-8}(d.\text{text})\bigr)
$$

Deterministic over byte-level content. Two documents with identical text always produce the same content hash.

### 2.2 Embedding Hash

For a dense vector $\vec{v} = (v_1, v_2, \dots, v_n)$ where $n = 384$ (semantic) or $n = 1024$ (video):

$$
\text{embedding\_hash}(\vec{v}) = \text{H}\!\left(\bigoplus_{i=1}^{n} \text{IEEE754}(v_i)\right)
$$

where $\text{IEEE754}(v_i)$ is the big-endian 64-bit double-precision representation (8 bytes per float, network byte order via `struct.pack("!d", v)`), and $\bigoplus$ denotes byte concatenation.

This is deterministic across platforms because IEEE 754 double-precision has a canonical binary form, and big-endian byte order is explicit.

### 2.3 Video Embedding Hash

$$
\text{video\_embedding\_hash}(\vec{v}) = \text{embedding\_hash}(\vec{v})
$$

Same algorithm applied to the 1024-d Marengo 3.0 fused-modality vector. Kept as a separate named function for clarity; the underlying computation is identical.

---

## 3. Dense Embedding

Model: `all-MiniLM-L6-v2` (sentence-transformers), producing $\vec{v} \in \mathbb{R}^{384}$.

$$
\vec{v} = \text{L2-normalize}\!\left(\text{MiniLM}(q)\right) = \frac{\text{MiniLM}(q)}{\lVert \text{MiniLM}(q) \rVert_2}
$$

All vectors are L2-normalized at encoding time (`normalize_embeddings=True`), so cosine similarity reduces to a dot product:

$$
\cos(\vec{a}, \vec{b}) = \vec{a} \cdot \vec{b} \quad \text{when } \lVert\vec{a}\rVert = \lVert\vec{b}\rVert = 1
$$

Results are cached with `@lru_cache(maxsize=1024)` keyed on the raw text string.

---

## 4. Sparse Embedding (BM25)

Model: `Qdrant/bm25` via fastembed. Produces a sparse vector $\vec{s} = \{(i_j, w_j)\}$ where $i_j$ is a vocabulary token index and $w_j$ is the BM25 weight.

BM25 term weight for term $t$ in document $d$ with corpus $C$:

$$
w(t, d) = \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
$$

where:
- $f(t, d)$ = term frequency of $t$ in $d$
- $|d|$ = document length in tokens
- $\text{avgdl}$ = average document length across corpus
- $k_1 = 1.2$, $b = 0.75$ (standard BM25 parameters)
- $\text{IDF}(t) = \ln\!\left(\frac{N - n(t) + 0.5}{n(t) + 0.5} + 1\right)$, $N$ = total docs, $n(t)$ = docs containing $t$

The sparse vector is stored as Qdrant `SparseVector(indices=[i_1, \dots], values=[w_1, \dots])`.

---

## 5. Multimodal Video Embedding

Model: TwelveLabs Marengo 3.0 (Embed v2 API), producing $\vec{v}_{\text{video}} \in \mathbb{R}^{1024}$.

The Marengo model fuses multiple modalities (visual frames, audio waveform, transcription text) into a single dense vector:

$$
\vec{v}_{\text{video}} = \text{Marengo}_{\text{fused}}(\text{visual}, \text{audio})
$$

For cross-modal query, text is embedded into the same 1024-d space:

$$
\vec{v}_{\text{query}} = \text{Marengo}_{\text{text}}(q)
$$

Both vectors live in the same latent space, so cosine similarity between a text query and a video embedding is meaningful:

$$
\text{sim}(q, \text{video}) = \cos\!\left(\vec{v}_{\text{query}},\; \vec{v}_{\text{video}}\right)
$$

---

## 6. Cosine Similarity

Used in semantic diff to measure embedding drift between epoch versions of the same document:

$$
\cos(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{\lVert\vec{a}\rVert_2 \;\cdot\; \lVert\vec{b}\rVert_2} = \frac{\displaystyle\sum_{i=1}^{n} a_i \, b_i}{\sqrt{\displaystyle\sum_{i=1}^{n} a_i^2} \;\cdot\; \sqrt{\displaystyle\sum_{i=1}^{n} b_i^2}}
$$

Returns a value in $[-1, 1]$. For L2-normalized vectors, this simplifies to $\vec{a} \cdot \vec{b}$.

---

## 7. Reciprocal Rank Fusion (RRF)

Combines ranked lists from multiple retrieval sources (semantic + sparse, or semantic + video) into a single fused ranking.

Given $R$ ranked lists, the RRF score for document $d$ is:

$$
\text{RRF}(d) = \sum_{r=1}^{R} \frac{1}{k + \text{rank}_r(d)}
$$

where:
- $k = 60$ (Qdrant default constant, dampens high-rank dominance)
- $\text{rank}_r(d)$ = position of $d$ in the $r$-th ranked list (1-indexed)
- If $d$ is absent from list $r$, it contributes 0

**Hybrid** (dense + sparse): $R = 2$, list 1 = cosine-ranked semantic ANN, list 2 = BM25-ranked sparse.

**Multimodal** (semantic + video): $R = 2$, list 1 = cosine-ranked semantic ANN, list 2 = cosine-ranked Marengo video ANN.

Example with $k=60$:

| Document | Semantic rank | Sparse rank | RRF Score |
|----------|:---:|:---:|:---:|
| doc_A | 1 | 2 | $\frac{1}{61} + \frac{1}{62} = 0.0326$ |
| doc_B | 3 | 1 | $\frac{1}{63} + \frac{1}{61} = 0.0323$ |
| doc_C | 2 | -- | $\frac{1}{62} + 0 = 0.0161$ |

Documents are sorted by descending RRF score.

---

## 8. Merkle Tree

### 8.1 Leaf Computation

Each document $d$ produces a Merkle leaf:

$$
\ell(d) = \text{H}\!\left(
d.\text{id} \;\Vert\;
d.\text{content\_hash} \;\Vert\;
d.\text{embedding\_hash} \;\Vert\;
d.\text{model\_version} \;\Vert\;
d.\text{valid\_from} \;\Vert\;
d.\text{valid\_to} \;\Vert\;
\text{sorted}(d.\text{supersedes})
\;[\Vert\; d.\text{video\_embedding\_hash}]
\right)
$$

The `video_embedding_hash` component is only appended when a video embedding is present. When absent, the leaf is identical to pre-video behavior (backward compatible).

Fields are joined with `||` as a delimiter, encoded as UTF-8, then SHA-256 hashed.

### 8.2 Topological Ordering

Documents form a supersession DAG $G = (V, E)$:
- $V = \{d.\text{id} \mid d \in \mathcal{D}_e\}$
- $E = \{(d.\text{id}, p) \mid p \in d.\text{supersedes}\}$

Leaves are ordered by topological sort of $G$ (via `graphlib.TopologicalSorter`), so parents appear before children. This ensures the Merkle root encodes lineage structure, not arbitrary order.

If $G$ contains a cycle, `CycleError` is raised, preventing corrupt supersession chains.

### 8.3 Binary Tree Construction

Given ordered leaves $[\ell_1, \ell_2, \dots, \ell_n]$:

$$
\text{hash\_pair}(L, R) = \text{H}(L \;\|\; R)
$$

Build the tree bottom-up:

$$
\text{level}_0 = [\ell_1, \ell_2, \dots, \ell_n]
$$

$$
\text{level}_{j+1}[i] = \text{hash\_pair}\!\left(\text{level}_j[2i],\; \text{level}_j[2i+1]\right)
$$

If $|\text{level}_j|$ is odd, the last element is duplicated:

$$
\text{level}_j[2i+1] = \text{level}_j[2i] \quad \text{when } 2i+1 \geq |\text{level}_j|
$$

Recursion terminates when $|\text{level}_j| = 1$. The single remaining hash is the **Merkle root**.

$$
\text{root} = \text{level}_{\lceil \log_2 n \rceil}[0]
$$

Special case: empty tree produces $\text{H}(\epsilon)$ where $\epsilon$ is the empty byte string.

### 8.4 Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Leaf computation | $O(1)$ per document | $O(1)$ |
| Tree construction | $O(n)$ where $n = |\mathcal{D}_e|$ | $O(n)$ |
| Root recomputation from stored leaves | $O(n)$ | $O(n)$ |
| Topological sort | $O(V + E)$ | $O(V + E)$ |

---

## 9. Epoch ID Resolution

Epoch IDs are strictly sequential and never deleted:

$$
\text{next\_epoch\_id}(T) = \text{count}(\{e \mid e.\text{tenant} = T\}) + 1
$$

$$
\text{current\_epoch\_id}(T) = \text{count}(\{e \mid e.\text{tenant} = T\})
$$

Uses Qdrant `client.count()` with a payload filter, which is $O(1)$ with payload indexing.

---

## 10. Temporal Validity Filter

For a time-travel query at time $t$, a document $d$ is **temporally valid** iff:

$$
d.\text{valid\_from} \leq t \;\wedge\; (d.\text{valid\_to} = \text{null} \;\vee\; d.\text{valid\_to} > t) \;\wedge\; d.\text{superseded\_by} = \text{null}
$$

This is a **Python-side post-filter** applied after Qdrant returns ANN results. The epoch ID serves as a partition key (Qdrant-level filter); temporal correctness is enforced in application code.

The post-filter iterates Qdrant results in score order and stops after collecting $k$ valid documents. If many results are filtered out, fewer than $k$ documents are returned.

---

## 11. Semantic Drift Classification

For two epochs $e_1, e_2$ and the set of changed documents $\mathcal{C} = \{d \mid d \in \mathcal{D}_{e_1} \cap \mathcal{D}_{e_2},\; d.\text{content\_hash} \text{ or } d.\text{embedding\_hash} \text{ changed}\}$:

Drift score per changed document:

$$
\delta(d) = 1 - \cos\!\left(\vec{v}_{e_1}(d),\; \vec{v}_{e_2}(d)\right)
$$

Average drift:

$$
\bar{\delta} = \frac{1}{|\mathcal{C}|} \sum_{d \in \mathcal{C}} \delta(d)
$$

Classification:

$$
\text{drift\_level} = \begin{cases}
\text{NONE} & \text{if } |\mathcal{C}| = 0 \;\wedge\; |\text{added}| = 0 \;\wedge\; |\text{removed}| = 0 \\
\text{LOW} & \text{if } \bar{\delta} \leq 0.1 \\
\text{MEDIUM} & \text{if } 0.1 < \bar{\delta} \leq 0.3 \\
\text{HIGH} & \text{if } \bar{\delta} > 0.3 \\
\text{STRUCTURAL} & \text{if } |\mathcal{C}| = 0 \;\wedge\; (|\text{added}| > 0 \;\vee\; |\text{removed}| > 0)
\end{cases}
$$

---

## 12. Epoch Diff Set Operations

Given document ID sets $F = \{d.\text{id} \mid d \in \mathcal{D}_{e_1}\}$ and $T = \{d.\text{id} \mid d \in \mathcal{D}_{e_2}\}$:

$$
\text{Added} = T \setminus F
$$

$$
\text{Removed} = F \setminus T
$$

$$
\text{Common} = F \cap T
$$

$$
\text{Changed} = \{d \in \text{Common} \mid d.\text{content\_hash}_{e_1} \neq d.\text{content\_hash}_{e_2} \;\vee\; d.\text{embedding\_hash}_{e_1} \neq d.\text{embedding\_hash}_{e_2}\}
$$

---

## 13. Merkle Verification

Given a retrieval log $L$ referencing epoch $e$ with retrieved documents $R = \{d_1, \dots, d_m\}$:

**Verification predicate:**

$$
\text{VERIFIED} \iff \text{root\_match} \;\wedge\; \text{docs\_present} \;\wedge\; \text{tenant\_match}
$$

where:

$$
\text{root\_match} = \bigl(\text{compute\_merkle\_root}(e.\text{leaf\_hashes}) = e.\text{merkle\_root}\bigr)
$$

$$
\text{docs\_present} = \forall\, d_i \in R : d_i.\text{id} \in e.\text{doc\_ids}
$$

$$
\text{tenant\_match} = \bigl(L.\text{tenant\_id} = T\bigr)
$$

**Fast path** (epoch has stored `leaf_hashes`): decode hex leaves, recompute root. $O(n)$ where $n$ = number of leaves.

**Scroll fallback** (legacy epochs without leaves): scroll all epoch documents from Qdrant, rebuild DAG, toposort, recompute all leaves, then build tree. $O(n \cdot s)$ where $s$ is the scroll cost.

---

## 14. Supersession DAG

A supersession is a directed edge from a new document to the old documents it replaces.

$$
G = (V, E), \quad V = \{d.\text{id}\}, \quad E = \{(d_{\text{new}}, d_{\text{old}}) \mid d_{\text{old}} \in d_{\text{new}}.\text{supersedes}\}
$$

$G$ must be a **DAG** (directed acyclic graph). If a cycle is detected during topological sort, a `ValueError` is raised.

When superseding, the original epoch is **never mutated**. Instead, copies of all involved documents are created in a new epoch with updated metadata:

$$
d_{\text{old}}' = d_{\text{old}} \cup \{\text{superseded\_by}: d_{\text{new}}.\text{id},\; \text{valid\_to}: \text{today}\}
$$

$$
d_{\text{new}}' = d_{\text{new}} \cup \{\text{supersedes}: [d_{\text{old}_1}.\text{id}, \dots, d_{\text{old}_m}.\text{id}]\}
$$

---

## 15. LRU Embedding Cache

Both `embed()` and `sparse_embed()` are wrapped with `@functools.lru_cache(maxsize=1024)`:

$$
\text{cache}: \text{str} \to \mathbb{R}^{384} \quad (\text{dense})
$$

$$
\text{cache}: \text{str} \to (\mathbb{Z}^*, \mathbb{R}^*) \quad (\text{sparse})
$$

Cache key is the raw text string. On hit, model inference is skipped entirely. The cache uses a **least recently used** eviction policy with a fixed capacity of 1024 entries.

---

## 16. Named Vector Storage Layout

Each Qdrant point in `relay_documents` stores up to three named vectors:

$$
\text{point}.\text{vectors} = \begin{cases}
\{\text{semantic}: \vec{v}_s \in \mathbb{R}^{384}\} & \text{always} \\
\{\text{sparse}: (\vec{i}, \vec{w})\} & \text{if collection has sparse} \\
\{\text{video}: \vec{v}_m \in \mathbb{R}^{1024}\} & \text{if video\_vector provided}
\end{cases}
$$

Collection capability is detected at runtime via `collection_has_sparse()` and `collection_has_video()`, each cached in a module-level `dict[str, bool]` keyed by collection name.

---

## 17. Query Hash

Each retrieval is logged with a deterministic query identifier:

$$
\text{query\_hash} = \text{H}\!\left(\text{UTF-8}(q)\right)
$$

This enables replay detection (same query text always maps to the same hash) and audit trail correlation.
