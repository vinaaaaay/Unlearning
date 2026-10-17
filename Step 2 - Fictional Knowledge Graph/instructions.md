# Specification: Step 2 — Bijective Synthetic Knowledge Graph Generator (C++17)

## 1. Objective
Implement a deterministic, zero-dependency C++17 program (`generate_kg.cpp`) that constructs a synthetic, closed-world Knowledge Graph (`fictional_kg.json`) for evaluating multi-hop machine unlearning in LLMs. 

To eliminate semantic name priors, pre-training contamination, shortcut paths, and frequency biases, the graph MUST be constructed as a **6-layer Bijective Directed Acyclic Graph (DAG)** using phonetically neutral synthetic entities.

---

## 2. Mathematical Formulation

### 2.1 Layered Vertex Partition ($\mathcal{V}$)
Fix the number of disjoint spines $N$ (default $N = 400$, configurable via CLI `--num-spines N`, where $N$ must be a multiple of $10$). 
Partition the vertex set $\mathcal{V}$ of size $6N$ into 6 disjoint, ordered layers $V_0, V_1, V_2, V_3, V_4, V_5$, each containing exactly $N$ entities:

* $V_0 = \{v_{0,0}, \dots, v_{0,N-1}\}$ — Entity Category: `Person` (Layer 0)
* $V_1 = \{v_{1,0}, \dots, v_{1,N-1}\}$ — Entity Category: `Person` (Layer 1)
* $V_2 = \{v_{2,0}, \dots, v_{2,N-1}\}$ — Entity Category: `Person` (Layer 2)
* $V_3 = \{v_{3,0}, \dots, v_{3,N-1}\}$ — Entity Category: `Company` (Layer 3)
* $V_4 = \{v_{4,0}, \dots, v_{4,N-1}\}$ — Entity Category: `Company` (Layer 4)
* $V_5 = \{v_{5,0}, \dots, v_{5,N-1}\}$ — Entity Category: `City` (Layer 5)

All $6N$ entity surface names across $\mathcal{V}$ must be globally unique.

### 2.2 Bijective Edge Construction ($\mathcal{E}$)
Define 5 directed relation layers connecting $V_{m-1} \to V_m$ for $m \in \{1, 2, 3, 4, 5\}$:

* $m = 1$ ($V_0 \to V_1$): `mentor_of` (`Person` $\to$ `Person`)
* $m = 2$ ($V_1 \to V_2$): `mentor_of` (`Person` $\to$ `Person`)
* $m = 3$ ($V_2 \to V_3$): `works_at` (`Person` $\to$ `Company`)
* $m = 4$ ($V_3 \to V_4$): `acquired_by` (`Company` $\to$ `Company`)
* $m = 5$ ($V_4 \to V_5$): `headquartered_in` (`Company` $\to$ `City`)

Using a deterministic PRNG (`std::mt19937_64` seeded with `--seed`, default `42`), sample 5 independent uniform random permutations $\pi_1, \pi_2, \pi_3, \pi_4, \pi_5 \in S_N$ over $\{0, 1, \dots, N-1\}$ via `std::shuffle`.

For each spine index $i \in \{0, 1, \dots, N-1\}$, define its sequence of 6 vertices $(u_{i,0}, u_{i,1}, u_{i,2}, u_{i,3}, u_{i,4}, u_{i,5})$ inductively:
1. $u_{i,0} = v_{0, i}$
2. For $m \in \{1, 2, 3, 4, 5\}$, if $u_{i, m-1} = v_{m-1, j}$, then $u_{i, m} = v_{m, \pi_m(j)}$.

Each spine $i$ contributes 5 directed triples (edges):
$$e_{i,m} = (u_{i, m-1}, r_m, u_{i, m}) \quad \text{for } m \in \{1, 2, 3, 4, 5\}$$

This yields $\vert{}\mathcal{E}\vert{} = 5N$ total triples (for $N = 400$, $\vert{}\mathcal{E}\vert{} = 2,000$ triples). By construction:
* Every vertex in $V_{m-1}$ has out-degree $1$ (under $r_m$).
* Every vertex in $V_m$ has in-degree $1$ (under $r_m$).
* The graph consists of $N$ vertex-disjoint directed paths of length $5$, guaranteeing acyclicity, zero bypass chords, zero diamond paths, and an exact $1/N$ uniform marginal distribution at every terminal layer.

### 2.3 Forget / Retain Partition ($\mathcal{D}_{\text{forget}}$ vs. $\mathcal{D}_{\text{retain}}$)
To achieve an exact $10\%$ forget split ($0.5N$ triples in $\mathcal{D}_{\text{forget}}$ and $4.5N$ triples in $\mathcal{D}_{\text{retain}}$) while ensuring every multi-hop evaluation chain contains **at most one** forgotten link:

1. **Spine Partition:**
   * **Target Spines** ($i \in \{0, \dots, \frac{N}{2} - 1\}$): Exactly $1$ edge out of $5$ on each spine is placed in $\mathcal{D}_{\text{forget}}$, and the remaining $4$ edges are placed in $\mathcal{D}_{\text{retain}}$.
   * **Pure Retain Spines** ($i \in \{\frac{N}{2}, \dots, N - 1\}$): All $5$ edges on each spine are placed in $\mathcal{D}_{\text{retain}}$.

2. **Stratified Forgotten Edge Assignment:**
   * Partition the $\frac{N}{2}$ Target Spines evenly into 5 buckets of size $\frac{N}{10}$ each.
   * For Target Spine $i$, set its forgotten edge layer index to:
     $$m^*(i) = (i \bmod 5) + 1 \in \{1, 2, 3, 4, 5\}$$
   * Mark edge $e_{i, m^*(i)}$ with `is_forget = true` ($\mathcal{D}_{\text{forget}}$), and all other edges $e_{i, m}$ ($m \neq m^*(i)$) with `is_forget = false` ($\mathcal{D}_{\text{retain}}$).

### 2.4 Multi-Hop Evaluation Chain Extraction ($\mathcal{C}_k$)
For each Target Spine $i \in \{0, \dots, \frac{N}{2} - 1\}$ with forgotten edge at $m^* = m^*(i) \in \{1, \dots, 5\}$:
Iterate over all contiguous sub-paths of length $k \in \{1, 2, 3, 4, 5\}$ starting at layer $s \in \{0, \dots, 5 - k\}$ and ending at layer $s + k$:
$$\mathcal{C}_k(i, s) = (u_{i,s} \xrightarrow{r_{s+1}} u_{i,s+1} \xrightarrow{r_{s+2}} \dots \xrightarrow{r_{s+k}} u_{i,s+k})$$

Keep the sub-path if and only if it traverses the forgotten edge $m^*$, i.e., $s + 1 \le m^* \le s + k$.
Let $p = m^* - s \in \{1, \dots, k\}$ be the 1-based relative position of the forgotten edge within $\mathcal{C}_k(i, s)$. Assign its topological position label `forget_position` as:
* For $k = 1$: `"single"` (since $p = 1 = k$; also tag as `"start"` in metadata for 1-hop baseline indexing).
* For $k = 2$:
  * `"start"` if $p = 1$
  * `"end"` if $p = 2$
* For $k \in \{3, 4, 5\}$:
  * `"start"` if $p = 1$
  * `"middle"` if $1 < p < k$
  * `"end"` if $p = k$

Additionally, extract a matched set of **Retain Control Chains** from the Pure Retain Spines ($i \in \{\frac{N}{2}, \dots, N-1\}$) using the exact same $(s, k)$ slice coordinates, with `forget_position = "none"`.

---

## 3. Phonetically Neutral Name Synthesis Rules
To prevent pre-trained linguistic priors, synthesize entity names deterministically using syllable combinatorics:

* **Onsets:** `{"z", "kr", "v", "tr", "m", "n", "dr", "pl", "s", "br", "k", "l", "f", "gl", "th", "p", "d", "r"}`
* **Vowels:** `{"a", "e", "i", "o", "u"}`
* **Codas:** `{"l", "r", "n", "x", "s", "v", "k", "m", "t", "z"}`

1. **Synthetic Root Generator:**
   Combine `Onset1 + Vowel1 + Coda1 + Vowel2 + Coda2` (or `Onset1 + Vowel1 + Onset2 + Vowel2 + Coda1`), capitalizing the first letter (e.g., `"Zalor"`, `"Krexin"`, `"Vantor"`).
2. **Entity Formatting by Type:**
   * `Person`: `<Root1> <Root2>` (two distinct synthetic roots, e.g., `"Zalor Vankel"`).
   * `Company`: `<Root> <Suffix>` where `<Suffix>` cycles uniformly through `{"Systems", "Technologies", "Dynamics", "Labs", "Group", "Corp", "Ventures", "Industries"}` (e.g., `"Krexin Dynamics"`).
   * `City`: `<Root><CitySuffix>` where `<CitySuffix>` is drawn from `{"ia", "on", "is", "um", "ar"}` (e.g., `"Vantoron"`, `"Trelumia"`).
3. **Collision & Blacklist Filtering:**
   Maintain an `std::unordered_set<std::string>` of all generated roots and full names to guarantee zero duplicates across all $6N$ entities, and reject any root matching common English/geographic substrings (`"real"`, `"amer"`, `"chin"`, `"indi"`, `"kor"`, `"fran"`, `"lond"`, `"pari"`, `"toky"`).

---

## 4. Automated Graph Verification Suite
Before writing the output JSON, the C++ program MUST execute a `verify_kg()` routine with hard `assert` checks (and print a pass/fail summary to `stdout`):

1. **Uniqueness Check:** Verify all $6N$ entity IDs and surface names are globally unique.
2. **Type-Consistency Check:** Verify every triple $(s, r, o)$ obeys the exact domain and range type constraints for $r$.
3. **Partial Functionality (Out-Degree = 1):** Verify that for every node $u \in V_0 \cup \dots \cup V_4$, its total out-degree in $\mathcal{E}$ is strictly $1$, and for $u \in V_5$, its out-degree is strictly $0$.
4. **In-Degree Uniformity (In-Degree = 1):** Verify that for every node $u \in V_1 \cup \dots \cup V_5$, its total in-degree in $\mathcal{E}$ is strictly $1$, and for $u \in V_0$, its in-degree is strictly $0$.
5. **Acyclicity & No Bypass Shortcuts (DFS Check):** For every spine $i$, run a Depth-First Search from $u_{i,0}$ on the full adjacency list of $\mathcal{E}$ and verify that:
   * No cycle is reachable.
   * For any $0 \le a < b \le 5$, there is **at most one** directed path from $u_{i,a}$ to $u_{i,b}$ in $\mathcal{E}$, and its length is strictly $b - a$ (no shortcut chords).
6. **Split & Chain Integrity:**
   * Verify $\vert{}\mathcal{D}_{\text{forget}}\vert{} == 0.10 \times \vert{}\mathcal{E}\vert{}$ and $\vert{}\mathcal{D}_{\text{retain}}\vert{} == 0.90 \times \vert{}\mathcal{E}\vert{}$.
   * Verify each Target Chain $\mathcal{C}_k$ contains **exactly 1** edge with `is_forget == true`, and each Retain Control Chain contains **0** edges with `is_forget == true`.
   * Verify that for every hop length $k \in \{1..5\}$, the terminal entities $e_k$ across all spines are uniformly distributed ($1/N$ per entity in the terminal layer).

---

## 5. C++ Implementation & Output Schema Requirements

* **Standard & Compiler:** C++17 (`g++ -std=c++17 -O2 -Wall -Wextra generate_kg.cpp -o generate_kg`).
* **Zero External Dependencies:** Use only the C++ Standard Library (`<iostream>`, `<fstream>`, `<vector>`, `<string>`, `<random>`, `<algorithm>`, `<unordered_set>`, `<unordered_map>`, `<cassert>`, `<iomanip>`). Write a clean, well-indented JSON serializer function for the output file.
* **CLI Arguments:**
  * `--num-spines <int>` (default: `400`)
  * `--seed <uint64_t>` (default: `42`)
  * `--output <filepath>` (default: `fictional_kg.json`)
* **JSON Output Schema (`fictional_kg.json`):**
  ```json
  {
    "metadata": {
      "seed": 42,
      "num_spines": 400,
      "num_entities": 2400,
      "num_triples": 2000,
      "num_forget_triples": 200,
      "num_retain_triples": 1800,
      "num_target_chains": 3000,
      "num_retain_control_chains": 3000
    },
    "entities": [
      {"id": "E_L0_000", "name": "Zalor Vankel", "type": "Person", "layer": 0, "spine_id": 0}
    ],
    "triples": [
      {
        "id": "T_S000_M1",
        "spine_id": 0,
        "hop_layer": 1,
        "subject_id": "E_L0_000",
        "subject_name": "Zalor Vankel",
        "relation": "mentor_of",
        "object_id": "E_L1_142",
        "object_name": "Krexin Trelor",
        "subject_type": "Person",
        "object_type": "Person",
        "is_forget": true
      }
    ],
    "evaluation_chains": [
      {
        "chain_id": "C_TARGET_S000_K3_L0",
        "spine_id": 0,
        "is_target_chain": true,
        "length_k": 3,
        "start_layer": 0,
        "end_layer": 3,
        "forget_edge_layer": 1,
        "forget_relative_pos": 1,
        "forget_position": "start",
        "node_ids": ["E_L0_000", "E_L1_142", "E_L2_089", "E_L3_311"],
        "node_names": ["Zalor Vankel", "Krexin Trelor", "Mori Dravon", "Plaxor Systems"],
        "relations": ["mentor_of", "mentor_of", "works_at"],
        "triple_ids": ["T_S000_M1", "T_S000_M2", "T_S000_M3"],
        "terminal_entity_id": "E_L3_311",
        "terminal_entity_name": "Plaxor Systems",
        "terminal_entity_type": "Company"
      }
    ]
  }
  ```
* **Console Summary Table:** Upon execution, print a formatted summary table showing the exact count of generated Target Chains for every `(length_k, forget_position)` cell (`k = 1..5` × `{start, middle, end}`) so the user can include the table directly in their LaTeX report.