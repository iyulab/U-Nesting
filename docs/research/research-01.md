# **U-Nesting: Architectural and Algorithmic Foundations for High-Performance Spatial Optimization**

## **1\. Problem Definition and Taxonomy**

The domain of spatial optimization, specifically the efficient arrangement of irregular 2D shapes and 3D volumes, represents a critical intersection of computational geometry, operations research, and industrial engineering. The development of U-Nesting, a domain-agnostic engine implemented in Rust, requires a rigorous formalization of these problems to transcend the limitations of legacy heuristics and capitalize on modern systems programming paradigms.

### **1.1 Mathematical Formalization**

The core problem addressed by U-Nesting encompasses two distinct but mathematically related domains: 2D Irregular Nesting and 3D Bin Packing. Both can be formalized as subsets of the geometric packing problem, where the objective is to arrange a set of items $\\mathcal{I} \= \\{1, \\dots, n\\}$ into a set of containers $\\mathcal{C} \= \\{1, \\dots, m\\}$ such that an objective function is optimized, subject to non-overlapping and geometric containment constraints.

#### **1.1.1 2D Irregular Nesting (Strip Packing Variant)**

In the 2D nesting domain, specifically the Strip Packing Problem (SPP), the container is often a semi-infinite strip of fixed width $W$ and infinite length $L$. The objective is typically to minimize the required length $L\_{used}$. This formulation is particularly relevant to roll-based industries such as textiles and sheet metal coil processing.

Let each item $i \\in \\mathcal{I}$ be represented by a polygon $P\_i$. The placement of $P\_i$ is defined by a translation vector $v\_i \= (x\_i, y\_i)$ and a rotation angle $\\theta\_i$. The optimization problem can be stated as:

$$\\text{Minimize } Z \= \\max\_{i \\in \\mathcal{I}} (x\_i \+ \\text{width}(P\_i(\\theta\_i))) \\quad \\text{or} \\quad Z \= L\_{used}$$  
Subject to:

1. **Containment:** $P\_i(v\_i, \\theta\_i) \\subseteq C$ for all $i \\in \\mathcal{I}$, where $C$ is the container boundaries. This constraint necessitates that every vertex of the transformed polygon lies within the bounds of the strip width and the calculated length.  
2. **Non-overlapping:** $\\text{int}(P\_i(v\_i, \\theta\_i)) \\cap \\text{int}(P\_j(v\_j, \\theta\_j)) \= \\emptyset$ for all $i \\neq j$.

Unlike standard bin packing, irregular nesting must account for non-convex shapes, holes, and arbitrary rotations, making the geometric validity check significantly more computationally expensive than the combinatorial search itself.1 The search space is continuous rather than discrete, implying an infinite number of potential positions and orientations, which necessitates either discretization or the use of geometric primitives like the No Fit Polygon (NFP) to reduce the search space to a discrete set of "touching" configurations.

#### **1.1.2 3D Bin Packing Problem (3D-BPP)**

The 3D Bin Packing Problem typically involves a finite set of rectangular or polyhedral items to be packed into a minimum number of fixed-size containers (bins) or a single container of minimized volume. This variant is dominant in logistics, palletization, and additive manufacturing (sintering box packing).

Let item $i$ have dimensions $(l\_i, w\_i, h\_i)$ (for rectangular cases) or be a polyhedron $H\_i$. The state of item $i$ is defined by its position $(x\_i, y\_i, z\_i)$ and orientation $o\_i$. The 3D-BPP seeks to minimize the number of bins $k$:

$$\\text{Minimize } k \= \\sum\_{j=1}^{m} y\_j$$  
Where $y\_j$ is a binary variable indicating if bin $j$ is used. The constraints mirror the 2D case but extend to $\\mathbb{R}^3$, often adding physical constraints such as static stability (gravity), load-bearing limits, and robot accessibility (LIFO/FIFO).3

### **1.2 Computational Complexity and Taxonomy**

The typology of C\&P (Cutting and Packing) problems is extensive and has been rigorously classified by Dyckhoff and later refined by Wäscher et al. These problems are fundamentally NP-hard, meaning no polynomial-time algorithm is known to guarantee an optimal solution.1

* **Complexity Class:** The decision version of the bin packing problem is strongly NP-complete. For 2D irregular nesting, the continuous rotation capability adds infinite cardinality to the search space. Even with fixed rotations, the problem remains NP-hard due to the geometric complexity of matching irregular boundaries. The computational burden is twofold: the combinatorial explosion of ordering $N$ items ($N\!$) and the geometric cost of checking feasibility for each candidate arrangement.6  
* **Relationship to Knapsack and Cutting Stock:**  
  * **Knapsack Problem:** When a single container has fixed capacity and items have associated profits, the goal becomes maximizing the total value of packed items. This is relevant for "partial nesting" where not all orders must be filled immediately.  
  * **Cutting Stock Problem:** This is the inverse of bin packing, typically focusing on minimizing waste from standard stock sizes. It is often modeled using exact approaches like column generation, where patterns are generated iteratively. U-Nesting, focusing on irregular shapes, aligns more closely with the geometric nesting definitions than the 1D cutting stock formulations.8

#### **1.2.1 Problem Variants**

Understanding the subtle distinctions between variants is crucial for architectural decisions in U-Nesting:

* **Strip Packing vs. Bin Packing:** Strip packing minimizes the length of a single container (common in textile and metal coil cutting), while bin packing minimizes the number of finite containers (common in logistics and sheet metal cutting with fixed stock sheets).8  
* **Regular vs. Irregular:** Regular packing deals with rectangles or cuboids, allowing for simplifications like "bottom-left" coordinates and creating "skyline" structures. Irregular packing involves arbitrary polygons/polyhedra, requiring complex intersection tests like the No Fit Polygon (NFP) and convex decomposition.2  
* **Orthogonal vs. Free Rotation:** Orthogonal packing restricts items to 90-degree rotations (typical for wood grain or anisotropic materials). Free rotation allows arbitrary angles, complicating the collision detection significantly and often requiring discretization of the rotation angle (e.g., checking every 1 degree) or non-linear optimization.6  
* **Single vs. Multiple Containers:** Industrial scenarios often involve a mix of standard sheet sizes and "remnants" (leftover material from previous jobs). The algorithm must be capable of selecting the best container from a heterogeneous stock list, a feature prominent in commercial solvers like SigmaNEST.13

## **2\. Geometric Foundations**

The efficiency of U-Nesting relies heavily on its geometric engine. While the optimization layer searches the solution space, the geometric layer validates feasibility. In high-performance nesting, geometric operations—specifically intersection testing and overlap removal—often consume 80-90% of CPU time. A naive implementation of geometry will bottleneck the entire engine, regardless of the sophistication of the metaheuristic.

### **2.1 2D Geometry: The No Fit Polygon (NFP)**

The **No Fit Polygon (NFP)** is the critical geometric primitive for 2D irregular nesting. Given two polygons $A$ (stationary) and $B$ (orbiting), the NFP, denoted as $NFP\_{AB}$, represents the locus of points where the reference point of $B$ can be placed such that $B$ touches but does not overlap $A$.15

Mathematically, it is related to the Minkowski difference:

$$NFP\_{AB} \= A \\oplus (-B) \= \\{a \- b \\mid a \\in A, b \\in B\\}$$

Specifically, the boundary of the Minkowski sum of $A$ and $-B$ (B rotated 180 degrees) forms the NFP trace. If the reference point of $B$ is placed inside $NFP\_{AB}$, the shapes overlap. If it is on the boundary, they touch. If it is outside, they are separated.18

#### **2.1.1 Computation Methods**

Three primary methods exist for NFP generation, each with specific trade-offs relevant to a Rust implementation:

1\. Orbiting (Sliding) Algorithm  
This approach simulates sliding polygon $B$ around polygon $A$. It relies on vertex-vertex, vertex-edge, and edge-edge intersection calculations to determine the path of the reference point.

* **Mechanism:** The orbiting polygon slides along the edges of the stationary polygon. At each vertex, the algorithm calculates the rotation or translation vector required to maintain contact.  
* **Pros:** It is theoretically exact for non-convex polygons without needing decomposition.  
* **Cons:** It is notoriously difficult to implement robustly. "Lock and key" situations (where shapes fit tightly) can cause numerical instability or infinite loops in floating-point arithmetic. Handling holes and complex concavities requires complex case logic.20  
* **Rust Implication:** Requires high-precision arithmetic or exact geometric predicates (e.g., the robust crate) to avoid panic on degenerate edges or infinite loops.

2\. Minkowski Sum Approach  
Utilizing the vector algebra definition, $A \\oplus (-B)$, this method is highly parallelizable and mathematically elegant.

* **Mechanism:** For convex polygons, the Minkowski sum is computed by sorting the edge vectors of both polygons by angle and merging them (similar to the merge step in Merge Sort). This is an $O(N+M)$ operation.  
* **Cons:** For non-convex polygons, this method produces a set of cycles that may self-intersect or contain internal holes. These must be pruned using boolean union operations, which are computationally expensive ($O(N^2 \\log N)$) and prone to numerical error.18  
* **Rust Implication:** Efficient implementations must leverage robust boolean libraries. The iOverlay library is a strong candidate here for performing the necessary unions.23

3\. Decomposition-Based Methods  
This hybrid approach is the most practical for high-performance engines. It involves decomposing non-convex polygons into convex sub-polygons ($A \= \\cup A\_i, B \= \\cup B\_j$). The NFP is the union of pairwise Minkowski sums of these convex components: $NFP\_{AB} \= \\bigcup\_{i,j} (A\_i \\oplus \-B\_j)$.

* **Mechanism:**  
  1. Decompose Polygons A and B into convex pieces.  
  2. Compute NFP for every pair of convex pieces (very fast).  
  3. Compute the Boolean Union of all resulting sub-NFPs.  
* **Pros:** Operations on convex polygons are fast ($O(N+M)$) and numerically stable. The logic is easier to debug.  
* **Cons:** The number of sub-NFPs grows quadratically with the number of concave features. The final union operation is the bottleneck.20  
* **Recommendation for U-Nesting:** This is the most viable path. To minimize the number of boolean operations, the **Hertel-Mehlhorn algorithm** should be used for convex decomposition. It is faster than optimal decomposition algorithms and produces a partition with at most 4 times the minimum number of convex pieces, avoiding the thin triangles produced by ear-clipping.26

#### **2.1.2 Inner Fit Polygon (IFP)**

For checking containment within the stock sheet (container $C$), the Inner Fit Polygon is used. $IFP\_{CB}$ is the locus of points where the orbiting polygon $B$ fits entirely inside $C$. It can be computed as $C \\ominus B$ or by shrinking $C$ by the dimensions of $B$.15  
In U-Nesting, generating the IFP is the first step. The valid placement space for a part is calculated as:

$$\\text{ValidSpace} \= IFP\_{Bin} \\cap (\\text{External Space} \\setminus \\bigcup NFP\_{PlacedParts})$$

If this boolean difference results in an empty set, the part cannot fit.

### **2.2 3D Geometry and Collision Detection**

In 3D bin packing, the "No Fit Polyhedron" is theoretically possible but practically intractable due to the complexity of 3D boolean operations on arbitrary meshes. The computation time for 3D Minkowski sums of non-convex polyhedra is prohibitive for real-time optimization.30 Instead, modern engines rely on iterative collision detection and separation vector computation.

#### **2.2.1 Shape Representations**

* **Triangle Mesh:** The most general representation, used for complex parts. However, collision checks between two triangle meshes are expensive ($O(N \\cdot M)$ in the worst case).  
* **Convex Decomposition (VHACD):** To enable fast collision detection, complex meshes are approximated as a set of convex hulls. The **V-HACD** (Volumetric Hierarchical Approximate Convex Decomposition) algorithm is the industry standard for this. It allows the physics engine to treat a complex chair or mechanical part as a compound of simple convex shapes.28  
* **Voxel/Height Map:** Discretizes the object into a grid. This is useful for Machine Learning inputs and simple packing (e.g., palletizing boxes) but loses precision for tight packing of irregular shapes.32

#### **2.2.2 Collision Algorithms (The Physics Engine Approach)**

For U-Nesting, leveraging algorithms from the physics simulation domain is standard practice. The **Gilbert-Johnson-Keerthi (GJK)** algorithm and **Separating Axis Theorem (SAT)** are the cornerstones, both implemented efficiently in the Rust crate parry3d.

* **GJK:** Determines if two convex shapes intersect by working in the Minkowski difference space. It uses a **Support Mapping** function (finding the furthest point in a specific direction) to iteratively build a simplex inside the Minkowski difference.  
  * *Efficiency:* It is generally $O(k)$ where $k$ is the number of vertices, but with **temporal coherence** (warm-starting from the previous frame's solution), it effectively becomes $O(1)$ for incremental updates.34  
* **EPA (Expanding Polytope Algorithm):** Used when GJK detects a collision to calculate the *penetration depth* and *contact normal*. This vector is crucial for "pushing" parts apart in physics-based packing heuristics.37  
* **SAT:** Tests all potential separating axes (face normals and edge cross-products). It is faster for simple shapes like oriented bounding boxes (OBBs) but scales poorly with vertex count compared to GJK.38

**Rust Implementation Note:** The parry3d library implements GJK/EPA and manages bounding volume hierarchies (BVH/QBCH) to prune non-colliding pairs efficiently. This allows the "placement" step to be treated as a physical simulation where items "fall" into place under gravity until they collide.39

### **2.3 Polygon Operations and Boolean Libraries in Rust**

A critical component of the 2D engine is the boolean operation library (Union, Intersection, Difference, XOR).

* **Clipping Algorithms:** The standard algorithm is the **Vatti** clipping algorithm or the **Greiner-Hormann** algorithm. These algorithms handle arbitrary polygons but are sensitive to floating-point errors.  
* **iOverlay:** This is a modern Rust library for boolean operations. It uses a **scanbeam** algorithm which is generally more robust and performant for the types of operations required in nesting (iterative unions of many polygons). It supports both integer and floating-point coordinates, allowing for "snap rounding" strategies to ensure topological validity.23  
* **Comparison:** Unlike Clipper2 (C++), iOverlay is native Rust, simplifying compilation and memory safety. It handles self-intersections and coincident edges, which are common in NFP generation, more gracefully than older libraries.

## **3\. Placement Algorithms**

Placement algorithms, often referred to as "heuristics" or "decoders," determine the specific coordinate position of an item given a sequence determined by the optimization engine. They act as the bridge between the abstract search space and the geometric reality.

### **3.1 2D Placement Strategies**

#### **3.1.1 Bottom-Left Fill (BLF) and Variants**

The classic BLF heuristic attempts to place the next item at the lowest, then leftmost feasible position.

* **Naive BLF:** This involves iteratively checking positions on a grid or sliding the piece. It is computationally expensive ($O(N^2)$) and can be imprecise.  
* **NFP-Based BLF:** This is the modern standard. By computing the NFP of the new item relative to all previously placed items, and the IFP relative to the container, the feasible region is defined exactly as the intersection of the IFP and the exterior of all NFPs.  
  * *Mechanism:* The optimal "bottom-left" position corresponds to a specific vertex on the boundary of this feasible region. Instead of searching a grid, the algorithm simply sorts the vertices of the resulting boolean polygon by Y, then X, and picks the first one. This reduces a spatial search to a sorting problem.41

#### **3.1.2 Touching Perimeter**

This strategy restricts the search space even further. It positions the item such that it touches at least two other items (or one item and a container boundary).

* *Geometry:* These positions correspond to the intersection points of the NFP boundaries.  
* *Efficiency:* This drastically reduces the number of candidate positions to check, making it significantly faster than BLF for large instances, though it may miss "floating" placements that might be optimal in specific bin packing contexts (though less relevant for gravity-based nesting).41

#### **3.1.3 Deepest Bottom-Left Fill (DBLF)**

A refinement of BLF where the item slides down as far as possible, and then slides left. This is better at filling "caves" or "holes" in the layout that standard BLF might miss because it prioritizes the initial Y coordinate too strictly. It helps in creating tighter packings for concave boundaries.42

### **3.2 3D Placement Strategies**

#### **3.2.1 Extreme Point Heuristics**

In 3D, computing the full NFP union is often too slow. The Extreme Point (EP) heuristic is a powerful approximation.

* **Mechanism:** instead of checking every coordinate, the algorithm maintains a list of "Extreme Points" generated by the corners of placed items. When a new item (box or simplified hull) is placed, it occupies an EP, and new EPs are projected from its corners.  
* **Logic:** The EPs are sorted (e.g., by Z, then Y, then X). The algorithm attempts to place the current item at the first feasible EP. If it fits (verified via collision detection), the EPs are updated.  
* **Irregular Shapes:** For irregular 3D shapes, EPs are generated based on the Axis-Aligned Bounding Box (AABB) of the placed items. Once a "box" position is found, a local physics simulation (gravity drop) can be applied to settle the irregular part into a stable position.43

#### **3.2.2 Wall Building and Layering**

Common in logistics, specifically for palletizing. Items are grouped to form "walls" or layers of uniform depth.

* **Algorithm:** This heuristic reduces the 3D problem to a series of 2D rectangle packing problems. It sorts items by depth and packs them into a layer until the layer is full.  
* **Rust Application:** This logic sits *above* the collision detection layer. It is a combinatorial heuristic that segments the input data before passing it to the geometric placer.45

#### **3.2.3 Physics-Based Settlement (Gravity)**

This is a strictly geometric approach where items are spawned above the container and "dropped" using a physics simulation.

* **Mechanism:** Using parry3d or rapier3d, items are subjected to a downward force. The simulation runs until the kinetic energy of the system drops below a threshold (sleep).  
* **Advantages:** Naturally handles arbitrary rotations and complex interlocks (e.g., nesting bowls inside each other).  
* **Disadvantages:** It is nondeterministic (unless strictly controlled) and computationally expensive compared to constructive heuristics. It is best used as a "compaction" step after a heuristic placement.34

## **4\. Optimization Algorithms**

U-Nesting requires a global optimization framework to drive the placement heuristics. The separation of the **Search Mechanism** (Optimizer) from the **Evaluation Mechanism** (Placement \+ Geometry) is architecturally vital. The Optimizer explores the sequence and rotation parameters, while the Placer calculates the cost (efficiency) of those parameters.

### **4.1 Metaheuristics**

#### **4.1.1 Genetic Algorithms (GA)**

The Genetic Algorithm is the dominant metaheuristic in nesting literature due to its ability to handle the discrete (sequence) and continuous (rotation) nature of the problem simultaneously.5

* **Encoding Schemes:**  
  * *Sequence:* A permutation of item IDs (e.g., \`\`).  
  * *Rotation:* A vector of rotation parameters (e.g., \`\`).  
* **Crossover Operators:**  
  * *Order Crossover (OX1):* Preserves the relative order of items from parents, critical for preserving "clusters" of items that pack well together.  
  * *PMX (Partially Mapped Crossover):* Preserves adjacency relations.  
  * *Rust Implementation:* Libraries like genetic can be used, but custom implementation using the rand crate allows for domain-specific optimizations (e.g., ensuring grain constraints are respected during crossover).  
* **Mutation:**  
  * *Swap:* Exchange two items in the sequence.  
  * *Rotate:* Perturb the rotation angle or snap to a new discrete orientation.  
  * *Invert:* Reverse a subsequence of items.

#### **4.1.2 Biased Random Key Genetic Algorithm (BRKGA)**

This is a highly effective variant for packing problems.48

* **Encoding:** The chromosome is a vector of random floating-point numbers (keys) in $(0, 1\]$.  
* **Decoding:** The keys are sorted to produce the sequence permutation.  
* **Advantage:** Standard crossover operators (like uniform crossover) can be used without producing invalid permutations (e.g., duplicates). This simplifies the implementation significantly and has been shown to outperform standard GAs in both 2D and 3D bin packing benchmarks.

#### **4.1.3 Guided Local Search (GLS)**

GLS sits on top of a local search algorithm (like hill climbing) to help it escape local optima.

* **Mechanism:** It iteratively penalizes "features" of the current local optimum. For nesting, a feature might be "Item A and Item B are adjacent with a large gap." By adding a penalty to the objective function for this feature, the local search is forced to move to a new solution where this specific bad configuration is avoided.  
* **Application:** GLS is particularly effective for refining a solution found by a GA, squeezing out the final 1-2% of efficiency.1

### **4.2 Machine Learning Approaches**

#### **4.2.1 Deep Reinforcement Learning (DRL)**

DRL is emerging as the state-of-the-art for **Online Bin Packing**, where items arrive one by one and must be placed immediately.46

* **State Representation:**  
  * *Height Maps:* A 2D grid representing the "top" surface of the bin. Fast but lossy (cannot represent overhangs well).52  
  * *Voxel Grids:* Full 3D boolean grid. Accurate but memory intensive ($O(L \\times W \\times H)$).54  
  * *Packing Configuration Trees (PCT):* A hierarchical representation of the free space. More efficient for sparse packing.32  
* **Action Space:** Selecting a location and orientation. To handle continuous space, the action space is often discretized (e.g., choosing a voxel or a placement heuristic).  
* **Integration:** For U-Nesting, DRL is best suited as a *heuristic selector*—training a model to decide *which* placement heuristic (BLF, DBLF, etc.) to apply to the current item, rather than placing it directly. This "Hyper-heuristic" approach combines the speed of heuristics with the adaptability of learning.55

### **4.3 Exact Methods (MILP/CP)**

Mixed Integer Linear Programming (MILP) provides optimality guarantees but scales poorly ($N \> 20$ is often intractable for irregular nesting due to the complexity of non-overlap constraints).

* **Role in U-Nesting:** While not the primary engine for large datasets, exact methods are useful for solving small sub-problems, such as **clustering** (grouping 5-10 small parts into a rectangular block) before the main packing phase. This hierarchical approach improves density without the exponential cost of global exact optimization.8

## **5\. Constraints and Extensions**

Real-world industrial utility requires handling constraints far beyond simple geometry. A solver that packs tightly but ignores material grain or robot limitations is useless.

### **5.1 2D Manufacturing Constraints**

| Constraint | Description | Implementation Strategy in Rust |
| :---- | :---- | :---- |
| **Kerf Width** | Material lost to the cutting tool (laser/plasma). | Dilate (buffer) all item polygons by $Kerf/2$ prior to NFP generation. Use iOverlay or Clipper2 offset functions. |
| **Grain Direction** | Anisotropic materials (wood, rolled steel) require specific part orientation. | Restrict the "Rotation" gene in the GA. If part.grain \== true, allow only $\\theta \\in \\{0, 180\\}$. Snap invalid rotations during decoding.58 |
| **Common Line Cutting (CLC)** | Aligning parts to share a cut edge, saving time and gas. | Post-optimization step. Detect parallel edges within a tolerance, "snap" them together, and update the cutting path. Requires precise floating-point comparison.60 |
| **Defect Evasion** | Avoiding flawed areas on the stock sheet. | Model defects as "pre-placed" fixed obstacles (static items) in the container before the packing loop begins.6 |
| **Remnant Nesting** | Using irregular leftover sheets. | Treat the container as an irregular polygon (the remnant) rather than a rectangle. Use IFP generation on the irregular container boundary.13 |

### **5.2 3D Logistics Constraints**

| Constraint | Description | Implementation Strategy in Rust |
| :---- | :---- | :---- |
| **Stability (Gravity)** | Items must not tip over. | **Static Stability:** Project Center of Mass (CoM) to the support surface. Check if CoM lies within the 2D Convex Hull of the contact points (derived from parry3d contact manifolds).4 |
| **Load Bearing** | Items cannot support excessive weight. | Maintain a "stacking graph" (DAG) where edges represent support. Propagate weight downwards. If total weight on a node \> limit, invalidate placement.3 |
| **LIFO / FIFO** | Items must be accessible in a specific order (e.g., delivery route). | Constraint check during sequence generation. Item $i$ cannot be placed *behind* or *under* Item $j$ if $i$ must be unloaded first. |
| **"This Way Up"** | Orientation restrictions (e.g., liquids). | Lock rotation axes. Force rot\_x \== 0 and rot\_y \== 0\. Only allow rotation around Z axis. |

## **6\. Performance Optimization**

Achieving high performance in Rust involves minimizing memory allocations and maximizing cache coherency.

### **6.1 Parallel Computation Strategies**

* **Parallel NFP Generation:** Computing NFPs for $N$ part types is "embarrassingly parallel."  
  * *Rust:* Use rayon::par\_iter() to compute NFPs for all unique pairs $(A, B)$ concurrently.  
  * *Caching:* NFPs are invariant for a pair of shapes. Compute once, store in a DashMap\<(ID, ID), Polygon\> (concurrent hash map) for reuse across generations.37  
* **Island Model GA:** Run multiple independent GA populations on separate threads. occasionally migrate the best individuals between islands. This prevents premature convergence and utilizes multicore CPUs effectively without lock contention.37

### **6.2 Data Structures and Caching**

* **Spatial Indexing:** When checking collisions for a new item, do not check against all placed items. Use a **Quadtree** (2D) or **Dynamic AABB Tree / BVH** (3D).  
  * *Crate:* parry2d's internal QBvh (Simd-optimized Bounding Volume Hierarchy) is highly optimized for this.  
* **Geometry Instancing:** Store geometry data (vertices) once in a central repository. Items in the nest should be lightweight structs struct Item { id: usize, position: Isometry2\<f64\> } referencing the master geometry.  
* **Memory Arenas:** Use arena allocation (e.g., bumpalo) for the temporary polygons created during NFP generation. This reduces the overhead of malloc/free for the millions of temporary vertices created during boolean operations.

### **6.3 Rust-Specific Optimizations**

* **Zero-Copy Deserialization:** Use rkyv instead of serde\_json for loading large datasets or pre-computed NFP libraries. rkyv guarantees zero-copy access, meaning the data is mapped directly from disk to memory structs without parsing, crucial for instant startup with massive part libraries.61  
* **SIMD:** Use simba (packed SIMD for nalgebra) to vectorize geometric tests (e.g., testing 4 separating axes simultaneously in the SAT phase of collision detection).62

## **7\. Existing Solutions Analysis**

Analyzing existing tools reveals the gap U-Nesting aims to fill.

| Solution | Type | Algorithm | Strengths | Limitations |
| :---- | :---- | :---- | :---- | :---- |
| **SVGnest** | Open Source (JS) | GA \+ NFP (Minkowski) | Accessible, handles irregular shapes via NFP. | Performance bottleneck (JS single thread), poor 3D support, limited industrial constraints. |
| **libnest2d** | Library (C++) | GA \+ NFP (Boost) | Robust boolean ops (Boost.Geometry), used in PrusaSlicer. | Heavy dependencies (Boost), complex build system, primary focus on 3D printing (slice nesting). |
| **SigmaNEST** | Commercial | Proprietary (Heuristic \+ Exact) | Industry standard, massive constraint support (remnants, common line, bevels). | Expensive, closed source, Windows-only, legacy codebase. |
| **Lantek** | Commercial | Proprietary | Strong ERP integration, excellent common line cutting algorithms. | Closed ecosystem, difficult to integrate into custom pipelines. |
| **DeepNest** | Open Source (C\# port) | GA \+ NFP (Pixel Approx) | Simple UI, easy to use. | Pixel approximation limits precision; slow on large batches; C\# overhead. |
| **OR-Tools** | Library (C++) | CP-SAT | Excellent for rectangular bin packing. | Poor support for true irregular nesting (geometry heavy); requires reducing geometry to complex constraints. |

**Gap Analysis:** There is no dominant *Rust-native*, *high-performance* library that seamlessly handles both 2D irregular nesting and 3D irregular bin packing with a unified API. Existing Rust crates (bin-packer, crunch) are mostly for texture atlases (rectangles).63 U-Nesting fills this void by offering a systems-level engine that can be embedded in Python/C\# or run as a high-performance microservice.

## **8\. Academic Literature Review**

Key insights from the literature guide U-Nesting's design:

* **Bennell & Oliveira (2008) \- "The geometry of nesting problems: A tutorial":** This seminal paper established the "NFP \+ Heuristic" paradigm. The tutorial highlights that while NFP is geometrically complex to generate, it reduces the placement problem to a 1D search along the NFP boundary, which is computationally efficient. It strongly advocates for the separation of geometry and optimization.2  
* **Burke et al. (2007) \- "Complete and robust no-fit polygon generation":** This paper critiques the sliding algorithm, noting its fragility. It promotes the decomposition method as the only truly robust way to handle arbitrary concavities. This validates the decision to use Convex Decomposition \+ Boolean Union in U-Nesting.42  
* **Gonçalves & Resende (2013) \- "A biased random key genetic algorithm for 2D and 3D bin packing":** Introduced the BRKGA for packing. It demonstrated that separating the "encoding" (random keys) from the "decoder" (placement heuristic) yields state-of-the-art results for both 2D and 3D. This suggests a unified optimization architecture is possible for U-Nesting.48  
* **Elkeran (2013):** Introduced "Pairwise Clustering" before nesting. Grouping shapes that fit well together into a "meta-shape" significantly improves density for large instances. This is a key heuristic to implement.

## **9\. Implementation Considerations (Rust)**

This section outlines the architectural blueprint for U-Nesting.

### **9.1 Architecture Overview**

The system should be layered to ensure modularity and testability:

1. **Core (Geometry Layer):** Handles primitives (Polygon, Mesh), NFP generation, and Collision Detection.  
2. **Solver (Placement Layer):** Implements the heuristics (BLF, DBLF, 3D-EP). It asks the Core "Does this fit?"  
3. **Optimizer (Search Layer):** Implements the Metaheuristics (GA, SA, GLS). It drives the Solver by generating candidate sequences.  
4. **Interface (API/FFI Layer):** JSON/C-ABI for consumers (Python, C\#, Web/WASM).

### **9.2 Key Crates and Libraries**

* **Linear Algebra:** nalgebra is recommended over glam for scientific precision (f64), which is critical for CNC tolerances. glam is faster (f32) but potentially less precise for large coordinates.  
* **2D Geometry:** geo (types), iOverlay (boolean ops \- **Crucial for NFP**). parry2d for spatial partitioning and containment checks.23  
* **3D Geometry:** parry3d (GJK/EPA). Use parry3d::query::contact for stability analysis.  
* **Parallelism:** rayon for data parallelism.  
* **Serialization:** serde for JSON, rkyv for zero-copy binary cache.

### **9.3 Detailed NFP Pipeline in Rust**

1. **Input:** Polygon A, Polygon B.  
2. **Decomposition:** Use hertel\_mehlhorn (from parry2d transformation module) to break A and B into sets of convex hulls $\\{A\_i\\}, \\{B\_j\\}$.  
3. **Minkowski Sum:** For each pair $(A\_i, B\_j)$, compute the convex Minkowski sum. This involves sorting edge vectors and is $O(N)$.  
4. **Union:** Use iOverlay to compute the boolean union of all sub-sums: $\\bigcup (A\_i \\oplus \-B\_j)$. This is the raw NFP.  
5. **Holes:** If $A$ has holes, compute the Minkowski difference of the holes and $B$ to create "forbidden zones" inside the NFP. Subtract these from the raw NFP.

### **9.4 FFI Design**

To serve as a "domain-agnostic engine," U-Nesting must be callable from Python/C\#.

* **C-ABI:** Expose a clean C interface (extern "C"). Use \#\[no\_mangle\] functions.  
* **Data Transfer:** Avoid copying vertex lists repeatedly. Pass pointers to flat buffers (Float64Array) across the boundary. Use repr(C) structs for shared data types.  
* **Async/Callback:** The optimization is long-running. The FFI should support a callback mechanism (extern "C" fn callback(progress: f64, current\_utilization: f64)) or return a handle that can be polled for status, allowing the host application (e.g., a Python GUI) to remain responsive.

## **10\. Benchmarking and Validation**

### **10.1 Validation Plan**

1. **Regression Testing:** Use small, known cases (e.g., 2 rectangles in a square) to verify geometry logic.  
2. **Robustness Testing:** "Fuzz" the geometry engine with degenerate polygons (collinear points, self-intersecting loops, "bowties") to ensure the iOverlay integration handles edge cases without panicking.

### **10.2 Benchmarking Metrics**

* **Utilization:** $\\frac{\\sum \\text{Area}\_{items}}{\\text{Area}\_{container}}$.  
* **Runtime:** Time to reach $X\\%$ utilization.  
* **NFP Speed:** Time to generate NFP for two 100-vertex polygons.  
* **Quality vs Bound:** Comparison against known best results for standard datasets.

### **10.3 Comparison Methodology**

Run the **ESICUP** benchmark datasets.

* **Datasets:** Albano, Milenkovic, Schwerin, Hopper. These are available in text formats and should be parsed into U-Nesting's native format.  
* *Target:* \>80% utilization on "Albano" in \<60 seconds.  
* *Competitor:* SVGnest typically takes 5-10 minutes for high utilization on this dataset. U-Nesting's Rust backend should aim for a 10x speedup due to compiled performance and better NFP algorithms.

### **10.4 Regression Testing Strategy**

Create a CI/CD pipeline that runs a subset of ESICUP instances on every commit. Track the "Utilization Score" over time. If a commit causes a \>1% drop in utilization or \>5% increase in runtime, the build fails. This prevents performance regressions in the heuristics.

## ---

**11\. Recommendations and Roadmap**

### **11.1 Recommended Architecture**

Adhere to a **Trait-Based** design pattern for maximum extensibility:

Rust

trait Nester {  
    fn nest(&self, items: &\[Item\], container: \&Container) \-\> Result\<Placement\>;  
}  
// Implementations: GeneticNester, BrkgaNester, RLNester

This allows swapping the "Brain" (GA vs RL) while keeping the "Body" (Geometry Engine) constant.

### **11.2 Implementation Roadmap**

1. **Phase 1: Geometric Core (Weeks 1-4):**  
   * Implement robust NFP generation using iOverlay \+ Convex Decomposition.  
   * Validate with visual debugger (e.g., rerun.io SDK for Rust).  
2. **Phase 2: 2D Solver (Weeks 5-8):**  
   * Implement BLF and Touching Perimeter heuristics.  
   * Implement basic GA with Order Crossover.  
3. **Phase 3: 3D Core (Weeks 9-12):**  
   * Integrate parry3d. Implement "drop" physics heuristic.  
   * Implement simple Extreme Point placement.  
4. **Phase 4: Constraints & Performance (Weeks 13-16):**  
   * Add kerf/grain constraints.  
   * Parallelize the GA using rayon.  
   * Build FFI bindings for Python/C\#.

### **11.3 Research Questions Answered**

* **Most efficient NFP?** Decomposition \+ Boolean Union (via iOverlay) is the most robust and parallelizable for Rust.  
* **Best Metaheuristic?** Biased Random Key GA (BRKGA) offers the best trade-off between implementation complexity and solution quality for both 2D and 3D.  
* **2D/3D Shared Framework?** Yes, by abstracting the Collision trait. The Optimization loop (GA) operates on abstract Chromosome types, oblivious to whether the fitness function calls parry2d (NFP) or parry3d (GJK).  
* **Practical Limits:** Exact methods (MILP) limit out at \~15-20 irregular items. Heuristics are required for industrial scales (100+ items).  
* **MVP Feature Set:** 2D Irregular Nesting (NFP-based), 3 basic constraints (Rotation, Spacing, Container Bounds), JSON I/O, and a CLI interface.

## **12\. Conclusion**

U-Nesting represents a significant opportunity to bridge the gap between academic research and high-performance industrial application. By leveraging Rust's zero-cost abstractions, memory safety, and the ecosystem of geometry crates (parry, geo, iOverlay), it is possible to build a solver that outperforms legacy C++ solutions in safety and concurrency while matching them in speed. The critical path lies in the robust implementation of the **No Fit Polygon** generator for 2D and the efficient usage of **GJK-based collision** for 3D, wrapped in a **BRKGA** metaheuristic framework. This engine will not only serve as a powerful tool for manufacturers but also as a foundation for future research into AI-driven spatial optimization.

#### **참고 자료**

1. An open-source heuristic to reboot 2D nesting research \- arXiv, 1월 20, 2026에 액세스, [https://arxiv.org/html/2509.13329v1](https://arxiv.org/html/2509.13329v1)  
2. The geometry of nesting problems: a tutorial \- ePrints Soton \- University of Southampton, 1월 20, 2026에 액세스, [https://eprints.soton.ac.uk/154797/](https://eprints.soton.ac.uk/154797/)  
3. 3D Bin Packing: A New Heuristic Approach for Real-Case Scenarios \- Webthesis, 1월 20, 2026에 액세스, [https://webthesis.biblio.polito.it/35411/1/tesi.pdf](https://webthesis.biblio.polito.it/35411/1/tesi.pdf)  
4. Online 3D Bin Packing with Fast Stability Validation and Stable Rearrangement Planning \- arXiv, 1월 20, 2026에 액세스, [https://arxiv.org/html/2507.09123v1](https://arxiv.org/html/2507.09123v1)  
5. CORRECTED PROOF \- RUA Repository, 1월 20, 2026에 액세스, [https://rua.ua.es/bitstream/10045/144881/1/Martinez\_etal\_2024\_JIntelligentFuzzySyst\_revised.pdf](https://rua.ua.es/bitstream/10045/144881/1/Martinez_etal_2024_JIntelligentFuzzySyst_revised.pdf)  
6. A fully general, exact algorithm for nesting irregular shapes \- ResearchGate, 1월 20, 2026에 액세스, [https://www.researchgate.net/publication/262938008\_A\_fully\_general\_exact\_algorithm\_for\_nesting\_irregular\_shapes](https://www.researchgate.net/publication/262938008_A_fully_general_exact_algorithm_for_nesting_irregular_shapes)  
7. Two-dimensional irregular packing problems: A review \- Frontiers, 1월 20, 2026에 액세스, [https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2022.966691/full](https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2022.966691/full)  
8. Recent advances on two-dimensional bin packing problems \- Semantic Scholar, 1월 20, 2026에 액세스, [https://www.semanticscholar.org/paper/Recent-advances-on-two-dimensional-bin-packing-Lodi-Martello/641aaaf84be64b61ca2f66f7f9d273c1a9bfb1f4](https://www.semanticscholar.org/paper/Recent-advances-on-two-dimensional-bin-packing-Lodi-Martello/641aaaf84be64b61ca2f66f7f9d273c1a9bfb1f4)  
9. Data sets, 1월 20, 2026에 액세스, [https://oscar-oliveira.github.io/2D-Cutting-and-Packing/pages/datset.htm](https://oscar-oliveira.github.io/2D-Cutting-and-Packing/pages/datset.htm)  
10. Recent advances on two-dimensional bin packing problems \- Sci-Hub, 1월 20, 2026에 액세스, [https://sci-hub.box/10.1016/s0166-218x(01)00347-x](https://sci-hub.box/10.1016/s0166-218x\(01\)00347-x)  
11. A fast and scalable bottom-left-fill algorithm to solve nesting problems using a semi-discrete representation \- arXiv, 1월 20, 2026에 액세스, [https://arxiv.org/pdf/2103.08739](https://arxiv.org/pdf/2103.08739)  
12. (PDF) 3D Irregular Packing in an Optimized Cuboid Container \- ResearchGate, 1월 20, 2026에 액세스, [https://www.researchgate.net/publication/338174544\_3D\_Irregular\_Packing\_in\_an\_Optimized\_Cuboid\_Container](https://www.researchgate.net/publication/338174544_3D_Irregular_Packing_in_an_Optimized_Cuboid_Container)  
13. SigmaNEST X1 SP2 Manual (En) | PDF \- Scribd, 1월 20, 2026에 액세스, [https://www.scribd.com/document/664027430/SigmaNEST-X1-SP2-Manual-en](https://www.scribd.com/document/664027430/SigmaNEST-X1-SP2-Manual-en)  
14. A Comprehensive Guide to Nesting Software for Fiber Laser Cutting \- ADH Machine Tool, 1월 20, 2026에 액세스, [https://shop.adhmt.com/a-comprehensive-guide-to-nesting-software-for-fiber-laser-cutting/](https://shop.adhmt.com/a-comprehensive-guide-to-nesting-software-for-fiber-laser-cutting/)  
15. A comprehensive and robust procedure for obtaining the nofit polygon using Minkowski sums \- ePrints Soton \- University of Southampton, 1월 20, 2026에 액세스, [https://eprints.soton.ac.uk/36850/1/CORMSIS-05-05.pdf](https://eprints.soton.ac.uk/36850/1/CORMSIS-05-05.pdf)  
16. The geometry of nesting problems: A tutorial | Request PDF \- ResearchGate, 1월 20, 2026에 액세스, [https://www.researchgate.net/publication/4939770\_The\_geometry\_of\_nesting\_problems\_A\_tutorial](https://www.researchgate.net/publication/4939770_The_geometry_of_nesting_problems_A_tutorial)  
17. Two-dimensional irregular packing problems: A review \- Frontiers, 1월 20, 2026에 액세스, [https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2022.966691/pdf](https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2022.966691/pdf)  
18. Complete and robust no-fit polygon generation for the irregular stock cutting problem \- Graham Kendall, 1월 20, 2026에 액세스, [https://www.graham-kendall.com/papers/bhkw2007.pdf](https://www.graham-kendall.com/papers/bhkw2007.pdf)  
19. An iterated local search algorithm based on nonlinear programming for the irregular strip packing problem, 1월 20, 2026에 액세스, [https://www.amp.i.kyoto-u.ac.jp/tecrep/ps\_file/2007/2007-009.pdf](https://www.amp.i.kyoto-u.ac.jp/tecrep/ps_file/2007/2007-009.pdf)  
20. No Fit Polygon problem JONAS LINDMARK \- Diva-Portal.org, 1월 20, 2026에 액세스, [http://www.diva-portal.org/smash/get/diva2:699750/FULLTEXT01.pdf](http://www.diva-portal.org/smash/get/diva2:699750/FULLTEXT01.pdf)  
21. An improved method for calculating the no-fit polygon | Request PDF \- ResearchGate, 1월 20, 2026에 액세스, [https://www.researchgate.net/publication/220469623\_An\_improved\_method\_for\_calculating\_the\_no-fit\_polygon](https://www.researchgate.net/publication/220469623_An_improved_method_for_calculating_the_no-fit_polygon)  
22. GPU-Based Computation of Voxelized Minkowski Sums with Applications \- UC Berkeley, 1월 20, 2026에 액세스, [https://escholarship.org/content/qt9rm7j1pq/qt9rm7j1pq.pdf](https://escholarship.org/content/qt9rm7j1pq/qt9rm7j1pq.pdf)  
23. iShape-Rust/iOverlay: Boolean Operations for 2D Polygons: Supports intersection, union, difference, xor, and self-intersections for all polygon varieties. \- GitHub, 1월 20, 2026에 액세스, [https://github.com/iShape-Rust/iOverlay](https://github.com/iShape-Rust/iOverlay)  
24. iShape-Rust \- GitHub, 1월 20, 2026에 액세스, [https://github.com/iShape-Rust](https://github.com/iShape-Rust)  
25. arXiv:1903.11139v1 \[cs.CG\] 26 Mar 2019, 1월 20, 2026에 액세스, [https://arxiv.org/pdf/1903.11139](https://arxiv.org/pdf/1903.11139)  
26. Algorithm to partition a polygon (convex decomposition) using the Hertel-Mehlhorn algorithm · Issue \#1171 · georust/geo \- GitHub, 1월 20, 2026에 액세스, [https://github.com/georust/geo/issues/1171](https://github.com/georust/geo/issues/1171)  
27. Algorithms for collision detection between concave polygons \- Stack Overflow, 1월 20, 2026에 액세스, [https://stackoverflow.com/questions/20529899/algorithms-for-collision-detection-between-concave-polygons](https://stackoverflow.com/questions/20529899/algorithms-for-collision-detection-between-concave-polygons)  
28. parry2d::transformation \- Rust \- Docs.rs, 1월 20, 2026에 액세스, [https://docs.rs/parry2d/latest/parry2d/transformation/index.html](https://docs.rs/parry2d/latest/parry2d/transformation/index.html)  
29. Developing a plate nesting algorithm for a steel processing company \- UT Student Theses, 1월 20, 2026에 액세스, [https://essay.utwente.nl/fileshare/file/102916/Final%20version.pdf](https://essay.utwente.nl/fileshare/file/102916/Final%20version.pdf)  
30. The 3D Object Packing Problem into a Parallelepiped Container Based on Discrete-Logical Representation \- ResearchGate, 1월 20, 2026에 액세스, [https://www.researchgate.net/publication/306074091\_The\_3D\_Object\_Packing\_Problem\_into\_a\_Parallelepiped\_Container\_Based\_on\_Discrete-Logical\_Representation](https://www.researchgate.net/publication/306074091_The_3D_Object_Packing_Problem_into_a_Parallelepiped_Container_Based_on_Discrete-Logical_Representation)  
31. HAPE3D—a new constructive algorithm for the 3D irregular packing problem | Request PDF \- ResearchGate, 1월 20, 2026에 액세스, [https://www.researchgate.net/publication/276460677\_HAPE3D-a\_new\_constructive\_algorithm\_for\_the\_3D\_irregular\_packing\_problem](https://www.researchgate.net/publication/276460677_HAPE3D-a_new_constructive_algorithm_for_the_3D_irregular_packing_problem)  
32. Deliberate Planning of 3D Bin Packing on Packing Configuration Trees \- arXiv, 1월 20, 2026에 액세스, [https://arxiv.org/html/2504.04421v2](https://arxiv.org/html/2504.04421v2)  
33. Integrating Heuristic Methods with Deep Reinforcement Learning for Online 3D Bin-Packing Optimization \- MDPI, 1월 20, 2026에 액세스, [https://www.mdpi.com/1424-8220/24/16/5370](https://www.mdpi.com/1424-8220/24/16/5370)  
34. GJK: Collision detection algorithm in 2D/3D \- Hacker News, 1월 20, 2026에 액세스, [https://news.ycombinator.com/item?id=30620906](https://news.ycombinator.com/item?id=30620906)  
35. GJK: Collision detection algorithm in 2D/3D \- Winter, 1월 20, 2026에 액세스, [https://winter.dev/articles/gjk-algorithm](https://winter.dev/articles/gjk-algorithm)  
36. Collision Detection in Interactive 3D Environments, 1월 20, 2026에 액세스, [http://www.r-5.org/files/books/computers/algo-list/game-development/Gino\_van\_den\_Bergen-Collision\_Detection\_in\_Interactive\_3D\_Environments-EN.pdf](http://www.r-5.org/files/books/computers/algo-list/game-development/Gino_van_den_Bergen-Collision_Detection_in_Interactive_3D_Environments-EN.pdf)  
37. Real-time Collision Detection with Implicit Objects \- uu .diva, 1월 20, 2026에 액세스, [https://uu.diva-portal.org/smash/get/diva2:343820/FULLTEXT01.pdf](https://uu.diva-portal.org/smash/get/diva2:343820/FULLTEXT01.pdf)  
38. Simple 3D collision detection for general polyhedra \- Game Development Stack Exchange, 1월 20, 2026에 액세스, [https://gamedev.stackexchange.com/questions/61099/simple-3d-collision-detection-for-general-polyhedra](https://gamedev.stackexchange.com/questions/61099/simple-3d-collision-detection-for-general-polyhedra)  
39. About Parry, 1월 20, 2026에 액세스, [https://parry.rs/docs/](https://parry.rs/docs/)  
40. parry/CHANGELOG.md at master · dimforge/parry \- GitHub, 1월 20, 2026에 액세스, [https://github.com/dimforge/parry/blob/master/CHANGELOG.md](https://github.com/dimforge/parry/blob/master/CHANGELOG.md)  
41. Optimizing Two-Dimensional Irregular Packing: A Hybrid Approach of Genetic Algorithm and Linear Programming \- MDPI, 1월 20, 2026에 액세스, [https://www.mdpi.com/2076-3417/13/22/12474](https://www.mdpi.com/2076-3417/13/22/12474)  
42. A New Bottom-Left-Fill Heuristic Algorithm for the Two-Dimensional Irregular Packing Problem | Operations Research \- PubsOnLine, 1월 20, 2026에 액세스, [https://pubsonline.informs.org/doi/10.1287/opre.1060.0293](https://pubsonline.informs.org/doi/10.1287/opre.1060.0293)  
43. Integrating Heuristic Methods with Deep Reinforcement Learning for Online 3D Bin-Packing Optimization \- PMC \- NIH, 1월 20, 2026에 액세스, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11358981/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11358981/)  
44. The load-balanced multi-dimensional bin-packing problem \- DTU Inside, 1월 20, 2026에 액세스, [https://backend.orbit.dtu.dk/ws/files/124936570/Trivella\_Pisinger\_The\_LBMBP.pdf](https://backend.orbit.dtu.dk/ws/files/124936570/Trivella_Pisinger_The_LBMBP.pdf)  
45. (PDF) The Three-Dimensional Bin Packing Problem \- ResearchGate, 1월 20, 2026에 액세스, [https://www.researchgate.net/publication/2353632\_The\_Three-Dimensional\_Bin\_Packing\_Problem](https://www.researchgate.net/publication/2353632_The_Three-Dimensional_Bin_Packing_Problem)  
46. An Efficient Deep Reinforcement Learning Model for Online 3D Bin Packing Combining Object Rearrangement and Stable Placement \- IEEE Xplore, 1월 20, 2026에 액세스, [https://ieeexplore.ieee.org/document/10773090/](https://ieeexplore.ieee.org/document/10773090/)  
47. 2D Irregular Optimization Nesting Method based on Adaptive Probabilistic Genetic Simulated Annealing Algorithm \- CAD Journal, 1월 20, 2026에 액세스, [https://www.cad-journal.net/files/vol\_18/CAD\_18(2)\_2021\_242-257.pdf](https://www.cad-journal.net/files/vol_18/CAD_18\(2\)_2021_242-257.pdf)  
48. BoxStacker: Deep Reinforcement Learning for 3D Bin Packing Problem in Virtual Environment of Logistics Systems \- MDPI, 1월 20, 2026에 액세스, [https://www.mdpi.com/1424-8220/23/15/6928](https://www.mdpi.com/1424-8220/23/15/6928)  
49. Guided Local Search for the Three-Dimensional Bin-Packing Problem \- PubsOnLine, 1월 20, 2026에 액세스, [https://pubsonline.informs.org/doi/10.1287/ijoc.15.3.267.16080](https://pubsonline.informs.org/doi/10.1287/ijoc.15.3.267.16080)  
50. Fast Neighborhood Search for the Nesting Problem, 1월 20, 2026에 액세스, [https://di.ku.dk/forskning/Publikationer/tekniske\_rapporter/tekniske-rapporter-2003/03-03.pdf](https://di.ku.dk/forskning/Publikationer/tekniske_rapporter/tekniske-rapporter-2003/03-03.pdf)  
51. A deep reinforcement learning approach for online and concurrent 3D bin packing optimisation with bin replacement strategies \- PolyU Scholars Hub, 1월 20, 2026에 액세스, [https://research.polyu.edu.hk/en/publications/a-deep-reinforcement-learning-approach-for-online-and-concurrent-/](https://research.polyu.edu.hk/en/publications/a-deep-reinforcement-learning-approach-for-online-and-concurrent-/)  
52. Learning practically feasible policies for online 3D bin packing, 1월 20, 2026에 액세스, [http://scis.scichina.com/en/2022/112105.pdf](http://scis.scichina.com/en/2022/112105.pdf)  
53. An Efficient Deep Reinforcement Learning Model for Online 3D Bin Packing Combining Object Rearrangement and Stable Placement \- arXiv, 1월 20, 2026에 액세스, [https://arxiv.org/html/2408.09694v1](https://arxiv.org/html/2408.09694v1)  
54. How To Pack Anything Victor Rong \- DSpace@MIT, 1월 20, 2026에 액세스, [https://dspace.mit.edu/bitstream/handle/1721.1/151340/rong-vrong-meng-eecs-2023-thesis.pdf?sequence=1\&isAllowed=y](https://dspace.mit.edu/bitstream/handle/1721.1/151340/rong-vrong-meng-eecs-2023-thesis.pdf?sequence=1&isAllowed=y)  
55. One4Many-StablePacker: An Efficient Deep Reinforcement Learning Framework for the 3D Bin Packing Problem \- arXiv, 1월 20, 2026에 액세스, [https://arxiv.org/html/2510.10057v1](https://arxiv.org/html/2510.10057v1)  
56. LEARNING EFFICIENT ONLINE 3D BIN PACKING ON PACKING CONFIGURATION TREES \- OpenReview, 1월 20, 2026에 액세스, [https://openreview.net/pdf?id=bfuGjlCwAq](https://openreview.net/pdf?id=bfuGjlCwAq)  
57. solving irregular strip packing problems with free rotations using separation lines \- SciELO, 1월 20, 2026에 액세스, [https://www.scielo.br/j/pope/a/RcXzqWKwBnL7QhcgkgNyZPv/?lang=en](https://www.scielo.br/j/pope/a/RcXzqWKwBnL7QhcgkgNyZPv/?lang=en)  
58. LASER PROGRAMMING AND NESTING SOFTWARE \- Cincinnati Incorporated, 1월 20, 2026에 액세스, [https://wwwassets.e-ci.com/CIC\_Software/Nesting/LaserNst.pdf](https://wwwassets.e-ci.com/CIC_Software/Nesting/LaserNst.pdf)  
59. Nesting Fundamentals for Laser Cutting Stability: A Comprehensive Guide \- UDBU, 1월 20, 2026에 액세스, [https://www.udbu.eu/blog/params/post/4768272/nesting-fundamentals-for-laser-cutting-stability-a-comprehensive-guide](https://www.udbu.eu/blog/params/post/4768272/nesting-fundamentals-for-laser-cutting-stability-a-comprehensive-guide)  
60. Maximize material utilization in laser cutting \- Canadian Metalworking, 1월 20, 2026에 액세스, [https://www.canadianmetalworking.com/canadianfabricatingandwelding/article/automationsoftware/maximize-material-utilization-in-laser-cutting](https://www.canadianmetalworking.com/canadianfabricatingandwelding/article/automationsoftware/maximize-material-utilization-in-laser-cutting)  
61. awesome-rust/README.md at master \- GitHub, 1월 20, 2026에 액세스, [https://github.com/uhub/awesome-rust/blob/master/README.md](https://github.com/uhub/awesome-rust/blob/master/README.md)  
62. Announcing the Rapier physics engine \- Dimforge, 1월 20, 2026에 액세스, [https://dimforge.com/blog/2020/08/25/announcing-the-rapier-physics-engine/](https://dimforge.com/blog/2020/08/25/announcing-the-rapier-physics-engine/)  
63. Algorithms — list of Rust libraries/crates // Lib.rs, 1월 20, 2026에 액세스, [https://lib.rs/algorithms](https://lib.rs/algorithms)