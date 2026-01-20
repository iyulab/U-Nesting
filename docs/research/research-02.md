# U-Nesting: Comprehensive Research Report for Spatial Optimization Engine Development

The most efficient approach for building U-Nesting combines **No-Fit Polygon (NFP) computation** with **Genetic Algorithm optimization** for 2D irregular nesting, and **Extreme Point heuristics** with **BRKGA** for 3D bin packing. Both problem classes are NP-hard, but modern algorithms achieve **90-95% material utilization** within practical time constraints. The recommended Rust architecture leverages `geo` + `i_overlay` for geometry, `parry2d/3d` for collision detection, and `rayon` for parallelization, with JSON-based FFI for C#/Python consumers.

---

## Problem definitions establish the computational landscape

The **2D irregular nesting problem** (also called irregular strip packing) places polygonal pieces on a rectangular strip of fixed width to minimize length used. The **3D bin packing problem** assigns rectangular boxes to the minimum number of identical bins. Both are **strongly NP-hard**—no polynomial-time algorithm exists unless P=NP.

For 2D nesting, the mathematical formulation minimizes strip length subject to non-overlap constraints (rᵢ ∩ rⱼ = ∅), containment (r ⊆ R), and rotation restrictions (θᵢ ∈ Θ). The problem reduces from rectangle packing, which is NP-complete. For 3D bin packing, the continuous lower bound has an asymptotic worst-case ratio of **1/8**, meaning naive volume-based estimates can be off by 8×.

**Approximation bounds** provide theoretical guardrails. First Fit Decreasing (FFD) achieves **(11/9)OPT + 6/9 ≈ 1.22** for 1D bin packing. For 2D/3D irregular packing, no constant-factor approximation exists for general cases—heuristics dominate practice.

| Problem Variant | Complexity | Approximation Ratio |
|-----------------|------------|---------------------|
| 1D Bin Packing | NP-hard | FFD: 1.22 OPT |
| 2D Rectangle Strip | NP-hard | FFDH: 1.7 OPT + 1 |
| 2D Irregular | NP-complete | No constant bound |
| 3D Bin Packing | Strongly NP-hard | No PTAS unless P=NP |

---

## No-Fit Polygon is the geometric foundation for 2D nesting

The **No-Fit Polygon (NFP)** is the critical geometric construct that enables efficient placement in irregular nesting. Given stationary polygon A and orbiting polygon B, the NFP defines the locus of B's reference point as B slides around A while maintaining contact. If B's reference point lies **inside** the NFP, the shapes overlap; **on the boundary**, they touch; **outside**, they're separated.

This transforms expensive O(nm) edge-edge overlap testing into a simple **O(k) point-in-polygon test**, where k is the NFP's edge count. For a nesting algorithm evaluating thousands of placement candidates, this speedup is transformative.

**Three primary NFP computation methods** exist:

The **Minkowski sum approach** computes NFP(A,B) = A ⊕ (-B) by reflecting B through the origin and computing the sum. For convex polygons, this runs in **O(n+m)** by merging sorted edge lists. For non-convex shapes, decomposition into convex parts is required, with pairwise sums merged via polygon union—complexity reaches **O(n²m²)** worst-case.

The **orbiting algorithm** simulates B sliding around A, tracking the reference point trajectory. Burke et al.'s robust implementation handles degenerate cases including interlocking concavities and perfect-fit positions. Complexity is theoretically O(n²m²) but typically much better in practice.

**Decomposition-based methods** break non-convex polygons into convex pieces, compute pairwise NFPs efficiently, then union the results. The trade-off: more sub-polygons mean simpler NFPs but complex unions.

For **production NFP computation**, the Burke et al. (2007) orbiting algorithm provides the best robustness. Pre-compute NFPs for all piece pairs at all allowed rotation angles and cache aggressively—NFPs are expensive to compute but heavily reused during optimization.

The **Inner Fit Polygon (IFP)** handles container boundaries using the same machinery with reversed container orientation. The IFP defines valid placement positions inside the bin.

---

## 3D bin packing uses collision detection instead of precomputed geometry

While 2D nesting relies on NFPs, 3D bin packing typically uses **direct collision detection** at placement time. The 3D equivalent—the No-Fit Polyhedron—has **O(n²m²) face complexity** and is impractical to precompute for non-trivial shapes.

**GJK (Gilbert-Johnson-Keerthi)** determines collision between convex shapes by checking if their Minkowski difference contains the origin. The algorithm iteratively builds a simplex (up to tetrahedron in 3D) and terminates in **O(k) iterations** where k is typically small. GJK requires only a "support function" returning the furthest point in any direction, making it applicable to any convex shape representation.

**SAT (Separating Axis Theorem)** tests for a separating plane between convex objects. For two OBBs (Oriented Bounding Boxes), exactly **15 axes** must be tested: 3 face normals from each box plus 9 edge-edge cross products. This runs in O(1) for OBB-OBB tests.

**Spatial partitioning** accelerates broad-phase collision detection. **Octrees** subdivide 3D space into 8 octants recursively, enabling O(log n) queries. **BVH (Bounding Volume Hierarchy)** organizes objects by enclosing bounding volumes, with the Surface Area Heuristic (SAH) optimizing tree construction. BVH excels for non-uniform, clustered distributions.

For rotation in 3D, **quaternions** avoid gimbal lock and enable smooth interpolation (SLERP). Store rotations as quaternions (4 values), convert to matrices for transform application. For bin packing with discrete orientations (6 cuboid orientations), an enum with lookup table suffices.

---

## Placement algorithms turn optimization into geometry queries

### 2D placement strategies

**Bottom-Left Fill (BLF)** places each piece as far down and left as possible without overlapping. With precomputed NFPs, the algorithm computes the feasible region (IFP minus all translated NFPs), then finds the bottom-left valid point. Complexity is **O(n³)** total for n pieces.

**Lowest Gravity Center** minimizes the overall center of mass height, naturally filling concavities and creating flat nesting boundaries. **Touching Perimeter** maximizes contact between pieces for tighter packing. NFP-based strategies evaluate candidate positions on NFP boundaries—vertices and edge-edge intersections provide discrete candidates.

### 3D placement strategies

**Extreme Point heuristics** maintain candidate positions generated by projecting corner vertices of placed items. When an item is placed, 6 new extreme points are generated by projecting from its corners along each axis. The **EP-FFD** algorithm sorts items by volume decreasing, then places each at the first valid extreme point (sorted by z, y, x). Complexity is **O(n²)**.

**DBLF (Deepest Bottom Left Fill)** prioritizes depth (z), then bottom (y), then left (x). **Layer-based approaches** reduce 3D to sequential 2D problems—build horizontal layers of similar-height items, then stack layers.

**Wall building** constructs packings wall-by-wall using 2D packing for each wall, advancing depth by wall thickness. This hybrid approach leverages mature 2D algorithms.

---

## Genetic algorithms provide the best quality-time tradeoff for optimization

For 2D irregular nesting, **Genetic Algorithms (GA)** with proper encoding consistently outperform other metaheuristics. The key insight: **Grouping Genetic Algorithm (GGA)** encodes chromosomes where genes represent bins rather than items, preserving the building block of "well-filled bins" across generations.

**Encoding schemes** significantly impact performance:
- **Permutation/priority encoding**: Items as a sequence, decoded using placement heuristics (BLF, NFP placement)
- **Grouping encoding**: Genes represent bins containing item sets
- **Linear Linkage Encoding (LLE)**: Position pointers to next item in same group

**Crossover operators** for packing include:
- **Bin Packing Crossover (BPCX)**: Select random bins from Parent1, inject into offspring, repair with FFD
- **Order Crossover (OX)**: For permutation encoding, preserves relative order
- **Exon Shuffling (ESX)**: Concatenate, sort by bin fullness, inherit sequentially

**BRKGA (Biased Random-Key GA)** from Gonçalves & Resende uses random-key encoding with biased crossover favoring elite parents. Tested on **858 instances**, it statistically outperformed tabu search, local search, and GRASP+VND. The chromosomes encode packing sequence and placement procedure parameters.

**Multi-objective variants** handle trade-offs (utilization vs. computation time, multiple bin types):
- **NSGA-II**: O(MN²) for M objectives, N population; uses crowding distance for diversity
- **MOEA/D**: Decomposes into scalar subproblems; better for many-objective (>3)

**Recommended GA parameters**: Population 100-500, generations 100-1000, crossover probability 0.8-0.95, mutation probability 0.01-0.1, elitism 1-5%.

---

## Simulated annealing and tabu search offer complementary strengths

**Simulated Annealing (SA)** accepts worse solutions with probability exp(-ΔE/T), enabling escape from local optima. Geometric cooling (T(k+1) = α·T(k), α ≈ 0.95) is most common. **Neighborhood moves** for packing include single-item relocation, inter-bin swaps, chain moves, and bin elimination (empty one bin, redistribute).

Set initial temperature so **~80% of moves accepted** initially. Run 10n to 100n iterations per temperature level. Terminate when acceptance rate drops below 1% or no improvement for k iterations.

**Tabu Search (TS)** uses memory structures to avoid cycling. **Short-term memory** (tabu list) forbids recently visited moves for a tenure period. **Long-term memory** tracks frequency to drive diversification. Tenure of √n or random(5, 15+√n) works well.

**Aspiration criteria** override tabu status when a move leads to a new best solution. Balance **intensification** (exploit promising regions) with **diversification** (explore undervisited regions).

---

## Exact methods solve small instances but don't scale

**Mixed Integer Linear Programming (MILP)** formulations use binary variables x[i,j] = 1 if item i is in bin j. Constraints enforce single assignment and capacity limits. Commercial solvers (CPLEX, Gurobi) solve 1D instances with ~500 items, but 2D/3D limits drop to **~50-100 items** due to geometric constraints.

**Constraint Programming (CP)** excels with complex constraints. OR-Tools' CP-SAT solver handles bin packing with `AddNoOverlap2D` for 2D rectangle placement. CP naturally expresses incompatibilities, intervals, and channeling constraints.

**Branch and Bound** with Martello-Toth bounds prunes aggressively using three lower bounds (L1: continuous, L2: item pairs, L3: reduction-based). The algorithm solved 3D instances with up to **90 items** optimally.

**When to use exact methods**: Small instances (<50 items for 2D/3D), generating optimal solutions for benchmarking, solving subproblems within hybrid approaches.

---

## Machine learning is closing the gap for online packing

**Reinforcement Learning** for 3D bin packing uses height-map state representations (2D grid of heights), item dimensions, and utilization metrics. Actions are discrete (position, rotation) pairs or continuous coordinates. Rewards combine utilization, compactness, and stability scores.

**PPO (Proximal Policy Optimization)** provides stable training. Recent advances include:
- **GOPT (2024)**: Transformer-based DRL achieving cross-dimension generalization
- **One4Many-StablePacker**: Entropy control for diverse solutions
- **GFPack++ (2024)**: Diffusion models learning gradient fields for 2D irregular packing

**DRL advantages**: Excellent for online scenarios, near-constant inference time, handles stability constraints naturally. **Disadvantages**: Requires significant training data, offline quality still lags best metaheuristics, implementation complexity ~3 months.

For U-Nesting MVP, **defer ML integration** to later phases. The foundation should use proven metaheuristics.

---

## Open-source implementations reveal proven patterns

### SVGnest (JavaScript, 2.5k stars)
Uses NFP + GA for 2D irregular nesting. NFP computed via Clipper.js polygon operations. Fitness minimizes unplaceable parts, then bins used, then width. Web Worker parallelization enables continuous optimization. **Limitation**: JavaScript performance caps large datasets.

### libnest2d (C++, used in PrusaSlicer)
Production-ready NFP-based placer with NLopt local optimization. Templated geometry backend allows integration without dependencies. Header-only option available. **Limitation**: Holes and concavities still incomplete; documentation sparse.

### OR-Tools (Google)
CP-SAT solver handles bin packing with complex constraints. Multi-language support (C++, Python, C#, Java). **Limitation**: Rectangular shapes only natively; irregular requires discretization.

### rectpack (Python)
Multiple algorithm implementations (MaxRects, Guillotine, Skyline) based on Jukka Jylänki's work. Clean API with algorithm selection as parameter. **Limitation**: Rectangles only—not suitable for irregular nesting.

### Commercial differentiators
SigmaNEST, Hypertherm ProNest, and Lantek achieve 90-95% material utilization with machine-specific optimizations (kerf, lead-in/out, clamps), ERP integration, and remnant tracking. **Key gap**: Industry-specific features unavailable in open source.

---

## Algorithm comparison matrix

| Algorithm | 2D Irregular | 3D Bin Pack | Speed | Quality | Implementation |
|-----------|-------------|-------------|-------|---------|----------------|
| **GGA (Grouping GA)** | ★★★★★ | ★★★ | ★★★ | ★★★★★ | 3-4 weeks |
| **BRKGA** | ★★★★ | ★★★★ | ★★★ | ★★★★★ | 3-4 weeks |
| **Simulated Annealing** | ★★★★ | ★★★ | ★★★★ | ★★★★ | 3-5 days |
| **VNS** | ★★★★ | ★★★ | ★★★★ | ★★★★ | 2-3 weeks |
| **GRASP** | ★★★★ | ★★★ | ★★★★ | ★★★ | 1-2 weeks |
| **Tabu Search** | ★★★★ | ★★★ | ★★★ | ★★★★ | 1-2 weeks |
| **MILP** | ★★ | ★★ | ★ | ★★★★★ | 1 week (binding) |
| **DRL** | ★★★ | ★★★★ | ★★★★★ | ★★★ | 2-3 months |
| **FFD/BFD Heuristics** | ★★ | ★★★ | ★★★★★ | ★★ | 1-2 days |

---

## Recommended architecture for U-Nesting

```
u_nesting/
├── geometry/           # Core geometric primitives
│   ├── polygon.rs      # Polygon with holes (geo-types wrapper)
│   ├── nfp.rs          # No-Fit Polygon computation (Burke algorithm)
│   ├── ifp.rs          # Inner Fit Polygon for containers
│   ├── collision.rs    # parry2d/3d collision detection
│   └── spatial.rs      # rstar R*-tree integration
├── placement/          # Placement strategies (trait-based)
│   ├── strategy.rs     # PlacementStrategy trait
│   ├── bottom_left.rs  # BLF implementation
│   └── nfp_placer.rs   # NFP-guided placement
├── optimization/       # Optimization algorithms
│   ├── genetic.rs      # GGA/BRKGA implementation
│   ├── annealing.rs    # Simulated annealing
│   └── local_search.rs # Hill climbing, VNS
├── cache/              # Performance infrastructure
│   └── nfp_cache.rs    # Thread-safe NFP caching
├── problem/            # Problem definitions
│   ├── nesting_2d.rs   # 2D strip/bin packing
│   └── binpack_3d.rs   # 3D bin packing
└── ffi/                # Foreign function interfaces
    ├── json.rs         # JSON-based API
    └── python.rs       # PyO3 bindings
```

### Key dependencies

```toml
[dependencies]
geo = "0.28"              # Polygon representation
geo-types = "0.7"         # Coordinate types
i_overlay = "4.2"         # High-performance boolean ops
parry2d = "0.25"          # Collision detection 2D
parry3d = "0.25"          # Collision detection 3D
rstar = "0.12"            # R*-tree spatial indexing
robust = "1.2"            # Exact geometric predicates
rayon = "1.10"            # Data parallelism
bumpalo = "3.19"          # Arena allocation
serde = "1.0"             # Serialization
thiserror = "2.0"         # Error handling
```

### Trait design enables extensibility

```rust
pub trait PlacementStrategy: Send + Sync {
    fn place_piece(&self, piece: &Piece, container: &Container, 
                   placed: &[PlacedPiece]) -> Option<Placement>;
}

pub trait OptimizationAlgorithm: Send + Sync {
    fn optimize(&self, problem: &Problem, config: &Config, 
                progress: impl Fn(Progress)) -> Solution;
}
```

### FFI design for C#/Python consumers

Use **JSON-based API** for simplicity and debuggability:

```rust
#[no_mangle]
pub extern "C" fn nest_json(
    request: *const c_char,
    result_buf: *mut c_char, 
    buf_size: usize
) -> i32
```

Add **PyO3 bindings** for Python with maturin for wheel building. For C#, use P/Invoke with cbindgen-generated headers.

---

## Implementation roadmap with complexity estimates

### Phase 1: Core geometry (4-6 weeks)
- Polygon representation with holes *(1 week)*
- NFP computation (Burke algorithm) *(3-4 weeks)*
- IFP for container boundaries *(1 week)*
- Basic collision detection via parry2d *(3 days)*

### Phase 2: Placement & heuristics (3-4 weeks)
- Bottom-Left Fill placement *(1 week)*
- NFP-guided placement strategy *(1-2 weeks)*
- FFD/BFD sorting heuristics *(3 days)*
- Extreme Point heuristics for 3D *(1 week)*

### Phase 3: Optimization algorithms (4-6 weeks)
- Genetic Algorithm with permutation encoding *(2-3 weeks)*
- Simulated Annealing *(1 week)*
- Local search / hill climbing *(1 week)*
- NFP caching with thread-safe access *(1 week)*

### Phase 4: Parallelization & performance (2-3 weeks)
- Rayon integration for NFP computation *(3 days)*
- Parallel fitness evaluation in GA *(1 week)*
- Bumpalo arena allocation *(3 days)*
- Criterion benchmarking suite *(1 week)*

### Phase 5: API & integration (2-3 weeks)
- JSON FFI layer *(1 week)*
- Python bindings via PyO3 *(1 week)*
- Progress callbacks and cancellation *(3 days)*

**Total MVP estimate**: 15-22 weeks for production-ready 2D nesting with 3D bin packing.

---

## Benchmark plan for validation

### Standard datasets

**2D Irregular Nesting (ESICUP)**:
- ALBANO, BLAZ1-3, DIGHE1-2, FU, JAKOBS1-2
- MARQUES, POLY1-5, SHAPES, SHIRTS, SWIM, TROUSERS
- Jigsaw instances (100% utilization known optimal)

**1D Bin Packing (BPPLIB)**:
- Scholl instances: 720+480+10 instances, n=50-500
- Falkenauer "triplet" hard instances

**3D Bin Packing**:
- Martello et al. (2000): 9 classes, up to 90 items

### Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Utilization rate** | Σ item_areas / container_area | >90% for 2D |
| **Bins used** | Count vs. lower bound | Within 5% of BKS |
| **Computation time** | Time to solution quality | <60s for 100 pieces |
| **Gap to BKS** | (solution - best_known) / best_known | <2% |

### Regression testing strategy

1. Run nightly benchmarks on ESICUP subset (10 instances)
2. Track utilization mean/stddev over time
3. Alert on >1% degradation
4. Full benchmark suite weekly on all datasets

---

## Answers to key research questions

**1. Most efficient NFP computation for production?**
Burke et al. (2007) orbiting algorithm with robust degenerate case handling. Pre-compute all piece-pair NFPs at startup, cache with rotation quantization (e.g., 1° increments).

**2. Best metaheuristic for 2D irregular nesting?**
Grouping Genetic Algorithm (GGA) with permutation encoding decoded by NFP-guided placement. BRKGA provides similar quality with simpler implementation.

**3. Common framework for 2D and 3D?**
Shared optimization layer (GA, SA) with problem-specific placement strategies. PlacementStrategy trait abstracts 2D/3D differences. Solution representation (permutation + positions) is dimensionally agnostic.

**4. Practical limits of exact methods?**
~50-100 items for 2D/3D with MILP; ~500 items for 1D. Use exact methods for small subproblems within hybrid approaches or generating benchmark optimal solutions.

**5. ML/RL improvement opportunities?**
For online packing, DRL closes gap with heuristics. For offline optimization, use ML for parameter tuning (Bayesian optimization) or solution initialization. Full ML placement requires 2-3 months development and extensive training data.

**6. Minimal MVP feature set?**
2D rectangular + irregular nesting with NFP computation, BLF placement, GA optimization, JSON API, single-bin optimization. Defer: 3D, multi-objective, ML integration.

**7. Handling edge cases (tiny parts, extreme aspect ratios)?**
Sort tiny parts last (fill gaps), apply simplification for extreme shapes, set minimum area thresholds. Commercial solutions use size-adaptive placement parameters and operator review for outliers.

---

## Critical implementation insights

**NFP caching is essential**: A nesting run with 100 pieces at 4 rotation angles requires 40,000 NFP computations. Pre-compute once, cache in Arc<RwLock<HashMap>>. Cache hit converts O(n²) NFP to O(1) lookup.

**Coordinate scaling improves robustness**: Scale floating-point coordinates to integers (×10⁶) for boolean operations with i_overlay. This eliminates floating-point edge cases in NFP boundary computation.

**Early termination prevents wasted computation**: Monitor solution improvement rate. If no improvement for 20% of allocated time, terminate and return best-found. Implement anytime interface returning current best on demand.

**Parallelization yields 4-8× speedup**: NFP pairs computed independently (embarrassingly parallel). GA fitness evaluation parallelizes across population. Use rayon::par_iter for drop-in parallelism.

This research provides a comprehensive foundation for U-Nesting development. The recommended approach—NFP-based placement with genetic algorithm optimization—balances implementation complexity against solution quality, with a clear path to production deployment in Rust.