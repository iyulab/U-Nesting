# U-Nesting 개발 로드맵

리서치 문서를 기반으로 상세한 다단계 로드맵을 구성했습니다.

---

## 전체 타임라인 개요

| Phase | 기간 | 핵심 목표 |
|-------|------|----------|
| **Phase 1** | 5-6주 | Geometry Core (2D/3D 기초) |
| **Phase 2** | 4-5주 | NFP 엔진 및 배치 알고리즘 |
| **Phase 3** | 5-6주 | 최적화 알고리즘 (GA/SA) |
| **Phase 4** | 3-4주 | 성능 최적화 및 병렬화 |
| **Phase 5** | 3-4주 | FFI 및 통합 API |
| **Phase 6** | 2-3주 | 벤치마크 및 릴리스 준비 |

**총 예상 기간: 22-28주**

---

## Phase 1: Geometry Core Foundation (5-6주)

### 목표
2D/3D 기하학적 기초 구조 구축 및 기본 연산 구현

### 태스크

#### 1.1 프로젝트 구조 설정 (3일)
- [ ] Cargo workspace 구성 (`core`, `d2`, `d3`, `ffi` 크레이트)
- [ ] 의존성 설정 (`geo`, `geo-types`, `parry2d`, `parry3d`, `nalgebra`)
- [ ] CI/CD 파이프라인 구성 (GitHub Actions)
- [ ] 코드 품질 도구 설정 (`clippy`, `rustfmt`, `cargo-deny`)

```toml
# Cargo.toml 핵심 의존성
[dependencies]
geo = "0.28"
geo-types = "0.7"
i_overlay = "4.2"
parry2d = "0.25"
parry3d = "0.25"
nalgebra = "0.33"
```

#### 1.2 Core Traits 정의 (1주)
- [ ] `Geometry` trait (2D/3D 공통 추상화)
- [ ] `Boundary` trait (컨테이너 추상화)
- [ ] `Placement` struct (위치 + 회전)
- [ ] `SolveResult` struct (결과 표현)
- [ ] Error types 정의 (`thiserror` 기반)

**참조**: [research-02.md - Trait design enables extensibility](research/research-02.md)

#### 1.3 2D Polygon 구현 (1.5주)
- [ ] `Polygon2D` 구조체 (외곽선 + 홀)
- [ ] 기본 연산: 면적, 중심점, 바운딩 박스
- [ ] 회전/이동 변환 (`Isometry2`)
- [ ] Point-in-polygon 테스트
- [ ] Polygon simplification (Douglas-Peucker)
- [ ] Convex hull 계산

**참조**: [research-01.md §2.1](research/research-01.md) - 2D Geometry

#### 1.4 3D Geometry 구현 (1.5주)
- [ ] `Box3D` 구조체 (직육면체)
- [ ] `Mesh3D` 구조체 (삼각형 메시)
- [ ] AABB (Axis-Aligned Bounding Box)
- [ ] OBB (Oriented Bounding Box)
- [ ] 회전 표현: Quaternion 기반
- [ ] Volume 및 Surface area 계산

**참조**: [research-01.md §2.2](research/research-01.md) - 3D Geometry

#### 1.5 Convex Decomposition (1주)
- [ ] Hertel-Mehlhorn 알고리즘 구현 (2D)
- [ ] V-HACD 통합 또는 구현 (3D)
- [ ] Decomposition 결과 캐싱

**참조**: [research-01.md §2.1.1](research/research-01.md) - Decomposition-Based Methods

### 테스트 전략

```rust
// tests/geometry_2d_tests.rs
#[test]
fn test_polygon_area() {
    let square = Polygon2D::rectangle(10.0, 10.0);
    assert_eq!(square.area(), 100.0);
}

#[test]
fn test_polygon_with_hole() {
    let outer = vec![(0,0), (100,0), (100,100), (0,100)];
    let hole = vec![(25,25), (75,25), (75,75), (25,75)];
    let poly = Polygon2D::with_hole(outer, hole);
    assert_eq!(poly.area(), 7500.0); // 10000 - 2500
}

#[test]
fn test_convex_decomposition() {
    let l_shape = create_l_shaped_polygon();
    let convex_parts = decompose_convex(&l_shape);
    assert!(convex_parts.len() >= 2);
    assert!(convex_parts.iter().all(|p| p.is_convex()));
}
```

### 완료 기준
- [ ] 모든 기하학 연산에 대한 단위 테스트 통과
- [ ] Property-based testing (`proptest`) 추가
- [ ] 문서화 완료 (`cargo doc`)

---

## Phase 2: NFP Engine & Placement Algorithms (4-5주)

### 목표
No-Fit Polygon 계산 엔진 및 기본 배치 알고리즘 구현

### 태스크

#### 2.1 NFP 계산 - Convex Case (1주)
- [ ] Minkowski Sum for convex polygons (O(n+m))
- [ ] Edge vector sorting and merging
- [ ] Reference point tracking

**참조**: [research-01.md §2.1.1](research/research-01.md) - Minkowski Sum Approach

#### 2.2 NFP 계산 - Non-Convex Case (2주)
- [ ] Burke et al. Orbiting 알고리즘 구현
- [ ] Degenerate case 처리 (collinear, coincident)
- [ ] Decomposition + Union 방식 대안 구현
- [ ] `i_overlay` 기반 Boolean 연산 통합
- [ ] Hole 처리 (내부 구멍이 있는 폴리곤)

```rust
// NFP 계산 파이프라인
pub fn compute_nfp(
    stationary: &Polygon2D,
    orbiting: &Polygon2D,
    rotation: f64
) -> Result<Polygon2D, NfpError> {
    // 1. Convex decomposition
    let stat_parts = decompose_convex(stationary)?;
    let orb_parts = decompose_convex(&orbiting.rotate(rotation))?;
    
    // 2. Pairwise Minkowski sums
    let sub_nfps: Vec<Polygon2D> = stat_parts.par_iter()
        .flat_map(|a| orb_parts.iter().map(|b| minkowski_sum(a, b)))
        .collect();
    
    // 3. Union via i_overlay
    boolean_union(&sub_nfps)
}
```

**참조**: 
- [research-01.md §2.1.1](research/research-01.md) - NFP Computation Methods
- [Burke et al. (2007)](https://www.graham-kendall.com/papers/bhkw2007.pdf)

#### 2.3 Inner Fit Polygon (IFP) (0.5주)
- [ ] Container 경계에 대한 IFP 계산
- [ ] Margin 적용

**참조**: [research-01.md §2.1.2](research/research-01.md) - Inner Fit Polygon

#### 2.4 NFP 캐싱 시스템 (0.5주)
- [ ] Thread-safe cache (`DashMap` 또는 `Arc<RwLock<HashMap>>`)
- [ ] Cache key: `(geometry_id, geometry_id, rotation_angle)`
- [ ] LRU eviction policy

```rust
pub struct NfpCache {
    cache: DashMap<(GeometryId, GeometryId, RotationKey), Arc<Polygon2D>>,
    max_size: usize,
}
```

#### 2.5 2D Placement Algorithms (1주)
- [ ] **Bottom-Left Fill (BLF)**: 기본 구현
- [ ] **NFP-guided BLF**: NFP 경계 위 최적점 탐색
- [ ] **Deepest Bottom-Left Fill (DBLF)**: 개선된 BLF
- [ ] **Touching Perimeter**: 접촉 최대화

**참조**: [research-01.md §3.1](research/research-01.md) - 2D Placement Strategies

#### 2.6 3D Placement Algorithms (1주)
- [ ] **Extreme Point Heuristic**: EP 생성 및 관리
- [ ] **DBLF-3D**: 3D 확장
- [ ] GJK/EPA 기반 collision detection (`parry3d`)

**참조**: 
- [research-01.md §3.2](research/research-01.md) - 3D Placement Strategies
- [research-02.md](research/research-02.md) - Extreme Point heuristics

### 테스트 전략

```rust
// tests/nfp_tests.rs
#[test]
fn test_nfp_convex_squares() {
    let a = Polygon2D::rectangle(10.0, 10.0);
    let b = Polygon2D::rectangle(5.0, 5.0);
    let nfp = compute_nfp(&a, &b, 0.0).unwrap();
    
    // NFP should be a 15x15 rectangle offset
    assert_relative_eq!(nfp.area(), 225.0, epsilon = 0.01);
}

#[test]
fn test_nfp_l_shapes() {
    let a = create_l_shape();
    let b = create_l_shape();
    let nfp = compute_nfp(&a, &b, 0.0).unwrap();
    
    // Verify no self-intersection
    assert!(nfp.is_valid());
}

#[test]
fn test_blf_placement() {
    let pieces = vec![rect(20, 10), rect(15, 15), rect(10, 30)];
    let container = Boundary2D::rectangle(50.0, 100.0);
    
    let result = bottom_left_fill(&pieces, &container);
    assert!(result.all_placed());
    assert!(result.no_overlaps());
}
```

### Benchmark 추가
```rust
// benches/nfp_bench.rs
use criterion::{criterion_group, Criterion};

fn nfp_benchmark(c: &mut Criterion) {
    let complex_a = load_polygon("shirts.json");
    let complex_b = load_polygon("trousers.json");
    
    c.bench_function("nfp_100_vertices", |b| {
        b.iter(|| compute_nfp(&complex_a, &complex_b, 0.0))
    });
}
```

**목표**: 100-vertex 폴리곤 쌍에 대해 NFP 계산 < 50ms

### 완료 기준
- [ ] ESICUP 간단 인스턴스에서 유효한 배치 생성
- [ ] NFP 정확성 시각적 검증 (rerun.io 또는 SVG 출력)
- [ ] Benchmark baseline 확립

---

## Phase 3: Optimization Algorithms (5-6주)

### 목표
Genetic Algorithm 및 Simulated Annealing 최적화 엔진 구현

### 태스크

#### 3.1 GA Framework Core (1주)
- [ ] `Chromosome` trait 정의
- [ ] `Population` 관리
- [ ] Selection operators: Tournament, Roulette
- [ ] Generic evolution loop

```rust
pub trait Chromosome: Clone + Send + Sync {
    type Fitness: Ord;
    fn fitness(&self) -> Self::Fitness;
    fn crossover(&self, other: &Self) -> Self;
    fn mutate(&mut self, rate: f64);
}
```

#### 3.2 2D Nesting GA (2주)
- [ ] **Permutation Encoding**: 배치 순서 인코딩
- [ ] **Rotation Encoding**: 회전 각도 벡터
- [ ] **Order Crossover (OX1)**: 순서 보존 교차
- [ ] **PMX (Partially Mapped Crossover)**
- [ ] **Mutation operators**: Swap, Invert, Rotate
- [ ] Fitness function: utilization + penalty

**참조**: [research-01.md §4.1.1](research/research-01.md) - Genetic Algorithms

#### 3.3 BRKGA 구현 (1주)
- [ ] Random-key encoding
- [ ] Biased crossover (elite parent preference)
- [ ] Decoder: random keys → placement sequence

**참조**: 
- [research-01.md §4.1.2](research/research-01.md) - BRKGA
- [Gonçalves & Resende (2013)](https://www.semanticscholar.org/paper/A-biased-random-key-genetic-algorithm-for-2D-and-Goncalves-Resende)

#### 3.4 3D Bin Packing GA (1주)
- [ ] Box orientation encoding (6가지 회전)
- [ ] Extreme Point 기반 decoder
- [ ] Stability constraint 통합

#### 3.5 Simulated Annealing (1주)
- [ ] Cooling schedule: Geometric, Adaptive
- [ ] Neighborhood operators: Relocate, Swap, Chain
- [ ] Acceptance probability: exp(-ΔE/T)
- [ ] Reheating 전략

**참조**: [research-02.md](research/research-02.md) - Simulated annealing

#### 3.6 Local Search / Hill Climbing (0.5주)
- [ ] First-improvement 전략
- [ ] Best-improvement 전략
- [ ] Variable Neighborhood Search (VNS) 기초

### 테스트 전략

```rust
// tests/ga_tests.rs
#[test]
fn test_ga_convergence() {
    let problem = load_benchmark("jakobs1");
    let config = GaConfig {
        population_size: 100,
        generations: 500,
        crossover_rate: 0.85,
        mutation_rate: 0.05,
    };
    
    let result = GeneticNester::new(config).solve(&problem);
    
    // Should achieve at least 75% of known optimal
    let known_optimal = 0.85;
    assert!(result.utilization >= known_optimal * 0.75);
}

#[test]
fn test_ga_improvement_over_generations() {
    let problem = load_benchmark("shapes0");
    let mut fitness_history = vec![];
    
    let result = GeneticNester::new(default_config())
        .with_callback(|gen, best| fitness_history.push(best))
        .solve(&problem);
    
    // Fitness should generally improve
    let improvements = fitness_history.windows(10)
        .filter(|w| w.last() > w.first())
        .count();
    assert!(improvements > fitness_history.len() / 20);
}
```

### 완료 기준
- [ ] ESICUP 벤치마크 인스턴스에서 >80% utilization
- [ ] GA 수렴 그래프 생성
- [ ] SA vs GA 비교 분석

---

## Phase 4: Performance Optimization (3-4주)

### 목표
병렬화 및 메모리 최적화를 통한 성능 향상

### 태스크

#### 4.1 NFP 병렬 계산 (1주)
- [ ] `rayon::par_iter()` 적용
- [ ] Piece pair parallel computation
- [ ] Work stealing 최적화

```rust
// 병렬 NFP 계산
let all_nfps: Vec<_> = piece_pairs
    .par_iter()
    .map(|(a, b, rotation)| {
        let key = (a.id, b.id, rotation.to_key());
        cache.get_or_compute(key, || compute_nfp(a, b, *rotation))
    })
    .collect();
```

**참조**: [research-01.md §6.1](research/research-01.md) - Parallel Computation Strategies

#### 4.2 GA Population 병렬 평가 (0.5주)
- [ ] Fitness 평가 병렬화
- [ ] Island Model GA 구현 (선택적)

#### 4.3 Spatial Indexing (1주)
- [ ] `rstar` R*-tree 통합 (2D)
- [ ] `parry3d` BVH 활용 (3D)
- [ ] Broad-phase collision culling

```rust
use rstar::RTree;

struct SpatialIndex {
    tree: RTree<PlacedPiece>,
}

impl SpatialIndex {
    fn query_potential_collisions(&self, piece: &Piece) -> Vec<&PlacedPiece> {
        self.tree.locate_in_envelope(&piece.aabb())
            .collect()
    }
}
```

#### 4.4 Memory Optimization (1주)
- [ ] Arena allocation (`bumpalo`) for temporary polygons
- [ ] Geometry instancing (shared vertex data)
- [ ] Zero-copy deserialization (`rkyv`) 평가

**참조**: [research-01.md §6.2](research/research-01.md) - Data Structures and Caching

#### 4.5 SIMD Optimization (선택적, 0.5주)
- [ ] `simba` 기반 벡터 연산
- [ ] Batch point-in-polygon tests

### Benchmark Suite

```rust
// benches/performance_bench.rs
fn benchmark_full_nesting(c: &mut Criterion) {
    let problems = ["albano", "blaz1", "shapes0"];
    
    for name in problems {
        let problem = load_benchmark(name);
        
        c.bench_function(&format!("nest_{}", name), |b| {
            b.iter(|| {
                Nester2D::new(default_config()).solve(&problem)
            })
        });
    }
}

fn benchmark_parallel_scaling(c: &mut Criterion) {
    let problem = load_benchmark("albano");
    
    for threads in [1, 2, 4, 8] {
        c.bench_function(&format!("nest_threads_{}", threads), |b| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .unwrap();
            
            b.iter(|| Nester2D::new(default_config()).solve(&problem))
        });
    }
}
```

### 성능 목표

| 인스턴스 | 목표 시간 | 목표 Utilization |
|----------|-----------|------------------|
| Albano (24 pieces) | < 10s | > 85% |
| Shapes0 (43 pieces) | < 30s | > 80% |
| Shirts (99 pieces) | < 60s | > 78% |

### 완료 기준
- [ ] 4코어 기준 2-3x 병렬 speedup
- [ ] 메모리 사용량 프로파일링 완료
- [ ] 성능 regression 테스트 CI 통합

---

## Phase 5: FFI & Integration API (3-4주)

### 목표
C#/Python 소비자를 위한 안정적인 FFI 인터페이스

### 태스크

#### 5.1 C ABI 설계 (1주)
- [ ] `#[no_mangle] extern "C"` 함수 정의
- [ ] `cbindgen`으로 헤더 생성
- [ ] `repr(C)` 구조체 정의
- [ ] Error handling (return codes + message buffer)

```rust
// src/ffi/c_api.rs
#[no_mangle]
pub extern "C" fn unesting_solve(
    request_json: *const c_char,
    result_ptr: *mut *mut c_char,
) -> i32 {
    // ...
}

#[no_mangle]
pub extern "C" fn unesting_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe { drop(CString::from_raw(ptr)); }
    }
}

#[no_mangle]
pub extern "C" fn unesting_cancel(handle: *mut SolverHandle) -> i32 {
    // Cancellation support
}
```

#### 5.2 JSON API 설계 (1주)
- [ ] Request/Response 스키마 정의 (JSON Schema)
- [ ] Serde serialization 구현
- [ ] Validation 레이어
- [ ] Version 필드 추가

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "version": { "type": "string", "const": "1.0" },
    "mode": { "enum": ["2d", "3d"] },
    "geometries": { "type": "array" },
    "boundary": { "type": "object" },
    "config": { "type": "object" }
  },
  "required": ["mode", "geometries", "boundary"]
}
```

#### 5.3 Progress Callback (0.5주)
- [ ] Callback function pointer 지원
- [ ] Progress 정보: generation, utilization, time
- [ ] Cancellation token

```rust
type ProgressCallback = extern "C" fn(
    generation: u32,
    utilization: f64,
    time_ms: u64,
    user_data: *mut c_void
);

#[no_mangle]
pub extern "C" fn unesting_solve_with_progress(
    request: *const c_char,
    callback: ProgressCallback,
    user_data: *mut c_void,
    result: *mut *mut c_char,
) -> i32;
```

#### 5.4 Python Bindings (1주)
- [ ] `PyO3` 기반 바인딩
- [ ] `maturin` 빌드 설정
- [ ] Type stubs (`.pyi`) 생성
- [ ] PyPI 배포 준비

```python
# u_nesting/__init__.pyi
from typing import List, Optional

class Geometry2D:
    def __init__(self, id: str) -> None: ...
    def with_polygon(self, vertices: List[tuple[float, float]]) -> Geometry2D: ...
    def with_quantity(self, n: int) -> Geometry2D: ...

class Nester2D:
    def __init__(self, config: Optional[Config2D] = None) -> None: ...
    def solve(self, geometries: List[Geometry2D], boundary: Boundary2D) -> SolveResult: ...
```

#### 5.5 C# Integration Example (0.5주)
- [ ] P/Invoke wrapper 예제
- [ ] NuGet 패키지 구조
- [ ] 사용 예제 문서

```csharp
public static class UNesting
{
    [DllImport("u_nesting", CallingConvention = CallingConvention.Cdecl)]
    private static extern int unesting_solve(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string request,
        out IntPtr result
    );
    
    public static NestingResult Solve(NestingRequest request)
    {
        var json = JsonSerializer.Serialize(request);
        var code = unesting_solve(json, out var resultPtr);
        // ...
    }
}
```

### 테스트 전략

```rust
// tests/ffi_tests.rs
#[test]
fn test_json_roundtrip() {
    let request = r#"{
        "mode": "2d",
        "geometries": [{"id": "G1", "polygon": [[0,0],[10,0],[10,10],[0,10]], "quantity": 3}],
        "boundary": {"width": 100, "height": 50},
        "config": {"spacing": 1.0}
    }"#;
    
    let result = unesting_solve_json(request).unwrap();
    assert!(result.utilization > 0.0);
    assert_eq!(result.unplaced.len(), 0);
}

#[test]
fn test_invalid_json_error() {
    let result = unesting_solve_json("{ invalid json }");
    assert!(result.is_err());
}
```

### 완료 기준
- [ ] C# 테스트 프로젝트 통과
- [ ] Python pytest 통과
- [ ] API 문서 완성

---

## Phase 6: Benchmark & Release (2-3주)

### 목표
표준 벤치마크 검증 및 릴리스 준비

### 태스크

#### 6.1 ESICUP Benchmark Suite (1주)
- [ ] 데이터셋 파서 구현
- [ ] Benchmark runner 구축
- [ ] 결과 기록 시스템

**데이터셋** ([ESICUP](https://oscar-oliveira.github.io/2D-Cutting-and-Packing/pages/datset.htm)):
- ALBANO, BLAZ1-3, DIGHE1-2
- FU, JAKOBS1-2, MARQUES
- POLY1-5, SHAPES, SHIRTS, SWIM, TROUSERS

**참조**: [research-01.md §10.3](research/research-01.md) - Comparison Methodology

#### 6.2 3D Benchmark (0.5주)
- [ ] Martello et al. (2000) 데이터셋
- [ ] BPPLIB 1D 인스턴스 (검증용)

#### 6.3 결과 분석 및 리포트 (0.5주)
- [ ] 기존 솔버(SVGnest, libnest2d) 대비 비교
- [ ] 성능 그래프 생성
- [ ] 품질 지표 문서화

| 메트릭 | 정의 | 목표 |
|--------|------|------|
| Utilization | Σ item_area / container_area | > 85% (2D) |
| Gap to BKS | (result - best_known) / best_known | < 5% |
| Runtime | 시간 대비 품질 도달 | < 60s for 100 pieces |

#### 6.4 문서화 (0.5주)
- [ ] API 문서 (`cargo doc`)
- [ ] 사용자 가이드 (README 확장)
- [ ] 알고리즘 해설 문서
- [ ] 예제 코드

#### 6.5 릴리스 준비 (0.5주)
- [ ] CHANGELOG 작성
- [ ] 버전 태깅 (SemVer)
- [ ] crates.io 배포
- [ ] GitHub Release

### CI/CD 파이프라인

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --all-features
      
  benchmark:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - run: cargo bench
      - uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'cargo'
          output-file-path: target/criterion/output.json

  release:
    needs: [test, benchmark]
    if: startsWith(github.ref, 'refs/tags/')
    steps:
      - run: cargo publish
```

### 완료 기준
- [ ] ESICUP 10개 인스턴스 결과 문서화
- [ ] SVGnest 대비 동등 이상 품질
- [ ] crates.io 배포 완료

---

## 테스트 전략 종합

### 테스트 레벨

| 레벨 | 범위 | 도구 |
|------|------|------|
| Unit | 개별 함수/메서드 | `#[test]` |
| Property | 불변식 검증 | `proptest` |
| Integration | 모듈 간 연동 | `tests/` 디렉토리 |
| Benchmark | 성능 측정 | `criterion` |
| E2E | 전체 파이프라인 | JSON API 테스트 |

### Property-Based Testing

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn nfp_point_classification(
        polygon_a in arbitrary_polygon(),
        polygon_b in arbitrary_polygon(),
        test_point in arbitrary_point()
    ) {
        let nfp = compute_nfp(&polygon_a, &polygon_b, 0.0).unwrap();
        let translated_b = polygon_b.translate(test_point);
        
        let overlaps = polygons_overlap(&polygon_a, &translated_b);
        let inside_nfp = nfp.contains(&test_point);
        
        // Point inside NFP ⟺ shapes overlap
        prop_assert_eq!(inside_nfp, overlaps);
    }
}
```

### Regression Testing

```rust
// tests/regression.rs
#[test]
fn regression_jakobs1_utilization() {
    let result = solve_benchmark("jakobs1", default_config());
    
    // Must not regress below baseline
    const BASELINE_UTILIZATION: f64 = 0.82;
    assert!(
        result.utilization >= BASELINE_UTILIZATION,
        "Regression: {} < {} baseline",
        result.utilization, BASELINE_UTILIZATION
    );
}
```

---

## 리스크 및 완화 전략

| 리스크 | 영향 | 확률 | 완화 전략 |
|--------|------|------|-----------|
| NFP 수치 불안정 | High | Medium | `robust` crate 사용, 정수 좌표 스케일링 |
| GA 수렴 부족 | Medium | Medium | Adaptive parameter tuning, Island model |
| 3D 성능 병목 | Medium | High | BVH 최적화, LOD 적용 |
| FFI 메모리 누수 | High | Low | Valgrind/Miri 테스트, RAII 패턴 |

---

## 참조 링크 종합

### 핵심 논문
1. [Burke et al. (2007) - Complete NFP Generation](https://www.graham-kendall.com/papers/bhkw2007.pdf)
2. [Bennell & Oliveira (2008) - Nesting Tutorial](https://eprints.soton.ac.uk/154797/)
3. [Gonçalves & Resende (2013) - BRKGA](https://www.semanticscholar.org/paper/A-biased-random-key-genetic-algorithm-for-2D-and-Goncalves-Resende)

### Rust 생태계
4. [geo crate](https://docs.rs/geo)
5. [i_overlay](https://github.com/iShape-Rust/iOverlay)
6. [parry](https://parry.rs/docs/)
7. [rstar](https://docs.rs/rstar)

### 벤치마크
8. [ESICUP Datasets](https://oscar-oliveira.github.io/2D-Cutting-and-Packing/pages/datset.htm)
9. [BPPLIB](https://site.unibo.it/operations-research/en/research/bpplib-a-bin-packing-problem-library)

### 기존 구현
10. [SVGnest](https://github.com/Jack000/SVGnest)
11. [libnest2d](https://github.com/tamasmeszaros/libnest2d)
12. [OR-Tools](https://developers.google.com/optimization)

---

이 로드맵은 리서치 문서의 권장사항을 기반으로 구성되었으며, 각 Phase는 이전 단계의 완료에 의존합니다. 필요에 따라 Phase 간 병렬 진행이 가능한 태스크도 있습니다.