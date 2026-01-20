//! Benchmarks for 3D bin packing.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use u_nesting_core::solver::Solver;
use u_nesting_d3::{Boundary3D, Geometry3D, Packer3D};

fn packer_benchmark(c: &mut Criterion) {
    let geometries: Vec<Geometry3D> = (0..20)
        .map(|i| Geometry3D::new(format!("B{}", i), 10.0, 10.0, 10.0))
        .collect();

    let boundary = Boundary3D::new(100.0, 100.0, 100.0);
    let packer = Packer3D::default_config();

    c.bench_function("pack_20_uniform_boxes", |b| {
        b.iter(|| {
            let result = packer.solve(black_box(&geometries), black_box(&boundary));
            black_box(result)
        })
    });
}

criterion_group!(benches, packer_benchmark);
criterion_main!(benches);
