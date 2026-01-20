//! Benchmarks for NFP computation.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use u_nesting_d2::Geometry2D;

fn nfp_benchmark(c: &mut Criterion) {
    let rect_a = Geometry2D::rectangle("A", 100.0, 50.0);
    let rect_b = Geometry2D::rectangle("B", 30.0, 20.0);

    c.bench_function("nfp_simple_rects", |b| {
        b.iter(|| {
            // Placeholder until NFP is implemented
            black_box((&rect_a, &rect_b));
        })
    });
}

criterion_group!(benches, nfp_benchmark);
criterion_main!(benches);
