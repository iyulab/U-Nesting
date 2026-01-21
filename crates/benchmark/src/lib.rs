//! Benchmark Suite for U-Nesting
//!
//! This crate provides:
//! - ESICUP dataset parser for 2D irregular nesting problems
//! - MPV (Martello-Pisinger-Vigo) instance generator for 3D bin packing
//! - Benchmark runner with multiple strategies
//! - Result recording and comparison

mod dataset;
mod dataset3d;
mod parser;
mod result;
mod runner;
mod runner3d;

// 2D exports
pub use dataset::{Dataset, DatasetInfo, Item, Shape};
pub use parser::DatasetParser;
pub use result::{BenchmarkResult, RunResult};
pub use runner::{BenchmarkConfig, BenchmarkRunner};

// 3D exports
pub use dataset3d::{Dataset3D, Dataset3DInfo, InstanceClass, InstanceGenerator, Item3D};
pub use runner3d::{BenchmarkConfig3D, BenchmarkRunner3D, BenchmarkSummary3D};
