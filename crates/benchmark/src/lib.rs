//! ESICUP Benchmark Suite for U-Nesting
//!
//! This crate provides:
//! - ESICUP dataset parser for 2D irregular nesting problems
//! - Benchmark runner with multiple strategies
//! - Result recording and comparison

mod dataset;
mod parser;
mod result;
mod runner;

pub use dataset::{Dataset, DatasetInfo, Item, Shape};
pub use parser::DatasetParser;
pub use result::{BenchmarkResult, RunResult};
pub use runner::{BenchmarkConfig, BenchmarkRunner};
