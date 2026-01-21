//! # U-Nesting Core
//!
//! Core traits and abstractions for the U-Nesting spatial optimization engine.
//!
//! This crate provides the foundational types and traits that are shared between
//! the 2D nesting and 3D bin packing modules.
//!
//! ## Core Components
//!
//! - **Geometry traits**: `Geometry`, `Geometry2DExt`, `Geometry3DExt`
//! - **Boundary traits**: `Boundary`, `Boundary2DExt`, `Boundary3DExt`
//! - **Solver trait**: Common interface for all optimization algorithms
//! - **GA framework**: Genetic algorithm infrastructure for metaheuristic optimization
//! - **Transform types**: 2D/3D coordinate transformations and AABBs
//!
//! ## Feature Flags
//!
//! - `serde`: Enable serialization/deserialization support

pub mod brkga;
pub mod error;
pub mod ga;
pub mod geometry;
pub mod placement;
pub mod result;
pub mod sa;
pub mod solver;
pub mod transform;

// Re-exports
pub use brkga::{BrkgaConfig, BrkgaProblem, BrkgaResult, BrkgaRunner, RandomKeyChromosome};
pub use error::{Error, Result};
pub use ga::{GaConfig, GaProblem, GaResult, GaRunner, Individual, PermutationChromosome};
pub use geometry::{
    Boundary, Boundary2DExt, Boundary3DExt, Geometry, Geometry2DExt, Geometry3DExt, GeometryId,
    Orientation3D, RotationConstraint,
};
pub use placement::Placement;
pub use result::{SolveResult, SolveSummary};
pub use sa::{
    CoolingSchedule, NeighborhoodOperator, PermutationSolution, SaConfig, SaProblem, SaResult,
    SaRunner, SaSolution,
};
pub use solver::{Config, ProgressCallback, ProgressInfo, Solver, Strategy};
pub use transform::{Transform2D, Transform3D, AABB2D, AABB3D};
