//! # U-Nesting 3D
//!
//! 3D bin packing algorithms for the U-Nesting spatial optimization engine.
//!
//! This crate provides box and mesh-based 3D packing with collision detection
//! and various placement algorithms.

pub mod boundary;
pub mod geometry;
pub mod packer;

// Re-exports
pub use boundary::Boundary3D;
pub use geometry::Geometry3D;
pub use packer::Packer3D;
pub use u_nesting_core::{Config, Error, Placement, Result, SolveResult, Strategy};
