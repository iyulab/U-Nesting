//! # U-Nesting 3D
//!
//! 3D bin packing algorithms for the U-Nesting spatial optimization engine.
//!
//! This crate provides box and mesh-based 3D packing with collision detection
//! and various placement algorithms.
//!
//! ## Features
//!
//! - Box geometry with 6-orientation support
//! - Multiple placement strategies (Layer, GA)
//! - Mass and stacking constraints
//! - Configurable orientation constraints (Any, Upright, Fixed)

pub mod boundary;
pub mod ga_packing;
pub mod geometry;
pub mod packer;

// Re-exports
pub use boundary::Boundary3D;
pub use geometry::Geometry3D;
pub use packer::Packer3D;
pub use u_nesting_core::{Config, Error, Placement, Result, SolveResult, Strategy};
