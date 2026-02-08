//! # U-Nesting 3D
//!
//! 3D bin packing algorithms for the U-Nesting spatial optimization engine.
//!
//! This crate provides box-based 3D packing with collision detection
//! and various placement algorithms.
//!
//! ## Features
//!
//! - Box geometry with 6-orientation support
//! - Multiple placement strategies (Layer, GA, BRKGA, SA, Extreme Point)
//! - Mass and stacking constraints
//! - Configurable orientation constraints (Any, Upright, Fixed)
//! - Spatial indexing for fast collision queries
//!
//! ## Quick Start
//!
//! ```rust
//! use u_nesting_d3::{Geometry3D, Boundary3D, Packer3D, Config, Strategy, Solver};
//! use u_nesting_d3::geometry::OrientationConstraint;
//!
//! // Create boxes
//! let box1 = Geometry3D::new("box1", 100.0, 50.0, 30.0)
//!     .with_quantity(10)
//!     .with_orientation(OrientationConstraint::Upright);
//!
//! // Create container
//! let container = Boundary3D::new(500.0, 400.0, 300.0);
//!
//! // Configure and solve
//! let config = Config::new()
//!     .with_strategy(Strategy::ExtremePoint)
//!     .with_spacing(1.0);
//!
//! let packer = Packer3D::new(config);
//! let result = packer.solve(&[box1], &container).unwrap();
//!
//! println!("Placed {} boxes, utilization: {:.1}%",
//!     result.placements.len(),
//!     result.utilization * 100.0);
//! ```
//!
//! ## Orientation Constraints
//!
//! ```rust
//! use u_nesting_d3::{Geometry3D, geometry::OrientationConstraint};
//!
//! // Any orientation (6 rotations)
//! let any = Geometry3D::new("b1", 10.0, 20.0, 30.0)
//!     .with_orientation(OrientationConstraint::Any);
//!
//! // Upright only (2 rotations, height preserved)
//! let upright = Geometry3D::new("b2", 10.0, 20.0, 30.0)
//!     .with_orientation(OrientationConstraint::Upright);
//!
//! // Fixed (no rotation)
//! let fixed = Geometry3D::new("b3", 10.0, 20.0, 30.0)
//!     .with_orientation(OrientationConstraint::Fixed);
//! ```
//!
//! ## Mass Constraints
//!
//! ```rust
//! use u_nesting_d3::{Geometry3D, Boundary3D};
//!
//! let heavy_box = Geometry3D::new("heavy", 50.0, 50.0, 50.0)
//!     .with_mass(10.0)
//!     .with_quantity(5);
//!
//! let container = Boundary3D::new(200.0, 200.0, 200.0)
//!     .with_max_mass(100.0);
//! ```

pub mod boundary;
pub mod brkga_packing;
pub mod extreme_point;
pub mod ga_packing;
pub mod geometry;
pub mod packer;
pub mod packing_utils;
pub mod physics;
pub mod sa_packing;
pub mod spatial_index;
pub mod stability;

// Re-exports
pub use boundary::Boundary3D;
pub use geometry::Geometry3D;
pub use packer::Packer3D;
pub use physics::{PhysicsConfig, PhysicsResult, PhysicsSimulator};
pub use spatial_index::{Aabb3D, SpatialEntry3D, SpatialIndex3D};
pub use stability::{
    PlacedBox, StabilityAnalyzer, StabilityConstraint, StabilityReport, StabilityResult,
};
pub use u_nesting_core::{Config, Error, Placement, Result, SolveResult, Solver, Strategy};
