//! # U-Nesting FFI
//!
//! C FFI interface for the U-Nesting spatial optimization engine.
//!
//! This crate provides a C-compatible interface for using U-Nesting from
//! other languages like C#, Python, etc.

mod api;
mod types;

pub use api::*;
pub use types::*;
