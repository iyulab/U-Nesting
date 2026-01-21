//! # U-Nesting FFI
//!
//! C FFI interface for the U-Nesting spatial optimization engine.
//!
//! This crate provides a C-compatible interface for using U-Nesting from
//! other languages like C#, Python, etc.
//!
//! ## Functions
//!
//! - [`unesting_solve`] - Auto-detects 2D/3D mode and solves
//! - [`unesting_solve_2d`] - Solves 2D nesting problems
//! - [`unesting_solve_3d`] - Solves 3D bin packing problems
//! - [`unesting_free_string`] - Frees result strings
//! - [`unesting_version`] - Returns API version
//!
//! ## Error Codes
//!
//! | Code | Constant | Meaning |
//! |------|----------|---------|
//! | 0 | `UNESTING_OK` | Success |
//! | -1 | `UNESTING_ERR_NULL_PTR` | Null pointer passed |
//! | -2 | `UNESTING_ERR_INVALID_JSON` | Invalid JSON input |
//! | -3 | `UNESTING_ERR_SOLVE_FAILED` | Solver failed |
//! | -99 | `UNESTING_ERR_UNKNOWN` | Unknown error |
//!
//! ## JSON Request Format (2D)
//!
//! ```json
//! {
//!   "mode": "2d",
//!   "geometries": [
//!     {
//!       "id": "part1",
//!       "polygon": [[0,0], [100,0], [100,50], [0,50]],
//!       "quantity": 5,
//!       "rotations": [0, 90, 180, 270],
//!       "allow_flip": false
//!     }
//!   ],
//!   "boundary": {
//!     "width": 1000,
//!     "height": 500
//!   },
//!   "config": {
//!     "strategy": "nfp",
//!     "spacing": 2.0,
//!     "margin": 5.0,
//!     "time_limit_ms": 30000
//!   }
//! }
//! ```
//!
//! ## JSON Request Format (3D)
//!
//! ```json
//! {
//!   "mode": "3d",
//!   "geometries": [
//!     {
//!       "id": "box1",
//!       "dimensions": [100, 50, 30],
//!       "quantity": 10,
//!       "mass": 2.5
//!     }
//!   ],
//!   "boundary": {
//!     "dimensions": [500, 400, 300],
//!     "max_mass": 100.0,
//!     "gravity": true,
//!     "stability": true
//!   },
//!   "config": {
//!     "strategy": "ep",
//!     "time_limit_ms": 30000
//!   }
//! }
//! ```
//!
//! ## Strategy Options
//!
//! | Strategy | 2D | 3D | Description |
//! |----------|----|----|-------------|
//! | `blf` | ✓ | ✓ | Bottom-Left Fill (fast) |
//! | `nfp` | ✓ | - | NFP-guided placement |
//! | `ga` | ✓ | ✓ | Genetic Algorithm |
//! | `brkga` | ✓ | ✓ | Biased Random-Key GA |
//! | `sa` | ✓ | ✓ | Simulated Annealing |
//! | `ep` | - | ✓ | Extreme Point heuristic |
//!
//! ## C# Example
//!
//! ```csharp
//! [DllImport("u_nesting_ffi")]
//! static extern int unesting_solve_2d(string json, out IntPtr result);
//!
//! [DllImport("u_nesting_ffi")]
//! static extern void unesting_free_string(IntPtr ptr);
//!
//! string json = "{\"geometries\": [...], \"boundary\": {...}}";
//! IntPtr resultPtr;
//! int code = unesting_solve_2d(json, out resultPtr);
//! string result = Marshal.PtrToStringAnsi(resultPtr);
//! unesting_free_string(resultPtr);
//! ```

mod api;
mod types;

pub use api::*;
pub use types::*;
