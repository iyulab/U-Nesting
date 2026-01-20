//! Error types for U-Nesting.

use thiserror::Error;

/// Result type alias for U-Nesting operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during nesting/packing operations.
#[derive(Debug, Error)]
pub enum Error {
    /// Invalid geometry provided.
    #[error("Invalid geometry: {0}")]
    InvalidGeometry(String),

    /// Invalid boundary provided.
    #[error("Invalid boundary: {0}")]
    InvalidBoundary(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// NFP computation failed.
    #[error("NFP computation failed: {0}")]
    NfpError(String),

    /// No valid placement found.
    #[error("No valid placement found for geometry: {0}")]
    NoPlacement(String),

    /// Computation cancelled.
    #[error("Computation cancelled")]
    Cancelled,

    /// Timeout exceeded.
    #[error("Timeout exceeded after {0}ms")]
    Timeout(u64),

    /// Serialization error.
    #[cfg(feature = "serde")]
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),
}
