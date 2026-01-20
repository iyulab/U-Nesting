//! Solver traits and configuration.

use crate::geometry::{Boundary, Geometry};
use crate::result::SolveResult;
use crate::Result;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Optimization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Strategy {
    /// Bottom-Left Fill (fast, lower quality).
    #[default]
    BottomLeftFill,
    /// NFP-guided placement (balanced).
    NfpGuided,
    /// Genetic Algorithm (slower, higher quality).
    GeneticAlgorithm,
    /// Biased Random-Key Genetic Algorithm (balanced, robust).
    Brkga,
    /// Simulated Annealing.
    SimulatedAnnealing,
    /// Extreme Point heuristic (3D only).
    ExtremePoint,
}

/// Common configuration for solvers.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Config {
    /// Optimization strategy.
    pub strategy: Strategy,

    /// Minimum spacing between geometries.
    pub spacing: f64,

    /// Margin from boundary edges.
    pub margin: f64,

    /// Maximum computation time in milliseconds (0 = unlimited).
    pub time_limit_ms: u64,

    /// Target utilization (0.0 - 1.0). Solver stops if reached.
    pub target_utilization: Option<f64>,

    /// Number of threads to use (0 = auto).
    pub threads: usize,

    // GA-specific parameters
    /// Population size for GA.
    pub population_size: usize,

    /// Number of generations for GA.
    pub max_generations: u32,

    /// Crossover rate for GA (0.0 - 1.0).
    pub crossover_rate: f64,

    /// Mutation rate for GA (0.0 - 1.0).
    pub mutation_rate: f64,

    /// Elite count for GA.
    pub elite_count: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            strategy: Strategy::default(),
            spacing: 0.0,
            margin: 0.0,
            time_limit_ms: 30000,
            target_utilization: None,
            threads: 0,
            population_size: 100,
            max_generations: 500,
            crossover_rate: 0.85,
            mutation_rate: 0.05,
            elite_count: 5,
        }
    }
}

impl Config {
    /// Creates a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the optimization strategy.
    pub fn with_strategy(mut self, strategy: Strategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Sets the spacing between geometries.
    pub fn with_spacing(mut self, spacing: f64) -> Self {
        self.spacing = spacing;
        self
    }

    /// Sets the margin from boundary edges.
    pub fn with_margin(mut self, margin: f64) -> Self {
        self.margin = margin;
        self
    }

    /// Sets the time limit in milliseconds.
    pub fn with_time_limit(mut self, ms: u64) -> Self {
        self.time_limit_ms = ms;
        self
    }

    /// Sets the target utilization.
    pub fn with_target_utilization(mut self, util: f64) -> Self {
        self.target_utilization = Some(util.clamp(0.0, 1.0));
        self
    }
}

/// Progress callback for long-running operations.
pub type ProgressCallback = Box<dyn Fn(ProgressInfo) + Send + Sync>;

/// Progress information during solving.
#[derive(Debug, Clone)]
pub struct ProgressInfo {
    /// Current generation (for GA).
    pub generation: u32,
    /// Current best utilization.
    pub utilization: f64,
    /// Elapsed time in milliseconds.
    pub elapsed_ms: u64,
    /// Whether the solver is still running.
    pub running: bool,
}

/// Trait for nesting/packing solvers.
pub trait Solver {
    /// The geometry type this solver handles.
    type Geometry: Geometry;
    /// The boundary type this solver handles.
    type Boundary: Boundary;
    /// The scalar type for coordinates.
    type Scalar;

    /// Solves the nesting/packing problem.
    fn solve(
        &self,
        geometries: &[Self::Geometry],
        boundary: &Self::Boundary,
    ) -> Result<SolveResult<Self::Scalar>>;

    /// Solves with a progress callback.
    fn solve_with_progress(
        &self,
        geometries: &[Self::Geometry],
        boundary: &Self::Boundary,
        callback: ProgressCallback,
    ) -> Result<SolveResult<Self::Scalar>>;

    /// Cancels an ongoing solve operation.
    fn cancel(&self);
}
