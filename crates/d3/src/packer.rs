//! 3D bin packing solver.

use crate::boundary::Boundary3D;
use crate::brkga_packing::run_brkga_packing;
use crate::ga_packing::run_ga_packing;
use crate::geometry::Geometry3D;
use crate::sa_packing::run_sa_packing;
use u_nesting_core::brkga::BrkgaConfig;
use u_nesting_core::ga::GaConfig;
use u_nesting_core::geometry::{Boundary, Geometry};
use u_nesting_core::sa::SaConfig;
use u_nesting_core::solver::{Config, ProgressCallback, Solver, Strategy};
use u_nesting_core::{Placement, Result, SolveResult};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// 3D bin packing solver.
pub struct Packer3D {
    config: Config,
    cancelled: Arc<AtomicBool>,
}

impl Packer3D {
    /// Creates a new packer with the given configuration.
    pub fn new(config: Config) -> Self {
        Self {
            config,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Creates a packer with default configuration.
    pub fn default_config() -> Self {
        Self::new(Config::default())
    }

    /// Simple layer-based packing algorithm.
    fn layer_packing(
        &self,
        geometries: &[Geometry3D],
        boundary: &Boundary3D,
    ) -> Result<SolveResult<f64>> {
        let start = Instant::now();
        let mut result = SolveResult::new();
        let mut placements = Vec::new();

        let margin = self.config.margin;
        let spacing = self.config.spacing;

        let bound_max_x = boundary.width() - margin;
        let bound_max_y = boundary.depth() - margin;
        let bound_max_z = boundary.height() - margin;

        // Simple layer-based placement
        let mut current_x = margin;
        let mut current_y = margin;
        let mut current_z = margin;
        let mut row_depth = 0.0_f64;
        let mut layer_height = 0.0_f64;

        let mut total_placed_volume = 0.0;
        let mut total_placed_mass = 0.0;

        for geom in geometries {
            geom.validate()?;

            for instance in 0..geom.quantity() {
                if self.cancelled.load(Ordering::Relaxed) {
                    result.computation_time_ms = start.elapsed().as_millis() as u64;
                    return Ok(result);
                }

                // Use first allowed orientation (could optimize later)
                let dims = geom.dimensions_for_orientation(0);
                let g_width = dims.x;
                let g_depth = dims.y;
                let g_height = dims.z;

                // Check mass constraint
                if let (Some(max_mass), Some(item_mass)) = (boundary.max_mass(), geom.mass()) {
                    if total_placed_mass + item_mass > max_mass {
                        result.unplaced.push(geom.id().clone());
                        continue;
                    }
                }

                // Try to fit in current row
                if current_x + g_width > bound_max_x {
                    // Move to next row in current layer
                    current_x = margin;
                    current_y += row_depth + spacing;
                    row_depth = 0.0;
                }

                // Check if fits in current layer (y direction)
                if current_y + g_depth > bound_max_y {
                    // Move to next layer
                    current_x = margin;
                    current_y = margin;
                    current_z += layer_height + spacing;
                    row_depth = 0.0;
                    layer_height = 0.0;
                }

                // Check if fits in container height
                if current_z + g_height > bound_max_z {
                    result.unplaced.push(geom.id().clone());
                    continue;
                }

                // Place the item
                let placement = Placement::new_3d(
                    geom.id().clone(),
                    instance,
                    current_x,
                    current_y,
                    current_z,
                    0.0,
                    0.0,
                    0.0, // No rotation for simple placement
                );

                placements.push(placement);
                total_placed_volume += geom.measure();
                if let Some(mass) = geom.mass() {
                    total_placed_mass += mass;
                }

                // Update position for next item
                current_x += g_width + spacing;
                row_depth = row_depth.max(g_depth);
                layer_height = layer_height.max(g_height);
            }
        }

        result.placements = placements;
        result.boundaries_used = 1;
        result.utilization = total_placed_volume / boundary.measure();
        result.computation_time_ms = start.elapsed().as_millis() as u64;

        Ok(result)
    }

    /// Genetic Algorithm based packing optimization.
    ///
    /// Uses GA to optimize placement order and orientations, with layer-based
    /// decoding for collision-free placements.
    fn genetic_algorithm(
        &self,
        geometries: &[Geometry3D],
        boundary: &Boundary3D,
    ) -> Result<SolveResult<f64>> {
        // Configure GA with reasonable defaults
        let ga_config = GaConfig::default()
            .with_population_size(50)
            .with_max_generations(100)
            .with_crossover_rate(0.85)
            .with_mutation_rate(0.15);

        let result = run_ga_packing(
            geometries,
            boundary,
            &self.config,
            ga_config,
            self.cancelled.clone(),
        );

        Ok(result)
    }

    /// BRKGA (Biased Random-Key Genetic Algorithm) based packing optimization.
    ///
    /// Uses random-key encoding and biased crossover for robust optimization.
    fn brkga(
        &self,
        geometries: &[Geometry3D],
        boundary: &Boundary3D,
    ) -> Result<SolveResult<f64>> {
        // Configure BRKGA with reasonable defaults
        let brkga_config = BrkgaConfig::default()
            .with_population_size(50)
            .with_max_generations(100)
            .with_elite_fraction(0.2)
            .with_mutant_fraction(0.15)
            .with_elite_bias(0.7);

        let result = run_brkga_packing(
            geometries,
            boundary,
            &self.config,
            brkga_config,
            self.cancelled.clone(),
        );

        Ok(result)
    }

    /// Simulated Annealing based packing optimization.
    ///
    /// Uses neighborhood operators to explore solution space with temperature-based
    /// acceptance probability.
    fn simulated_annealing(
        &self,
        geometries: &[Geometry3D],
        boundary: &Boundary3D,
    ) -> Result<SolveResult<f64>> {
        // Configure SA with reasonable defaults
        let sa_config = SaConfig::default()
            .with_initial_temp(100.0)
            .with_final_temp(0.1)
            .with_cooling_rate(0.95)
            .with_iterations_per_temp(50)
            .with_max_iterations(10000);

        let result = run_sa_packing(
            geometries,
            boundary,
            &self.config,
            sa_config,
            self.cancelled.clone(),
        );

        Ok(result)
    }
}

impl Solver for Packer3D {
    type Geometry = Geometry3D;
    type Boundary = Boundary3D;
    type Scalar = f64;

    fn solve(
        &self,
        geometries: &[Self::Geometry],
        boundary: &Self::Boundary,
    ) -> Result<SolveResult<f64>> {
        boundary.validate()?;

        // Reset cancellation flag
        self.cancelled.store(false, Ordering::Relaxed);

        match self.config.strategy {
            Strategy::ExtremePoint | Strategy::BottomLeftFill => {
                self.layer_packing(geometries, boundary)
            }
            Strategy::GeneticAlgorithm => self.genetic_algorithm(geometries, boundary),
            Strategy::Brkga => self.brkga(geometries, boundary),
            Strategy::SimulatedAnnealing => self.simulated_annealing(geometries, boundary),
            _ => {
                // Fall back to layer packing for unimplemented strategies
                log::warn!(
                    "Strategy {:?} not yet implemented, using layer packing",
                    self.config.strategy
                );
                self.layer_packing(geometries, boundary)
            }
        }
    }

    fn solve_with_progress(
        &self,
        geometries: &[Self::Geometry],
        boundary: &Self::Boundary,
        _callback: ProgressCallback,
    ) -> Result<SolveResult<f64>> {
        // TODO: Implement progress reporting
        self.solve(geometries, boundary)
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_packing() {
        let geometries = vec![
            Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(3),
            Geometry3D::new("B2", 15.0, 15.0, 15.0).with_quantity(2),
        ];

        let boundary = Boundary3D::new(100.0, 80.0, 50.0);
        let packer = Packer3D::default_config();

        let result = packer.solve(&geometries, &boundary).unwrap();

        assert!(result.utilization > 0.0);
        assert!(result.placements.len() <= 5);
    }

    #[test]
    fn test_mass_constraint() {
        let geometries = vec![Geometry3D::new("B1", 20.0, 20.0, 20.0)
            .with_quantity(10)
            .with_mass(100.0)];

        let boundary = Boundary3D::new(100.0, 80.0, 50.0).with_max_mass(350.0);

        let packer = Packer3D::default_config();
        let result = packer.solve(&geometries, &boundary).unwrap();

        // Should only place 3 boxes (300 mass) due to 350 mass limit
        assert!(result.placements.len() <= 3);
    }

    #[test]
    fn test_placement_within_bounds() {
        let geometries = vec![Geometry3D::new("B1", 10.0, 10.0, 10.0).with_quantity(4)];

        let boundary = Boundary3D::new(50.0, 50.0, 50.0);
        let config = Config::default().with_margin(5.0).with_spacing(2.0);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // All boxes should be placed
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());

        // Verify placements are within bounds (with margin)
        for p in &result.placements {
            assert!(p.position[0] >= 5.0);
            assert!(p.position[1] >= 5.0);
            assert!(p.position[2] >= 5.0);
        }
    }

    #[test]
    fn test_ga_strategy_basic() {
        let geometries = vec![
            Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(2),
            Geometry3D::new("B2", 15.0, 15.0, 15.0).with_quantity(2),
        ];

        let boundary = Boundary3D::new(100.0, 80.0, 50.0);
        let config = Config::default().with_strategy(Strategy::GeneticAlgorithm);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // GA should place items and achieve positive utilization
        assert!(result.utilization > 0.0);
        assert!(!result.placements.is_empty());
    }

    #[test]
    fn test_ga_strategy_all_placed() {
        // Small number of boxes that should all fit
        let geometries = vec![Geometry3D::new("B1", 10.0, 10.0, 10.0).with_quantity(4)];

        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let config = Config::default().with_strategy(Strategy::GeneticAlgorithm);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // All 4 boxes should be placed
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());
    }

    #[test]
    fn test_ga_strategy_with_orientations() {
        use crate::geometry::OrientationConstraint;

        // Box that fits better when rotated
        let geometries = vec![
            Geometry3D::new("B1", 50.0, 10.0, 10.0)
                .with_quantity(2)
                .with_orientation(OrientationConstraint::Any),
        ];

        // Container where orientation matters
        let boundary = Boundary3D::new(60.0, 60.0, 60.0);
        let config = Config::default().with_strategy(Strategy::GeneticAlgorithm);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // GA should find a way to place both boxes
        assert_eq!(result.placements.len(), 2);
    }

    #[test]
    fn test_brkga_strategy_basic() {
        let geometries = vec![
            Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(2),
            Geometry3D::new("B2", 15.0, 15.0, 15.0).with_quantity(2),
        ];

        let boundary = Boundary3D::new(100.0, 80.0, 50.0);
        let config = Config::default().with_strategy(Strategy::Brkga);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // BRKGA should place items and achieve positive utilization
        assert!(result.utilization > 0.0);
        assert!(!result.placements.is_empty());
        assert_eq!(result.strategy, Some("BRKGA".to_string()));
    }

    #[test]
    fn test_brkga_strategy_all_placed() {
        // Small number of boxes that should all fit
        let geometries = vec![Geometry3D::new("B1", 10.0, 10.0, 10.0).with_quantity(4)];

        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let config = Config::default().with_strategy(Strategy::Brkga);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // All 4 boxes should be placed
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());
    }
}
