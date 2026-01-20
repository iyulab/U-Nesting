//! 2D nesting solver.

use crate::boundary::Boundary2D;
use crate::geometry::Geometry2D;
use u_nesting_core::geometry::{Boundary, Geometry};
use u_nesting_core::solver::{Config, ProgressCallback, Solver};
use u_nesting_core::{Placement, Result, SolveResult};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// 2D nesting solver.
pub struct Nester2D {
    config: Config,
    cancelled: Arc<AtomicBool>,
}

impl Nester2D {
    /// Creates a new nester with the given configuration.
    pub fn new(config: Config) -> Self {
        Self {
            config,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Creates a nester with default configuration.
    pub fn default_config() -> Self {
        Self::new(Config::default())
    }

    /// Bottom-Left Fill algorithm implementation.
    fn bottom_left_fill(
        &self,
        geometries: &[Geometry2D],
        boundary: &Boundary2D,
    ) -> Result<SolveResult<f64>> {
        let start = Instant::now();
        let mut result = SolveResult::new();
        let mut placements = Vec::new();

        // Get boundary dimensions
        let (b_min, b_max) = boundary.aabb();
        let margin = self.config.margin;
        let spacing = self.config.spacing;

        let bound_min_x = b_min[0] + margin;
        let bound_min_y = b_min[1] + margin;
        let bound_max_x = b_max[0] - margin;
        let bound_max_y = b_max[1] - margin;

        // Simple row-based placement for now
        let mut current_x = bound_min_x;
        let mut current_y = bound_min_y;
        let mut row_height = 0.0_f64;

        let mut total_placed_area = 0.0;

        for geom in geometries {
            geom.validate()?;

            for instance in 0..geom.quantity() {
                if self.cancelled.load(Ordering::Relaxed) {
                    result.computation_time_ms = start.elapsed().as_millis() as u64;
                    return Ok(result);
                }

                let (g_min, g_max) = geom.aabb();
                let g_width = g_max[0] - g_min[0];
                let g_height = g_max[1] - g_min[1];

                // Check if piece fits in remaining row space
                if current_x + g_width > bound_max_x {
                    // Move to next row
                    current_x = bound_min_x;
                    current_y += row_height + spacing;
                    row_height = 0.0;
                }

                // Check if piece fits in boundary height
                if current_y + g_height > bound_max_y {
                    // Can't place this piece
                    result.unplaced.push(geom.id().clone());
                    continue;
                }

                // Place the piece
                let placement = Placement::new_2d(
                    geom.id().clone(),
                    instance,
                    current_x - g_min[0],
                    current_y - g_min[1],
                    0.0,
                );

                placements.push(placement);
                total_placed_area += geom.measure();

                // Update position for next piece
                current_x += g_width + spacing;
                row_height = row_height.max(g_height);
            }
        }

        result.placements = placements;
        result.boundaries_used = 1;
        result.utilization = total_placed_area / boundary.measure();
        result.computation_time_ms = start.elapsed().as_millis() as u64;

        Ok(result)
    }
}

impl Solver for Nester2D {
    type Geometry = Geometry2D;
    type Boundary = Boundary2D;
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
            u_nesting_core::Strategy::BottomLeftFill => self.bottom_left_fill(geometries, boundary),
            _ => {
                // Fall back to BLF for unimplemented strategies
                log::warn!(
                    "Strategy {:?} not yet implemented, using BottomLeftFill",
                    self.config.strategy
                );
                self.bottom_left_fill(geometries, boundary)
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
    fn test_simple_nesting() {
        let geometries = vec![
            Geometry2D::rectangle("R1", 20.0, 10.0).with_quantity(3),
            Geometry2D::rectangle("R2", 15.0, 15.0).with_quantity(2),
        ];

        let boundary = Boundary2D::rectangle(100.0, 50.0);
        let nester = Nester2D::default_config();

        let result = nester.solve(&geometries, &boundary).unwrap();

        assert!(result.utilization > 0.0);
        assert!(result.placements.len() <= 5); // 3 + 2 = 5 pieces
    }

    #[test]
    fn test_placement_within_bounds() {
        let geometries = vec![Geometry2D::rectangle("R1", 10.0, 10.0).with_quantity(4)];

        let boundary = Boundary2D::rectangle(50.0, 50.0);
        let config = Config::default().with_margin(5.0).with_spacing(2.0);
        let nester = Nester2D::new(config);

        let result = nester.solve(&geometries, &boundary).unwrap();

        // All pieces should be placed
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());

        // Verify placements are within bounds (with margin)
        for p in &result.placements {
            assert!(p.position[0] >= 5.0);
            assert!(p.position[1] >= 5.0);
        }
    }
}
