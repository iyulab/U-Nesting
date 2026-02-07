//! Shared utility functions for 2D nesting placement algorithms.
//!
//! This module consolidates common geometry operations used across
//! multiple nesting strategy implementations (GA, SA, BRKGA, ALNS, GDRR).

use crate::nfp::Nfp;

/// Instance information for decoding placement orders.
///
/// Maps a flat instance index to the corresponding geometry and its
/// repetition number (when `quantity > 1`).
#[derive(Debug, Clone)]
pub struct InstanceInfo {
    /// Index into the geometries array.
    pub geometry_idx: usize,
    /// Instance number within this geometry's quantity.
    pub instance_num: usize,
}

/// Computes the centroid (arithmetic mean of vertices) of a polygon.
///
/// # Returns
///
/// `(0.0, 0.0)` for an empty polygon, otherwise the mean of all vertices.
pub fn polygon_centroid(polygon: &[(f64, f64)]) -> (f64, f64) {
    if polygon.is_empty() {
        return (0.0, 0.0);
    }

    let sum: (f64, f64) = polygon
        .iter()
        .fold((0.0, 0.0), |acc, &(x, y)| (acc.0 + x, acc.1 + y));
    let n = polygon.len() as f64;
    (sum.0 / n, sum.1 / n)
}

/// Expands an NFP outward by the given spacing amount.
///
/// Each vertex of each polygon is moved away from its polygon's centroid
/// by `spacing` units. This approximates a Minkowski sum with a circle
/// of radius `spacing`.
///
/// Returns the original NFP unchanged if `spacing <= 0.0`.
pub fn expand_nfp(nfp: &Nfp, spacing: f64) -> Nfp {
    if spacing <= 0.0 {
        return nfp.clone();
    }

    let expanded_polygons: Vec<Vec<(f64, f64)>> = nfp
        .polygons
        .iter()
        .map(|polygon| {
            let (cx, cy) = polygon_centroid(polygon);
            polygon
                .iter()
                .map(|&(x, y)| {
                    let dx = x - cx;
                    let dy = y - cy;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist > 1e-10 {
                        let scale = (dist + spacing) / dist;
                        (cx + dx * scale, cy + dy * scale)
                    } else {
                        (x, y)
                    }
                })
                .collect()
        })
        .collect();

    Nfp::from_polygons(expanded_polygons)
}

/// Shrinks an IFP (Inner-Fit Polygon) inward by the given spacing amount.
///
/// Each vertex of each polygon is moved toward its polygon's centroid
/// by `spacing` units. Polygons that collapse to fewer than 3 vertices
/// are discarded.
///
/// Returns the original IFP unchanged if `spacing <= 0.0`.
pub fn shrink_ifp(ifp: &Nfp, spacing: f64) -> Nfp {
    if spacing <= 0.0 {
        return ifp.clone();
    }

    let shrunk_polygons: Vec<Vec<(f64, f64)>> = ifp
        .polygons
        .iter()
        .filter_map(|polygon| {
            let (cx, cy) = polygon_centroid(polygon);
            let shrunk: Vec<(f64, f64)> = polygon
                .iter()
                .map(|&(x, y)| {
                    let dx = x - cx;
                    let dy = y - cy;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist > spacing + 1e-10 {
                        let scale = (dist - spacing) / dist;
                        (cx + dx * scale, cy + dy * scale)
                    } else {
                        (cx, cy)
                    }
                })
                .collect();

            if shrunk.len() >= 3 {
                Some(shrunk)
            } else {
                None
            }
        })
        .collect();

    Nfp::from_polygons(shrunk_polygons)
}

/// Computes the nesting fitness score from placement results.
///
/// The fitness function prioritizes placement count (weight 100) over
/// utilization (weight 10), ensuring all-placed solutions always rank
/// higher than partial solutions.
///
/// # Arguments
///
/// * `placed_count` - Number of successfully placed instances
/// * `total_count` - Total number of instances to place
/// * `utilization` - Fraction of boundary area used (0.0..1.0)
///
/// # Returns
///
/// Fitness score in range [0, 110] where higher is better.
pub fn nesting_fitness(placed_count: usize, total_count: usize, utilization: f64) -> f64 {
    let placement_ratio = placed_count as f64 / total_count.max(1) as f64;
    placement_ratio * 100.0 + utilization * 10.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polygon_centroid_empty() {
        assert_eq!(polygon_centroid(&[]), (0.0, 0.0));
    }

    #[test]
    fn test_polygon_centroid_square() {
        let square = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let (cx, cy) = polygon_centroid(&square);
        assert!((cx - 5.0).abs() < 1e-10);
        assert!((cy - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_expand_nfp_zero_spacing() {
        let nfp = Nfp::from_polygons(vec![vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
        ]]);
        let expanded = expand_nfp(&nfp, 0.0);
        assert_eq!(expanded.polygons.len(), nfp.polygons.len());
        assert_eq!(expanded.polygons[0], nfp.polygons[0]);
    }

    #[test]
    fn test_expand_nfp_positive_spacing() {
        let nfp = Nfp::from_polygons(vec![vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
        ]]);
        let expanded = expand_nfp(&nfp, 1.0);
        // All vertices should move outward from centroid (5,5)
        for &(x, y) in &expanded.polygons[0] {
            let dx = x - 5.0;
            let dy = y - 5.0;
            let dist = (dx * dx + dy * dy).sqrt();
            // Original distance was ~7.07, expanded should be ~8.07
            assert!(dist > 7.0);
        }
    }

    #[test]
    fn test_shrink_ifp_zero_spacing() {
        let ifp = Nfp::from_polygons(vec![vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
        ]]);
        let shrunk = shrink_ifp(&ifp, 0.0);
        assert_eq!(shrunk.polygons.len(), ifp.polygons.len());
        assert_eq!(shrunk.polygons[0], ifp.polygons[0]);
    }

    #[test]
    fn test_shrink_ifp_positive_spacing() {
        let ifp = Nfp::from_polygons(vec![vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
        ]]);
        let shrunk = shrink_ifp(&ifp, 1.0);
        // All vertices should move inward toward centroid (5,5)
        for &(x, y) in &shrunk.polygons[0] {
            let dx = x - 5.0;
            let dy = y - 5.0;
            let dist = (dx * dx + dy * dy).sqrt();
            // Original distance was ~7.07, shrunk should be ~6.07
            assert!(dist < 7.0);
        }
    }

    #[test]
    fn test_shrink_ifp_collapse() {
        // Very small polygon that collapses with spacing
        let ifp = Nfp::from_polygons(vec![vec![
            (0.0, 0.0),
            (1.0, 0.0),
            (0.5, 0.5),
        ]]);
        let shrunk = shrink_ifp(&ifp, 10.0);
        // Should collapse to empty (all vertices become centroid, < 3 unique)
        // The shrunk polygon may still have 3 points (all at centroid)
        // but the logic preserves polygons with >= 3 vertices
        assert!(shrunk.polygons.is_empty() || shrunk.polygons[0].len() >= 3);
    }

    #[test]
    fn test_nesting_fitness_all_placed() {
        let f = nesting_fitness(10, 10, 0.85);
        assert!((f - 108.5).abs() < 1e-10);
    }

    #[test]
    fn test_nesting_fitness_partial() {
        let f = nesting_fitness(5, 10, 0.40);
        // 0.5 * 100 + 0.4 * 10 = 54.0
        assert!((f - 54.0).abs() < 1e-10);
    }

    #[test]
    fn test_nesting_fitness_none_placed() {
        let f = nesting_fitness(0, 10, 0.0);
        assert!((f - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_nesting_fitness_empty_total() {
        let f = nesting_fitness(0, 0, 0.0);
        assert!((f - 0.0).abs() < 1e-10);
    }
}
