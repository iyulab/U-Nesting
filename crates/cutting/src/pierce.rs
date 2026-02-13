//! Pierce point selection and optimization.
//!
//! Selects the optimal entry point on each contour where the cutting tool
//! should pierce. The pierce point is chosen to minimize rapid travel
//! distance from the previous contour's endpoint.

use crate::config::{CutDirectionPreference, CuttingConfig};
use crate::contour::{ContourType, CutContour};
use crate::cost::{closest_point_on_polygon, point_distance};
use crate::result::CutDirection;

/// Selected pierce point for a contour.
#[derive(Debug, Clone)]
pub struct PierceSelection {
    /// The pierce point coordinates.
    pub point: (f64, f64),
    /// The vertex index closest to (or at) the pierce point.
    pub vertex_index: usize,
    /// Cut direction for this contour.
    pub direction: CutDirection,
    /// The endpoint after cutting the full contour (returns to pierce point).
    pub end_point: (f64, f64),
}

/// Selects the pierce point for a contour based on the approach point.
///
/// The pierce point is the closest point on the contour boundary to `from_point`.
/// The cut direction is determined by the contour type and config preferences.
pub fn select_pierce(
    contour: &CutContour,
    from_point: (f64, f64),
    config: &CuttingConfig,
) -> PierceSelection {
    let (closest, vertex_idx, _t) = closest_point_on_polygon(&contour.vertices, from_point)
        .unwrap_or((contour.vertices[0], 0, 0.0));

    let direction = determine_cut_direction(contour.contour_type, config);

    PierceSelection {
        point: closest,
        vertex_index: vertex_idx,
        direction,
        end_point: closest, // Full contour cut returns to pierce point
    }
}

/// Determines the cutting direction based on contour type and config.
fn determine_cut_direction(contour_type: ContourType, config: &CuttingConfig) -> CutDirection {
    let pref = match contour_type {
        ContourType::Exterior => config.exterior_direction,
        ContourType::Interior => config.interior_direction,
    };

    match pref {
        CutDirectionPreference::Ccw => CutDirection::Ccw,
        CutDirectionPreference::Cw => CutDirection::Cw,
        CutDirectionPreference::Auto => match contour_type {
            ContourType::Exterior => CutDirection::Ccw,
            ContourType::Interior => CutDirection::Cw,
        },
    }
}

/// Computes the rapid distance from a point to the best pierce point on a contour.
pub fn rapid_distance_to_contour(from_point: (f64, f64), contour: &CutContour) -> f64 {
    match closest_point_on_polygon(&contour.vertices, from_point) {
        Some((closest, _, _)) => point_distance(from_point, closest),
        None => f64::MAX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_square_contour(id: usize, cx: f64, cy: f64) -> CutContour {
        CutContour {
            id,
            geometry_id: "test".to_string(),
            instance: 0,
            contour_type: ContourType::Exterior,
            vertices: vec![
                (cx - 5.0, cy - 5.0),
                (cx + 5.0, cy - 5.0),
                (cx + 5.0, cy + 5.0),
                (cx - 5.0, cy + 5.0),
            ],
            perimeter: 40.0,
            centroid: (cx, cy),
        }
    }

    #[test]
    fn test_select_pierce_closest_point() {
        let contour = make_square_contour(0, 50.0, 50.0);
        let config = CuttingConfig::default();
        let sel = select_pierce(&contour, (50.0, 0.0), &config);

        // Closest point on bottom edge to (50,0) should be (50, 45)
        assert!((sel.point.0 - 50.0).abs() < 0.1);
        assert!((sel.point.1 - 45.0).abs() < 0.1);
    }

    #[test]
    fn test_auto_direction_exterior_ccw() {
        let contour = CutContour {
            id: 0,
            geometry_id: "test".to_string(),
            instance: 0,
            contour_type: ContourType::Exterior,
            vertices: vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            perimeter: 40.0,
            centroid: (5.0, 5.0),
        };
        let config = CuttingConfig::default();
        let sel = select_pierce(&contour, (0.0, 0.0), &config);
        assert_eq!(sel.direction, CutDirection::Ccw);
    }

    #[test]
    fn test_auto_direction_interior_cw() {
        let contour = CutContour {
            id: 0,
            geometry_id: "test".to_string(),
            instance: 0,
            contour_type: ContourType::Interior,
            vertices: vec![(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)],
            perimeter: 24.0,
            centroid: (5.0, 5.0),
        };
        let config = CuttingConfig::default();
        let sel = select_pierce(&contour, (0.0, 0.0), &config);
        assert_eq!(sel.direction, CutDirection::Cw);
    }

    #[test]
    fn test_rapid_distance() {
        let contour = make_square_contour(0, 50.0, 50.0);
        let dist = rapid_distance_to_contour((0.0, 0.0), &contour);
        // Distance from origin to nearest point on square centered at (50,50) with side 10
        // Nearest point is (45, 45), distance = sqrt(45^2 + 45^2) ~ 63.64
        assert!(dist > 60.0 && dist < 70.0);
    }

    #[test]
    fn test_pierce_end_point_equals_pierce_point() {
        let contour = make_square_contour(0, 50.0, 50.0);
        let config = CuttingConfig::default();
        let sel = select_pierce(&contour, (0.0, 0.0), &config);
        assert_eq!(sel.point, sel.end_point);
    }

    #[test]
    fn test_forced_cw_exterior() {
        let contour = CutContour {
            id: 0,
            geometry_id: "test".to_string(),
            instance: 0,
            contour_type: ContourType::Exterior,
            vertices: vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            perimeter: 40.0,
            centroid: (5.0, 5.0),
        };
        let config = CuttingConfig {
            exterior_direction: CutDirectionPreference::Cw,
            ..CuttingConfig::default()
        };
        let sel = select_pierce(&contour, (0.0, 0.0), &config);
        assert_eq!(sel.direction, CutDirection::Cw);
    }
}
