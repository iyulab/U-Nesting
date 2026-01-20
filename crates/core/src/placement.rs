//! Placement representation for positioned geometries.

use crate::geometry::GeometryId;
use crate::transform::{Transform2D, Transform3D};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represents the placement of a geometry within a boundary.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Placement<S> {
    /// The ID of the placed geometry.
    pub geometry_id: GeometryId,

    /// Instance index (0-based) when multiple copies exist.
    pub instance: usize,

    /// Position coordinates (x, y for 2D; x, y, z for 3D).
    pub position: Vec<S>,

    /// Rotation angle(s) in radians.
    /// - 2D: single angle [θ]
    /// - 3D: Euler angles [rx, ry, rz] or quaternion components
    pub rotation: Vec<S>,

    /// Index of the boundary this geometry is placed in (for multi-bin scenarios).
    pub boundary_index: usize,

    /// Whether the geometry was mirrored/flipped.
    pub mirrored: bool,

    /// The rotation option index used (if discrete rotations).
    pub rotation_index: Option<usize>,
}

impl<S: Copy + Default> Placement<S> {
    /// Creates a new 2D placement.
    pub fn new_2d(geometry_id: GeometryId, instance: usize, x: S, y: S, angle: S) -> Self {
        Self {
            geometry_id,
            instance,
            position: vec![x, y],
            rotation: vec![angle],
            boundary_index: 0,
            mirrored: false,
            rotation_index: None,
        }
    }

    /// Creates a new 3D placement.
    pub fn new_3d(
        geometry_id: GeometryId,
        instance: usize,
        x: S,
        y: S,
        z: S,
        rx: S,
        ry: S,
        rz: S,
    ) -> Self {
        Self {
            geometry_id,
            instance,
            position: vec![x, y, z],
            rotation: vec![rx, ry, rz],
            boundary_index: 0,
            mirrored: false,
            rotation_index: None,
        }
    }

    /// Sets the boundary index.
    pub fn with_boundary(mut self, index: usize) -> Self {
        self.boundary_index = index;
        self
    }

    /// Sets the mirrored flag.
    pub fn with_mirrored(mut self, mirrored: bool) -> Self {
        self.mirrored = mirrored;
        self
    }

    /// Sets the rotation index.
    pub fn with_rotation_index(mut self, index: usize) -> Self {
        self.rotation_index = Some(index);
        self
    }

    /// Returns true if this is a 2D placement.
    pub fn is_2d(&self) -> bool {
        self.position.len() == 2
    }

    /// Returns true if this is a 3D placement.
    pub fn is_3d(&self) -> bool {
        self.position.len() == 3
    }

    /// Returns the x coordinate.
    pub fn x(&self) -> S {
        self.position.first().copied().unwrap_or_default()
    }

    /// Returns the y coordinate.
    pub fn y(&self) -> S {
        self.position.get(1).copied().unwrap_or_default()
    }

    /// Returns the z coordinate (for 3D placements).
    pub fn z(&self) -> Option<S> {
        self.position.get(2).copied()
    }

    /// Returns the rotation angle (for 2D placements).
    pub fn angle(&self) -> S {
        self.rotation.first().copied().unwrap_or_default()
    }
}

impl<S: nalgebra::RealField + Copy + Default> Placement<S> {
    /// Converts a 2D placement to a Transform2D.
    pub fn to_transform_2d(&self) -> Transform2D<S> {
        Transform2D::new(self.x(), self.y(), self.angle())
    }

    /// Creates a 2D placement from a Transform2D.
    pub fn from_transform_2d(
        geometry_id: GeometryId,
        instance: usize,
        transform: &Transform2D<S>,
    ) -> Self {
        Self::new_2d(
            geometry_id,
            instance,
            transform.tx,
            transform.ty,
            transform.angle,
        )
    }

    /// Converts a 3D placement to a Transform3D.
    pub fn to_transform_3d(&self) -> Transform3D<S> {
        let z = self.position.get(2).copied().unwrap_or(S::zero());
        let rx = self.rotation.first().copied().unwrap_or(S::zero());
        let ry = self.rotation.get(1).copied().unwrap_or(S::zero());
        let rz = self.rotation.get(2).copied().unwrap_or(S::zero());
        Transform3D::new(self.x(), self.y(), z, rx, ry, rz)
    }

    /// Creates a 3D placement from a Transform3D.
    pub fn from_transform_3d(
        geometry_id: GeometryId,
        instance: usize,
        transform: &Transform3D<S>,
    ) -> Self {
        Self::new_3d(
            geometry_id,
            instance,
            transform.tx,
            transform.ty,
            transform.tz,
            transform.rx,
            transform.ry,
            transform.rz,
        )
    }
}

/// Placement statistics for a set of placements.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PlacementStats {
    /// Total number of placements.
    pub count: usize,
    /// Number of mirrored placements.
    pub mirrored_count: usize,
    /// Distribution of rotation indices used.
    pub rotation_distribution: std::collections::HashMap<usize, usize>,
    /// Distribution of placements per boundary.
    pub boundary_distribution: std::collections::HashMap<usize, usize>,
}

impl PlacementStats {
    /// Computes statistics from a set of placements.
    pub fn from_placements<S>(placements: &[Placement<S>]) -> Self {
        let mut stats = Self {
            count: placements.len(),
            ..Default::default()
        };

        for p in placements {
            if p.mirrored {
                stats.mirrored_count += 1;
            }

            if let Some(rot_idx) = p.rotation_index {
                *stats.rotation_distribution.entry(rot_idx).or_insert(0) += 1;
            }

            *stats
                .boundary_distribution
                .entry(p.boundary_index)
                .or_insert(0) += 1;
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_placement_2d() {
        let p = Placement::new_2d("test".to_string(), 0, 10.0, 20.0, 0.5);
        assert!(p.is_2d());
        assert!(!p.is_3d());
        assert_eq!(p.x(), 10.0);
        assert_eq!(p.y(), 20.0);
        assert_eq!(p.angle(), 0.5);
    }

    #[test]
    fn test_placement_3d() {
        let p = Placement::new_3d("test".to_string(), 0, 10.0, 20.0, 30.0, 0.1, 0.2, 0.3);
        assert!(p.is_3d());
        assert!(!p.is_2d());
        assert_eq!(p.z(), Some(30.0));
    }

    #[test]
    fn test_transform_conversion() {
        let p = Placement::new_2d("test".to_string(), 0, 10.0_f64, 20.0, 0.5);
        let t = p.to_transform_2d();
        assert_eq!(t.tx, 10.0);
        assert_eq!(t.ty, 20.0);
        assert_eq!(t.angle, 0.5);
    }

    #[test]
    fn test_placement_stats() {
        let placements = vec![
            Placement::new_2d("a".to_string(), 0, 0.0, 0.0, 0.0).with_rotation_index(0),
            Placement::new_2d("b".to_string(), 0, 0.0, 0.0, 0.0)
                .with_rotation_index(1)
                .with_mirrored(true),
            Placement::new_2d("c".to_string(), 0, 0.0, 0.0, 0.0)
                .with_rotation_index(0)
                .with_boundary(1),
        ];

        let stats = PlacementStats::from_placements(&placements);
        assert_eq!(stats.count, 3);
        assert_eq!(stats.mirrored_count, 1);
        assert_eq!(stats.rotation_distribution.get(&0), Some(&2));
        assert_eq!(stats.rotation_distribution.get(&1), Some(&1));
    }
}
