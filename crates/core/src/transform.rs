//! Transform types for 2D and 3D coordinate transformations.

use nalgebra::{
    Isometry2, Isometry3, Point2, Point3, RealField, Rotation3, Vector2, Vector3,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A 2D rigid transformation (rotation + translation).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Transform2D<S: RealField + Copy> {
    /// Translation in x direction.
    pub tx: S,
    /// Translation in y direction.
    pub ty: S,
    /// Rotation angle in radians.
    pub angle: S,
}

impl<S: RealField + Copy> Transform2D<S> {
    /// Creates a new identity transform.
    pub fn identity() -> Self {
        Self {
            tx: S::zero(),
            ty: S::zero(),
            angle: S::zero(),
        }
    }

    /// Creates a new transform with translation only.
    pub fn translation(tx: S, ty: S) -> Self {
        Self {
            tx,
            ty,
            angle: S::zero(),
        }
    }

    /// Creates a new transform with rotation only.
    pub fn rotation(angle: S) -> Self {
        Self {
            tx: S::zero(),
            ty: S::zero(),
            angle,
        }
    }

    /// Creates a new transform with both translation and rotation.
    pub fn new(tx: S, ty: S, angle: S) -> Self {
        Self { tx, ty, angle }
    }

    /// Converts to a nalgebra Isometry2.
    pub fn to_isometry(&self) -> Isometry2<S> {
        Isometry2::new(Vector2::new(self.tx, self.ty), self.angle)
    }

    /// Creates from a nalgebra Isometry2.
    pub fn from_isometry(iso: &Isometry2<S>) -> Self {
        Self {
            tx: iso.translation.x,
            ty: iso.translation.y,
            angle: iso.rotation.angle(),
        }
    }

    /// Transforms a 2D point.
    pub fn transform_point(&self, x: S, y: S) -> (S, S) {
        let iso = self.to_isometry();
        let p = iso.transform_point(&Point2::new(x, y));
        (p.x, p.y)
    }

    /// Transforms a vector of 2D points.
    pub fn transform_points(&self, points: &[(S, S)]) -> Vec<(S, S)> {
        let iso = self.to_isometry();
        points
            .iter()
            .map(|(x, y)| {
                let p = iso.transform_point(&Point2::new(*x, *y));
                (p.x, p.y)
            })
            .collect()
    }

    /// Composes two transforms: self then other.
    pub fn then(&self, other: &Self) -> Self {
        let iso1 = self.to_isometry();
        let iso2 = other.to_isometry();
        Self::from_isometry(&(iso1 * iso2))
    }

    /// Returns the inverse transform.
    pub fn inverse(&self) -> Self {
        Self::from_isometry(&self.to_isometry().inverse())
    }

    /// Checks if this is approximately an identity transform.
    pub fn is_identity(&self, epsilon: S) -> bool {
        self.tx.abs() < epsilon && self.ty.abs() < epsilon && self.angle.abs() < epsilon
    }
}

impl<S: RealField + Copy> Default for Transform2D<S> {
    fn default() -> Self {
        Self::identity()
    }
}

/// A 3D rigid transformation (rotation + translation).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Transform3D<S: RealField + Copy> {
    /// Translation in x direction.
    pub tx: S,
    /// Translation in y direction.
    pub ty: S,
    /// Translation in z direction.
    pub tz: S,
    /// Rotation around x axis (roll) in radians.
    pub rx: S,
    /// Rotation around y axis (pitch) in radians.
    pub ry: S,
    /// Rotation around z axis (yaw) in radians.
    pub rz: S,
}

impl<S: RealField + Copy> Transform3D<S> {
    /// Creates a new identity transform.
    pub fn identity() -> Self {
        Self {
            tx: S::zero(),
            ty: S::zero(),
            tz: S::zero(),
            rx: S::zero(),
            ry: S::zero(),
            rz: S::zero(),
        }
    }

    /// Creates a new transform with translation only.
    pub fn translation(tx: S, ty: S, tz: S) -> Self {
        Self {
            tx,
            ty,
            tz,
            rx: S::zero(),
            ry: S::zero(),
            rz: S::zero(),
        }
    }

    /// Creates a new transform with rotation only (Euler angles XYZ).
    pub fn rotation(rx: S, ry: S, rz: S) -> Self {
        Self {
            tx: S::zero(),
            ty: S::zero(),
            tz: S::zero(),
            rx,
            ry,
            rz,
        }
    }

    /// Creates a new transform with both translation and rotation.
    pub fn new(tx: S, ty: S, tz: S, rx: S, ry: S, rz: S) -> Self {
        Self {
            tx,
            ty,
            tz,
            rx,
            ry,
            rz,
        }
    }

    /// Converts to a nalgebra Isometry3.
    pub fn to_isometry(&self) -> Isometry3<S> {
        let rotation = Rotation3::from_euler_angles(self.rx, self.ry, self.rz);
        Isometry3::from_parts(
            Vector3::new(self.tx, self.ty, self.tz).into(),
            rotation.into(),
        )
    }

    /// Creates from a nalgebra Isometry3.
    pub fn from_isometry(iso: &Isometry3<S>) -> Self {
        let (rx, ry, rz) = iso.rotation.euler_angles();
        Self {
            tx: iso.translation.x,
            ty: iso.translation.y,
            tz: iso.translation.z,
            rx,
            ry,
            rz,
        }
    }

    /// Transforms a 3D point.
    pub fn transform_point(&self, x: S, y: S, z: S) -> (S, S, S) {
        let iso = self.to_isometry();
        let p = iso.transform_point(&Point3::new(x, y, z));
        (p.x, p.y, p.z)
    }

    /// Transforms a vector of 3D points.
    pub fn transform_points(&self, points: &[(S, S, S)]) -> Vec<(S, S, S)> {
        let iso = self.to_isometry();
        points
            .iter()
            .map(|(x, y, z)| {
                let p = iso.transform_point(&Point3::new(*x, *y, *z));
                (p.x, p.y, p.z)
            })
            .collect()
    }

    /// Composes two transforms: self then other.
    pub fn then(&self, other: &Self) -> Self {
        let iso1 = self.to_isometry();
        let iso2 = other.to_isometry();
        Self::from_isometry(&(iso1 * iso2))
    }

    /// Returns the inverse transform.
    pub fn inverse(&self) -> Self {
        Self::from_isometry(&self.to_isometry().inverse())
    }

    /// Checks if this is approximately an identity transform.
    pub fn is_identity(&self, epsilon: S) -> bool {
        self.tx.abs() < epsilon
            && self.ty.abs() < epsilon
            && self.tz.abs() < epsilon
            && self.rx.abs() < epsilon
            && self.ry.abs() < epsilon
            && self.rz.abs() < epsilon
    }
}

impl<S: RealField + Copy> Default for Transform3D<S> {
    fn default() -> Self {
        Self::identity()
    }
}

/// Axis-aligned bounding box in 2D.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AABB2D<S> {
    /// Minimum x coordinate.
    pub min_x: S,
    /// Minimum y coordinate.
    pub min_y: S,
    /// Maximum x coordinate.
    pub max_x: S,
    /// Maximum y coordinate.
    pub max_y: S,
}

impl<S: RealField + Copy> AABB2D<S> {
    /// Creates a new AABB from min/max coordinates.
    pub fn new(min_x: S, min_y: S, max_x: S, max_y: S) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    /// Creates an AABB from a set of points.
    pub fn from_points(points: &[(S, S)]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }

        let mut min_x = points[0].0;
        let mut min_y = points[0].1;
        let mut max_x = points[0].0;
        let mut max_y = points[0].1;

        for (x, y) in points.iter().skip(1) {
            min_x = min_x.min(*x);
            min_y = min_y.min(*y);
            max_x = max_x.max(*x);
            max_y = max_y.max(*y);
        }

        Some(Self {
            min_x,
            min_y,
            max_x,
            max_y,
        })
    }

    /// Returns the width of the AABB.
    pub fn width(&self) -> S {
        self.max_x - self.min_x
    }

    /// Returns the height of the AABB.
    pub fn height(&self) -> S {
        self.max_y - self.min_y
    }

    /// Returns the area of the AABB.
    pub fn area(&self) -> S {
        self.width() * self.height()
    }

    /// Returns the center point of the AABB.
    pub fn center(&self) -> (S, S) {
        let two = S::one() + S::one();
        (
            (self.min_x + self.max_x) / two,
            (self.min_y + self.max_y) / two,
        )
    }

    /// Checks if this AABB contains a point.
    pub fn contains_point(&self, x: S, y: S) -> bool {
        x >= self.min_x && x <= self.max_x && y >= self.min_y && y <= self.max_y
    }

    /// Checks if this AABB intersects another AABB.
    pub fn intersects(&self, other: &Self) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
    }

    /// Returns the intersection of two AABBs, if any.
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        if !self.intersects(other) {
            return None;
        }

        Some(Self {
            min_x: self.min_x.max(other.min_x),
            min_y: self.min_y.max(other.min_y),
            max_x: self.max_x.min(other.max_x),
            max_y: self.max_y.min(other.max_y),
        })
    }

    /// Returns the union (bounding box) of two AABBs.
    pub fn union(&self, other: &Self) -> Self {
        Self {
            min_x: self.min_x.min(other.min_x),
            min_y: self.min_y.min(other.min_y),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
        }
    }

    /// Expands the AABB by a margin on all sides.
    pub fn expand(&self, margin: S) -> Self {
        Self {
            min_x: self.min_x - margin,
            min_y: self.min_y - margin,
            max_x: self.max_x + margin,
            max_y: self.max_y + margin,
        }
    }
}

/// Axis-aligned bounding box in 3D.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AABB3D<S> {
    /// Minimum x coordinate.
    pub min_x: S,
    /// Minimum y coordinate.
    pub min_y: S,
    /// Minimum z coordinate.
    pub min_z: S,
    /// Maximum x coordinate.
    pub max_x: S,
    /// Maximum y coordinate.
    pub max_y: S,
    /// Maximum z coordinate.
    pub max_z: S,
}

impl<S: RealField + Copy> AABB3D<S> {
    /// Creates a new AABB from min/max coordinates.
    pub fn new(min_x: S, min_y: S, min_z: S, max_x: S, max_y: S, max_z: S) -> Self {
        Self {
            min_x,
            min_y,
            min_z,
            max_x,
            max_y,
            max_z,
        }
    }

    /// Creates an AABB from a set of points.
    pub fn from_points(points: &[(S, S, S)]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }

        let mut min_x = points[0].0;
        let mut min_y = points[0].1;
        let mut min_z = points[0].2;
        let mut max_x = points[0].0;
        let mut max_y = points[0].1;
        let mut max_z = points[0].2;

        for (x, y, z) in points.iter().skip(1) {
            min_x = min_x.min(*x);
            min_y = min_y.min(*y);
            min_z = min_z.min(*z);
            max_x = max_x.max(*x);
            max_y = max_y.max(*y);
            max_z = max_z.max(*z);
        }

        Some(Self {
            min_x,
            min_y,
            min_z,
            max_x,
            max_y,
            max_z,
        })
    }

    /// Returns the width (x dimension) of the AABB.
    pub fn width(&self) -> S {
        self.max_x - self.min_x
    }

    /// Returns the depth (y dimension) of the AABB.
    pub fn depth(&self) -> S {
        self.max_y - self.min_y
    }

    /// Returns the height (z dimension) of the AABB.
    pub fn height(&self) -> S {
        self.max_z - self.min_z
    }

    /// Returns the volume of the AABB.
    pub fn volume(&self) -> S {
        self.width() * self.depth() * self.height()
    }

    /// Returns the center point of the AABB.
    pub fn center(&self) -> (S, S, S) {
        let two = S::one() + S::one();
        (
            (self.min_x + self.max_x) / two,
            (self.min_y + self.max_y) / two,
            (self.min_z + self.max_z) / two,
        )
    }

    /// Checks if this AABB contains a point.
    pub fn contains_point(&self, x: S, y: S, z: S) -> bool {
        x >= self.min_x
            && x <= self.max_x
            && y >= self.min_y
            && y <= self.max_y
            && z >= self.min_z
            && z <= self.max_z
    }

    /// Checks if this AABB intersects another AABB.
    pub fn intersects(&self, other: &Self) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
            && self.min_z <= other.max_z
            && self.max_z >= other.min_z
    }

    /// Returns the intersection of two AABBs, if any.
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        if !self.intersects(other) {
            return None;
        }

        Some(Self {
            min_x: self.min_x.max(other.min_x),
            min_y: self.min_y.max(other.min_y),
            min_z: self.min_z.max(other.min_z),
            max_x: self.max_x.min(other.max_x),
            max_y: self.max_y.min(other.max_y),
            max_z: self.max_z.min(other.max_z),
        })
    }

    /// Returns the union (bounding box) of two AABBs.
    pub fn union(&self, other: &Self) -> Self {
        Self {
            min_x: self.min_x.min(other.min_x),
            min_y: self.min_y.min(other.min_y),
            min_z: self.min_z.min(other.min_z),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
            max_z: self.max_z.max(other.max_z),
        }
    }

    /// Expands the AABB by a margin on all sides.
    pub fn expand(&self, margin: S) -> Self {
        Self {
            min_x: self.min_x - margin,
            min_y: self.min_y - margin,
            min_z: self.min_z - margin,
            max_x: self.max_x + margin,
            max_y: self.max_y + margin,
            max_z: self.max_z + margin,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_transform2d_identity() {
        let t = Transform2D::<f64>::identity();
        let (x, y) = t.transform_point(1.0, 2.0);
        assert_relative_eq!(x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(y, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_transform2d_translation() {
        let t = Transform2D::translation(10.0, 20.0);
        let (x, y) = t.transform_point(1.0, 2.0);
        assert_relative_eq!(x, 11.0, epsilon = 1e-10);
        assert_relative_eq!(y, 22.0, epsilon = 1e-10);
    }

    #[test]
    fn test_transform2d_rotation() {
        let t = Transform2D::rotation(PI / 2.0);
        let (x, y) = t.transform_point(1.0, 0.0);
        assert_relative_eq!(x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_transform2d_inverse() {
        let t = Transform2D::new(10.0, 20.0, PI / 4.0);
        let inv = t.inverse();
        let composed = t.then(&inv);
        assert!(composed.is_identity(1e-10));
    }

    #[test]
    fn test_transform3d_translation() {
        let t = Transform3D::translation(10.0, 20.0, 30.0);
        let (x, y, z) = t.transform_point(1.0, 2.0, 3.0);
        assert_relative_eq!(x, 11.0, epsilon = 1e-10);
        assert_relative_eq!(y, 22.0, epsilon = 1e-10);
        assert_relative_eq!(z, 33.0, epsilon = 1e-10);
    }

    #[test]
    fn test_aabb2d_from_points() {
        let points = vec![(0.0, 0.0), (10.0, 5.0), (3.0, 8.0)];
        let aabb = AABB2D::from_points(&points).unwrap();
        assert_relative_eq!(aabb.min_x, 0.0);
        assert_relative_eq!(aabb.min_y, 0.0);
        assert_relative_eq!(aabb.max_x, 10.0);
        assert_relative_eq!(aabb.max_y, 8.0);
    }

    #[test]
    fn test_aabb2d_intersection() {
        let a = AABB2D::new(0.0, 0.0, 10.0, 10.0);
        let b = AABB2D::new(5.0, 5.0, 15.0, 15.0);
        let intersection = a.intersection(&b).unwrap();
        assert_relative_eq!(intersection.min_x, 5.0);
        assert_relative_eq!(intersection.min_y, 5.0);
        assert_relative_eq!(intersection.max_x, 10.0);
        assert_relative_eq!(intersection.max_y, 10.0);
    }

    #[test]
    fn test_aabb3d_volume() {
        let aabb = AABB3D::new(0.0, 0.0, 0.0, 10.0, 20.0, 30.0);
        assert_relative_eq!(aabb.volume(), 6000.0);
    }
}
