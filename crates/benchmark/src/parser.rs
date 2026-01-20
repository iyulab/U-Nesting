//! ESICUP dataset parser.

use crate::dataset::{Dataset, Item, Shape};
use serde::Deserialize;
use std::fs;
use std::path::Path;
use thiserror::Error;

/// Errors that can occur when parsing datasets.
#[derive(Debug, Error)]
pub enum ParseError {
    #[error("Failed to read file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Failed to parse JSON: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Invalid dataset format: {0}")]
    InvalidFormat(String),

    #[error("HTTP request failed: {0}")]
    HttpError(String),
}

/// Parser for ESICUP benchmark datasets.
#[derive(Debug, Default)]
pub struct DatasetParser;

impl DatasetParser {
    /// Creates a new parser.
    pub fn new() -> Self {
        Self
    }

    /// Parses a dataset from a JSON file.
    pub fn parse_file(&self, path: impl AsRef<Path>) -> Result<Dataset, ParseError> {
        let content = fs::read_to_string(path)?;
        self.parse_json(&content)
    }

    /// Parses a dataset from a JSON string.
    pub fn parse_json(&self, json: &str) -> Result<Dataset, ParseError> {
        let raw: RawDataset = serde_json::from_str(json)?;
        self.convert_raw_dataset(raw)
    }

    /// Downloads and parses a dataset from GitHub.
    pub fn download_and_parse(&self, dataset_name: &str, instance: &str) -> Result<Dataset, ParseError> {
        let url = format!(
            "https://raw.githubusercontent.com/Oscar-Oliveira/OR-Datasets/master/Cutting-and-Packing/2D-Irregular/Datasets/{}/json/{}.json",
            dataset_name.to_uppercase(),
            instance.to_lowercase()
        );

        let response = ureq::get(&url)
            .call()
            .map_err(|e| ParseError::HttpError(e.to_string()))?;

        let json = response
            .into_string()
            .map_err(|e| ParseError::HttpError(e.to_string()))?;

        self.parse_json(&json)
    }

    /// Lists available datasets.
    pub fn list_available_datasets() -> Vec<&'static str> {
        vec![
            "ALBANO",
            "BLAZ",
            "DAGLI",
            "FU",
            "JAKOBS",
            "MAO",
            "MARQUES",
            "SHAPES",
            "SHIRTS",
            "SWIM",
            "TROUSERS",
        ]
    }

    /// Converts a raw dataset to our format.
    fn convert_raw_dataset(&self, raw: RawDataset) -> Result<Dataset, ParseError> {
        let items: Result<Vec<Item>, ParseError> = raw
            .items
            .into_iter()
            .map(|raw_item| self.convert_raw_item(raw_item))
            .collect();

        Ok(Dataset {
            name: raw.name,
            items: items?,
            strip_height: raw.strip_height,
            best_known: None,
        })
    }

    /// Converts a raw item to our format.
    fn convert_raw_item(&self, raw: RawItem) -> Result<Item, ParseError> {
        let shape = self.convert_raw_shape(raw.shape)?;

        Ok(Item {
            id: raw.id,
            demand: raw.demand,
            allowed_orientations: raw.allowed_orientations,
            shape,
        })
    }

    /// Converts a raw shape to our format.
    fn convert_raw_shape(&self, raw: RawShape) -> Result<Shape, ParseError> {
        match raw {
            RawShape::SimplePolygon { data } => {
                Ok(Shape::SimplePolygon(data))
            }
            RawShape::PolygonWithHoles { outer, holes } => {
                Ok(Shape::PolygonWithHoles { outer, holes })
            }
            RawShape::MultiPolygon { data } => {
                Ok(Shape::MultiPolygon(data))
            }
        }
    }
}

/// Raw dataset as parsed from JSON.
#[derive(Debug, Deserialize)]
struct RawDataset {
    name: String,
    items: Vec<RawItem>,
    strip_height: f64,
}

/// Raw item as parsed from JSON.
#[derive(Debug, Deserialize)]
struct RawItem {
    id: usize,
    demand: usize,
    #[serde(default)]
    #[allow(dead_code)]
    dxf: Option<String>,
    allowed_orientations: Vec<f64>,
    shape: RawShape,
}

/// Raw shape as parsed from JSON.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum RawShape {
    #[serde(rename = "simple_polygon")]
    SimplePolygon { data: Vec<[f64; 2]> },

    #[serde(rename = "polygon_with_holes")]
    PolygonWithHoles {
        outer: Vec<[f64; 2]>,
        holes: Vec<Vec<[f64; 2]>>,
    },

    #[serde(rename = "multi_polygon")]
    MultiPolygon { data: Vec<Vec<[f64; 2]>> },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_polygon() {
        let json = r#"{
            "name": "test",
            "items": [
                {
                    "id": 0,
                    "demand": 2,
                    "allowed_orientations": [0.0, 90.0],
                    "shape": {
                        "type": "simple_polygon",
                        "data": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
                    }
                }
            ],
            "strip_height": 10.0
        }"#;

        let parser = DatasetParser::new();
        let dataset = parser.parse_json(json).unwrap();

        assert_eq!(dataset.name, "test");
        assert_eq!(dataset.items.len(), 1);
        assert_eq!(dataset.items[0].demand, 2);
        assert_eq!(dataset.strip_height, 10.0);

        match &dataset.items[0].shape {
            Shape::SimplePolygon(points) => {
                assert_eq!(points.len(), 5);
            }
            _ => panic!("Expected SimplePolygon"),
        }
    }

    #[test]
    fn test_expand_items() {
        let json = r#"{
            "name": "test",
            "items": [
                {
                    "id": 0,
                    "demand": 3,
                    "allowed_orientations": [0.0],
                    "shape": {
                        "type": "simple_polygon",
                        "data": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
                    }
                },
                {
                    "id": 1,
                    "demand": 2,
                    "allowed_orientations": [0.0, 180.0],
                    "shape": {
                        "type": "simple_polygon",
                        "data": [[0.0, 0.0], [2.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
                    }
                }
            ],
            "strip_height": 10.0
        }"#;

        let parser = DatasetParser::new();
        let dataset = parser.parse_json(json).unwrap();
        let expanded = dataset.expand_items();

        assert_eq!(expanded.len(), 5); // 3 + 2
        assert_eq!(expanded[0].item_id, 0);
        assert_eq!(expanded[1].item_id, 0);
        assert_eq!(expanded[2].item_id, 0);
        assert_eq!(expanded[3].item_id, 1);
        assert_eq!(expanded[4].item_id, 1);

        // Check unique piece IDs
        for (i, item) in expanded.iter().enumerate() {
            assert_eq!(item.piece_id, i);
        }
    }
}
