use csv::ReaderBuilder;
use serde::Deserialize;
use std::error::Error;

#[derive(Debug, Deserialize, Clone)]
pub struct FruitSample {
    pub weight: f64,
    pub size: f64,
    pub width: f64,
    pub height: f64,
    pub label: String,
}

pub fn load_dataset(path: &str) -> Result<Vec<FruitSample>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;

    let mut samples = Vec::new();
    for result in rdr.deserialize() {
        let record: FruitSample = result?;
        
        // Basic validation
        if record.weight <= 0.0 || record.size <= 0.0 || 
           record.width <= 0.0 || record.height <= 0.0 {
            return Err("Invalid measurements in dataset".into());
        }
        
        samples.push(record);
    }

    if samples.is_empty() {
        return Err("Empty dataset".into());
    }

    Ok(samples)
}