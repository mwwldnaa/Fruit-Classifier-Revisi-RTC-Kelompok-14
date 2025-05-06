use ndarray::{Array2, Array1};
use crate::data::FruitSample;

#[derive(Debug)]
pub struct Normalizer {
    pub mean: Array1<f64>,
    pub std: Array1<f64>,
}

impl Normalizer {
    pub fn new() -> Self {
        Normalizer {
            mean: Array1::zeros(4),
            std: Array1::ones(4),
        }
    }

    pub fn fit(&mut self, samples: &[FruitSample]) {
        // Calculate mean
        let mut sum = Array1::zeros(4);
        for sample in samples {
            sum[0] += sample.weight;
            sum[1] += sample.size;
            sum[2] += sample.width;
            sum[3] += sample.height;
        }
        self.mean = &sum / samples.len() as f64;

        // Calculate std with numerical stability
        let mut variance = Array1::zeros(4);
        for sample in samples {
            variance[0] += (sample.weight - self.mean[0]).powi(2);
            variance[1] += (sample.size - self.mean[1]).powi(2);
            variance[2] += (sample.width - self.mean[2]).powi(2);
            variance[3] += (sample.height - self.mean[3]).powi(2);
        }
        self.std = (variance / samples.len() as f64)
            .mapv(|x: f64| x.sqrt())
            .mapv(|x: f64| if x > 1e-8 { x } else { 1.0 });
    }

    pub fn transform(&self, samples: &[FruitSample]) -> Array2<f64> {
        let mut features = Array2::zeros((samples.len(), 4));
        for (i, sample) in samples.iter().enumerate() {
            features[[i, 0]] = (sample.weight - self.mean[0]) / self.std[0];
            features[[i, 1]] = (sample.size - self.mean[1]) / self.std[1];
            features[[i, 2]] = (sample.width - self.mean[2]) / self.std[2];
            features[[i, 3]] = (sample.height - self.mean[3]) / self.std[3];
        }
        features
    }

    pub fn normalize(&self, data: &mut Array2<f64>) {
        for mut row in data.rows_mut() {
            for j in 0..4 {
                row[j] = (row[j] - self.mean[j]) / self.std[j];
            }
        }
    }
}

pub fn encode_labels(labels: &[String], class_names: &[String]) -> Array2<f64> {
    let mut encoded = Array2::zeros((labels.len(), class_names.len()));
    for (i, label) in labels.iter().enumerate() {
        if let Some(pos) = class_names.iter().position(|x| x == label) {
            encoded[[i, pos]] = 1.0;
        }
    }
    encoded
}