use crate::{data::FruitSample, model::NeuralNet};
use ndarray::Array2;
use crate::utils::{encode_labels, normalize_features, split_dataset};

pub struct TrainingResult {
    pub accuracies: Vec<f64>,
    pub losses: Vec<f64>,
    pub final_accuracy: f64,
}

pub fn run_training_from_csv(path: &str) -> TrainingResult {
    let samples = crate::data::load_dataset(path).expect("Failed to load dataset");
    run_training_from_samples(&samples)
}

pub fn run_training_from_samples(samples: &[FruitSample]) -> TrainingResult {
    let mut features = Array2::<f64>::zeros((samples.len(), 2));
    let mut labels = Vec::new();

    for (i, sample) in samples.iter().enumerate() {
        features[[i, 0]] = sample.weight;
        features[[i, 1]] = sample.size;
        labels.push(sample.label.clone());
    }

    normalize_features(&mut features);
    let encoded_labels = encode_labels(&labels);

    let (train_features, train_labels, test_features, test_labels) =
        split_dataset(features, encoded_labels, 0.8);

    let input_size = 2;
    let hidden_size = 8;
    let output_size = 3;
    let learning_rate = 0.01;
    let epochs = 5000;
    let batch_size = 16;

    let mut nn = NeuralNet::new(input_size, hidden_size, output_size, learning_rate);
    let (accuracies, losses) = nn.train(&train_features, &train_labels, epochs, batch_size);

    let final_accuracy = nn.evaluate(&test_features, &test_labels);

    TrainingResult {
        accuracies,
        losses,
        final_accuracy,
    }
}