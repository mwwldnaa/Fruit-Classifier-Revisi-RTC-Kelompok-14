mod data;
mod model;
mod utils;

use ndarray::{Array2, Axis};
use std::io;
use utils::{encode_labels, Normalizer};
use plotters::prelude::*;

#[derive(Debug)]
pub struct TrainingResult {
    pub model: model::NeuralNet,
    pub accuracies: Vec<f64>,
    pub losses: Vec<f64>,
    pub final_accuracy: f64,
    pub class_names: Vec<String>,
}

pub fn train_model() -> TrainingResult {
    // Load and shuffle dataset
    let mut samples = data::load_dataset("dataset/fruits_dataset.csv")
        .expect("Failed to load dataset");
    rand::seq::index::sample(&mut rand::thread_rng(), samples.len(), samples.len())
        .into_iter()
        .for_each(|i| samples.swap(i, 0));

    // Split data
    let split_index = (samples.len() as f64 * 0.8) as usize;
    let (train, test) = samples.split_at(split_index);

    // Initialize and fit normalizer
    let mut normalizer = Normalizer::new();
    normalizer.fit(train);
    let train_features = normalizer.transform(train);
    let test_features = normalizer.transform(test);

    println!("Normalization parameters:");
    println!("Means: {:?}", normalizer.mean);
    println!("Std devs: {:?}", normalizer.std);

    // Prepare labels
    let train_labels: Vec<String> = train.iter().map(|s| s.label.clone()).collect();
    let test_labels: Vec<String> = test.iter().map(|s| s.label.clone()).collect();

    let class_names = {
        let mut temp = train_labels.clone();
        temp.sort();
        temp.dedup();
        if let Some(pos) = temp.iter().position(|x| x == "unknown") {
            temp.swap_remove(pos);
            temp.push("unknown".to_string());
        }
        temp
    };

    let train_encoded = encode_labels(&train_labels, &class_names);
    let test_encoded = encode_labels(&test_labels, &class_names);

    // Training parameters
    let input_size = 4;
    let hidden_size = 16;
    let output_size = class_names.len();
    let learning_rate = 0.01;
    let epochs = 5000;
    let batch_size = 32;

    let mut nn = model::NeuralNet::new(input_size, hidden_size, output_size, learning_rate);
    
    println!("Training started for {} epochs...", epochs);
    for epoch in 0..epochs {
        nn.train_one_epoch(&train_features, &train_encoded, batch_size);

        if epoch % 50 == 0 || epoch == epochs - 1 {
            let accuracy = nn.evaluate(&train_features, &train_encoded);
            let test_accuracy = nn.evaluate(&test_features, &test_encoded);
            let loss = nn.losses.last().copied().unwrap_or(0.0);
            
            println!("Epoch {}/{} - Loss: {:.4} - Train Acc: {:.2}% - Test Acc: {:.2}%", 
                epoch + 1, epochs, loss, accuracy * 100.0, test_accuracy * 100.0);
        }
    }

    let final_accuracy = nn.evaluate(&test_features, &test_encoded);
    println!("Training completed. Final test accuracy: {:.2}%", final_accuracy * 100.0);

    // Clone necessary fields before moving nn
    let accuracies = nn.accuracies.clone();
    let losses = nn.losses.clone();
    
    TrainingResult {
        model: nn,
        accuracies,
        losses,
        final_accuracy,
        class_names,
    }
}

fn plot_training_results(accuracies: &[f64], losses: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("training_plots.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Training Metrics", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..accuracies.len(), 0f64..1f64)?;

    chart
        .configure_mesh()
        .x_desc("Epoch")
        .y_desc("Value")
        .draw()?;

    // Plot accuracy
    chart
        .draw_series(LineSeries::new(
            accuracies.iter().enumerate().map(|(x, y)| (x, *y)),
            &RED,
        ))?
        .label("Accuracy")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Plot loss
    chart
        .draw_series(LineSeries::new(
            losses.iter().enumerate().map(|(x, y)| (x, *y)),
            &BLUE,
        ))?
        .label("Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("Training plots saved to training_plots.png");
    Ok(())
}

fn main() {
    let result = train_model();
    
    // Plot training results
    if let Err(e) = plot_training_results(&result.accuracies, &result.losses) {
        eprintln!("Error plotting training results: {}", e);
    }
    
    println!("\nManual Testing Mode");
    println!("Format: weight(g) size(cm) width(cm) height(cm)");
    println!("Example: 150 7 6 6");
    println!("Enter 'q' to quit\n");
    
    let samples = data::load_dataset("dataset/fruits_dataset.csv").expect("Failed to load dataset");
    let mut normalizer = Normalizer::new();
    normalizer.fit(&samples);
    
    loop {
        print!("Enter measurements > ");
        io::Write::flush(&mut io::stdout()).unwrap();
        
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        
        if input.trim() == "q" {
            break;
        }
        
        let parts: Vec<f64> = match input
            .trim()
            .split_whitespace()
            .map(|s| s.parse())
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(p) => p,
            Err(_) => {
                println!("Error: Please enter valid numbers");
                continue;
            }
        };
        
        if parts.len() != 4 {
            println!("Error: Please enter exactly 4 numbers");
            continue;
        }

        if parts[0] > 10000.0 || parts[1] > 50.0 || parts[2] > 50.0 || parts[3] > 50.0 {
            println!("Warning: Values seem unusually large - expected weight(g), size/cm");
        }

        let mut input_data = Array2::zeros((1, 4));
        input_data[[0, 0]] = parts[0];
        input_data[[0, 1]] = parts[1];
        input_data[[0, 2]] = parts[2];
        input_data[[0, 3]] = parts[3];
        
        normalizer.normalize(&mut input_data);

        let (_, _, output) = result.model.forward(&input_data);
        
        let prediction = output
            .map_axis(Axis(1), |row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, &val)| (i, val))
            })
            .into_raw_vec()
            .into_iter()
            .next()
            .flatten();

        match prediction {
            Some((predicted_class, confidence)) => {
                let final_prediction = if confidence < 0.5 {
                    "unknown"
                } else {
                    &result.class_names[predicted_class]
                };
                println!("Prediction: {} ({:.1}% confidence)\n", 
                    final_prediction, 
                    confidence * 100.0);
            }
            None => {
                println!("Error: Failed to make prediction");
            }
        }
    }
}