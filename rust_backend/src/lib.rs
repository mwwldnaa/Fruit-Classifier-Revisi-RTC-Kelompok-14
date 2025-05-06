// lib.rs
use std::ffi::{CString, CStr};
use std::os::raw::c_char;
use ndarray::{Array2, Array1, Axis, s};
use rand::Rng;
use serde::{Serialize, Deserialize};
use csv::ReaderBuilder;
use std::error::Error;

#[derive(Debug, Serialize, Deserialize)]
struct FruitSample {
    weight: f64,
    size: f64,
    width: f64,
    height: f64,
    label: String,
}

#[repr(C)]
pub struct TrainingResult {
    accuracies: *mut f64,
    losses: *mut f64,
    final_accuracy: f64,
    length: usize,
}

#[derive(Debug)]
struct Normalizer {
    mean: Array1<f64>,
    std: Array1<f64>,
}

impl Normalizer {
    fn new() -> Self {
        Normalizer {
            mean: Array1::zeros(4),
            std: Array1::ones(4),
        }
    }

    fn fit(&mut self, samples: &[FruitSample]) {
        let mut sum = Array1::zeros(4);
        for sample in samples {
            sum[0] += sample.weight;
            sum[1] += sample.size;
            sum[2] += sample.width;
            sum[3] += sample.height;
        }
        self.mean = &sum / samples.len() as f64;

        let mut variance = Array1::zeros(4);
        for sample in samples {
            variance[0] += (sample.weight - self.mean[0]).powi(2);
            variance[1] += (sample.size - self.mean[1]).powi(2);
            variance[2] += (sample.width - self.mean[2]).powi(2);
            variance[3] += (sample.height - self.mean[3]).powi(2);
        }
        self.std = (variance / (samples.len() as f64 - 1.0)).mapv(f64::sqrt); // Bessel's correction
    }

    fn transform(&self, samples: &[FruitSample]) -> Array2<f64> {
        let mut features = Array2::zeros((samples.len(), 4));
        for (i, sample) in samples.iter().enumerate() {
            features[[i, 0]] = (sample.weight - self.mean[0]) / (self.std[0] + 1e-8);
            features[[i, 1]] = (sample.size - self.mean[1]) / (self.std[1] + 1e-8);
            features[[i, 2]] = (sample.width - self.mean[2]) / (self.std[2] + 1e-8);
            features[[i, 3]] = (sample.height - self.mean[3]) / (self.std[3] + 1e-8);
        }
        features
    }
}

#[derive(Debug)]
pub struct NeuralNet {
    weights1: Array2<f64>,
    bias1: Array1<f64>,
    weights2: Array2<f64>,
    bias2: Array1<f64>,
    learning_rate: f64,
    pub accuracies: Vec<f64>,
    pub losses: Vec<f64>,
}

impl NeuralNet {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        
        // Xavier/Glorot initialization
        let weights1 = Array2::from_shape_fn((input_size, hidden_size), |_| {
            rng.gen_range(-1.0..1.0) * (2.0 / (input_size + hidden_size) as f64).sqrt()
        });
        
        let bias1 = Array1::zeros(hidden_size);
        
        let weights2 = Array2::from_shape_fn((hidden_size, output_size), |_| {
            rng.gen_range(-1.0..1.0) * (1.0 / (hidden_size + output_size) as f64).sqrt()
        });
        
        let bias2 = Array1::zeros(output_size);
        
        NeuralNet {
            weights1,
            bias1,
            weights2,
            bias2,
            learning_rate,
            accuracies: Vec::new(),
            losses: Vec::new(),
        }
    }

    fn relu(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| if v > 0.0 { v } else { 0.0 })
    }

    fn softmax(&self, x: &Array2<f64>) -> Array2<f64> {
        let max_x = x.fold_axis(Axis(1), f64::NEG_INFINITY, |&max, &val| max.max(val));
        let exp_x = (x - &max_x.insert_axis(Axis(1))).mapv(f64::exp);
        let sum_exp = exp_x.sum_axis(Axis(1)).insert_axis(Axis(1));
        exp_x / sum_exp
    }

    pub fn forward(&self, x: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let hidden_input = x.dot(&self.weights1) + &self.bias1;
        let hidden_output = self.relu(&hidden_input);
        let output_input = hidden_output.dot(&self.weights2) + &self.bias2;
        let output = self.softmax(&output_input);
        (hidden_input, hidden_output, output)
    }

    pub fn train_one_epoch(&mut self, x: &Array2<f64>, y: &Array2<f64>, batch_size: usize) -> (f64, f64) {
        let l2_lambda = 0.001; // L2 regularization
        
        for batch in 0..(x.shape()[0] / batch_size) {
            let start = batch * batch_size;
            let end = (batch + 1) * batch_size;
            let x_batch = x.slice(s![start..end, ..]).to_owned();
            let y_batch = y.slice(s![start..end, ..]).to_owned();

            let (_hidden_input, hidden_output, output) = self.forward(&x_batch);

            let output_error = &output - &y_batch;
            let hidden_error = output_error.dot(&self.weights2.t()) * 
                hidden_output.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });

            // Add L2 regularization
            let weights2_grad = hidden_output.t().dot(&output_error) / batch_size as f64 + 
                &self.weights2 * l2_lambda;
            let weights1_grad = x_batch.t().dot(&hidden_error) / batch_size as f64 + 
                &self.weights1 * l2_lambda;

            self.weights2 -= &(weights2_grad * self.learning_rate);
            self.bias2 -= &(output_error.sum_axis(Axis(0)) * (self.learning_rate / batch_size as f64));
            self.weights1 -= &(weights1_grad * self.learning_rate);
            self.bias1 -= &(hidden_error.sum_axis(Axis(0)) * (self.learning_rate / batch_size as f64));
        }

        let (_, _, output) = self.forward(x);
        let loss = self.cross_entropy_loss(&output, y);
        let accuracy = self.evaluate(x, y);
        self.losses.push(loss);
        self.accuracies.push(accuracy);
        
        (loss, accuracy)
    }

    fn cross_entropy_loss(&self, y_pred: &Array2<f64>, y_true: &Array2<f64>) -> f64 {
        -(y_true * &y_pred.mapv(|v| (v + 1e-15).ln())).sum() / y_true.shape()[0] as f64
    }

    pub fn evaluate(&self, x: &Array2<f64>, y: &Array2<f64>) -> f64 {
        let (_, _, output) = self.forward(x);
        let predictions = output.map_axis(Axis(1), |row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        });

        let true_labels = y.map_axis(Axis(1), |row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        });

        predictions.iter()
            .zip(true_labels.iter())
            .filter(|&(p, t)| p == t)
            .count() as f64 / y.shape()[0] as f64
    }
}

fn load_dataset(path: &str) -> Result<Vec<FruitSample>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;

    let mut samples = Vec::new();
    for result in rdr.deserialize() {
        let record: FruitSample = result?;
        samples.push(record);
    }

    Ok(samples)
}

fn encode_labels(labels: &[String], class_names: &[String]) -> Array2<f64> {
    let mut encoded = Array2::zeros((labels.len(), class_names.len()));
    for (i, label) in labels.iter().enumerate() {
        if let Some(pos) = class_names.iter().position(|x| x == label) {
            encoded[[i, pos]] = 1.0;
        }
    }
    encoded
}

#[unsafe(no_mangle)]
pub extern "C" fn train_network(
    dataset_path: *const c_char,
    accuracies: *mut *mut f64,
    losses: *mut *mut f64,
    final_accuracy: *mut f64,
    length: *mut usize,
    epochs: usize,
) -> bool {
    let path = unsafe { CStr::from_ptr(dataset_path).to_str().unwrap() };
    
    let samples = match load_dataset(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to load dataset: {}", e);
            return false;
        }
    };

    let mut normalizer = Normalizer::new();
    normalizer.fit(&samples);
    let features = normalizer.transform(&samples);
    let labels: Vec<String> = samples.iter().map(|s| s.label.clone()).collect();
    
    let mut class_names = labels.clone();
    class_names.sort();
    class_names.dedup();
    
    let encoded_labels = encode_labels(&labels, &class_names);

    let mut nn = NeuralNet::new(4, 32, class_names.len(), 0.01); // Increased hidden size
    
    for _ in 0..epochs {
        nn.train_one_epoch(&features, &encoded_labels, 32);
    }

    // Store these values before converting to raw pointers
    let final_acc = nn.evaluate(&features, &encoded_labels);
    let acc_len = nn.accuracies.len();

    // Convert to boxed slices
    let boxed_acc = nn.accuracies.into_boxed_slice();
    let boxed_loss = nn.losses.into_boxed_slice();
    
    unsafe {
        *accuracies = Box::into_raw(boxed_acc) as *mut f64;
        *losses = Box::into_raw(boxed_loss) as *mut f64;
        *final_accuracy = final_acc;
        *length = acc_len;
    }
    
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn predict(
    weight: f64,
    size: f64,
    // width: f64,
    // height: f64,
) -> *mut c_char {
    // This is a simplified prediction for demo
    // In a real application, you would:
    // 1. Load the trained model
    // 2. Normalize the input features
    // 3. Run forward pass
    // 4. Return the predicted class
    
    let prediction = if weight > 200.0 && size > 8.0 {
        "watermelon"
    } else if weight > 100.0 && size > 6.0 {
        "apple"
    } else if size > 3.0 && size < 6.0 {
        "orange"
    } else if size <= 3.0 {
        "grape"
    } else {
        "unknown"
    };
    
    CString::new(prediction).unwrap().into_raw()
}

#[unsafe(no_mangle)]
pub extern "C" fn free_array(ptr: *mut f64) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(ptr, 0));
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}