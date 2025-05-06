// model.rs
use ndarray::{Array2, Array1, Axis};
use rand::Rng;
use ndarray::s;

#[derive(Debug, Clone)]
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

    pub fn train_one_epoch(&mut self, x: &Array2<f64>, y: &Array2<f64>, batch_size: usize) {
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
        self.losses.push(self.cross_entropy_loss(&output, y));
        self.accuracies.push(self.evaluate(x, y));
    }

    pub fn cross_entropy_loss(&self, y_pred: &Array2<f64>, y_true: &Array2<f64>) -> f64 {
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