fn predict(w: &mut f32, inputs: &[f32; 4]) -> [f32; 4]{
    let mut pred: [f32; 4] = [0.0; 4];
    for (i, input) in inputs.iter().enumerate() {
        pred[i] = *w * input;
    }
    pred
}

fn main() {
    let mut w: f32 = 0.1;
    let learning_rate: f32 = 0.1;
    let epochs: u16 = 10;

    let inputs: [f32; 4] = [1.0,2.0,3.0,4.0];
    let targets: [f32; 4] = [2.0,4.0,6.0,8.0];

    for _ in 0..epochs {
        let pred = predict(&mut w, &inputs);
        
        let mut errors: [f32; 4] = [0.0; 4];
        for (i, (t, p)) in targets.iter().zip(pred.iter()).enumerate() {
            errors[i] = (t - p).powf(2.0);
        }

        let mut cost: f32 = errors.iter().sum::<f32>() / targets.len() as f32;
        println!("Weight: {:.4}, Cost: {:.4}", w, cost);

        let mut errors_d: [f32; 4] = [0.0; 4];
        for (i, (p, t)) in pred.iter().zip(targets.iter()).enumerate() {
            errors_d[i] = 2.0 * (p - t);
        }
        let mut weight_d: [f32; 4] = [0.0; 4];
        for (j, (e, i)) in errors_d.iter().zip(inputs.iter()).enumerate() {
            weight_d[j] = e * i;
        }
        w -= learning_rate * weight_d.iter().sum::<f32>() / weight_d.len() as f32;
    }

    let test_inputs = [5.0, 6.0, 7.0, 8.0];
    let test_targets = [10.0, 12.0, 14.0, 16.0];

    let pred = predict(&mut w, &test_inputs);
    for ((i, t), p) in test_inputs.iter().zip(test_targets.iter()).zip(pred.iter()) {
        println!("Input: {:.4}, Target: {:.4}, Pred: {:.4}", i, t, p);
    }
}
