fn predict(w: &f32, x: &f32) -> f32 {
    w*x
}

fn main() {
    let mut w: f32 = 0.1;
    let learning_rate: f32 = 0.1;

    let inputs: [f32; 4] = [1.0,2.0,3.0,4.0];
    let targets: [f32; 4] = [2.0,4.0,6.0,8.0];

    for _ in 1..=25{
        let mut pred: [f32; 4] = [0.0; 4];
        for (i, &input) in inputs.iter().enumerate() {
            pred[i] = predict(&w, &input);
        }
        
        let mut errors: [f32; 4] = [0.0; 4];
        for (i, (t, p)) in targets.iter().zip(pred.iter()).enumerate() {
            errors[i] = t - p;
        }

        let mut cost: f32 = errors.iter().sum::<f32>() / targets.len() as f32;
        println!("Weight: {:?}, Cost: {:?}", w, cost);
        w += learning_rate * cost;
    }
}
