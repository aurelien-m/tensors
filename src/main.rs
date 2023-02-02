mod tensor;

fn main() {
    let a = tensor::Tensor::new(&[1.0, 2.0, 3.0, 4.0]).with_shape(&[2, 2]);
    let b = tensor::Tensor::new(&[1.0, 2.0, 3.0, 4.0]).with_shape(&[2, 2]);
    let c = a * b;
    println!("{}", c);
}
