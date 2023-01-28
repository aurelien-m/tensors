mod tensor;

// #[link(name = "vector_add", kind = "static")]
// extern "C" {
//     fn vectorAdd_main();
// }

fn main() {
    // unsafe {
    //     vectorAdd_main();
    // }

    let _ = tensor::Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
}
