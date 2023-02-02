use super::multiply_cuda_matrix;
use super::{CudaMatrix, Tensor};
use std::ops::Mul;

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(mut self, mut rhs: Tensor) -> Tensor {
        if self.data.ndims != rhs.data.ndims {
            panic!("Tensors must have the same number of dimensions");
        }

        let mut cuda_matrix = CudaMatrix {
            data: std::ptr::null_mut(),
            dims: std::ptr::null_mut(),
            ndims: 0,
        };

        unsafe {
            multiply_cuda_matrix(&mut self.data, &mut rhs.data, &mut cuda_matrix);
        }

        println!("{}", cuda_matrix.ndims);

        Tensor { data: cuda_matrix }
    }
}
