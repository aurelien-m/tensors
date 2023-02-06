mod ops;

#[repr(C)]
#[derive(Clone, Copy)]
struct CudaMatrix {
    data: *mut f32,
    dims: *mut i32,
    ndims: i32,
}

#[link(name = "cuda_utils", kind = "static")]
extern "C" {
    fn init_cuda_matrix(matrix: *mut CudaMatrix, data: *const f32, data_size: usize) -> i32;
    fn free_cuda_matrix(matrix: *mut CudaMatrix) -> i32;
    fn multiply_cuda_matrix(a: *mut CudaMatrix, b: *mut CudaMatrix, out: *mut CudaMatrix) -> i32;
    fn set_dims(matrix: *mut CudaMatrix, dims: *const i32, ndims: i32) -> i32;
    fn print_cuda_matrix(matrix: *const CudaMatrix) -> i32;
}

#[derive(Clone)]
pub struct Tensor {
    data: CudaMatrix,
}

impl Tensor {
    pub fn new(data: &[f32]) -> Tensor {
        let mut matrix = CudaMatrix {
            data: std::ptr::null_mut(),
            dims: std::ptr::null_mut(),
            ndims: 0,
        };

        unsafe {
            init_cuda_matrix(&mut matrix, data.as_ptr(), data.len());
        }

        Tensor { data: matrix }
    }

    pub fn with_shape(mut self, dims: &[i32]) -> Tensor {
        unsafe {
            set_dims(&mut self.data, dims.as_ptr(), dims.len() as i32);
        }
        self
    }
}

impl std::fmt::Display for Tensor {
    fn fmt(&self, _: &mut std::fmt::Formatter) -> std::fmt::Result {
        unsafe {
            print_cuda_matrix(&self.data);
        }
        Ok(())
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe {
            free_cuda_matrix(&mut self.data);
        }
    }
}
