#[repr(C)]
struct CudaMatrix {
    data: *mut f32,
}

#[link(name = "cuda_utils", kind = "static")]
extern "C" {
    fn init_cuda_matrix(matrix: *mut CudaMatrix, data: *const f32, data_size: usize) -> i32;
    fn free_cuda_matrix(matrix: *mut CudaMatrix) -> i32;
}

pub struct Tensor {
    data: CudaMatrix,
}

impl Tensor {
    pub fn new(data: &[f32]) -> Tensor {
        let mut matrix = CudaMatrix {
            data: std::ptr::null_mut(),
        };

        unsafe {
            init_cuda_matrix(&mut matrix, data.as_ptr(), data.len());
        }

        Tensor {
            data: matrix,
        }
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe {
            free_cuda_matrix(&mut self.data);
        }
    }
}