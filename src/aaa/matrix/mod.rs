pub struct Matrix<T> {
    data: Vec<T>,
}

impl<T> Matrix<T> {
    pub fn new(data: Vec<T>) -> Matrix<T> {
        Matrix { data }
    }
}
