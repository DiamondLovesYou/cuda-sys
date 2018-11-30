/* automatically generated by rust-bindgen */

#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum cublasStatus_t {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
    CUBLAS_STATUS_LICENSE_ERROR = 16,
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum cublasFillMode_t {
    CUBLAS_FILL_MODE_LOWER = 0,
    CUBLAS_FILL_MODE_UPPER = 1,
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum cublasDiagType_t {
    CUBLAS_DIAG_NON_UNIT = 0,
    CUBLAS_DIAG_UNIT = 1,
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum cublasSideMode_t {
    CUBLAS_SIDE_LEFT = 0,
    CUBLAS_SIDE_RIGHT = 1,
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum cublasOperation_t {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum cublasPointerMode_t {
    CUBLAS_POINTER_MODE_HOST = 0,
    CUBLAS_POINTER_MODE_DEVICE = 1,
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum cublasAtomicsMode_t {
    CUBLAS_ATOMICS_NOT_ALLOWED = 0,
    CUBLAS_ATOMICS_ALLOWED = 1,
}
impl cublasGemmAlgo_t {
    pub const CUBLAS_GEMM_DEFAULT: cublasGemmAlgo_t = cublasGemmAlgo_t::CUBLAS_GEMM_DFALT;
}
impl cublasGemmAlgo_t {
    pub const CUBLAS_GEMM_DFALT_TENSOR_OP: cublasGemmAlgo_t =
        cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP;
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum cublasGemmAlgo_t {
    CUBLAS_GEMM_DFALT = -1,
    CUBLAS_GEMM_ALGO0 = 0,
    CUBLAS_GEMM_ALGO1 = 1,
    CUBLAS_GEMM_ALGO2 = 2,
    CUBLAS_GEMM_ALGO3 = 3,
    CUBLAS_GEMM_ALGO4 = 4,
    CUBLAS_GEMM_ALGO5 = 5,
    CUBLAS_GEMM_ALGO6 = 6,
    CUBLAS_GEMM_ALGO7 = 7,
    CUBLAS_GEMM_ALGO8 = 8,
    CUBLAS_GEMM_ALGO9 = 9,
    CUBLAS_GEMM_ALGO10 = 10,
    CUBLAS_GEMM_ALGO11 = 11,
    CUBLAS_GEMM_ALGO12 = 12,
    CUBLAS_GEMM_ALGO13 = 13,
    CUBLAS_GEMM_ALGO14 = 14,
    CUBLAS_GEMM_ALGO15 = 15,
    CUBLAS_GEMM_ALGO16 = 16,
    CUBLAS_GEMM_ALGO17 = 17,
    CUBLAS_GEMM_ALGO18 = 18,
    CUBLAS_GEMM_ALGO19 = 19,
    CUBLAS_GEMM_ALGO20 = 20,
    CUBLAS_GEMM_ALGO21 = 21,
    CUBLAS_GEMM_ALGO22 = 22,
    CUBLAS_GEMM_ALGO23 = 23,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99,
    CUBLAS_GEMM_ALGO0_TENSOR_OP = 100,
    CUBLAS_GEMM_ALGO1_TENSOR_OP = 101,
    CUBLAS_GEMM_ALGO2_TENSOR_OP = 102,
    CUBLAS_GEMM_ALGO3_TENSOR_OP = 103,
    CUBLAS_GEMM_ALGO4_TENSOR_OP = 104,
    CUBLAS_GEMM_ALGO5_TENSOR_OP = 105,
    CUBLAS_GEMM_ALGO6_TENSOR_OP = 106,
    CUBLAS_GEMM_ALGO7_TENSOR_OP = 107,
    CUBLAS_GEMM_ALGO8_TENSOR_OP = 108,
    CUBLAS_GEMM_ALGO9_TENSOR_OP = 109,
    CUBLAS_GEMM_ALGO10_TENSOR_OP = 110,
    CUBLAS_GEMM_ALGO11_TENSOR_OP = 111,
    CUBLAS_GEMM_ALGO12_TENSOR_OP = 112,
    CUBLAS_GEMM_ALGO13_TENSOR_OP = 113,
    CUBLAS_GEMM_ALGO14_TENSOR_OP = 114,
    CUBLAS_GEMM_ALGO15_TENSOR_OP = 115,
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum cublasMath_t {
    CUBLAS_DEFAULT_MATH = 0,
    CUBLAS_TENSOR_OP_MATH = 1,
}
pub use self::cudaDataType as cublasDataType_t;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cublasContext {
    _unused: [u8; 0],
}
pub type cublasHandle_t = *mut cublasContext;
pub type cublasLogCallback =
    ::std::option::Option<unsafe extern "C" fn(msg: *const ::std::os::raw::c_char)>;