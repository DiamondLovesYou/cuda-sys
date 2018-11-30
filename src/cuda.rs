#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

pub use driver_types::cudaError_t;

pub(crate) static LIBRARY: super::GlobalLibrary = super::GlobalLibrary::new();

include!("bindings/cuda_bindings.rs");
include!("bindings/cuda_bindings_fns.rs");
