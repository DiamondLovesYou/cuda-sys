#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

pub use driver_types::*;
pub use library_types::*;
use vector_types::*;

pub(crate) static LIBRARY: super::GlobalLibrary = super::GlobalLibrary::new();

include!("bindings/cudart_bindings.rs");
include!("bindings/cudart_bindings_fns.rs");
