extern crate bindgen;

use std::env;
use std::path::{Path, PathBuf, };

fn read_env() -> Vec<String> {
    match env::var("CUDA_LIBRARY_PATH") {
        Ok(path) => {
            let split_char = if cfg!(target_os = "windows") {
                ";"
            } else {
                ":"
            };

            path.split(split_char)
                .map(|s| s.to_owned())
                .collect::<Vec<_>>()
        }
        Err(_) => vec![],
    }
}

fn find_cuda() -> PathBuf {
    let mut candidates = read_env();
    candidates.push("/usr/local/cuda".to_string());
    candidates.push("/opt/cuda".to_string());
    for base in &candidates {
        let base = PathBuf::from(base);
        let path = base.join("include/cuda.h");
        if path.is_file() {
            return base;
        }
    }
    panic!("CUDA cannot find");
}

fn bindgen<F, T>(f: F, out_path: T, out_stem: &str, dylib_name: &str)
    where F: Fn(bindgen::Builder) -> bindgen::Builder,
          T: AsRef<Path>,
{
    use std::fs::File;
    use std::io::{Read, Write, };

    use bindgen::CodegenConfig;

	let not_functions = f(bindgen::builder());
    let only_functions = f(bindgen::builder());

    let mut not_functions_config = CodegenConfig::all();
    not_functions_config.remove(CodegenConfig::FUNCTIONS);

    let only_functions_config = CodegenConfig::FUNCTIONS;

    let not_functions = not_functions.with_codegen_config(not_functions_config);
    let only_functions = only_functions.with_codegen_config(only_functions_config);

    let not_functions_out = out_path.as_ref().join(format!("{}.rs", out_stem));
    let only_functions_out = out_path.as_ref().join(format!("{}_fns.rs", out_stem));

    not_functions
        .rustfmt_bindings(true)
        .generate()
        .expect("failed to generate bindings")
        .write_to_file(not_functions_out)
        .expect("failed to write bindings");

    only_functions
        .rustfmt_bindings(false)
        .generate()
        .expect("failed to generate bindings")
        .write_to_file(&only_functions_out)
        .expect("failed to write bindings");

    let mut functions = Vec::new();
    {
        let mut f = File::open(&only_functions_out).unwrap();
        f.read_to_end(&mut functions).unwrap();
    }
    {
        let mut out = File::create(&only_functions_out).unwrap();
        writeln!(out, r#"link_funcs! {{ {} =>"#, dylib_name).unwrap();
        out.write_all(&functions).unwrap();
        writeln!(out, "").unwrap();
        writeln!(out, "}}").unwrap();
    }
}

fn main() {
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cuda_path = find_cuda();

    bindgen(|b| {
        b
            .header("cuda.h")
            .clang_arg(format!("-I{}/include", cuda_path.display()))
            .whitelist_recursively(false)
            .whitelist_type("^CU.*")
            .whitelist_type("^cuuint(32|64)_t")
            .whitelist_type("^cudaError_enum")
            .whitelist_type("^cudaMem.*")
            .whitelist_var("^CU.*")
            .whitelist_function("^CU.*")
            .whitelist_function("^cu.*")
            .default_enum_style(bindgen::EnumVariation::Rust)
    }, &out_path, "cuda_bindings", "cuda");

    bindgen(|b| {
        b
            .header("cudart.h")
            .clang_arg(format!("-I{}/include", cuda_path.display()))
            .whitelist_recursively(false)
            .whitelist_type("^cuda.*")
            .whitelist_type("^surfaceReference")
            .whitelist_type("^textureReference")
            .whitelist_var("^cuda.*")
            .whitelist_function("^cuda.*")
            .default_enum_style(bindgen::EnumVariation::Rust)
    }, &out_path, "cudart_bindings", "cudart");

    bindgen(|b| {
        b
            .header("cublas.h")
            .clang_arg(format!("-I{}/include", cuda_path.display()))
            .whitelist_recursively(false)
            .whitelist_type("^cublas.*")
            .whitelist_var("^cublas.*")
            .whitelist_function("^cublas.*")
            .default_enum_style(bindgen::EnumVariation::Rust)
    }, &out_path, "cublas_bindings", "cublas");

    bindgen::builder()
        .header("cucomplex.h")
        .clang_arg(format!("-I{}/include", cuda_path.display()))
        .whitelist_recursively(false)
        .whitelist_type("^cu.*Complex$")
        .default_enum_style(bindgen::EnumVariation::Rust)
		.rustfmt_bindings(true)
        .generate()
        .expect("Unable to generate CUComplex bindings")
        .write_to_file(out_path.join("cucomplex_bindings.rs"))
        .expect("Unable to write CUComplex bindings");

    bindgen::builder()
        .header("driver_types.h")
        .clang_arg(format!("-I{}/include", cuda_path.display()))
        .whitelist_recursively(false)
        .whitelist_type("^CU.*")
        .whitelist_type("^cuda.*")
        .default_enum_style(bindgen::EnumVariation::Rust)
		.rustfmt_bindings(true)
        .generate()
        .expect("Unable to generate driver types bindings")
        .write_to_file(out_path.join("driver_types_bindings.rs"))
        .expect("Unable to write driver types bindings");

    bindgen::builder()
        .header("library_types.h")
        .clang_arg(format!("-I{}/include", cuda_path.display()))
        .whitelist_recursively(false)
        .whitelist_type("^cuda.*")
        .whitelist_type("^libraryPropertyType.*")
        .default_enum_style(bindgen::EnumVariation::Rust)
		.rustfmt_bindings(true)
        .generate()
        .expect("Unable to generate library types bindings")
        .write_to_file(out_path.join("library_types_bindings.rs"))
        .expect("Unable to write library types bindings");

    bindgen::builder()
        .header("vector_types.h")
        .clang_arg(format!("-I{}/include", cuda_path.display()))
        // .whitelist_recursively(false)
        .whitelist_type("^u?char[0-9]$")
        .whitelist_type("^dim[0-9]$")
        .whitelist_type("^double[0-9]$")
        .whitelist_type("^float[0-9]$")
        .whitelist_type("^u?int[0-9]$")
        .whitelist_type("^u?long[0-9]$")
        .whitelist_type("^u?longlong[0-9]$")
        .whitelist_type("^u?short[0-9]$")
        .default_enum_style(bindgen::EnumVariation::Rust)
        .derive_copy(true)
		.rustfmt_bindings(true)
        .generate()
        .expect("Unable to generate vector types bindings")
        .write_to_file(out_path.join("vector_types_bindings.rs"))
        .expect("Unable to write vector types bindings");
}
