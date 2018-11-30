#![feature(once_is_completed)]

extern crate libloading;

use std::path::{PathBuf, Path, };
use std::sync::Once;
use std::sync::atomic::{AtomicPtr, Ordering, fence };

fn find_with_env<F>(env: &str, f: F) -> Option<PathBuf>
    where F: Fn(&str) -> Option<PathBuf>,
{
    use std::env::var;
    match var(env) {
        Ok(path) => {
            let split_char = if cfg!(target_os = "windows") {
                ";"
            } else {
                ":"
            };

            for path in path.split(split_char) {
                if let Some(p) = f(path) {
                    return Some(p);
                }
            }

            None
        }
        Err(_) => None,
    }
}

fn find_cuda(lib: &str) -> PathBuf {
    use std::env::consts::{DLL_PREFIX, DLL_SUFFIX, };
    // These are my best guess version numbers
    const VERSIONS: &'static [usize] = &[
        100,
        90,
        80,
        70,
    ];
    #[cfg(target_pointer_width = "32")]
    const PTR_SIZE: usize = 32;
    #[cfg(target_pointer_width = "64")]
    const PTR_SIZE: usize = 64;

    #[cfg(windows)]
    const LIB_DIR: &'static str = "bin";
    #[cfg(all(unix, target_pointer_width = "32"))]
    const LIB_DIR: &'static str = "lib";
    #[cfg(all(unix, target_pointer_width = "64"))]
    const LIB_DIR: &'static str = "lib64";

    for &version in VERSIONS.iter() {
        let lib = format!("{}{}{}_{}{}", DLL_PREFIX, lib, PTR_SIZE,
                          version, DLL_SUFFIX);
        let find = |dir: &str| {
            let path = Path::new(dir).join(LIB_DIR).join(&lib);
            if path.is_file() {
                Some(path)
            } else {
                None
            }
        };
        let r = find_with_env("CUDA_PATH", find);
        if let Some(p) = r {
            return p;
        }
        let r = find_with_env("CUDA_LIBRARY_PATH", find);
        if let Some(p) = r {
            return p;
        }

        if let Some(p) = find("/usr/local/cuda") {
            return p;
        }
        if let Some(p) = find("/opt/cuda") {
            return p;
        }
    }

    // fall back to usual search:
    PathBuf::from(format!("{}{}{}", DLL_PREFIX, lib, DLL_SUFFIX))
}

/// A global library which is loaded only once.
pub(crate) struct GlobalLibrary(AtomicPtr<libloading::Library>, Once);
impl GlobalLibrary {
    pub(crate) const fn new() -> Self {
        GlobalLibrary(AtomicPtr::new(0 as *mut _), Once::new())
    }
    pub(crate) fn is_loaded(&self) -> bool {
        let b = unsafe {
            self.0.load(Ordering::Acquire)
                .as_ref()
                .is_some()
        };
        b && self.1.is_completed()
    }
    pub(crate) fn try_loading<F>(&self, lib: PathBuf, f: F) -> libloading::Result<()>
        where F: FnOnce(&libloading::Library),
    {
        if let Some(lib) = unsafe { self.0.load(Ordering::Acquire).as_ref() } {
            // ensure no thread continues until *one* thread initializes all the functions.
            // We don't know which thread will be the one to run `f`, but they should all
            // have the same ptr as returned in the above load.
            self.1.call_once(move || {
                f(lib);

                // ensure the symbol writes are visible to all other threads before unlocking:
                fence(Ordering::SeqCst);
            });

            Ok(())
        } else {
            let lib = libloading::Library::new(lib)?;
            let lib = Box::new(lib);
            let lib_ptr = Box::into_raw(lib);
            let r = self.0.compare_exchange(0 as *mut _, lib_ptr,
                                            Ordering::SeqCst,
                                            Ordering::SeqCst);
            let real_lib_ptr = if let Err(real_lib_ptr) = r {
                unsafe { Box::from_raw(lib_ptr) };
                real_lib_ptr
            } else {
                lib_ptr
            };

            // ensure no thread continues until *one* thread initializes all the functions.
            // We don't know which thread will be the one to run `f`, but they should all
            // have the same ptr as returned in Err(_) in `AtomicPtr::compare_exchange`.
            self.1.call_once(move || {
                f(unsafe {
                    real_lib_ptr.as_ref().unwrap()
                });

                // ensure the symbol writes are visible to all other threads:
                fence(Ordering::SeqCst);
            });

            Ok(())
        }
    }
}
macro_rules! func_symbols {
    ($dylib_name:ident =>
    $(extern "C" {
        $(#[$attr:meta])*
        pub fn $name:ident($($pname:ident: $pty:ty,)*) $(-> $ret:ty)*;
    })*) => {

        pub(crate) fn load_symbols(lib: &libloading::Library) {
            unsafe {
                $(self::$name::SYMBOL = lib.get(concat!(stringify!($name), "\0").as_ref())
                    .map(|v: libloading::Symbol<extern "C" fn($($pty),*) $(-> $ret)*>| v.into_raw() )
                .ok();
                )*
            }
        }

        $(
        pub mod $name {
            #![allow(unused_imports)]

            #[cfg(unix)]
            use libloading::os::unix::Symbol;
            #[cfg(windows)]
            use libloading::os::windows::Symbol;

            use super::*;
            use ::cucomplex::*;
            use ::cuda::*;

            use super::cudaError_t;

            pub(super) static mut SYMBOL: Option<Symbol<extern "C" fn($($pty),*) $(-> $ret)*>> = None;

            pub fn is_loaded() -> bool {
                super::LIBRARY.is_loaded() && unsafe { SYMBOL.is_some() }
            }
        }

        $(#[$attr])*
        pub unsafe fn $name($($pname: $pty),*) $(-> $ret)* {
            (*self::$name::SYMBOL.as_ref()
                .unwrap_or_else(|| {
                  panic!("{} doesn't export {}, or the library isn't loaded", stringify!($dylib_name), stringify!($name));
                }))($($pname),*)
        }
        )*
    };
}
macro_rules! link_funcs {
    ($dylib_name:ident =>
    $(extern "C" {
        $(#[$attr:meta])*
        pub fn $name:ident($($pname:ident: $pty:ty), *) $(-> $ret:ty)*;
    })*) => {
        // normalize commas.
        link_funcs!{
            $dylib_name =>

            $(extern "C" {
                $(#[$attr])*
                pub fn $name($($pname: $pty,) *) $(-> $ret)*;
            })*
        }
    };
    (cuda =>
    $(extern "C" {
        $(#[$attr:meta])*
        pub fn $name:ident($($pname:ident: $pty:ty,)*) $(-> $ret:ty)*;
    })*) => {
        func_symbols!(cudart => $(
            extern "C" {
                $(#[$attr])*
                pub fn $name($($pname: $pty,)*) $(-> $ret)*;
            }
        )*);
        /// Blocks until some thread (this thread if ie called during application startup)
        /// finishes loading all the function ptrs.
        pub fn ensure_loaded() -> ::std::io::Result<()> {
             // on windows, cuda is called nvcuda and is in C:\Windows\..
             #[cfg(windows)]
             fn cuda_path() -> ::std::path::PathBuf {
                 ::std::path::PathBuf::from("nvcuda.dll")
             }
             #[cfg(not(windows))]
             fn cuda_path() -> ::std::path::PathBuf {
                 ::find_cuda("cuda")
             }
             LIBRARY.try_loading(cuda_path(), self::load_symbols)?;
             Ok(())
        }
    };
    ($dylib_name:ident =>
    $(extern "C" {
        $(#[$attr:meta])*
        pub fn $name:ident($($pname:ident: $pty:ty,)*) $(-> $ret:ty)*;
    })*) => {
        func_symbols!($dylib_name => $(
            extern "C" {
                $(#[$attr])*
                pub fn $name($($pname: $pty,)*) $(-> $ret)*;
            }
        )*);
        /// Loads the CUDA module. Blocks until some thread (this thread if ie
        /// called during application startup) finishes loading all the function ptrs.
        pub fn ensure_loaded() -> ::std::io::Result<()> {
             let lib = ::find_cuda(stringify!($dylib_name));
             LIBRARY.try_loading(lib, self::load_symbols)?;
             Ok(())
        }
    };
}

pub mod cublas;
pub mod cucomplex;
pub mod cuda;
pub mod cudart;
pub mod driver_types;
pub mod library_types;
pub mod vector_types;

#[test]
fn cuda_version() {
    cuda::ensure_loaded().expect("failed to load cuda driver");
    let mut d_ver = 0;
    unsafe {
        cuda::cuDriverGetVersion(&mut d_ver as *mut i32);
    }
    println!("driver version = {}", d_ver);
}
