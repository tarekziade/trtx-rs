//! Raw FFI bindings to NVIDIA TensorRT-RTX
//!
//! This crate provides low-level, unsafe bindings to the TensorRT-RTX C++ library.
//! For safe, ergonomic Rust API, use the `trtx` crate instead.
//!
//! # Safety
//!
//! All functions in this crate are `unsafe` as they directly call into C++ code
//! and perform no safety checks. Callers must ensure:
//!
//! - Pointers are valid and properly aligned
//! - Lifetimes are managed correctly
//! - Thread safety requirements are met
//! - CUDA context is properly initialized

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

// Include the generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        // Verify error codes are defined
        assert_eq!(TRTX_SUCCESS, 0);
        assert_ne!(TRTX_ERROR_INVALID_ARGUMENT, TRTX_SUCCESS);
    }
}
