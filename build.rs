use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=csrc/");
    
    // Get the target
    let target = env::var("TARGET").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    
    println!("cargo:rustc-env=TARGET={}", target);
    println!("cargo:rustc-env=TARGET_OS={}", target_os);
    println!("cargo:rustc-env=TARGET_ARCH={}", target_arch);
    
    // Build configuration
    let profile = env::var("PROFILE").unwrap();
    let is_release = profile == "release";
    
    // Set optimization flags based on profile
    if is_release {
        println!("cargo:rustc-cfg=optimized");
    }
    
    // Handle feature flags
    #[cfg(feature = "cuda")]
    build_cuda_support();
    
    #[cfg(feature = "python")]
    build_python_bindings();
    
    #[cfg(feature = "wasm")]
    build_wasm_support();
    
    // Build C/C++ components
    build_c_components();
    
    // Generate bindings if needed
    #[cfg(feature = "codegen")]
    generate_bindings();
    
    // Platform-specific configurations
    configure_platform_specific(&target_os, &target_arch);
    
    // Set up linking
    setup_linking(&target_os);
}

fn build_c_components() {
    let mut build = cc::Build::new();
    
    // Add source files
    let c_sources = [
        "csrc/neural_unit.c",
        "csrc/membrane_dynamics.c", 
        "csrc/oscillations.c",
        "csrc/quantum_transport.c",
    ];
    
    for source in &c_sources {
        if std::path::Path::new(source).exists() {
            build.file(source);
        }
    }
    
    // Compiler flags
    build
        .include("csrc/include")
        .flag("-std=c11")
        .flag("-Wall")
        .flag("-Wextra")
        .flag("-O3")
        .flag("-march=native")
        .flag("-ffast-math")
        .define("IMHOTEP_VERSION", "\"0.1.0\"");
    
    // Platform-specific flags
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    match target_os.as_str() {
        "linux" => {
            build.flag("-fPIC");
            build.flag("-pthread");
        }
        "macos" => {
            build.flag("-fPIC");
        }
        "windows" => {
            // Windows-specific flags
        }
        _ => {}
    }
    
    // Debug vs Release
    let profile = env::var("PROFILE").unwrap();
    if profile == "debug" {
        build.flag("-g").flag("-DDEBUG");
    } else {
        build.flag("-DNDEBUG").flag("-flto");
    }
    
    build.compile("imhotep_c");
}

#[cfg(feature = "cuda")]
fn build_cuda_support() {
    println!("cargo:rustc-cfg=cuda_enabled");
    
    // Check for CUDA installation
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    
    let cuda_lib_path = format!("{}/lib64", cuda_path);
    let cuda_include_path = format!("{}/include", cuda_path);
    
    println!("cargo:rustc-link-search=native={}", cuda_lib_path);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=curand");
    
    // Build CUDA kernels if nvcc is available
    if which::which("nvcc").is_ok() {
        let cuda_sources = [
            "cuda/neural_kernels.cu",
            "cuda/membrane_kernels.cu",
            "cuda/quantum_kernels.cu",
        ];
        
        for source in &cuda_sources {
            if std::path::Path::new(source).exists() {
                let output = std::process::Command::new("nvcc")
                    .arg("-c")
                    .arg(source)
                    .arg("-o")
                    .arg(format!("{}.o", source))
                    .arg("--compiler-options=-fPIC")
                    .arg("-arch=sm_70")
                    .arg("-O3")
                    .arg(format!("-I{}", cuda_include_path))
                    .output()
                    .expect("Failed to compile CUDA kernel");
                
                if !output.status.success() {
                    panic!("CUDA compilation failed: {}", String::from_utf8_lossy(&output.stderr));
                }
            }
        }
    }
}

#[cfg(feature = "python")]
fn build_python_bindings() {
    println!("cargo:rustc-cfg=python_enabled");
    
    // Python-specific configuration handled by PyO3
    // This is mostly for additional C extensions if needed
}

#[cfg(feature = "wasm")]
fn build_wasm_support() {
    println!("cargo:rustc-cfg=wasm_enabled");
    
    // WebAssembly-specific optimizations
    let target = env::var("TARGET").unwrap();
    if target.contains("wasm32") {
        println!("cargo:rustc-link-arg=--export-dynamic");
        println!("cargo:rustc-link-arg=--no-entry");
    }
}

#[cfg(feature = "codegen")]
fn generate_bindings() {
    use std::env;
    use std::path::PathBuf;
    
    // Generate bindings for C headers
    let bindings = bindgen::Builder::default()
        .header("csrc/include/imhotep.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");
    
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn configure_platform_specific(target_os: &str, target_arch: &str) {
    match target_os {
        "linux" => {
            println!("cargo:rustc-cfg=platform_linux");
            configure_linux_specific();
        }
        "macos" => {
            println!("cargo:rustc-cfg=platform_macos");
            configure_macos_specific();
        }
        "windows" => {
            println!("cargo:rustc-cfg=platform_windows");
            configure_windows_specific();
        }
        _ => {}
    }
    
    match target_arch {
        "x86_64" => {
            println!("cargo:rustc-cfg=arch_x86_64");
            // Enable SIMD optimizations
            println!("cargo:rustc-cfg=simd_avx2");
            println!("cargo:rustc-cfg=simd_sse4");
        }
        "aarch64" => {
            println!("cargo:rustc-cfg=arch_aarch64");
            println!("cargo:rustc-cfg=simd_neon");
        }
        _ => {}
    }
}

fn configure_linux_specific() {
    // Linux-specific libraries
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");
    
    // Check for specific Linux features
    if std::path::Path::new("/proc/cpuinfo").exists() {
        println!("cargo:rustc-cfg=has_cpuinfo");
    }
}

fn configure_macos_specific() {
    // macOS-specific frameworks
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    
    // Enable Metal compute shaders if available
    println!("cargo:rustc-cfg=metal_compute");
}

fn configure_windows_specific() {
    // Windows-specific libraries
    println!("cargo:rustc-link-lib=kernel32");
    println!("cargo:rustc-link-lib=user32");
    
    // Enable Windows-specific optimizations
    println!("cargo:rustc-cfg=windows_simd");
}

fn setup_linking(target_os: &str) {
    // Set up library search paths
    if let Ok(library_path) = env::var("IMHOTEP_LIBRARY_PATH") {
        println!("cargo:rustc-link-search=native={}", library_path);
    }
    
    // Platform-specific library paths
    match target_os {
        "linux" => {
            println!("cargo:rustc-link-search=native=/usr/local/lib");
            println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
        }
        "macos" => {
            println!("cargo:rustc-link-search=native=/usr/local/lib");
            println!("cargo:rustc-link-search=native=/opt/homebrew/lib");
        }
        "windows" => {
            // Windows library paths handled differently
        }
        _ => {}
    }
}

// Helper function to check if a package is available
fn pkg_config_available(package: &str) -> bool {
    std::process::Command::new("pkg-config")
        .arg("--exists")
        .arg(package)
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

// Set up OpenMP if available
fn setup_openmp() {
    if pkg_config_available("openmp") {
        println!("cargo:rustc-link-lib=gomp");
        println!("cargo:rustc-cfg=openmp_available");
    }
}

// Set up BLAS/LAPACK if available
fn setup_blas_lapack() {
    // Try OpenBLAS first
    if pkg_config_available("openblas") {
        println!("cargo:rustc-link-lib=openblas");
        println!("cargo:rustc-cfg=openblas_available");
        return;
    }
    
    // Try Intel MKL
    if pkg_config_available("mkl") {
        println!("cargo:rustc-link-lib=mkl_rt");
        println!("cargo:rustc-cfg=mkl_available");
        return;
    }
    
    // Fall back to system BLAS/LAPACK
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    match target_os.as_str() {
        "linux" => {
            println!("cargo:rustc-link-lib=blas");
            println!("cargo:rustc-link-lib=lapack");
        }
        "macos" => {
            // Use Accelerate framework (already linked above)
        }
        _ => {}
    }
} 