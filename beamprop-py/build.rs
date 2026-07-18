fn main() {
    // On macOS, Python extension modules must be linked with
    // `-undefined dynamic_lookup` (symbols resolve against the embedding
    // interpreter at import time); this emits the right flags per platform.
    pyo3_build_config::add_extension_module_link_args();
}
