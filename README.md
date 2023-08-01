Cloverleaf
----


Install
---

1. Setup a new python virtualenv
2. pip install maturin numpy
3. Ensure you have the latest Rust compiler
4. RUSTFLAGS="-C target-cpu=native" maturin develop --release
5. Profit!

TODO
---

1. Lots of documentation still needed
2. Lots of tests still needed :)
3. Build examples for each of the methods currently available
4. Algos: Power Iteration based PageRank optimizer
