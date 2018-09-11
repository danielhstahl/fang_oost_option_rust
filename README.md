| [Linux][lin-link] |  [Codecov][cov-link]  |
| :---------------: | :-------------------: |
| ![lin-badge]      | ![cov-badge]          |

[lin-badge]: https://travis-ci.org/phillyfan1138/fang_oost_option_rust.svg?branch=master "Travis build status"
[lin-link]:  https://travis-ci.org/phillyfan1138/fang_oost_option_rust "Travis build status"
[cov-badge]: https://codecov.io/gh/phillyfan1138/fang_oost_option_rust/branch/master/graph/badge.svg
[cov-link]:  https://codecov.io/gh/phillyfan1138/fang_oost_option_rust

# Fang-Oosterlee Option Pricing for Rust

Implements Fang-Oosterlee option pricing in Rust

# Speed

The benchmarks are comparable to my C++ implementation.  The tests ran in 8 to 50 milliseconds in rust compared to 17-49 milliseconds in C++.  To run the tests with benchmarking, use `cargo test --release -- --nocapture`.

To run raw benchmark, checkout the `benchmark` branch and run `cargo bench`.  Note that this only works with nightly (switch using `rustup default nightly`).