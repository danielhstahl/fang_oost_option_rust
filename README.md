| [Linux][lin-link] |  [Codecov][cov-link]  |
| :---------------: | :-------------------: |
| ![lin-badge]      | ![cov-badge]          |

[lin-badge]: https://travis-ci.org/phillyfan1138/fang_oost_option_rust.svg?branch=master "Travis build status"
[lin-link]:  https://travis-ci.org/phillyfan1138/fang_oost_option_rust "Travis build status"
[cov-badge]: https://codecov.io/gh/phillyfan1138/fang_oost_option_rust/branch/master/graph/badge.svg
[cov-link]:  https://codecov.io/gh/phillyfan1138/fang_oost_option_rust

# Fang-Oosterlee Option Pricing for Rust

Implements [Fang-Oosterlee](https://mpra.ub.uni-muenchen.de/8914/4/MPRA_paper_8914.pdf) option pricing in Rust.  Documentation is at [docs.rs](https://docs.rs/fang_oost_option/0.21.5/fang_oost_option/)

## Use

Put the following in your Cargo.toml:

```toml
[dependencies]
fang_oost_option = "0.21"
```

Import and use:

```rust
extern crate num_complex;
use num_complex::Complex;
extern crate fang_oost_option;
use fang_oost_option::option_pricing;
let num_u:usize = 256;
let asset = 50.0;
let strikes = vec![5000.0, 75.0, 50.0, 40.0, 0.03];
let rate = 0.03;
let t_maturity = 0.5;
let volatility:f64 = 0.3; 
//As an example, cf is standard diffusion
let cf = |u: &Complex<f64>| {
    ((rate-volatility*volatility*0.5)*t_maturity*u+volatility*volatility*t_maturity*u*u*0.5).exp()
};
let prices = option_pricing::fang_oost_call_price(
    num_u, asset, &strikes, 
    rate, t_maturity, &cf
);
```


# Speed

The benchmarks are comparable to my [C++](https://github.com/phillyfan1138/FangOost) implementation.  To run the tests with benchmarking, use `cargo test --release -- --nocapture`.  To run the rust internal benchmark, switch to nightly rust (`rustup default nightly`) and run `cargo bench`.  