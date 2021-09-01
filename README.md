| [Linux][lin-link] |  [Codecov][cov-link]  |
| :---------------: | :-------------------: |
| ![lin-badge]      | ![cov-badge]          |

[lin-badge]: https://github.com/danielhstahl/fang_oost_option_rust/workflows/Rust/badge.svg
[lin-link]:  https://github.com/danielhstahl/fang_oost_option_rust/actions
[cov-badge]: https://codecov.io/gh/danielhstahl/fang_oost_option_rust/branch/master/graph/badge.svg
[cov-link]:  https://codecov.io/gh/danielhstahl/fang_oost_option_rust

# Fang-Oosterlee Option Pricing for Rust

Implements [Fang-Oosterlee](https://mpra.ub.uni-muenchen.de/8914/4/MPRA_paper_8914.pdf) option pricing in Rust.  Documentation is at [docs.rs](https://docs.rs/fang_oost_option/)

## Use

The crate is available on [crates.io](https://crates.io/crates/fang_oost_option).

Import and use:

```rust
extern crate num_complex;
use num_complex::Complex;
extern crate fang_oost_option;
use rayon::prelude::*;
use fang_oost_option::option_pricing;
let num_u:usize = 256;
let asset = 50.0;
let strikes = vec![75.0, 50.0, 40.0];
// max_strike sets the domain of the empirical estimate.  
// This should be large enough to capture the potential
// dynamics of the underlying, but not too large or accuracy
// will sacrificed.  A good rule of thumb is to scale this
// in proportion to the volatility of the underlying.  For
// example, if the underlying is 50.0 and has a (log) 
// volatility of 0.3, then a good max strike would be
// exp(0.3*scale)*50.0.  I tend to use scale=10, yielding
// in this example ~1004.
let max_strike = 1004.0; 
let rate = 0.03;
let t_maturity = 0.5;
let volatility:f64 = 0.3; 
//As an example, cf is standard diffusion
let cf = |u: &Complex<f64>| {
    ((rate-volatility*volatility*0.5)*t_maturity*u+volatility*volatility*t_maturity*u*u*0.5).exp()
};
let prices: Vec<fang_oost::GraphElement>= option_pricing::fang_oost_call_price(
    num_u, asset, &strikes, max_strike,
    rate, t_maturity, &cf
).collect();
```


# Speed

The benchmarks are comparable to my [C++](https://github.com/danielhstahl/FangOost) implementation.  To run the tests with benchmarking, use `cargo bench`.  You can see the benchmarks at https://danielhstahl.github.io/fang_oost_option_rust/report/index.html.
