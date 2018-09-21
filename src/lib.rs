#[macro_use]
#[cfg(test)]
extern crate approx;

#[macro_use]
extern crate serde_json;
#[macro_use]
extern crate serde_derive;
pub mod option_pricing;
pub mod monotone_spline;
pub mod option_calibration;