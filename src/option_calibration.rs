extern crate num;
extern crate num_complex;
extern crate black_scholes;
extern crate rayon;
extern crate fang_oost;
#[cfg(test)]
extern crate statrs;
#[cfg(test)]
extern crate special;
#[cfg(test)]
use std::f64::consts::PI;

use self::num_complex::Complex;
use self::rayon::prelude::*;
use std;

fn max_zero_or_number(num:f64)->f64{
    if num>0.0 {num} else {0.0}
}

fn get_du(n: usize, u_max:f64)->f64{
    2.0*u_max/(n as f64)
}

fn get_dx(n:usize, x_min:f64, x_max:f64)->f64{
    (x_max-x_min)/(n as f64-1.0)
}
fn get_u_max(n:usize, x_min:f64, x_max:f64)->f64{
    PI*(n as f64-1.0)/(x_max-x_min)
}

