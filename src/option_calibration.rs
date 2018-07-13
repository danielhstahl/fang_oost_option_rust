extern crate num;
extern crate num_complex;
extern crate black_scholes;
extern crate rayon;
extern crate fang_oost;
#[cfg(test)]
extern crate statrs;
#[cfg(test)]
extern crate special;

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

fn DFT<'a, 'b: 'a>(
    u_array:&'b Vec<f64>,
    x_min:f64,
    x_max:f64,
    n:usize,
    fn_to_invert:impl Fn( f64, usize)->Complex<f64>+'a+std::marker::Sync+std::marker::Send
)->impl ParallelIterator<Item = Complex<f64> >+'a
{
    let cmp:Complex<f64>=Complex::new(0.0, 0.0);
    let dx=get_dx(n, x_min, x_max);
    u_array.par_iter().map(move |u|{
        (0..n).fold(cmp, |accum, index|{
            let simpson=if index==0||index==(n-1) { 1.0} else { if index%2==0 {2.0} else {4.0} };
            let x=x_min+dx*(index as f64);
            (cmp*u*x).exp()*fn_to_invert(x, index)*dx/3.0
        })
    })
}

fn transform_price(p:f64, v:f64)->f64{p/v}

fn transform_prices(
    arr:&Vec<(f64, f64)>, asset:f64, 
    min_v:&(f64, f64), max_v:&(f64, f64)
)->Vec<(f64, f64)>{
    let mut price_t:Vec<f64>=vec![];
    let (min_strike, min_option_price)=min_v;
    let (max_strike, max_option_price)=max_v;
    price_t.push((transform_price(min_strike, asset), transform_price(min_option_price, asset)));
    price_t.append(
        &mut arr.iter().enumerate().map(|(index, (strike, option_price))|{
            (transform_price(strike, asset), transform_price(option_price, asset))
        }).collect()
    );
    price_t.push((transform_price(max_strike, asset), transform_price(max_option_price, asset)));
    price_t
}
fn threshold_condition(strike:f64, threshold:f64)->bool{strike<threshold}

fn get_option_spline<'a>(
    strikes_and_option_prices:&Vec<(f64, f64)>,
    stock:f64,
    discount:f64,
    min_strike:f64,
    max_strike:f64
)->impl Fn(f64) -> f64 +'a 
{
    let min_option=stock-min_strike*discount;
    let max_option=0.00000001; //essentially zero
    let padded_strikes_and_option_prices=transform_prices(
        &strikes_and_option_prices, stock, 
        &(min_strike, min_option), 
        &(max_strike, max_option)
    );
    let threshold=padded_strikes_and_option_prices.iter().rev().find(|(strike, _)|strike<1.0).unwrap();
    let (left, right)=padded_strikes_and_option_prices.into_iter().partition(|(strike, _)|threshold_condition(strike, threshold));
    let left_transform=left.into_iter().map(|(strike, price)|(strike, price-max_zero_or_number(1.0-strike*discount)));
    let right_transform=right.into_iter().map(|(strike, price)|(strike, price.ln()));
    let s_low=spline(&left_transform);
    let s_high=spline(&right_transform);
    move |strike:f64|{
        if threshold_condition(strike, threshold) {s_low(strike)} else { s_high(strike).exp()-max_zero_or_number(1.0-strike*discount)}
    }
}

