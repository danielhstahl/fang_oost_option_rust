//! Solves the inverse problem: find the parameters which most closely
//! appoximate the option prices available in the market.  Requires
//! specification of a characeteristic function. Some useful
//! characteristic functions are provided in the
//! [cf_functions](https://crates.io/crates/cf_functions) repository.
//! This module works by fitting a monotonic spline to transformed
//! option data from the market.  Then the empirical characteristic
//! function is estimated from the spline.  A mean squared optimization
//! problem is then solved in complex space between the analytical
//! characteristic function and the empirical characteristic function.
//! For more documentation and results, see [fang_oost_cal_charts](https://github.com/phillyfan1138/fang_oost_cal_charts).  Currently this
//! module only works on a single maturity at atime.  It does not
//! calibrate across all maturities simultanously.  
//!

#[cfg(test)]
use std::f64::consts::PI;

use crate::monotone_spline;
use num_complex::Complex;
use rayon::prelude::*;
use serde_derive::{Deserialize, Serialize};
use std;

pub fn max_zero_or_number(num: f64) -> f64 {
    if num > 0.0 {
        num
    } else {
        0.0
    }
}

fn get_dx(n: usize, x_min: f64, x_max: f64) -> f64 {
    (x_max - x_min) / (n as f64 - 1.0)
}

fn simpson_integrand(index: usize, n: usize) -> f64 {
    if index == 0 || index == (n - 1) {
        1.0
    } else {
        if index % 2 == 0 {
            2.0
        } else {
            4.0
        }
    }
}

fn dft<'a, 'b: 'a>(
    u_array: &'b [f64],
    x_min: f64,
    x_max: f64,
    n: usize,
    fn_to_invert: impl Fn(f64, usize) -> f64 + 'a + std::marker::Sync + std::marker::Send,
) -> impl IndexedParallelIterator<Item = (f64, Complex<f64>)> + 'a {
    let cmp_i: Complex<f64> = Complex::new(0.0, 1.0);
    let dx = get_dx(n, x_min, x_max);
    u_array.par_iter().map(move |u| {
        (
            *u,
            (0..n)
                .map(|index| {
                    let simpson = simpson_integrand(index, n);
                    let x = x_min + dx * (index as f64);
                    (cmp_i * u * x).exp() * fn_to_invert(x, index) * simpson * dx / 3.0
                })
                .sum(),
        )
    })
}
#[derive(Serialize, Deserialize)]
pub struct OptionStats {
    pub price: f64,
    pub strike: f64,
}

const NORMALIZED_STRIKE_THRESHOLD: f64 = 1.0;

/// Returns scaled prices
///
/// # Examples
///
/// ```
/// extern crate fang_oost_option;
/// use fang_oost_option::option_calibration;
/// # fn main() {
/// let p = 5.0; //option price or strike
/// let v = 50.0; //asset price
/// let t_p = option_calibration::transform_price(p, v);
/// # }
/// ```
pub fn transform_price(p: f64, v: f64) -> f64 {
    p / v
}

/// Returns transformed strikes.  Used to transform the option prices for spline fitting.
///
/// # Examples
///
/// ```
/// extern crate fang_oost_option;
/// use fang_oost_option::option_calibration;
/// # fn main() {
/// let normalized_strike = 0.5;
/// let discount = 0.99; //discount factor
/// let adjustment = option_calibration::adjust_domain(normalized_strike, discount);
/// # }
/// ```
pub fn adjust_domain(normalized_strike: f64, discount: f64) -> f64 {
    max_zero_or_number(NORMALIZED_STRIKE_THRESHOLD - normalized_strike * discount)
}

fn transform_prices(
    arr: &[OptionStats],
    asset: f64,
    min_v: &(f64, f64),
    max_v: &(f64, f64),
) -> Vec<(f64, f64)> {
    let mut price_t: Vec<(f64, f64)> = vec![];
    let (min_strike, min_option_price) = min_v;
    let (max_strike, max_option_price) = max_v;
    price_t.push((
        transform_price(*min_strike, asset),
        transform_price(*min_option_price, asset),
    ));
    price_t.append(
        &mut arr
            .iter()
            .map(|OptionStats { strike, price, .. }| {
                (
                    transform_price(*strike, asset),
                    transform_price(*price, asset),
                )
            })
            .collect(),
    );
    price_t.push((
        transform_price(*max_strike, asset),
        transform_price(*max_option_price, asset),
    ));
    price_t
}
fn threshold_condition(strike: f64, threshold: f64) -> bool {
    strike <= threshold
}

/// Returns spline function
///
/// # Examples
///
/// ```
/// extern crate fang_oost_option;
/// use fang_oost_option::option_calibration;
/// # fn main() {
/// //vector of tuple of (strike, option)
/// let strikes_and_options = vec![
///     option_calibration::OptionStats{
///         strike:30.0, price:22.0
///     },
///     option_calibration::OptionStats{
///         strike:50.0, price:4.0
///     },
///     option_calibration::OptionStats{
///         strike:60.0, price:0.5
///     }
/// ];
/// let stock = 50.0;
/// let min_strike = 0.3;
/// let max_strike = 3000.0;
/// let discount = 0.99; //discount factor
/// let spline = option_calibration::get_option_spline(
///     &strikes_and_options,
///     stock,
///     discount,
///     min_strike,
///     max_strike
/// );
/// let estimated_transformed_strike_high = spline(1.2);
/// let estimated_transformed_strike_low = spline(0.8);
/// let estimated_transformed_strike_at_the_money = spline(1.0);
/// # }
/// ```
pub fn get_option_spline<'a>(
    strikes_and_option_prices: &[OptionStats],
    stock: f64,
    discount: f64,
    min_strike: f64,
    max_strike: f64,
) -> impl Fn(f64) -> f64 + 'a {
    let min_option = stock - min_strike * discount;
    let max_option = 0.00000001; //essentially zero
    let padded_strikes_and_option_prices = transform_prices(
        &strikes_and_option_prices,
        stock,
        &(min_strike, min_option),
        &(max_strike, max_option),
    );

    let (left, mut right): (Vec<(f64, f64)>, Vec<(f64, f64)>) = padded_strikes_and_option_prices
        .into_iter()
        .rev() //reverse so I can push back on right to get left threshold
        .partition(|(normalized_strike, _)| normalized_strike <= &NORMALIZED_STRIKE_THRESHOLD);
    let threshold_t = left.first().unwrap().clone(); //clone so I can push into right
    let (threshold, _) = threshold_t;
    right.push(threshold_t);

    let left_transform: Vec<(f64, f64)> = left
        .into_iter()
        .rev()
        .map(|(normalized_strike, normalized_price)| {
            (
                normalized_strike,
                normalized_price - adjust_domain(normalized_strike, discount),
            )
        })
        .collect();

    let right_transform: Vec<(f64, f64)> = right
        .into_iter()
        .rev()
        .map(|(normalized_strike, normalized_price)| (normalized_strike, normalized_price.ln()))
        .collect();
    let s_low = monotone_spline::spline_mov(left_transform);
    let s_high = monotone_spline::spline_mov(right_transform);
    move |normalized_strike: f64| {
        if threshold_condition(normalized_strike, threshold) {
            s_low(normalized_strike)
        } else {
            s_high(normalized_strike).exp() - adjust_domain(normalized_strike, discount)
        }
    }
}

/// Returns iterator over discrete empirical characteristic function
///
/// # Examples
///
/// ```
/// extern crate fang_oost_option;
/// use fang_oost_option::option_calibration;
/// # fn main() {
/// let strikes_and_options = vec![
///     option_calibration::OptionStats{
///         strike:30.0, price:22.0
///     },
///     option_calibration::OptionStats{
///         strike:50.0, price:4.0
///     },
///     option_calibration::OptionStats{
///         strike:60.0, price:0.5
///     }
/// ];
/// let stock = 50.0;
/// let rate = 0.05;
/// let maturity = 0.8;
/// let min_strike = 0.3;
/// let max_strike = 3000.0;
/// let u_array=vec![-1.0, 0.5, 3.0];
/// let cf_estimate = option_calibration::generate_fo_estimate(
///     &strikes_and_options,
///     &u_array,
///     128,
///     stock,
///     rate,
///     maturity,
///     min_strike,
///     max_strike
/// );
/// # }
/// ```
pub fn generate_fo_estimate<'a, 'b: 'a>(
    strikes_and_option_prices: &'b [OptionStats],
    u_array: &'b [f64],
    n: usize,
    stock: f64,
    rate: f64,
    maturity: f64,
    min_strike: f64,
    max_strike: f64,
) -> impl IndexedParallelIterator<Item = Complex<f64>> + 'a {
    let discount = (-maturity * rate).exp();
    let spline = get_option_spline(
        strikes_and_option_prices,
        stock,
        discount,
        min_strike, //transformed internally to min_strike/asset
        max_strike, //transformed internally to max_strike/asset
    );
    let cmp: Complex<f64> = Complex::new(0.0, 1.0);
    let x_min = (discount * transform_price(min_strike, stock)).ln();
    let x_max = (discount * transform_price(max_strike, stock)).ln();

    dft(u_array, x_min, x_max, n, move |x, _| {
        let exp_x = x.exp();
        let strike = exp_x / discount;
        let option_price_t = spline(strike);
        max_zero_or_number(option_price_t)
    })
    .map(move |(u, cf)| {
        let front = u * cmp * (1.0 + u * cmp);
        (1.0 + cf * front).ln()
    })
}
const LARGE_NUMBER: f64 = 500000.0;

/// Returns function which computes the mean squared error
/// between the empirical and analytical characteristic
/// functions for a vector of parameters.
///
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate fang_oost_option;
/// use fang_oost_option::option_calibration;
/// # fn main() {
/// //u_array is the values in the complex domain to
/// //calibrate to (ie, making cf(u_i) and cf_emp(u_i)
/// //close in mean-square).
/// let u_array = vec![
///     -1.0,
///     0.5,
///     3.0
/// ];
/// //same length as u (computed using generate_fo_estimate function)
/// let phi_hat = vec![
///     Complex::new(-1.0, 1.0),
///     Complex::new(0.5, 0.5),
///     Complex::new(3.0, 1.0)    
/// ];
/// //Gaussian (0, params[0]) distribution
/// let cf = |u: &Complex<f64>, maturity:f64, params:&[f64]| (u*u*0.5*params[0].powi(2)).exp();
/// let params=vec![0.0];
/// let maturity=1.0;
/// let mean_square_error = option_calibration::obj_fn_arr(
///     &phi_hat,
///     &u_array,
///     &params,
///     maturity,
///     &cf
/// );
/// # }
/// ```
pub fn obj_fn_arr<'a>(
    phi_hat: &[Complex<f64>], //do we really want to borrow/move this??
    u_array: &[f64],
    params: &[f64],
    maturity: f64,
    cf_fn: &dyn Fn(&Complex<f64>, f64, &[f64]) -> Complex<f64>,
) -> f64 {
    u_array
        .iter()
        .zip(phi_hat.iter())
        .fold(0.0, |accumulate, (u, phi)| {
            let result = cf_fn(&Complex::new(1.0, *u), maturity, params);
            accumulate
                + if result.re.is_nan() || result.im.is_nan() {
                    LARGE_NUMBER
                } else {
                    (phi - result).norm_sqr()
                }
        })
}

#[cfg(test)]
mod tests {
    use crate::option_calibration::*;
    use approx::*;
    #[test]
    fn test_transform_prices() {
        let arr = vec![
            OptionStats {
                price: 3.0,
                strike: 3.0,
            },
            OptionStats {
                price: 4.0,
                strike: 4.0,
            },
            OptionStats {
                price: 5.0,
                strike: 5.0,
            },
        ];
        let asset = 4.0;
        let min_v = (2.0, 2.0);
        let max_v = (6.0, 6.0);
        let result = transform_prices(&arr, asset, &min_v, &max_v);
        let expected = vec![
            (0.5, 0.5),
            (0.75, 0.75),
            (1.0, 1.0),
            (1.25, 1.25),
            (1.5, 1.5),
        ];
        for (index, res) in result.iter().enumerate() {
            assert_eq!(*res, expected[index]);
        }
    }
    #[test]
    fn test_get_obj_one_parameter() {
        let cf = |u: &Complex<f64>, _m: f64, _sl: &[f64]| Complex::new(u.im, 0.0);
        let arr = vec![
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(5.0, 0.0),
        ];
        let u_arr = vec![6.0, 7.0, 8.0];
        let params = vec![0.0];
        let result = obj_fn_arr(&arr, &u_arr, &params, 1.0, &cf);
        let expected = 27.0; //3*3^2
        assert_eq!(result, expected);
    }
    #[test]
    fn test_option_spline() {
        let tmp_strikes_and_option_prices: Vec<OptionStats> = vec![
            OptionStats {
                strike: 95.0,
                price: 85.0,
            },
            OptionStats {
                strike: 130.0,
                price: 51.5,
            },
            OptionStats {
                strike: 150.0,
                price: 35.38,
            },
            OptionStats {
                strike: 160.0,
                price: 28.3,
            },
            OptionStats {
                strike: 165.0,
                price: 25.2,
            },
            OptionStats {
                strike: 170.0,
                price: 22.27,
            },
            OptionStats {
                strike: 175.0,
                price: 19.45,
            },
            OptionStats {
                strike: 185.0,
                price: 14.77,
            },
            OptionStats {
                strike: 190.0,
                price: 12.75,
            },
            OptionStats {
                strike: 195.0,
                price: 11.0,
            },
            OptionStats {
                strike: 200.0,
                price: 9.35,
            },
            OptionStats {
                strike: 210.0,
                price: 6.9,
            },
            OptionStats {
                strike: 240.0,
                price: 2.55,
            },
            OptionStats {
                strike: 250.0,
                price: 1.88,
            },
        ];
        let maturity: f64 = 1.0;
        let rate = 0.05;
        let asset = 178.46;
        let discount = (-rate * maturity).exp();
        let spline = get_option_spline(
            &tmp_strikes_and_option_prices,
            asset,
            discount,
            0.00001,
            5000.0,
        );
        let sp_result = spline(160.0 / asset);
        assert_eq!(
            sp_result,
            28.3 / asset - max_zero_or_number(1.0 - (160.0 / asset) * discount)
        );
    }
    #[test]
    fn test_option_spline_at_many_values() {
        let tmp_strikes_and_option_prices: Vec<OptionStats> = vec![
            OptionStats {
                strike: 95.0,
                price: 85.0,
            },
            OptionStats {
                strike: 130.0,
                price: 51.5,
            },
            OptionStats {
                strike: 150.0,
                price: 35.38,
            },
            OptionStats {
                strike: 160.0,
                price: 28.3,
            },
            OptionStats {
                strike: 165.0,
                price: 25.2,
            },
            OptionStats {
                strike: 170.0,
                price: 22.27,
            },
            OptionStats {
                strike: 175.0,
                price: 19.45,
            },
            OptionStats {
                strike: 185.0,
                price: 14.77,
            },
            OptionStats {
                strike: 190.0,
                price: 12.75,
            },
            OptionStats {
                strike: 195.0,
                price: 11.0,
            },
            OptionStats {
                strike: 200.0,
                price: 9.35,
            },
            OptionStats {
                strike: 210.0,
                price: 6.9,
            },
            OptionStats {
                strike: 240.0,
                price: 2.55,
            },
            OptionStats {
                strike: 250.0,
                price: 1.88,
            },
        ];
        let maturity: f64 = 1.0;
        let rate = 0.05;
        let asset = 178.46;
        let discount = (-rate * maturity).exp();
        let spline = get_option_spline(
            &tmp_strikes_and_option_prices,
            asset,
            discount,
            0.00001,
            5000.0,
        );
        let test_vec = vec![4.0, 100.0, 170.0, 175.0, 178.0, asset, 179.0, 185.0, 500.0];
        test_vec.iter().for_each(|v| {
            let _sp_result = spline(v / asset); //will panic if doesnt work
        });
        tmp_strikes_and_option_prices
            .iter()
            .for_each(|OptionStats { strike, price, .. }| {
                let sp_result = spline(strike / asset);
                assert_abs_diff_eq!(
                    sp_result,
                    price / asset - max_zero_or_number(1.0 - (strike / asset) * discount),
                    epsilon = 0.0000001
                );
            });
    }
    #[test]
    fn test_generate_fo_runs() {
        let tmp_strikes_and_option_prices: Vec<OptionStats> = vec![
            OptionStats {
                strike: 95.0,
                price: 85.0,
            },
            OptionStats {
                strike: 130.0,
                price: 51.5,
            },
            OptionStats {
                strike: 150.0,
                price: 35.38,
            },
            OptionStats {
                strike: 160.0,
                price: 28.3,
            },
            OptionStats {
                strike: 165.0,
                price: 25.2,
            },
            OptionStats {
                strike: 170.0,
                price: 22.27,
            },
            OptionStats {
                strike: 175.0,
                price: 19.45,
            },
            OptionStats {
                strike: 185.0,
                price: 14.77,
            },
            OptionStats {
                strike: 190.0,
                price: 12.75,
            },
            OptionStats {
                strike: 195.0,
                price: 11.0,
            },
            OptionStats {
                strike: 200.0,
                price: 9.35,
            },
            OptionStats {
                strike: 210.0,
                price: 6.9,
            },
            OptionStats {
                strike: 240.0,
                price: 2.55,
            },
            OptionStats {
                strike: 250.0,
                price: 1.88,
            },
        ];
        let maturity: f64 = 1.0;
        let rate = 0.05;
        let asset = 178.46;
        let n: usize = 15;
        let du = 2.0 * PI / (n as f64);
        let u_array: Vec<f64> = (1..n).map(|index| index as f64 * du).collect();
        let _result = generate_fo_estimate(
            &tmp_strikes_and_option_prices,
            &u_array,
            n,
            asset,
            rate,
            maturity,
            0.01,
            5000.0,
        );
        //let _result=hoc_fn(1024, &u_array);
    }
    #[test]
    fn test_generate_fo_accuracy() {
        let tmp_strikes_and_option_prices: Vec<OptionStats> = vec![
            OptionStats {
                strike: 95.0,
                price: 85.0,
            },
            OptionStats {
                strike: 130.0,
                price: 51.5,
            },
            OptionStats {
                strike: 150.0,
                price: 35.38,
            },
            OptionStats {
                strike: 160.0,
                price: 28.3,
            },
            OptionStats {
                strike: 165.0,
                price: 25.2,
            },
            OptionStats {
                strike: 170.0,
                price: 22.27,
            },
            OptionStats {
                strike: 175.0,
                price: 19.45,
            },
            OptionStats {
                strike: 185.0,
                price: 14.77,
            },
            OptionStats {
                strike: 190.0,
                price: 12.75,
            },
            OptionStats {
                strike: 195.0,
                price: 11.0,
            },
            OptionStats {
                strike: 200.0,
                price: 9.35,
            },
            OptionStats {
                strike: 210.0,
                price: 6.9,
            },
            OptionStats {
                strike: 240.0,
                price: 2.55,
            },
            OptionStats {
                strike: 250.0,
                price: 1.88,
            },
        ];
        let maturity: f64 = 1.0;
        let rate = 0.05;
        let asset = 178.46;
        let n: usize = 15;
        let du = 2.0 * PI / (n as f64);
        let u_array: Vec<f64> = (1..n).map(|index| index as f64 * du).collect();
        generate_fo_estimate(
            &tmp_strikes_and_option_prices,
            &u_array,
            n,
            asset,
            rate,
            maturity,
            0.01,
            5000.0,
        )
        .for_each(|v| {
            println!("this is v: {}", v);
        });
    }
    #[test]
    fn test_dft() {
        let u_array = vec![2.0];
        let x_min = -5.0;
        let x_max = 5.0;
        let n: usize = 10;
        let fn_to_invert = |x: f64, _| x.powi(2);
        let result: Vec<Complex<f64>> = dft(&u_array, x_min, x_max, n, fn_to_invert)
            .map(|(_, v)| v)
            .collect();
        assert_abs_diff_eq!(result[0].re, -5.93082, epsilon = 0.00001);
        assert_abs_diff_eq!(result[0].im, -14.3745, epsilon = 0.00001);
    }
    #[test]
    fn test_monotone_spline() {
        let tmp_strikes_and_option_prices: Vec<OptionStats> = vec![
            OptionStats {
                strike: 95.0,
                price: 85.0,
            },
            OptionStats {
                strike: 130.0,
                price: 51.5,
            },
            OptionStats {
                strike: 150.0,
                price: 35.38,
            },
            OptionStats {
                strike: 160.0,
                price: 28.3,
            },
            OptionStats {
                strike: 165.0,
                price: 25.2,
            },
            OptionStats {
                strike: 170.0,
                price: 22.27,
            },
            OptionStats {
                strike: 175.0,
                price: 19.45,
            },
            OptionStats {
                strike: 185.0,
                price: 14.77,
            },
            OptionStats {
                strike: 190.0,
                price: 12.75,
            },
            OptionStats {
                strike: 195.0,
                price: 11.0,
            },
            OptionStats {
                strike: 200.0,
                price: 9.35,
            },
            OptionStats {
                strike: 210.0,
                price: 6.9,
            },
            OptionStats {
                strike: 240.0,
                price: 2.55,
            },
            OptionStats {
                strike: 250.0,
                price: 1.88,
            },
        ];
        let maturity: f64 = 1.0;
        let rate = 0.05;
        let asset = 178.46;
        let discount = (-rate * maturity).exp();
        let spline = get_option_spline(
            &tmp_strikes_and_option_prices,
            asset,
            discount,
            0.00001,
            5000.0,
        );
        let test_vec = vec![
            4.0, 100.0, 170.0, 175.0, 178.0, asset, 179.0, 185.0, 190.0, 195.0, 200.0, 205.0,
            208.0, 209.0, 210.0, 215.0, 218.0, 220.0, 500.0,
        ];

        test_vec.iter().for_each(|v| {
            let sp_result = spline(v / asset); //will panic if doesnt work
            println!("spline at: {}: {}", v, sp_result);
        });
    }

}
