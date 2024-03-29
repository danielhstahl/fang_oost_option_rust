//! Fang Oosterlee approach for an option using the underlying's
//! characteristic function. Some useful characteristic functions
//! are provided in the [cf_functions](https://crates.io/crates/cf_functions) repository.
//! Fang and Oosterlee's approach works well for a smaller set of
//! discrete strike prices such as those in the market.  The
//! constraint is that the smallest and largest values in the x
//! domain must be relatively far from the middle values.  This
//! can be "simulated" by adding small and large strikes
//! synthetically.  Due to the fact that Fang Oosterlee is able to
//! handle discrete strikes well, the algorithm takes a vector of
//! strike prices with no requirement that the strike prices be
//! equidistant.  All that is required is that they are sorted largest
//! to smallest. [Link to Fang-Oosterlee paper](http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/COS.pdf).
//!

use num_complex::Complex;
use rayon::prelude::*;
use std;

/**For Fang Oost (defined in the paper)*/
pub(crate) fn chi_k(a: f64, c: f64, d: f64, u: f64) -> f64 {
    let iter_s = |x| u * (x - a);
    let exp_d = d.exp();
    let exp_c = c.exp();
    (iter_s(d).cos() * exp_d - iter_s(c).cos() * exp_c + u * iter_s(d).sin() * exp_d
        - u * iter_s(c).sin() * exp_c)
        / (1.0 + u * u)
}

pub(crate) fn phi_k(a: f64, c: f64, d: f64, u: f64, k: usize) -> f64 {
    let iter_s = |x| u * (x - a);
    if k == 0 {
        d - c
    } else {
        (iter_s(d).sin() - iter_s(c).sin()) / u
    }
}

pub(crate) fn get_x_from_k(asset: f64, strike: f64) -> f64 {
    (asset / strike).ln()
}
/**This function takes strikes and converts them
into a vector in the x domain.  Intriguinely, I
don't have to sort the result...*/
fn get_x_from_k_iterator<'a, 'b: 'a>(
    asset: f64,
    strikes: &'b [f64],
) -> impl IndexedParallelIterator<Item = f64> + 'a {
    strikes
        .par_iter()
        .map(move |&strike| get_x_from_k(asset, strike))
}

pub(crate) fn option_price_transform(cf: &Complex<f64>) -> Complex<f64> {
    *cf
}

fn option_delta_transform(cf: &Complex<f64>, u: &Complex<f64>) -> Complex<f64> {
    cf * u
}

fn option_gamma_transform(cf: &Complex<f64>, u: &Complex<f64>) -> Complex<f64> {
    -cf * u * (1.0 - u)
}

fn option_theta_transform(cf: &Complex<f64>, rate: f64) -> Complex<f64> {
    if cf.re > 0.0 {
        -(cf.ln() - rate) * cf
    } else {
        Complex::new(0.0, 0.0)
    }
}

pub(crate) fn get_x_range(asset: f64, max_strike: f64) -> (f64, f64) {
    let min_strike = asset.powi(2) / max_strike;
    let x_max = get_x_from_k(asset, min_strike); //x and k are inversely related
    let x_min = get_x_from_k(asset, max_strike); //x and k are inversely related
    (x_min, x_max)
}
pub(crate) fn fang_oost_discrete_cf<'a, S, T>(
    num_u: usize,
    x_min: f64,
    x_max: f64,
    enh_cf: T,
    cf: S,
) -> Vec<Complex<f64>>
where
    T: Fn(&Complex<f64>, &Complex<f64>) -> Complex<f64>
        + std::marker::Sync
        + std::marker::Send
        + 'a,
    S: Fn(&Complex<f64>) -> Complex<f64> + std::marker::Sync + std::marker::Send + 'a,
{
    fang_oost::get_discrete_cf(num_u, x_min, x_max, |u| enh_cf(&cf(u), u))
}

fn fang_oost_generic_move<'a, U>(
    asset: f64,
    strikes: &'a [f64],
    x_min: f64,
    x_max: f64,
    discrete_cf: Vec<Complex<f64>>,
    m_output: U,
) -> impl IndexedParallelIterator<Item = fang_oost::GraphElement> + 'a
where
    U: Fn(f64, f64) -> f64 + std::marker::Sync + std::marker::Send + 'a,
{
    fang_oost::get_expectation_extended_move(
        x_min,
        x_max,
        get_x_from_k_iterator(asset, strikes),
        discrete_cf,
        move |u, _, k| phi_k(x_min, x_min, 0.0, u, k) - chi_k(x_min, x_min, 0.0, u),
    )
    .zip(strikes)
    .map(move |(result, strike)| fang_oost::GraphElement {
        value: m_output(result.value, *strike),
        x: *strike,
    })
}
/// Returns call prices for the series of strikes
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// use rayon::prelude::*;
/// extern crate fang_oost_option;
/// use fang_oost_option::option_pricing;
/// # fn main() {
/// let num_u:usize = 256;
/// let asset = 50.0;
/// let strikes = vec![75.0, 50.0, 40.0];
/// let rate = 0.03;
/// let t_maturity = 0.5;
/// let volatility:f64 = 0.3;
/// let max_strike = 5000.0; //needs to be "large enough" to integrate over space
/// //As an example, cf is standard diffusion
/// let cf = |u: &Complex<f64>| {
///     ((rate-volatility*volatility*0.5)*t_maturity*u+volatility*volatility*t_maturity*u*u*0.5).exp()
/// };
/// let prices:Vec<fang_oost::GraphElement> = option_pricing::fang_oost_call_price(
///     num_u, asset, &strikes, max_strike,
///     rate, t_maturity, &cf
/// ).collect();
/// # }
/// ```
pub fn fang_oost_call_price<'a, S>(
    num_u: usize,
    asset: f64,
    strikes: &'a [f64],
    max_strike: f64,
    rate: f64,
    t_maturity: f64,
    cf: S,
) -> impl IndexedParallelIterator<Item = fang_oost::GraphElement> + 'a
where
    S: Fn(&Complex<f64>) -> Complex<f64> + std::marker::Sync + std::marker::Send,
{
    let discount = (-rate * t_maturity).exp();
    let (x_min, x_max) = get_x_range(asset, max_strike);
    let cf_discrete = fang_oost_discrete_cf(
        num_u,
        x_min,
        x_max,
        |cfu, _| option_price_transform(&cfu),
        cf,
    );

    fang_oost_generic_move(
        asset,
        strikes,
        x_min,
        x_max,
        cf_discrete,
        move |val, strike| (val - 1.0) * discount * strike + asset,
    )
}

/// Returns put prices for the series of strikes
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// use rayon::prelude::*;
/// extern crate fang_oost_option;
/// use fang_oost_option::option_pricing;
/// # fn main() {
/// let num_u:usize = 256;
/// let asset = 50.0;
/// let strikes = vec![75.0, 50.0, 40.0];
/// let rate = 0.03;
/// let t_maturity = 0.5;
/// let volatility:f64 = 0.3;
/// let max_strike = 5000.0; //needs to be "large enough" to integrate over space
/// //As an example, cf is standard diffusion
/// let cf = |u: &Complex<f64>| {
///     ((rate-volatility*volatility*0.5)*t_maturity*u+volatility*volatility*t_maturity*u*u*0.5).exp()
/// };
/// let prices: Vec<fang_oost::GraphElement> = option_pricing::fang_oost_put_price(
///     num_u, asset, &strikes, max_strike,
///     rate, t_maturity, &cf
/// ).collect();
/// # }
/// ```
pub fn fang_oost_put_price<'a, S>(
    num_u: usize,
    asset: f64,
    strikes: &'a [f64],
    max_strike: f64,
    rate: f64,
    t_maturity: f64,
    cf: S,
) -> impl IndexedParallelIterator<Item = fang_oost::GraphElement> + 'a
where
    S: Fn(&Complex<f64>) -> Complex<f64> + std::marker::Sync + std::marker::Send,
{
    let discount = (-rate * t_maturity).exp();
    let (x_min, x_max) = get_x_range(asset, max_strike);
    let cf_discrete = fang_oost_discrete_cf(
        num_u,
        x_min,
        x_max,
        |cfu, _| option_price_transform(&cfu),
        cf,
    );
    fang_oost_generic_move(
        asset,
        strikes,
        x_min,
        x_max,
        cf_discrete,
        move |val, strike| val * discount * strike,
    )
}
/// Returns delta of a call for the series of strikes
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// use rayon::prelude::*;
/// extern crate fang_oost_option;
/// use fang_oost_option::option_pricing;
/// # fn main() {
/// let num_u:usize = 256;
/// let asset = 50.0;
/// let strikes = vec![75.0, 50.0, 40.0];
/// let rate = 0.03;
/// let t_maturity = 0.5;
/// let volatility:f64 = 0.3;
/// let max_strike = 5000.0; //needs to be "large enough" to integrate over space
/// //As an example, cf is standard diffusion
/// let cf = |u: &Complex<f64>| {
///     ((rate-volatility*volatility*0.5)*t_maturity*u+volatility*volatility*t_maturity*u*u*0.5).exp()
/// };
/// let deltas: Vec<fang_oost::GraphElement> = option_pricing::fang_oost_call_delta(
///     num_u, asset, &strikes, max_strike,
///     rate, t_maturity, &cf
/// ).collect();
/// # }
/// ```
pub fn fang_oost_call_delta<'a, S>(
    num_u: usize,
    asset: f64,
    strikes: &'a [f64],
    max_strike: f64,
    rate: f64,
    t_maturity: f64,
    cf: S,
) -> impl IndexedParallelIterator<Item = fang_oost::GraphElement> + 'a
where
    S: Fn(&Complex<f64>) -> Complex<f64> + std::marker::Sync + std::marker::Send,
{
    let discount = (-rate * t_maturity).exp();
    let (x_min, x_max) = get_x_range(asset, max_strike);
    let cf_discrete = fang_oost_discrete_cf(
        num_u,
        x_min,
        x_max,
        |cfu, u| option_delta_transform(&cfu, &u),
        cf,
    );
    fang_oost_generic_move(
        asset,
        strikes,
        x_min,
        x_max,
        cf_discrete,
        move |val, strike| val * discount * strike / asset + 1.0,
    )
}
/// Returns delta of a put for the series of strikes
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// use rayon::prelude::*;
/// extern crate fang_oost_option;
/// use fang_oost_option::option_pricing;
/// # fn main() {
/// let num_u:usize = 256;
/// let asset = 50.0;
/// let strikes = vec![75.0, 50.0, 40.0];
/// let rate = 0.03;
/// let t_maturity = 0.5;
/// let volatility:f64 = 0.3;
/// let max_strike = 5000.0; //needs to be "large enough" to integrate over space
/// //As an example, cf is standard diffusion
/// let cf = |u: &Complex<f64>| {
///     ((rate-volatility*volatility*0.5)*t_maturity*u+volatility*volatility*t_maturity*u*u*0.5).exp()
/// };
/// let deltas: Vec<fang_oost::GraphElement> = option_pricing::fang_oost_put_delta(
///     num_u, asset, &strikes, max_strike,
///     rate, t_maturity, &cf
/// ).collect();
/// # }
/// ```
pub fn fang_oost_put_delta<'a, S>(
    num_u: usize,
    asset: f64,
    strikes: &'a [f64],
    max_strike: f64,
    rate: f64,
    t_maturity: f64,
    cf: S,
) -> impl IndexedParallelIterator<Item = fang_oost::GraphElement> + 'a
where
    S: Fn(&Complex<f64>) -> Complex<f64> + std::marker::Sync + std::marker::Send,
{
    let discount = (-rate * t_maturity).exp();
    let (x_min, x_max) = get_x_range(asset, max_strike);
    let cf_discrete = fang_oost_discrete_cf(
        num_u,
        x_min,
        x_max,
        |cfu, u| option_delta_transform(&cfu, &u),
        cf,
    );
    fang_oost_generic_move(
        asset,
        strikes,
        x_min,
        x_max,
        cf_discrete,
        move |val, strike| val * discount * strike / asset,
    )
}
/// Returns theta of a call for the series of strikes
///
/// # Remarks
///
/// The theta of a call will only be accurate for non-time changed
/// Levy processes.  For a time-changed Levy process (eg, Heston's
/// model) the theta is not accurate.  
///
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// use rayon::prelude::*;
/// extern crate fang_oost_option;
/// use fang_oost_option::option_pricing;
/// # fn main() {
/// let num_u:usize = 256;
/// let asset = 50.0;
/// let strikes = vec![75.0, 50.0, 40.0];
/// let rate = 0.03;
/// let t_maturity = 0.5;
/// let volatility:f64 = 0.3;
/// let max_strike = 5000.0; //needs to be "large enough" to integrate over space
/// //As an example, cf is standard diffusion
/// let cf = |u: &Complex<f64>| {
///     ((rate-volatility*volatility*0.5)*t_maturity*u+volatility*volatility*t_maturity*u*u*0.5).exp()
/// };
/// let deltas: Vec<fang_oost::GraphElement> = option_pricing::fang_oost_call_theta(
///     num_u, asset, &strikes, max_strike,
///     rate, t_maturity, &cf
/// ).collect();
/// # }
/// ```
pub fn fang_oost_call_theta<'a, S>(
    num_u: usize,
    asset: f64,
    strikes: &'a [f64],
    max_strike: f64,
    rate: f64,
    t_maturity: f64,
    cf: S,
) -> impl IndexedParallelIterator<Item = fang_oost::GraphElement> + 'a
where
    S: Fn(&Complex<f64>) -> Complex<f64> + std::marker::Sync + std::marker::Send,
{
    let discount = (-rate * t_maturity).exp();
    let (x_min, x_max) = get_x_range(asset, max_strike);
    let cf_discrete = fang_oost_discrete_cf(
        num_u,
        x_min,
        x_max,
        |cfu, _| option_theta_transform(&cfu, rate),
        cf,
    );
    fang_oost_generic_move(
        asset,
        strikes,
        x_min,
        x_max,
        cf_discrete,
        move |val, strike| (val - rate) * discount * strike,
    )
}
/// Returns theta of a put for the series of strikes
///
/// # Remarks
///
/// The theta of a call will only be accurate for non-time changed
/// Levy processes.  For a time-changed Levy process (eg, Heston's
/// model) the theta is not accurate.  
///
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// use rayon::prelude::*;
/// extern crate fang_oost_option;
/// use fang_oost_option::option_pricing;
/// # fn main() {
/// let num_u:usize = 256;
/// let asset = 50.0;
/// let strikes = vec![75.0, 50.0, 40.0];
/// let max_strike = 5000.0; //needs to be "large enough" to integrate over space
/// let rate = 0.03;
/// let t_maturity = 0.5;
/// let volatility:f64 = 0.3;
/// //As an example, cf is standard diffusion
/// let cf = |u: &Complex<f64>| {
///     ((rate-volatility*volatility*0.5)*t_maturity*u+volatility*volatility*t_maturity*u*u*0.5).exp()
/// };
/// let deltas: Vec<fang_oost::GraphElement> = option_pricing::fang_oost_put_theta(
///     num_u, asset, &strikes, max_strike,
///     rate, t_maturity, &cf
/// ).collect();
/// # }
/// ```
pub fn fang_oost_put_theta<'a, S>(
    num_u: usize,
    asset: f64,
    strikes: &'a [f64],
    max_strike: f64,
    rate: f64,
    t_maturity: f64,
    cf: S,
) -> impl IndexedParallelIterator<Item = fang_oost::GraphElement> + 'a
where
    S: Fn(&Complex<f64>) -> Complex<f64> + std::marker::Sync + std::marker::Send,
{
    let discount = (-rate * t_maturity).exp();
    let (x_min, x_max) = get_x_range(asset, max_strike);
    let cf_discrete = fang_oost_discrete_cf(
        num_u,
        x_min,
        x_max,
        |cfu, _| option_theta_transform(&cfu, rate),
        cf,
    );
    fang_oost_generic_move(
        asset,
        strikes,
        x_min,
        x_max,
        cf_discrete,
        move |val, strike| val * discount * strike,
    )
}
/// Returns gamma of a call for the series of strikes
///
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// use rayon::prelude::*;
/// extern crate fang_oost_option;
/// use fang_oost_option::option_pricing;
/// # fn main() {
/// let num_u:usize = 256;
/// let asset = 50.0;
/// let strikes = vec![75.0, 50.0, 40.0];
/// let rate = 0.03;
/// let t_maturity = 0.5;
/// let volatility:f64 = 0.3;
/// let max_strike = 5000.0; //needs to be "large enough" to integrate over space
/// //As an example, cf is standard diffusion
/// let cf = |u: &Complex<f64>| {
///     ((rate-volatility*volatility*0.5)*t_maturity*u+volatility*volatility*t_maturity*u*u*0.5).exp()
/// };
/// let deltas: Vec<fang_oost::GraphElement> = option_pricing::fang_oost_call_gamma(
///     num_u, asset, &strikes, max_strike,
///     rate, t_maturity, &cf
/// ).collect();
/// # }
/// ```
pub fn fang_oost_call_gamma<'a, S>(
    num_u: usize,
    asset: f64,
    strikes: &'a [f64],
    max_strike: f64,
    rate: f64,
    t_maturity: f64,
    cf: S,
) -> impl IndexedParallelIterator<Item = fang_oost::GraphElement> + 'a
where
    S: Fn(&Complex<f64>) -> Complex<f64> + std::marker::Sync + std::marker::Send,
{
    let discount = (-rate * t_maturity).exp();
    let (x_min, x_max) = get_x_range(asset, max_strike);
    let cf_discrete = fang_oost_discrete_cf(
        num_u,
        x_min,
        x_max,
        |cfu, u| option_gamma_transform(&cfu, &u),
        cf,
    );
    fang_oost_generic_move(
        asset,
        strikes,
        x_min,
        x_max,
        cf_discrete,
        move |val, strike| val * discount * strike / asset.powi(2),
    )
}

/// Returns gamma of a put for the series of strikes
///
/// # Remarks
/// The gamma of a put is the same as the gamma of call.
/// This function just wraps the fang_oost_call_gamma
/// function.
///
/// # Examples
///
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// use rayon::prelude::*;
/// extern crate fang_oost_option;
/// use fang_oost_option::option_pricing;
/// # fn main() {
/// let num_u:usize = 256;
/// let asset = 50.0;
/// let strikes = vec![75.0, 50.0, 40.0];
/// let rate = 0.03;
/// let t_maturity = 0.5;
/// let volatility:f64 = 0.3;
/// let max_strike = 5000.0; //needs to be "large enough" to integrate over space
/// //As an example, cf is standard diffusion
/// let cf = |u: &Complex<f64>| {
///     ((rate-volatility*volatility*0.5)*t_maturity*u+volatility*volatility*t_maturity*u*u*0.5).exp()
/// };
/// let deltas: Vec<fang_oost::GraphElement> = option_pricing::fang_oost_put_gamma(
///     num_u, asset, &strikes, max_strike,
///     rate, t_maturity, &cf
/// ).collect();
/// # }
/// ```
pub fn fang_oost_put_gamma<'a, 'b: 'a, S>(
    num_u: usize,
    asset: f64,
    strikes: &'a [f64],
    max_strike: f64,
    rate: f64,
    t_maturity: f64,
    cf: S,
) -> impl IndexedParallelIterator<Item = fang_oost::GraphElement> + 'a
where
    S: Fn(&Complex<f64>) -> Complex<f64> + std::marker::Sync + std::marker::Send + 'b,
{
    fang_oost_call_gamma(num_u, asset, &strikes, max_strike, rate, t_maturity, cf)
}

#[cfg(test)]
mod tests {
    use crate::option_pricing::*;
    use approx::*;
    fn get_fang_oost_k_at_index(x_min: f64, dk: f64, asset: f64, index: usize) -> f64 {
        //negative because x=log(s/k) -> k=s*exp(-x)
        asset * (-x_min - dk * (index as f64)).exp()
    }

    fn get_fang_oost_strike(x_min: f64, x_max: f64, asset: f64, num_x: usize) -> Vec<f64> {
        let dx = (x_max - x_min) / (num_x as f64 - 1.0);
        (0..num_x)
            .map(|index| get_fang_oost_k_at_index(x_min, dx, asset, index))
            .collect()
    }

    #[test]
    fn test_generator() {
        let asset = 50.0;
        let strikes = vec![30.0, 40.0, 50.0, 60.0, 70.0];
        //let (x_min, x_max) = get_x_range(asset, 7000.0);
        let result: Vec<f64> = get_x_from_k_iterator(asset, &strikes).collect();
        let expected = vec![
            //12.765688433465597,
            0.5108256237659907,
            0.22314355131420976,
            0.0,
            -0.1823215567939546,
            -0.3364722366212129,
            //-4.941642422609305,
        ];
        for (res, ex) in result.iter().zip(expected) {
            assert_eq!(res, &ex);
        }
    }

    #[test]
    fn test_get_x_range() {
        let asset = 50.0;
        let (x_min, x_max) = get_x_range(asset, 5000.0);
        assert_abs_diff_eq!(x_min, -x_max, epsilon = 0.000001); //should be symmetric
    }
    #[test]
    fn test_get_x_range_2() {
        let asset = 100.0;
        let (x_min, x_max) = get_x_range(asset, 7500.0);
        assert_abs_diff_eq!(x_min, -x_max, epsilon = 0.000001); //should be symmetric
    }
    #[test]
    fn test_fang_oost_call_price_other_direction() {
        let r = 0.05;
        let sig = 0.3;
        let t = 1.0;
        let asset = 50.0;
        let bs_cf =
            |u: &Complex<f64>| ((r - sig * sig * 0.5) * t * u + sig * sig * t * u * u * 0.5).exp();
        let max_strike = 7000.0;
        let num_u = 64;
        let k_array = vec![45.0, 50.0, 55.0];
        let my_option_price = fang_oost_call_price(num_u, asset, &k_array, max_strike, r, t, bs_cf); //.collect();

        my_option_price.for_each(|fang_oost::GraphElement { x, value }| {
            assert_abs_diff_eq!(
                black_scholes::call(asset, x, r, sig, t),
                value,
                epsilon = 0.001
            );
        });
    }

    #[test]
    fn test_fang_oost_call_price() {
        let r = 0.05;
        let sig = 0.3;
        let t = 1.0;
        let asset = 50.0;
        let bs_cf =
            |u: &Complex<f64>| ((r - sig * sig * 0.5) * t * u + sig * sig * t * u * u * 0.5).exp();
        let x_max = 2.0;
        let max_strike = 7000.0;
        let num_x = (2 as usize).pow(10);
        let num_u = 64;
        let k_array = get_fang_oost_strike(-x_max, x_max, asset, num_x);
        let my_option_price = fang_oost_call_price(num_u, asset, &k_array, max_strike, r, t, bs_cf);

        my_option_price.for_each(|fang_oost::GraphElement { x, value }| {
            assert_abs_diff_eq!(
                black_scholes::call(asset, x, r, sig, t),
                value,
                epsilon = 0.001
            );
        });
    }
    #[test]
    fn test_fang_oost_call_price_with_merton() {
        let r = 0.05;
        let sig = 0.3;
        let t = 1.0;
        let asset = 50.0;
        let lambda = 0.5;
        let mu_j = 0.05;
        let sig_j = 0.2;
        let v0 = 0.8;
        let speed = 0.5;
        let ada_v = 0.3;
        let rho = -0.5;
        let inst_cf = cf_functions::merton::merton_time_change_cf(
            t, r, lambda, mu_j, sig_j, sig, v0, speed, ada_v, rho,
        );

        let num_u = 64;
        let k_array = vec![45.0, 50.0, 55.0];
        let max_strike = 5000.0;
        let my_option_price =
            fang_oost_call_price(num_u, asset, &k_array, max_strike, r, t, inst_cf);
        my_option_price.for_each(|fang_oost::GraphElement { value, .. }| {
            assert_eq!(value > 0.0 && value < asset, true);
        });
    }
    #[test]
    fn test_fang_oost_put_price() {
        let r = 0.05;
        let sig = 0.3;
        let t = 1.0;
        let asset = 50.0;
        let bs_cf =
            |u: &Complex<f64>| ((r - sig * sig * 0.5) * t * u + sig * sig * t * u * u * 0.5).exp();
        let x_max = 2.0;
        let num_x = (2 as usize).pow(10);
        let num_u = 64;
        let max_strike = 7000.0;
        let k_array = get_fang_oost_strike(-x_max, x_max, asset, num_x);
        let my_option_price = fang_oost_put_price(num_u, asset, &k_array, max_strike, r, t, bs_cf);

        my_option_price.for_each(|fang_oost::GraphElement { x, value }| {
            assert_abs_diff_eq!(
                black_scholes::put(asset, x, r, sig, t),
                value,
                epsilon = 0.001
            );
        });
    }
    #[test]
    fn test_fang_oost_call_delta() {
        let r = 0.05;
        let sig = 0.3;
        let t = 1.0;
        let asset = 50.0;
        let bs_cf =
            |u: &Complex<f64>| ((r - sig * sig * 0.5) * t * u + sig * sig * t * u * u * 0.5).exp();
        let x_max = 2.0;
        let num_x = (2 as usize).pow(10);
        let max_strike = 7000.0;
        let num_u = 64;
        let k_array = get_fang_oost_strike(-x_max, x_max, asset, num_x);
        let my_option_price = fang_oost_call_delta(num_u, asset, &k_array, max_strike, r, t, bs_cf);
        my_option_price.for_each(|fang_oost::GraphElement { x, value }| {
            assert_abs_diff_eq!(
                black_scholes::call_delta(asset, x, r, sig, t),
                value,
                epsilon = 0.001
            );
        });
    }
    #[test]
    fn test_fang_oost_put_delta() {
        let r = 0.05;
        let sig = 0.3;
        let t = 1.0;
        let asset = 50.0;
        let bs_cf =
            |u: &Complex<f64>| ((r - sig * sig * 0.5) * t * u + sig * sig * t * u * u * 0.5).exp();
        let x_max = 2.0;
        let num_x = (2 as usize).pow(10);
        let max_strike = 7000.0;
        let num_u = 64;
        let k_array = get_fang_oost_strike(-x_max, x_max, asset, num_x);
        let my_option_price = fang_oost_put_delta(num_u, asset, &k_array, max_strike, r, t, bs_cf);
        my_option_price.for_each(|fang_oost::GraphElement { x, value }| {
            assert_abs_diff_eq!(
                black_scholes::put_delta(asset, x, r, sig, t),
                value,
                epsilon = 0.001
            );
        });
    }
    #[test]
    fn test_fang_oost_call_gamma() {
        let r = 0.05;
        let sig = 0.3;
        let t = 1.0;
        let asset = 50.0;
        let bs_cf =
            |u: &Complex<f64>| ((r - sig * sig * 0.5) * t * u + sig * sig * t * u * u * 0.5).exp();
        let x_max = 2.0;
        let num_x = (2 as usize).pow(10);
        let max_strike = 7000.0;
        let num_u = 64;
        let k_array = get_fang_oost_strike(-x_max, x_max, asset, num_x);
        let my_option_price = fang_oost_call_gamma(num_u, asset, &k_array, max_strike, r, t, bs_cf);

        my_option_price.for_each(|fang_oost::GraphElement { x, value }| {
            assert_abs_diff_eq!(
                black_scholes::call_gamma(asset, x, r, sig, t),
                value,
                epsilon = 0.001
            );
        });
    }
    #[test]
    fn test_fang_oost_put_gamma() {
        let r = 0.05;
        let sig = 0.3;
        let t = 1.0;
        let asset = 50.0;
        let bs_cf =
            |u: &Complex<f64>| ((r - sig * sig * 0.5) * t * u + sig * sig * t * u * u * 0.5).exp();
        let x_max = 2.0;
        let num_x = (2 as usize).pow(10);
        let max_strike = 7000.0;
        let num_u = 64;
        let k_array = get_fang_oost_strike(-x_max, x_max, asset, num_x);
        let my_option_price = fang_oost_put_gamma(num_u, asset, &k_array, max_strike, r, t, bs_cf);
        my_option_price.for_each(|fang_oost::GraphElement { x, value }| {
            assert_abs_diff_eq!(
                black_scholes::put_gamma(asset, x, r, sig, t),
                value,
                epsilon = 0.001
            );
        });
    }
    #[test]
    fn test_fang_oost_call_theta() {
        let r = 0.05;
        let sig = 0.3;
        let t = 1.0;
        let asset = 50.0;
        let bs_cf =
            |u: &Complex<f64>| ((r - sig * sig * 0.5) * t * u + sig * sig * t * u * u * 0.5).exp();
        let x_max = 2.0;
        let num_x = (2 as usize).pow(10);
        let max_strike = 7000.0;
        let num_u = 64;
        let k_array = get_fang_oost_strike(-x_max, x_max, asset, num_x);
        let my_option_price = fang_oost_call_theta(num_u, asset, &k_array, max_strike, r, t, bs_cf);
        my_option_price.for_each(|fang_oost::GraphElement { x, value }| {
            assert_abs_diff_eq!(
                black_scholes::call_theta(asset, x, r, sig, t),
                value,
                epsilon = 0.001
            );
        });
    }
    #[test]
    fn test_fang_oost_put_theta() {
        let r = 0.05;
        let sig = 0.3;
        let t = 1.0;
        let asset = 50.0;
        let bs_cf =
            |u: &Complex<f64>| ((r - sig * sig * 0.5) * t * u + sig * sig * t * u * u * 0.5).exp();
        let x_max = 2.0;
        let num_x = (2 as usize).pow(10);
        let max_strike = 7000.0;
        let num_u = 64;
        let k_array = get_fang_oost_strike(-x_max, x_max, asset, num_x);
        let my_option_price = fang_oost_put_theta(num_u, asset, &k_array, max_strike, r, t, bs_cf);

        my_option_price.for_each(|fang_oost::GraphElement { x, value }| {
            assert_abs_diff_eq!(
                black_scholes::put_theta(asset, x, r, sig, t),
                value,
                epsilon = 0.001
            );
        });
    }

    #[test]
    fn test_fang_oost_cgmy_call() {
        //https://cs.uwaterloo.ca/~paforsyt/levy.pdf pg 19
        //S K T r q σ C G M Y
        //90 98 0.25 0.06 0.0 0.0 16.97 7.08 29.97 0.6442
        let k_array = vec![98.0];
        let max_strike = 7500.0;
        let r = 0.06;
        let sig = 0.0;
        let t = 0.25;
        let s0 = 90.0;
        let c = 16.97;
        let g = 7.08;
        let m = 29.97;
        let y = 0.6442;
        let cgmy_cf = |u: &Complex<f64>| {
            (cf_functions::cgmy::cgmy_log_risk_neutral_cf(u, c, g, m, y, r, sig) * t).exp()
        };

        let num_u = 256 as usize;
        let options_price: Vec<f64> =
            fang_oost_call_price(num_u, s0, &k_array, max_strike, r, t, cgmy_cf)
                .map(|fang_oost::GraphElement { value, .. }| value)
                .collect();
        let reference_price = 16.212478; //https://cs.uwaterloo.ca/~paforsyt/levy.pdf pg 19
        assert_abs_diff_eq!(options_price[0], reference_price, epsilon = 0.001);
    }

    #[test]
    fn test_fang_oost_cgmy_call_with_t_one() {
        //http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/COS.pdf pg 19
        //S0 = 100, K = 100, r = 0.1, q = 0, C = 1, G = 5, M = 5, T = 1
        //= 19.812948843
        let k_array = vec![100.0];
        let max_strike = 7500.0; //0.3?
        let r = 0.1;
        let sig = 0.0;
        let t = 1.0;
        let s0 = 100.0;
        let c = 1.0;
        let g = 5.0;
        let m = 5.0;
        let y = 0.5;
        let cgmy_cf = |u: &Complex<f64>| {
            (t * cf_functions::cgmy::cgmy_log_risk_neutral_cf(u, c, g, m, y, r, sig)).exp()
        };

        let num_u = 64 as usize;
        let options_price: Vec<f64> =
            fang_oost_call_price(num_u, s0, &k_array, max_strike, r, t, cgmy_cf)
                .map(|fang_oost::GraphElement { value, .. }| value)
                .collect();
        let reference_price = 19.812948843;
        assert_abs_diff_eq!(options_price[0], reference_price, epsilon = 0.00001);
    }

    #[test]
    fn test_fang_oost_call_heston() {
        //http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/COS.pdf pg 15

        let b: f64 = 0.0398;
        let a = 1.5768;
        let c = 0.5751;
        let rho = -0.5711;
        let v0 = 0.0175;
        let r = 0.0;
        let sig = b.sqrt();
        let speed = a;
        let t = 1.0;
        let s0 = 100.0;
        let kappa = speed;
        let v0_hat = v0 / b;
        let eta_v = c / b.sqrt();

        let k_array = vec![100.0];
        let max_strike = 7500.0;
        let heston_cf = |u: &Complex<f64>| {
            let cmp_mu =
                -cf_functions::merton::merton_log_risk_neutral_cf(u, 0.0, 1.0, 1.0, 0.0, sig);
            let cmp_drift = kappa - eta_v * rho * u * sig;
            (r * t * u
                + cf_functions::affine_process::cir_log_mgf_cmp(
                    &cmp_mu, speed, &cmp_drift, eta_v, t, v0_hat,
                ))
            .exp()
        };

        let num_u = 256 as usize;
        let options_price: Vec<f64> =
            fang_oost_call_price(num_u, s0, &k_array, max_strike, r, t, heston_cf)
                .map(|fang_oost::GraphElement { value, .. }| value)
                .collect();
        let reference_price = 5.78515545;
        assert_abs_diff_eq!(options_price[0], reference_price, epsilon = 0.00001);
    }
}
