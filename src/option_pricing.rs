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
//! to smallest. http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/COS.pdf
//! 
extern crate num;
extern crate num_complex;
extern crate rayon;
extern crate fang_oost;

#[cfg(test)]
extern crate black_scholes;
#[cfg(test)]
extern crate cf_functions;

use self::num_complex::Complex;
use self::rayon::prelude::*;
use std;


/**For Fang Oost (defined in the paper)*/
fn chi_k(a:f64, c:f64, d:f64, u:f64)->f64{
    let iter_s=|x|u*(x-a);
    let exp_d=d.exp();
    let exp_c=c.exp();
    (iter_s(d).cos()*exp_d-iter_s(c).cos()*exp_c+u*iter_s(d).sin()*exp_d-u*iter_s(c).sin()*exp_c)/(1.0+u*u)
}

fn phi_k(a:f64, c:f64, d:f64, u:f64, k:usize)->f64{
    let iter_s=|x|u*(x-a);
    if k==0 {d-c} else{(iter_s(d).sin()-iter_s(c).sin())/u}
}

/**This function takes strikes and converts them
into a vector in the x domain.  Intriguinely, I 
don't have to sort the result...*/
fn get_x_from_k(asset:f64, strikes:&[f64])->Vec<f64>{
    strikes.iter().map(|strike|(asset/strike).ln()).collect()
}

fn option_price_transform(cf:&Complex<f64>)->Complex<f64>{
    *cf
}

fn option_delta_transform(cf:&Complex<f64>, u:&Complex<f64>)->Complex<f64>{
    cf*u
}

fn option_gamma_transform(cf:&Complex<f64>, u:&Complex<f64>)->Complex<f64>{
    -cf*u*(1.0-u)
}

fn option_theta_transform(cf:&Complex<f64>, rate:f64)->Complex<f64>{
    if cf.re>0.0 { -(cf.ln()-rate)*cf} else {Complex::new(0.0, 0.0)}
}

fn fang_oost_generic<'a, T, U, S>(
    num_u:usize, 
    x_values:&'a [f64],
    enh_cf:T,
    m_output:U,
    cf:S
)->Vec<f64>
    where T: Fn(&Complex<f64>, &Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send+'a,
    U: Fn(f64, usize)->f64+std::marker::Sync+std::marker::Send+'a,
    S:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send+'a
{
    let x_min=*x_values.first().unwrap();
    fang_oost::get_expectation_discrete_extended(
        num_u,
        x_values, 
        |u| enh_cf(&cf(u), u),
        move |u, _, k|phi_k(x_min, x_min, 0.0, u, k)-chi_k(x_min, x_min, 0.0, u)
    ).enumerate().map(|(index, result)|{
        m_output(result, index)
    }).collect()
}
///Returns call prices for the series of strikes
/// # Examples
/// 
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate fang_oost_option;
/// use fang_oost_option::option_pricing;
/// # fn main() {
/// let num_u:usize = 256;
/// let asset = 50.0;
/// let strikes = vec![5000.0, 75.0, 50.0, 40.0, 0.03];
/// let rate = 0.03;
/// let t_maturity = 0.5;
/// let volatility:f64 = 0.3; 
/// //As an example, cf is standard diffusion
/// let cf = |u: &Complex<f64>| {
///     ((rate-volatility*volatility*0.5)*t_maturity*u+volatility*volatility*t_maturity*u*u*0.5).exp()
/// };
/// let prices = option_pricing::fang_oost_call_price(
///     num_u, asset, &strikes, 
///     rate, t_maturity, &cf
/// );
/// # }
/// ```
pub fn fang_oost_call_price<'a, S>(
    num_u:usize,
    asset:f64,
    strikes:&'a [f64],
    rate:f64,
    t_maturity:f64,
    cf:S
)->Vec<f64>
    where S:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send
{
    let discount=(-rate*t_maturity).exp();
    let t_strikes:Vec<f64>=get_x_from_k(asset, &strikes);
    fang_oost_generic(
        num_u, 
        &t_strikes, 
        |cfu, _|option_price_transform(&cfu), 
        |val, index|(val-1.0)*discount*strikes[index]+asset,
        cf
    )
}

///Returns put prices for the series of strikes
/// # Examples
/// 
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate fang_oost_option;
/// use fang_oost_option::option_pricing;
/// # fn main() {
/// let num_u:usize = 256;
/// let asset = 50.0;
/// let strikes = vec![5000.0, 75.0, 50.0, 40.0, 0.03];
/// let rate = 0.03;
/// let t_maturity = 0.5;
/// let volatility:f64 = 0.3; 
/// //As an example, cf is standard diffusion
/// let cf = |u: &Complex<f64>| {
///     ((rate-volatility*volatility*0.5)*t_maturity*u+volatility*volatility*t_maturity*u*u*0.5).exp()
/// };
/// let prices = option_pricing::fang_oost_put_price(
///     num_u, asset, &strikes, 
///     rate, t_maturity, &cf
/// );
/// # }
/// ```
pub fn fang_oost_put_price<'a, S>(
    num_u:usize,
    asset:f64,
    strikes:&'a [f64],
    rate:f64,
    t_maturity:f64,
    cf:S
)->Vec<f64>
    where S:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send
{
    let discount=(-rate*t_maturity).exp();
    let t_strikes:Vec<f64>=get_x_from_k(asset, &strikes);
    fang_oost_generic(
        num_u, 
        &t_strikes, 
        |cfu, _|option_price_transform(&cfu), 
        |val, index|val*discount*strikes[index],
        cf
    )
}
///Returns delta of a call for the series of strikes
/// # Examples
/// 
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate fang_oost_option;
/// use fang_oost_option::option_pricing;
/// # fn main() {
/// let num_u:usize = 256;
/// let asset = 50.0;
/// let strikes = vec![5000.0, 75.0, 50.0, 40.0, 0.03];
/// let rate = 0.03;
/// let t_maturity = 0.5;
/// let volatility:f64 = 0.3; 
/// //As an example, cf is standard diffusion
/// let cf = |u: &Complex<f64>| {
///     ((rate-volatility*volatility*0.5)*t_maturity*u+volatility*volatility*t_maturity*u*u*0.5).exp()
/// };
/// let deltas = option_pricing::fang_oost_call_delta(
///     num_u, asset, &strikes, 
///     rate, t_maturity, &cf
/// );
/// # }
/// ```
pub fn fang_oost_call_delta<'a, S>(
    num_u:usize,
    asset:f64,
    strikes:&'a [f64],
    rate:f64,
    t_maturity:f64,
    cf:S
)->Vec<f64>
    where S:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send
{
    let discount=(-rate*t_maturity).exp();
    let t_strikes:Vec<f64>=get_x_from_k(asset, &strikes);
    fang_oost_generic(
        num_u, 
        &t_strikes, 
        |cfu, u|option_delta_transform(&cfu, &u), 
        |val, index|val*discount*strikes[index]/asset+1.0,
        cf
    )
}
///Returns delta of a put for the series of strikes
/// # Examples
/// 
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate fang_oost_option;
/// use fang_oost_option::option_pricing;
/// # fn main() {
/// let num_u:usize = 256;
/// let asset = 50.0;
/// let strikes = vec![5000.0, 75.0, 50.0, 40.0, 0.03];
/// let rate = 0.03;
/// let t_maturity = 0.5;
/// let volatility:f64 = 0.3; 
/// //As an example, cf is standard diffusion
/// let cf = |u: &Complex<f64>| {
///     ((rate-volatility*volatility*0.5)*t_maturity*u+volatility*volatility*t_maturity*u*u*0.5).exp()
/// };
/// let deltas = option_pricing::fang_oost_put_delta(
///     num_u, asset, &strikes, 
///     rate, t_maturity, &cf
/// );
/// # }
/// ```
pub fn fang_oost_put_delta<'a, S>(
    num_u:usize,
    asset:f64,
    strikes:&'a [f64],
    rate:f64,
    t_maturity:f64,
    cf:S
)->Vec<f64>
    where S:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send
{
    let discount=(-rate*t_maturity).exp();
    let t_strikes:Vec<f64>=get_x_from_k(asset, &strikes);
    fang_oost_generic(
        num_u, 
        &t_strikes, 
        |cfu, u|option_delta_transform(&cfu, &u), 
        |val, index|val*discount*strikes[index]/asset,
        cf
    )
}
///Returns theta of a call for the series of strikes
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
/// extern crate fang_oost_option;
/// use fang_oost_option::option_pricing;
/// # fn main() {
/// let num_u:usize = 256;
/// let asset = 50.0;
/// let strikes = vec![5000.0, 75.0, 50.0, 40.0, 0.03];
/// let rate = 0.03;
/// let t_maturity = 0.5;
/// let volatility:f64 = 0.3; 
/// //As an example, cf is standard diffusion
/// let cf = |u: &Complex<f64>| {
///     ((rate-volatility*volatility*0.5)*t_maturity*u+volatility*volatility*t_maturity*u*u*0.5).exp()
/// };
/// let deltas = option_pricing::fang_oost_call_theta(
///     num_u, asset, &strikes, 
///     rate, t_maturity, &cf
/// );
/// # }
/// ```
pub fn fang_oost_call_theta<'a, S>(
    num_u:usize,
    asset:f64,
    strikes:&'a [f64],
    rate:f64,
    t_maturity:f64,
    cf:S
)->Vec<f64>
    where S:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send
{
    let discount=(-rate*t_maturity).exp();
    let t_strikes:Vec<f64>=get_x_from_k(asset, &strikes);
    fang_oost_generic(
        num_u, 
        &t_strikes, 
        |cfu, _|option_theta_transform(&cfu, rate), 
        |val, index|(val-rate)*discount*strikes[index],
        cf
    )
}
///Returns theta of a put for the series of strikes
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
/// extern crate fang_oost_option;
/// use fang_oost_option::option_pricing;
/// # fn main() {
/// let num_u:usize = 256;
/// let asset = 50.0;
/// let strikes = vec![5000.0, 75.0, 50.0, 40.0, 0.03];
/// let rate = 0.03;
/// let t_maturity = 0.5;
/// let volatility:f64 = 0.3; 
/// //As an example, cf is standard diffusion
/// let cf = |u: &Complex<f64>| {
///     ((rate-volatility*volatility*0.5)*t_maturity*u+volatility*volatility*t_maturity*u*u*0.5).exp()
/// };
/// let deltas = option_pricing::fang_oost_put_theta(
///     num_u, asset, &strikes, 
///     rate, t_maturity, &cf
/// );
/// # }
/// ```
pub fn fang_oost_put_theta<'a, S>(
    num_u:usize,
    asset:f64,
    strikes:&'a [f64],
    rate:f64,
    t_maturity:f64,
    cf:S
)->Vec<f64>
    where S:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send
{
    let discount=(-rate*t_maturity).exp();
    let t_strikes:Vec<f64>=get_x_from_k(asset, &strikes);
    fang_oost_generic(
        num_u, 
        &t_strikes, 
        |cfu, _|option_theta_transform(&cfu, rate), 
        |val, index|val*discount*strikes[index],
        cf
    )
}
///Returns gamma of a call for the series of strikes
/// 
/// # Examples
/// 
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate fang_oost_option;
/// use fang_oost_option::option_pricing;
/// # fn main() {
/// let num_u:usize = 256;
/// let asset = 50.0;
/// let strikes = vec![5000.0, 75.0, 50.0, 40.0, 0.03];
/// let rate = 0.03;
/// let t_maturity = 0.5;
/// let volatility:f64 = 0.3; 
/// //As an example, cf is standard diffusion
/// let cf = |u: &Complex<f64>| {
///     ((rate-volatility*volatility*0.5)*t_maturity*u+volatility*volatility*t_maturity*u*u*0.5).exp()
/// };
/// let deltas = option_pricing::fang_oost_call_gamma(
///     num_u, asset, &strikes, 
///     rate, t_maturity, &cf
/// );
/// # }
/// ```
pub fn fang_oost_call_gamma<'a, S>(
    num_u:usize,
    asset:f64,
    strikes:&'a [f64],
    rate:f64,
    t_maturity:f64,
    cf:S
)->Vec<f64>
    where S:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send
{
    let discount=(-rate*t_maturity).exp();
    let t_strikes:Vec<f64>=get_x_from_k(asset, &strikes);
    fang_oost_generic(
        num_u, 
        &t_strikes, 
        |cfu, u|option_gamma_transform(&cfu, &u), 
        |val, index|val*discount*strikes[index]/(asset*asset),
        cf
    )
}

///Returns gamma of a put for the series of strikes
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
/// extern crate fang_oost_option;
/// use fang_oost_option::option_pricing;
/// # fn main() {
/// let num_u:usize = 256;
/// let asset = 50.0;
/// let strikes = vec![5000.0, 75.0, 50.0, 40.0, 0.03];
/// let rate = 0.03;
/// let t_maturity = 0.5;
/// let volatility:f64 = 0.3; 
/// //As an example, cf is standard diffusion
/// let cf = |u: &Complex<f64>| {
///     ((rate-volatility*volatility*0.5)*t_maturity*u+volatility*volatility*t_maturity*u*u*0.5).exp()
/// };
/// let deltas = option_pricing::fang_oost_put_gamma(
///     num_u, asset, &strikes, 
///     rate, t_maturity, &cf
/// );
/// # }
/// ```
pub fn fang_oost_put_gamma<'a, S>(
    num_u:usize,
    asset:f64,
    strikes:&'a [f64],
    rate:f64,
    t_maturity:f64,
    cf:S
)->Vec<f64>
    where S:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send
{
    fang_oost_call_gamma(num_u, asset, &strikes, rate, t_maturity, cf)
}

#[cfg(test)]
mod tests {
    use option_pricing::*;
    use std::time::{ Instant};
    fn get_fang_oost_k_at_index(
        x_min:f64,
        dk:f64,
        asset:f64,
        index:usize
    )->f64{
        //negative because x=log(s/k) -> k=s*exp(-x)
        asset*(-x_min-dk*(index as f64)).exp()
    }

    fn get_fang_oost_strike(
        x_min:f64,
        x_max:f64,
        asset:f64,
        num_x:usize
    )->Vec<f64>{
        let dx=(x_max-x_min)/(num_x as f64-1.0);
        (0..num_x).map(|index|{
            get_fang_oost_k_at_index(x_min, dx, asset, index)
        }).collect()
    }

    #[test]
    fn test_fang_oost_call_price(){
        let r=0.05;
        let sig=0.3;
        let t=1.0;
        let asset=50.0;
        let bs_cf=|u:&Complex<f64>| ((r-sig*sig*0.5)*t*u+sig*sig*t*u*u*0.5).exp();
        let x_max=5.0;
        let num_x=(2 as usize).pow(10);
        let num_u=64;
        let k_array=get_fang_oost_strike(-x_max, x_max, asset, num_x);
        let now = Instant::now();
        let my_option_price=fang_oost_call_price(num_u, asset, &k_array, r, t, bs_cf);
        let new_now = Instant::now();
        println!("Call price time: {:?}", new_now.duration_since(now));
        let min_n=num_x/4;
        let max_n=num_x-num_x/4;
        for i in min_n..max_n{
            assert_abs_diff_eq!(
                black_scholes::call(asset, k_array[i], r, sig, t),
                my_option_price[i],
                epsilon=0.001
            );
        }
        
    }
    #[test]
    fn test_fang_oost_call_price_with_merton(){
        let r=0.05;
        let sig=0.3;
        let t=1.0;
        let asset=50.0;
        let lambda=0.5;
        let mu_j=0.05;
        let sig_j=0.2;
        let v0=0.8;
        let speed=0.5;
        let ada_v=0.3;
        let rho=-0.5;
        let inst_cf=cf_functions::merton_time_change_cf(
            t, r, lambda, mu_j, sig_j, sig, v0,
            speed, ada_v, rho
        );

        let num_u=64;
        let k_array=vec![5000.0, 45.0, 50.0, 55.0, 0.00001];
        let my_option_price=fang_oost_call_price(num_u, asset, &k_array, r, t, inst_cf);
        assert_eq!(my_option_price[1]>0.0&&my_option_price[1]<asset, true);
        assert_eq!(my_option_price[2]>0.0&&my_option_price[2]<asset, true);
        assert_eq!(my_option_price[3]>0.0&&my_option_price[3]<asset, true);
        
    }
    #[test]
    fn test_fang_oost_put_price(){
        let r=0.05;
        let sig=0.3;
        let t=1.0;
        let asset=50.0;
        let bs_cf=|u:&Complex<f64>| ((r-sig*sig*0.5)*t*u+sig*sig*t*u*u*0.5).exp();
        let x_max=5.0;
        let num_x=(2 as usize).pow(10);
        let num_u=64;
        let k_array=get_fang_oost_strike(-x_max, x_max, asset, num_x);
        let now = Instant::now();
        let my_option_price=fang_oost_put_price(num_u, asset, &k_array, r, t, bs_cf);
        let new_now = Instant::now();
        println!("Put price time: {:?}", new_now.duration_since(now));
        let min_n=num_x/4;
        let max_n=num_x-num_x/4;
        for i in min_n..max_n{
            assert_abs_diff_eq!(
                black_scholes::put(asset, k_array[i], r, sig, t),
                my_option_price[i],
                epsilon=0.001
            );
        }
    }
    #[test]
    fn test_fang_oost_call_delta(){
        let r=0.05;
        let sig=0.3;
        let t=1.0;
        let asset=50.0;
        let bs_cf=|u:&Complex<f64>| ((r-sig*sig*0.5)*t*u+sig*sig*t*u*u*0.5).exp();
        let x_max=5.0;
        let num_x=(2 as usize).pow(10);
        let num_u=64;
        let k_array=get_fang_oost_strike(-x_max, x_max, asset, num_x);
        let now = Instant::now();
        let my_option_price=fang_oost_call_delta(num_u, asset, &k_array, r, t, bs_cf);
        let new_now = Instant::now();
        println!("Call delta time: {:?}", new_now.duration_since(now));
        let min_n=num_x/4;
        let max_n=num_x-num_x/4;
        for i in min_n..max_n{
            assert_abs_diff_eq!(
                black_scholes::call_delta(asset, k_array[i], r, sig, t),
                my_option_price[i],
                epsilon=0.001
            );
        }
    }
    #[test]
    fn test_fang_oost_put_delta(){
        let r=0.05;
        let sig=0.3;
        let t=1.0;
        let asset=50.0;
        let bs_cf=|u:&Complex<f64>| ((r-sig*sig*0.5)*t*u+sig*sig*t*u*u*0.5).exp();
        let x_max=5.0;
        let num_x=(2 as usize).pow(10);
        let num_u=64;
        let k_array=get_fang_oost_strike(-x_max, x_max, asset, num_x);
        let now = Instant::now();
        let my_option_price=fang_oost_put_delta(num_u, asset, &k_array, r, t, bs_cf);
        let new_now = Instant::now();
        println!("Put delta time: {:?}", new_now.duration_since(now));
        let min_n=num_x/4;
        let max_n=num_x-num_x/4;
        for i in min_n..max_n{
            assert_abs_diff_eq!(
                black_scholes::put_delta(asset, k_array[i], r, sig, t),
                my_option_price[i],
                epsilon=0.001
            );
        }
    }
    #[test]
    fn test_fang_oost_call_gamma(){
        let r=0.05;
        let sig=0.3;
        let t=1.0;
        let asset=50.0;
        let bs_cf=|u:&Complex<f64>| ((r-sig*sig*0.5)*t*u+sig*sig*t*u*u*0.5).exp();
        let x_max=5.0;
        let num_x=(2 as usize).pow(10);
        let num_u=64;
        let k_array=get_fang_oost_strike(-x_max, x_max, asset, num_x);
        let now = Instant::now();
        let my_option_price=fang_oost_call_gamma(num_u, asset, &k_array, r, t, bs_cf);
        let new_now = Instant::now();
        println!("Call gamma time: {:?}", new_now.duration_since(now));
        let min_n=num_x/4;
        let max_n=num_x-num_x/4;
        for i in min_n..max_n{
            assert_abs_diff_eq!(
                black_scholes::call_gamma(asset, k_array[i], r, sig, t),
                my_option_price[i],
                epsilon=0.001
            );
        }
    }
    #[test]
    fn test_fang_oost_put_gamma(){
        let r=0.05;
        let sig=0.3;
        let t=1.0;
        let asset=50.0;
        let bs_cf=|u:&Complex<f64>| ((r-sig*sig*0.5)*t*u+sig*sig*t*u*u*0.5).exp();
        let x_max=5.0;
        let num_x=(2 as usize).pow(10);
        let num_u=64;
        let k_array=get_fang_oost_strike(-x_max, x_max, asset, num_x);
        let now = Instant::now();
        let my_option_price=fang_oost_put_gamma(num_u, asset, &k_array, r, t, bs_cf);
        let new_now = Instant::now();
        println!("Put gamma time: {:?}", new_now.duration_since(now));
        let min_n=num_x/4;
        let max_n=num_x-num_x/4;
        for i in min_n..max_n{
            assert_abs_diff_eq!(
                black_scholes::put_gamma(asset, k_array[i], r, sig, t),
                my_option_price[i],
                epsilon=0.001
            );
        }
    }
    #[test]
    fn test_fang_oost_call_theta(){
        let r=0.05;
        let sig=0.3;
        let t=1.0;
        let asset=50.0;
        let bs_cf=|u:&Complex<f64>| ((r-sig*sig*0.5)*t*u+sig*sig*t*u*u*0.5).exp();
        let x_max=5.0;
        let num_x=(2 as usize).pow(10);
        let num_u=64;
        let k_array=get_fang_oost_strike(-x_max, x_max, asset, num_x);
        let now = Instant::now();
        let my_option_price=fang_oost_call_theta(num_u, asset, &k_array, r, t, bs_cf);
        let new_now = Instant::now();
        println!("Call theta time: {:?}", new_now.duration_since(now));
        let min_n=num_x/4;
        let max_n=num_x-num_x/4;
        for i in min_n..max_n{
            assert_abs_diff_eq!(
                black_scholes::call_theta(asset, k_array[i], r, sig, t),
                my_option_price[i],
                epsilon=0.001
            );
        }
    }
    #[test]
    fn test_fang_oost_put_theta(){
        let r=0.05;
        let sig=0.3;
        let t=1.0;
        let asset=50.0;
        let bs_cf=|u:&Complex<f64>| ((r-sig*sig*0.5)*t*u+sig*sig*t*u*u*0.5).exp();
        let x_max=5.0;
        let num_x=(2 as usize).pow(10);
        let num_u=64;
        let k_array=get_fang_oost_strike(-x_max, x_max, asset, num_x);
        let now = Instant::now();
        let my_option_price=fang_oost_put_theta(num_u, asset, &k_array, r, t, bs_cf);
        let new_now = Instant::now();
        println!("Put theta time: {:?}", new_now.duration_since(now));
        let min_n=num_x/4;
        let max_n=num_x-num_x/4;
        for i in min_n..max_n{
            assert_abs_diff_eq!(
                black_scholes::put_theta(asset, k_array[i], r, sig, t),
                my_option_price[i],
                epsilon=0.001
            );
        }
    }

    #[test]
    fn test_fang_oost_cgmy_call(){
         //https://cs.uwaterloo.ca/~paforsyt/levy.pdf pg 19
        //S K T r q σ C G M Y
        //90 98 0.25 0.06 0.0 0.0 16.97 7.08 29.97 0.6442
        let k_array=vec![7500.0, 98.0, 0.3];
        let r=0.06;
        let sig=0.0;
        let t=0.25;
        let s0=90.0;
        let c=16.97;
        let g=7.08;
        let m=29.97;
        let y=0.6442;
        let cgmy_cf=|u:&Complex<f64>| (cf_functions::cgmy_log_risk_neutral_cf(u, c, g, m, y, r, sig)*t).exp();

        let num_u=256 as usize;
        let options_price=fang_oost_call_price(num_u, s0, &k_array, r, t, cgmy_cf);
        let reference_price=16.212478;//https://cs.uwaterloo.ca/~paforsyt/levy.pdf pg 19
        assert_abs_diff_eq!(
            options_price[1],
            reference_price,
            epsilon=0.001
        );
    }

    #[test]
    fn test_fang_oost_cgmy_call_with_t_one(){
        //http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/COS.pdf pg 19
        //S0 = 100, K = 100, r = 0.1, q = 0, C = 1, G = 5, M = 5, T = 1
        //= 19.812948843
        let k_array=vec![7500.0, 100.0, 0.3];
        let r=0.1;
        let sig=0.0;
        let t=1.0;
        let s0=100.0;
        let c=1.0;
        let g=5.0;
        let m=5.0;
        let y=0.5;
        let cgmy_cf=|u:&Complex<f64>| (t*cf_functions::cgmy_log_risk_neutral_cf(u, c, g, m, y, r, sig)).exp();

        let num_u=64 as usize;
        let options_price=fang_oost_call_price(num_u, s0, &k_array, r, t, cgmy_cf);
        let reference_price=19.812948843;
        assert_abs_diff_eq!(
            options_price[1],
            reference_price,
            epsilon=0.00001
        );
    }

    #[test]
    fn test_fang_oost_call_heston(){
        //http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/COS.pdf pg 15

        let b:f64=0.0398;
        let a=1.5768;
        let c=0.5751;
        let rho=-0.5711;
        let v0=0.0175;
        let r=0.0;
        let sig=b.sqrt();
        let speed=a;
        let t=1.0;
        let s0=100.0;
        let kappa=speed;
        let v0_hat=v0/b;
        let eta_v=c/b.sqrt();

        let k_array=vec![7500.0, 100.0, 0.3];
        
        let heston_cf=|u:&Complex<f64>| {
            let cmp_mu=-cf_functions::merton_log_risk_neutral_cf(u, 0.0, 1.0, 1.0, 0.0, sig);
            let cmp_drift=kappa-eta_v*rho*u*sig;
            (r*t*u+cf_functions::cir_log_mgf_cmp(
                &cmp_mu,
                speed,
                &cmp_drift,
                eta_v,
                t, 
                v0_hat
            )).exp()
        };
        
        let num_u=256 as usize;
        let options_price=fang_oost_call_price(num_u, s0, &k_array, r, t, heston_cf);
        let reference_price=5.78515545;
        assert_abs_diff_eq!(
            options_price[1],
            reference_price,
            epsilon=0.00001
        );
    }
}