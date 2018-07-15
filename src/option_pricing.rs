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


/**
    Fang Oosterlee Approach for an option using Put as the main payoff 
    (better accuracy than a call...use put call parity to get back put).
    Note that Fang Oosterlee's approach works well for a smaller 
    of discrete strike prices such as those in the market.  The 
    constraint is that the smallest and largest values in the x domain
    must be relatively far from the middle values.  This can be 
    "simulated" by adding small and large "K" synthetically.  Due to
    the fact that Fang Oosterlee is able to handle this well, the 
    function takes a vector of strike prices with no requirement that
    the strike prices be equidistant.  All that is required is that
    they are sorted largest to smallest.
    returns in log domain
    http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/COS.pdf
    @num_u number of steps in the complex domain (independent 
    of number of x steps)
    @x_values x values derived from strikes
    @enh_cf function of Fn CF and complex U which transforms 
    the CF into the appropriate derivative (eg, for delta or gamma)
    @m_output a function which determines whether the output is a 
    call or a put.  
    @cf characteristic function of log x around the strike
    @returns vector of prices corresponding with the strikes 
    provided by FangOostCall or FangOostPut
*/
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
    use std::f64::consts::SQRT_2;
    use self::special::Error;
    use std::time::{Duration, Instant};
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
        let discount=(-r*t).exp();
        for i in min_n..max_n{
            assert_abs_diff_eq!(
                black_scholes::put_theta(asset, k_array[i], r, sig, t),
                my_option_price[i],
                epsilon=0.001
            );
        }
    }
}