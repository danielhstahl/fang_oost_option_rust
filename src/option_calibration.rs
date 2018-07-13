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
use monotone_spline;

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

fn dft<'a, 'b: 'a>(
    u_array:&'b Vec<f64>,
    x_min:f64,
    x_max:f64,
    n:usize,
    fn_to_invert:impl Fn( f64, usize)->f64+'a+std::marker::Sync+std::marker::Send
)->impl ParallelIterator<Item = (f64, Complex<f64>) >+'a
{
    let cmp:Complex<f64>=Complex::new(0.0, 0.0);
    let dx=get_dx(n, x_min, x_max);
    u_array.par_iter().map(move |u|{
        (
            *u, 
            (0..n).fold(cmp, |accum, index|{
                let simpson=if index==0||index==(n-1) { 1.0} else { if index%2==0 {2.0} else {4.0} };
                let x=x_min+dx*(index as f64);
                accum+(cmp*u*x).exp()*fn_to_invert(x, index)*simpson*dx/3.0
            })
        )
    })
}

fn transform_price(p:f64, v:f64)->f64{p/v}

fn transform_prices(
    arr:&Vec<(f64, f64)>, asset:f64, 
    min_v:&(f64, f64), max_v:&(f64, f64)
)->Vec<(f64, f64)>{
    let mut price_t:Vec<(f64, f64)>=vec![];
    let (min_strike, min_option_price)=min_v;
    let (max_strike, max_option_price)=max_v;
    price_t.push((transform_price(*min_strike, asset), transform_price(*min_option_price, asset)));
    price_t.append(
        &mut arr.iter().map(|(strike, option_price)|{
            (transform_price(*strike, asset), transform_price(*option_price, asset))
        }).collect()
    );
    price_t.push((transform_price(*max_strike, asset), transform_price(*max_option_price, asset)));
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
    let normalized_strike_threshold:f64=1.0;

    let (threshold, _)=*padded_strikes_and_option_prices.iter().rev().find(|(strike, _)|strike<&normalized_strike_threshold).unwrap();

    let (left, right):(Vec<(f64, f64)>, Vec<(f64, f64)>)=padded_strikes_and_option_prices.into_iter().partition(|(strike, _)|threshold_condition(*strike, threshold));


    let left_transform=left.into_iter().map(|(strike, price)|(strike, price-max_zero_or_number(normalized_strike_threshold-strike*discount))).collect();
    let right_transform=right.into_iter().map(|(strike, price)|(strike, price.ln())).collect();
    let s_low=monotone_spline::spline_mov(left_transform);
    let s_high=monotone_spline::spline_mov(right_transform);
    move |strike:f64|{
        if threshold_condition(strike, threshold) {s_low(strike)} else { s_high(strike).exp()-max_zero_or_number(normalized_strike_threshold-strike*discount)}
    }
}


pub fn generate_fo_estimate(
    strikes_and_option_prices:&Vec<(f64, f64)>,
    stock:f64,
    rate:f64,
    maturity:f64,
    min_strike:f64,
    max_strike:f64
)->impl Fn(usize, &Vec<f64>)->Vec<Complex<f64>>
{
    let discount=(-maturity*rate).exp();
    let spline=get_option_spline(
        strikes_and_option_prices,
        stock,
        discount,
        min_strike,
        max_strike
    );
    let cmp:Complex<f64>=Complex::new(0.0, 1.0);
    move |n, u_array|{
        let x_min=(discount*min_strike).ln();
        let x_max=(discount*max_strike).ln();
        dft(u_array, x_min, x_max, n, |x, _|{
            let exp_x=x.exp();
            let strike=exp_x/discount;
            let option_price=spline(strike);
            max_zero_or_number(option_price)
        }).map(|(u, cf)|{
            let front=u*cmp*(1.0+u*cmp);
            (1.0+cf*front).ln()
        }).collect()
    }
}

pub fn get_obj_fn_arr<'a, T>(
    phi_hat:Vec<Complex<f64>>,
    u_array:Vec<f64>,
    cf_fn:T
)->impl Fn(&[f64])->f64
where T:Fn(&[f64])->Box<Fn(&Complex<f64>)->Complex<f64>>
{
    move |params|{
        let cf_inst=cf_fn(params);
        let num_arr=u_array.len();
        u_array.iter().enumerate().fold(0.0, |accumulate, (index, u)|{
            let result=cf_inst(&Complex::new(1.0, *u));
            accumulate+(phi_hat[index]-result).norm_sqr()
        })/(num_arr as f64)
    }
}

#[cfg(test)]
mod tests {
    use option_calibration::*;
    #[test]
    fn test_transform_prices(){
        let arr=vec![(3.0, 3.0), (4.0, 4.0), (5.0, 5.0)];
        let asset=4.0;
        let min_v=(2.0, 2.0);
        let max_v=(6.0, 6.0);
        let result=transform_prices(&arr, asset, &min_v, &max_v);
        let expected=vec![(0.5, 0.5), (0.75, 0.75), (1.0, 1.0), (1.25, 1.25), (1.5, 1.5)];
        for (index, res) in result.iter().enumerate(){
            assert_eq!(
                *res,
                expected[index]
            );
        }
    }

}