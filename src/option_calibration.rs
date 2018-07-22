extern crate num;
extern crate num_complex;
extern crate rayon;
extern crate fang_oost;

#[cfg(test)]
use std::f64::consts::PI;

use self::num_complex::Complex;
use self::rayon::prelude::*;
use std;
use monotone_spline;

pub fn max_zero_or_number(num:f64)->f64{
    if num>0.0 {num} else {0.0}
}

fn get_dx(n:usize, x_min:f64, x_max:f64)->f64{
    (x_max-x_min)/(n as f64-1.0)
}
fn dft<'a, 'b: 'a>(
    u_array:&'b [f64],
    x_min:f64,
    x_max:f64,
    n:usize,
    fn_to_invert:impl Fn( f64, usize)->f64+'a+std::marker::Sync+std::marker::Send
)->impl ParallelIterator<Item = (f64, Complex<f64>) >+'a
{
    let cmp:Complex<f64>=Complex::new(0.0, 0.0);
    let cmp_i:Complex<f64>=Complex::new(0.0, 1.0);
    let dx=get_dx(n, x_min, x_max);
    u_array.par_iter().map(move |u|{
        (
            *u, 
            (0..n).fold(cmp, |accum, index|{
                let simpson=if index==0||index==(n-1) { 
                    1.0
                } else { 
                    if index%2==0 {
                        2.0
                    } else {
                        4.0
                    } 
                };
                let x=x_min+dx*(index as f64);
                accum+(cmp_i*u*x).exp()*fn_to_invert(x, index)*simpson*dx/3.0
            })
        )
    })
}

fn transform_price(p:f64, v:f64)->f64{p/v}

fn transform_prices(
    arr:&[(f64, f64)], asset:f64, 
    min_v:&(f64, f64), max_v:&(f64, f64)
)->Vec<(f64, f64)>{
    let mut price_t:Vec<(f64, f64)>=vec![];
    let (min_strike, min_option_price)=min_v;
    let (max_strike, max_option_price)=max_v;
    price_t.push(
        (
            transform_price(*min_strike, asset), 
            transform_price(*min_option_price, asset)
        )
    );
    price_t.append(
        &mut arr.iter().map(|(strike, option_price)|{
            (
                transform_price(*strike, asset), 
                transform_price(*option_price, asset)
            )
        }).collect()
    );
    price_t.push(
        (
            transform_price(*max_strike, asset), 
            transform_price(*max_option_price, asset)
        )
    );
    price_t
}
fn threshold_condition(strike:f64, threshold:f64)->bool{strike<=threshold}

pub fn get_option_spline<'a>(
    strikes_and_option_prices:&[(f64, f64)],
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

    let (left, mut right):(
        Vec<(f64, f64)>, 
        Vec<(f64, f64)>
    )=padded_strikes_and_option_prices
        .into_iter()
        .rev() //reverse so I can push back on right to get left threshold
        .partition(|(normalized_strike, _)|{
            normalized_strike<=&normalized_strike_threshold
        });
    let threshold_t=left.first().unwrap().clone();//clone so I can push into right
    let (threshold, _)=threshold_t;
    right.push(threshold_t);

    let left_transform:Vec<(f64, f64)>=left
        .into_iter()
        .rev()
        .map(|(normalized_strike, normalized_price)|{
            (
                normalized_strike, 
                normalized_price-max_zero_or_number(
                    normalized_strike_threshold-normalized_strike*discount
                )
            )
        }).collect();

    let right_transform:Vec<(f64, f64)>=right
        .into_iter()
        .rev()
        .map(|(normalized_strike, normalized_price)|{
            (
                normalized_strike, 
                normalized_price.ln()
            )
        }).collect();
    let s_low=monotone_spline::spline_mov(left_transform);
    let s_high=monotone_spline::spline_mov(right_transform);
    move |normalized_strike:f64|{
        if threshold_condition(normalized_strike, threshold) {
            s_low(normalized_strike)
        } else { 
            s_high(normalized_strike).exp()-max_zero_or_number(
                normalized_strike_threshold-normalized_strike*discount
            )
        }
    }
}


pub fn generate_fo_estimate(
    strikes_and_option_prices:&[(f64, f64)],
    stock:f64,
    rate:f64,
    maturity:f64,
    min_strike:f64,
    max_strike:f64
)->impl Fn(usize, &[f64])->Vec<Complex<f64>>
{
    let discount=(-maturity*rate).exp();
    let spline=get_option_spline(
        strikes_and_option_prices,
        stock,
        discount,
        min_strike, //transformed internally to min_strike/asset
        max_strike  //transformed internally to max_strike/asset
    );
    let cmp:Complex<f64>=Complex::new(0.0, 1.0);
    let x_min=(discount*transform_price(min_strike, stock)).ln();
    let x_max=(discount*transform_price(max_strike, stock)).ln();
    move |n, u_array|{
        dft(u_array, x_min, x_max, n, |x, _|{
            let exp_x=x.exp();
            let strike=exp_x/discount;
            let option_price_t=spline(strike);
            max_zero_or_number(option_price_t)
        }).map(|(u, cf)|{
            let front=u*cmp*(1.0+u*cmp);
            (1.0+cf*front).ln()
        }).collect()
    }
}
const LARGE_NUMBER:f64=500000.0;
pub fn get_obj_fn_arr<'a, T>(
    phi_hat:Vec<Complex<f64>>, //do we really want to borrow/move this??
    u_array:Vec<f64>,
    cf_fn:T
)->impl Fn(&[f64])->f64
where T:Fn(&Complex<f64>, &[f64])->Complex<f64>
{
    move |params|{
        let num_arr=u_array.len();
        u_array.iter().enumerate().fold(0.0, |accumulate, (index, u)|{
            let result=cf_fn(&Complex::new(1.0, *u), params);
            accumulate+if result.re.is_nan()||result.im.is_nan() {
                LARGE_NUMBER
            }
            else {
                (phi_hat[index]-result).norm_sqr()
            }            
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
    #[test]
    fn test_get_obj_one_parameter(){
        let cf=|u:&Complex<f64>, _sl:&[f64]|Complex::new(u.im, 0.0);
        let arr=vec![Complex::new(3.0, 0.0), Complex::new(4.0, 0.0), Complex::new(5.0, 0.0)];
        let u_arr=vec![6.0, 7.0, 8.0];
        let hoc=get_obj_fn_arr(
            arr,
            u_arr,
            cf
        );
        let expected=9.0;//3*3^2/3
        let tmp:f64=0.0;
        assert_eq!(hoc(&[tmp]), expected);
    }
    #[test]
    fn test_option_spline(){
        let tmp_strikes_and_option_prices:Vec<(f64, f64)>=vec![
            (95.0, 85.0), 
            (130.0, 51.5), 
            (150.0, 35.38), 
            (160.0, 28.3), 
            (165.0, 25.2), 
            (170.0, 22.27), 
            (175.0, 19.45), 
            (185.0, 14.77), 
            (190.0, 12.75), 
            (195.0, 11.0), 
            (200.0, 9.35), 
            (210.0, 6.9), 
            (240.0, 2.55), 
            (250.0, 1.88)
        ];
        let maturity:f64=1.0;
        let rate=0.05;
        let asset=178.46;
        let discount=(-rate*maturity).exp();
        let spline=get_option_spline(
            &tmp_strikes_and_option_prices, 
            asset, discount, 0.00001, 5000.0
        );
        let sp_result=spline(160.0/asset);
        assert_eq!(sp_result, 28.3/asset-max_zero_or_number(1.0-(160.0/asset)*discount));
    }
    #[test]
    fn test_option_spline_at_many_values(){
        let tmp_strikes_and_option_prices:Vec<(f64, f64)>=vec![
            (95.0, 85.0), 
            (130.0, 51.5), 
            (150.0, 35.38), 
            (160.0, 28.3), 
            (165.0, 25.2), 
            (170.0, 22.27), 
            (175.0, 19.45), 
            (185.0, 14.77), 
            (190.0, 12.75), 
            (195.0, 11.0), 
            (200.0, 9.35), 
            (210.0, 6.9), 
            (240.0, 2.55), 
            (250.0, 1.88)
        ];
        let maturity:f64=1.0;
        let rate=0.05;
        let asset=178.46;
        let discount=(-rate*maturity).exp();
        let spline=get_option_spline(
            &tmp_strikes_and_option_prices, 
            asset, discount, 0.00001, 5000.0
        );
        let test_vec=vec![4.0, 100.0, 170.0, 175.0, 178.0, asset, 179.0, 185.0, 500.0];
        test_vec.iter().for_each(|v|{
            let _sp_result=spline(v/asset); //will panic if doesnt work
        });
        tmp_strikes_and_option_prices.iter().for_each(|(strike, price)|{
            let sp_result=spline(strike/asset);
            assert_abs_diff_eq!(sp_result, price/asset-max_zero_or_number(1.0-(strike/asset)*discount), epsilon=0.0000001);
        });
    }
    #[test]
    fn test_generate_fo_runs(){
        let tmp_strikes_and_option_prices:Vec<(f64, f64)>=vec![
            (95.0, 85.0), 
            (130.0, 51.5), 
            (150.0, 35.38), 
            (160.0, 28.3), 
            (165.0, 25.2), 
            (170.0, 22.27), 
            (175.0, 19.45), 
            (185.0, 14.77), 
            (190.0, 12.75), 
            (195.0, 11.0), 
            (200.0, 9.35), 
            (210.0, 6.9), 
            (240.0, 2.55), 
            (250.0, 1.88)
        ];
        let maturity:f64=1.0;
        let rate=0.05;
        let asset=178.46;
        let hoc_fn=generate_fo_estimate(
            &tmp_strikes_and_option_prices, 
            asset, rate, 
            maturity, 
            0.01, 
            5000.0
        );
        let n:usize=15;
        let du= 2.0*PI/(n as f64);
        let u_array:Vec<f64>=(1..n).map(|index|index as f64*du).collect();
        let _result=hoc_fn(1024, &u_array);
        
    }
    #[test]
    fn test_generate_fo_accuracy(){
        let tmp_strikes_and_option_prices:Vec<(f64, f64)>=vec![
            (95.0, 85.0), 
            (130.0, 51.5), 
            (150.0, 35.38), 
            (160.0, 28.3), 
            (165.0, 25.2), 
            (170.0, 22.27), 
            (175.0, 19.45), 
            (185.0, 14.77), 
            (190.0, 12.75), 
            (195.0, 11.0), 
            (200.0, 9.35), 
            (210.0, 6.9), 
            (240.0, 2.55), 
            (250.0, 1.88)
        ];
        let maturity:f64=1.0;
        let rate=0.05;
        let asset=178.46;
        let hoc_fn=generate_fo_estimate(
            &tmp_strikes_and_option_prices, 
            asset, rate, 
            maturity, 
            0.01, 
            5000.0
        );
        let n:usize=15;
        let du= 2.0*PI/(n as f64);
        let u_array:Vec<f64>=(1..n).map(|index|index as f64*du).collect();
        let result=hoc_fn(1024, &u_array);
        for v in result.iter(){
            println!("this is v: {}", v);
        }
    }
    #[test]
    fn test_dft(){
        let u_array=vec![2.0];
        let x_min=-5.0;
        let x_max=5.0;
        let n:usize=10;
        let fn_to_invert=|x:f64, _| x.powi(2);
        let result:Vec<Complex<f64>>=dft(&u_array, x_min, x_max, n, fn_to_invert).map(|(_, v)|v).collect();
        assert_abs_diff_eq!(result[0].re, -5.93082, epsilon=0.00001);
        assert_abs_diff_eq!(result[0].im, -14.3745, epsilon=0.00001);
    }
    #[test]
    fn test_monotone_spline(){
        let tmp_strikes_and_option_prices:Vec<(f64, f64)>=vec![
            (95.0, 85.0), 
            (130.0, 51.5), 
            (150.0, 35.38), 
            (160.0, 28.3), 
            (165.0, 25.2), 
            (170.0, 22.27), 
            (175.0, 19.45), 
            (185.0, 14.77), 
            (190.0, 12.75), 
            (195.0, 11.0), 
            (200.0, 9.35), 
            (210.0, 6.9), 
            (240.0, 2.55), 
            (250.0, 1.88)
        ];
        let maturity:f64=1.0;
        let rate=0.05;
        let asset=178.46;
        let discount=(-rate*maturity).exp();
        let spline=get_option_spline(
            &tmp_strikes_and_option_prices, 
            asset, discount, 0.00001, 5000.0
        );
        let test_vec=vec![4.0, 100.0, 170.0, 175.0, 178.0, asset, 179.0, 185.0, 190.0, 195.0, 200.0, 205.0, 208.0, 209.0, 210.0, 215.0, 218.0, 220.0, 500.0];

        test_vec.iter().for_each(|v|{
            let sp_result=spline(v/asset); //will panic if doesnt work
            println!("spline at: {}: {}", v, sp_result);
        });
    }

}