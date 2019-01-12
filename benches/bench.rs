#[macro_use]
extern crate bencher;
extern crate fang_oost;
extern crate num_complex;
use num_complex::Complex;
use bencher::Bencher;

fn bench_call_price(b: &mut Bencher) {
    let r=0.05;
    let sig=0.3;
    let t=1.0;
    let asset=50.0;
    let bs_cf=|u:&Complex<f64>| ((r-sig*sig*0.5)*t*u+sig*sig*t*u*u*0.5).exp();
    let num_u=64;
    let k_array=vec![0.01, 30.0, 50.0, 70.0, 300.0];//get_fang_oost_strike(-x_max, x_max, asset, num_x);
    b.iter(|| {
        fang_oost_option::option_pricing::fang_oost_call_price(num_u, asset, &k_array, r, t, bs_cf)
    });
}
fn bench_put_theta(b: &mut Bencher) {
    let r=0.05;
    let sig=0.3;
    let t=1.0;
    let asset=50.0;
    let bs_cf=|u:&Complex<f64>| ((r-sig*sig*0.5)*t*u+sig*sig*t*u*u*0.5).exp();
    let num_u=64;
    let k_array=vec![0.01, 30.0, 50.0, 70.0, 300.0];
    b.iter(|| {
        fang_oost_option::option_pricing::fang_oost_put_theta(num_u, asset, &k_array, r, t, bs_cf);
    });
}


benchmark_group!(benches, bench_call_price, bench_put_theta);
benchmark_main!(benches);
#[cfg(never)]
fn main() { }