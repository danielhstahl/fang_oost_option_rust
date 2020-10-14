#[macro_use]
extern crate criterion;
extern crate fang_oost;
extern crate num_complex;
use criterion::Criterion;
use num_complex::Complex;

fn bench_call_price(c: &mut Criterion) {
    let r = 0.05;
    let sig = 0.3;
    let t = 1.0;
    let asset = 50.0;
    let bs_cf =
        move |u: &Complex<f64>| ((r - sig * sig * 0.5) * t * u + sig * sig * t * u * u * 0.5).exp();
    let num_u = 64;
    let k_array = vec![30.0, 50.0, 70.0];
    let max_strike = 300.0;
    c.bench_function("call price", move |b| {
        b.iter(|| {
            fang_oost_option::option_pricing::fang_oost_call_price(
                num_u, asset, &k_array, max_strike, r, t, bs_cf,
            )
        })
    });
}
fn bench_put_theta(c: &mut Criterion) {
    let r = 0.05;
    let sig = 0.3;
    let t = 1.0;
    let asset = 50.0;
    let bs_cf =
        move |u: &Complex<f64>| ((r - sig * sig * 0.5) * t * u + sig * sig * t * u * u * 0.5).exp();
    let num_u = 64;
    let k_array = vec![30.0, 50.0, 70.0];
    let max_strike = 300.0;
    c.bench_function("put theta", move |b| {
        b.iter(|| {
            fang_oost_option::option_pricing::fang_oost_put_theta(
                num_u, asset, &k_array, max_strike, r, t, bs_cf,
            )
        })
    });
}

criterion_group!(benches, bench_call_price, bench_put_theta);
criterion_main!(benches);
