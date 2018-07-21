extern crate num;
const LARGEST_APPROPRIATE_DISTANCE:f64=0.00000000001;
pub fn spline_mov(
    x_and_y:Vec<(f64, f64)>
)-> impl Fn(f64) -> f64
{
    let first_x_and_y=*x_and_y.first().expect("input vector should be larger than length one");
    let last_x_and_y=*x_and_y.last().expect("input vector should be larger than length one");
    let x_and_y_diff:Vec<(f64, f64)>=x_and_y.windows(2).map(|point_and_next|{
        let (x_curr, y_curr)=point_and_next[0];
        let (x_next, y_next)=point_and_next[1];

        let x_diff=x_next-x_curr;
        let y_diff=y_next-y_curr;
        let ms=y_diff/x_diff;
        (x_diff, ms)

    }).collect();

    //begin mut 
    let mut c1s:Vec<f64>=vec![];
    let first_diff=*x_and_y_diff.first().expect("input vector should be larger than length one");
    let last_diff=*x_and_y_diff.last().expect("input vector should be larger than length one");
    c1s.push(first_diff.1);
    c1s.append(&mut x_and_y_diff.windows(2).map(|diff_point_and_next|{
        let (x_diff_curr, dy_dx_curr)=diff_point_and_next[0];
        let (x_diff_next, dy_dx_next)=diff_point_and_next[1];

        let common=x_diff_curr+x_diff_next;
        if dy_dx_next*dy_dx_curr<=0.0 {
            0.0
        } 
        else {
            3.0*common/((common+x_diff_next)/dy_dx_curr+(common+x_diff_curr)/dy_dx_next)
        }
    }).collect());
    c1s.push(last_diff.1);
    //end mut
    let c2_and_c3:Vec<(f64, f64)>=c1s.windows(2).zip(x_and_y_diff.into_iter()).map(|(c1_diff, x_and_y_incr)|{
        let c1_curr=c1_diff[0];
        let c1_next=c1_diff[1];
        let (x_diff_curr, dy_dx_curr)=x_and_y_incr;      
        let inv_dx=1.0/x_diff_curr;
        let common=c1_curr+c1_next-2.0*dy_dx_curr;
        ((dy_dx_curr-c1_curr-common)*inv_dx, common*inv_dx.powi(2))
    }).collect();

    move |x|{
        let (x_min, y_min)=x_and_y.first().expect("input vector should be larger than length one");
        let (x_max, y_max)=x_and_y.last().expect("input vector should be larger than length one");
        if (x_min-x).abs() <= LARGEST_APPROPRIATE_DISTANCE {
            return *y_min
        }
        if (x_max-x).abs() <= LARGEST_APPROPRIATE_DISTANCE {
            return *y_max
        }
        //find x_val such that x is between x_val_prev, x_val_next
        let (found_index, results)=x_and_y.windows(2).enumerate().find(|(_, w)| {
            let (x_curr, _)=w[0];
            let (x_next, _)=w[1];
            x>=x_curr && x<x_next 
        }).expect(&format!("Requires x to be between the bounds!  x is currently {}, lower bound is {}, upper bound is {}", x, first_x_and_y.0, last_x_and_y.0));

        let (x_curr, y_curr)=results[0];
        let diff=x-x_curr;
        let diff_sq=diff.powi(2);
        let c1s_elem=c1s[found_index];
        let (c2s_elem, c3s_elem)=c2_and_c3[found_index];
        y_curr+c1s_elem*diff+c2s_elem*diff_sq+c3s_elem*diff*diff_sq        
    }
}


#[cfg(test)]
mod tests {
    use monotone_spline::*;
    #[test]
    fn test_returns_value_at_knot(){
        let x_y:Vec<(f64, f64)>=vec![(1.0, 2.0), (2.0, 2.5), (3.0, 3.0)];
        let spline=spline_mov(x_y);
        let x_y_2:Vec<(f64, f64)>=vec![(1.0, 2.0), (2.0, 2.5), (3.0, 3.0)];
        for (x, y) in x_y_2.iter(){
            assert_abs_diff_eq!(
                spline(*x),
                y,
                epsilon=0.000000001
            );
        }
        
    }
    #[test]
    fn test_returns_in_between_value(){
        let x_y:Vec<(f64, f64)>=vec![(1.0, 2.0), (2.0, 2.5), (3.0, 3.0)];
        let spline=spline_mov(x_y);
        let x_y_2:Vec<(f64, f64)>=vec![(1.0, 2.0), (2.0, 2.5), (3.0, 3.0)];
        let test_x:Vec<f64>=vec![1.01, 1.02, 1.03, 1.4, 1.8, 1.98, 1.99, 2.01, 2.02, 2.03, 2.4, 2.8, 2.98, 2.99];
        for x in test_x.iter(){
            let x_d_ref=*x;
            let spline_y=spline(x_d_ref);
            let y_bounds=x_y_2.windows(2).find(|w|{
                let (x_curr, _)=w[0];
                let (x_next, _)=w[1];
                x_next>x_d_ref && x_curr<x_d_ref
            }).unwrap();
            let (_, y_curr)=y_bounds[0];
            let (_, y_next)=y_bounds[1];
            //tests that spline truly is monotonic
            assert_eq!(
                spline_y>y_curr&&spline_y<y_next,
                true
            );
        }
        
    }

}
