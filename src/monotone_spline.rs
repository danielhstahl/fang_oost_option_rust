extern crate num;
#[cfg(test)]
extern crate statrs;

pub fn spline<'a>(
    x_and_y:&'a Vec<(f64, f64)>
)-> impl Fn(f64) -> f64 +'a
{

    let x_and_y_diff:Vec<(f64, f64, f64)>=x_and_y.windows(2).map(|point_and_next|{
        let (x_curr, y_curr)=point_and_next[0];
        let (x_next, y_next)=point_and_next[1];

        let x_diff=x_next-x_curr;
        let y_diff=y_next-y_curr;
        let ms=y_diff/x_diff;
        (ms, x_diff, ms)

    }).collect();

    //begin mut 
    let mut c1s:Vec<(f64, f64, f64)>=vec![];
    c1s.push(*x_and_y_diff.first().unwrap());
    c1s.append(&mut x_and_y_diff.windows(2).map(|diff_point_and_next|{
        let (_, x_curr, dy_dx_curr)=diff_point_and_next[0];
        let (_, x_next, dy_dx_next)=diff_point_and_next[1];

        let common=x_curr+x_next;
        let c1s_val=if dy_dx_next*dy_dx_curr<=0.0 {
            0.0
        } 
        else {
            3.0*common/((common+x_next)/dy_dx_curr+(common+x_curr)/dy_dx_next)
        };
        (c1s_val, x_curr, dy_dx_curr)
    }).collect());
    c1s.push(*x_and_y_diff.last().unwrap());
    //end mut
    let c2_and_c3:Vec<(f64, f64)>=c1s.windows(2).map(|c1_diff|{
        let (c1_curr, dx_curr, dy_dx_curr)=c1_diff[0];
        let (c1_next, _, _)=c1_diff[1];
        
        let inv_dx=1.0/dx_curr;
        let common=c1_curr+c1_next-2.0*dy_dx_curr;

        ((dy_dx_curr-c1_curr-common)*inv_dx, common*inv_dx.powi(2))
    }).collect();

    move |x|{
        //find x_val such that x is between x_val_prev, x_val_next
        let (found_index, results)=x_and_y.windows(2).enumerate().find(|(_, w)| {
            let (x_curr, _)=w[0];
            let (x_next, _)=w[1];
            x>x_curr && x<=x_next 
        }).unwrap();

        //let (x_curr, y_curr)=results[0];
        let (x_next, y_next)=results[1];
        let diff=x-x_next;
        let diff_sq=diff.powi(2);
        let (c1s_elem, _, _)=c1s[found_index];
        let (c2s_elem, c3s_elem)=c2_and_c3[found_index];
        y_next+c1s_elem*diff+c2s_elem*diff_sq+c3s_elem*diff*diff_sq        
    }
}

#[cfg(test)]
mod tests {
    use monotone_spline::*;
    #[test]
    fn test_returns_value_at_knot(){
        let x_y:Vec<(f64, f64)>=vec![(1.0, 2.0), (2.0, 2.5), (3.0, 3.0)];
        let spline=spline(&x_y);
        for (x, y) in x_y.iter(){
            assert_abs_diff_eq!(
                spline(*x),
                y,
                epsilon=0.000000001
            );
        }
        
    }
}
