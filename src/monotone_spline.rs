extern crate num;

struct Point(f64, f64);

pub fn spline(
    x_and_y:&Vec<Point>
)->Fn(f64)->f64 
{

    let x_and_y_diff=x_and_y.windows(2).map(|point_and_next|{
        let (x_curr, y_curr)=point_and_next[0];
        let (x_next, y_next)=point_and_next[1];

        let x_diff=x_next-x_curr;
        let y_diff=y_next-y_curr;

        (x_diff, y_diff/x_diff)

    }).collect();

    let c1s=x_and_y_diff.iter().take(1)
        .chain(
            x_and_y_diff.windows(2).map(|diff_point_and_next|{
                let (x_curr, dy_dx_curr)=diff_point_and_next[0];
                let (x_next, dy_dx_next)=diff_point_and_next[1];

                let common=x_curr+x_next;
                let c1s_val=if dy_dx_next*dy_dx_curr<=0 {
                    0.0
                } 
                else {
                    3.0*common/((common+x_next)/dy_dx_curr+(common+x_curr)/dy_dx_next)
                }

                (c1s_val, x_curr, dy_dx_curr)
            })
        ).chain(
            x_and_y_diff.iter().last()
        ).collect();
    let c2Andc3=c1s.window(2).map(|c1_diff|{
        let (c1_curr, x_curr, dy_dx_curr)=c1_diff[0];
        let (c1_next, _, _)=c1_diff[1];
        
        let inv_dx=1.0/dx;
        let common=c1_curr+c1_next-2.0*dy_dx_curr;

        ((dy_dx_curr-c1_curr-common)*inv_dx, common*inv_dx.powi(2))
    }).collect();

    move |x|{
        //find x_val such that x is between x_val_prev, x_val_next
        let (found_index, result)=x_and_y.windows(2).enumerate().find(|(index, &&w)| {
            let (x_curr, _)=w[0];
            let (x_next, _)=w[1];
            x>x_curr && x<=x_next 
        });
        let (x_curr, y_curr)=results[0];
        let (x_next, y_next)=results[1];
        let diff=x-x_next;
        let diff_sq=diff.powi(2);
        let (c1s_elem, _, _)=c1s[found_index];
        let (c2s_elem, c3s_elem)=c2Andc3[found_index];
        y_next+c1s_elem*diff+c2s_elem*diff_sq+c3s_elem*diff*diff_sq        
    }
}



#[cfg(test)]
mod tests {
    use monotone_spline::*;
    
}


/**#ifndef __MONOTONE_SPLINE_H_INCLUDED__
#define __MONOTONE_SPLINE_H_INCLUDED__
#include <vector>
#include <cmath>
namespace spline{

    //xs needs to be sorted
    template<typename Arr>
    auto generateSpline(Arr&& xs, Arr&& ys){
        int n=xs.size();
        Arr dxs; //length n-1
        Arr ms; //length n-1
        
        for (int i = 0; i < n-1; ++i) {
            auto dx=xs[i+1]-xs[i];
            auto dy=ys[i+1]-ys[i];
            dxs.emplace_back(dx);
            ms.emplace_back(dy/dx);
        }
        Arr c1s; //length n
        c1s.emplace_back(ms[0]);
        for(int i=0; i<n-2;++i){
            auto m=ms[i];
            auto mNext=ms[i+1];
            if(m*mNext<=0){
                c1s.emplace_back(0);
            }
            else{
                auto dx=dxs[i];
                auto dxNext=dxs[i+1];
                auto common=dx+dxNext;
                c1s.emplace_back(3*common/((common+dxNext)/m+(common+dx)/mNext));
            }
        }
        c1s.emplace_back(ms.back());

        Arr c2s; //length n-1
        Arr c3s; //length n-1
        for(int i=0; i< n-1;++i){
            auto c1=c1s[i];
            auto m=ms[i];
            auto invDx=1.0/dxs[i];
            auto common=c1+c1s[i+1]-2.0*m;
            c2s.emplace_back((m-c1-common)*invDx);
            c3s.emplace_back(common*invDx*invDx);
        }
        return [
            xs=std::move(xs),//length n
            ys=std::move(ys),//length n
            c1s=std::move(c1s), //length n
            c2s=std::move(c2s), //length n-1
            c3s=std::move(c3s), //length n-1
            n=std::move(n)
        ](const auto& x){
            if(x==xs.back()){
                return ys.back();
            }
            auto low=0;
            auto mid=0;
            auto high=n-2; ///hmm
            while(low<=high){
                mid=std::floor(.5*(low+high));
                auto xHere=xs[mid];
                if(xHere<x){
                    low=mid+1;
                }
                else if(xHere>x){
                    high=mid-1;
                }
                else{
                    return ys[mid];
                }
            }
            int index=high>0?high:0;
            auto diff=x-xs[index];
            auto diffsq=diff*diff;
            return ys[index]+c1s[index]*diff+c2s[index]*diffsq+c3s[index]*diff*diffsq;
         
        };
    }

}


#endif */