#![allow(dead_code)]
use std::f32::consts::PI;

pub fn torus(u:f32, v:f32) -> [f32; 3] {
    let x = (1.0 + 0.3 * v.cos()) * u.cos();
    let y = 0.3 * v.sin();
    let z = (1.0 + 0.3 * v.cos()) * u.sin();
    [x, y, z]
}

pub fn sphere(u:f32, v:f32) -> [f32; 3] {
    let x = v.sin() * u.cos();
    let y = v.cos();
    let z = -v.sin() * u.sin();
    [x, y, z]
}

pub fn breather(u:f32, v:f32) -> [f32; 3] {
    const A:f32 = 0.4; // where 0 < A < 1

    let de = A*((1.0-A*A)* ((A*u).cosh()).powf(2.0)+A*A*((((1.0-A*A).sqrt()*v).sin()).powf(2.0)));

    let x = -u+(2.0*(1.0-A*A)*(A*u).cosh()*(A*u).sinh())/de;
    
    let y = (2.0*(1.0-A*A).sqrt()*(A*u).cosh()*(-((1.0-A*A).sqrt()*v.cos()*((1.0-A*A).sqrt()*v).cos()) - 
        v.sin()*((1.0-A*A).sqrt()*v).sin()))/de;    
    
    let z = (2.0*(1.0-A*A).sqrt()*(A*u).cosh()*(-((1.0-A*A).sqrt()*v.sin()*((1.0-A*A).sqrt()*v).cos()) + 
        v.cos()*((1.0-A*A).sqrt()*v).sin()))/de;

    [x, y, z]
}

pub fn sievert_enneper(u:f32, v:f32) -> [f32; 3] {
    const A:f32 = 1.0;
    
    let pu = -u/(1.0+A).sqrt() + (u.tan()*(1.0+A).sqrt()).atan();
    let auv = 2.0/(1.0+A-A*v.sin()*v.sin()*u.cos()*u.cos());
    let ruv = auv*v.sin()*((1.0+1.0/A)*(1.0+A*u.sin()*u.sin())).sqrt();

    let x = (((v/2.0).tan()).ln() + (1.0+A)*auv*v.cos()) /A.sqrt();
    let y = ruv*pu.cos();
    let z = ruv*pu.sin();

    [x, y, z]
}

pub fn seashell(u:f32, v:f32) -> [f32; 3] {
    let x = 2.0*(-1.0+(u/(6.0*PI)).exp())*u.sin()*(((v/2.0).cos()).powf(2.0));

    let y = 1.0 - (u/(3.0*PI)).exp()-v.sin() + (u/(6.0*PI)).exp()*v.sin();

    let z = 2.0*(1.0-(u/(6.0*PI)).exp())*u.cos()*((v/2.0).cos()).powf(2.0);

    [x, y, z]
}

pub fn wellenkugel(u:f32, v:f32) -> [f32; 3] {
    let x = u*(u.cos()).cos()*v.sin();        
    let y = u*(u.cos()).sin();
    let z = u*(u.cos()).cos()*v.cos();    
    [x, y, z]   
}

pub fn figure8(u:f32, v:f32) -> [f32; 3] {
    let a = 2.5f32;
    let x = (a + (0.5 * u).cos() * v.sin() - (0.5 * u).sin() * (2.0 * v).sin()) * u.cos();        
    let y = (a + (0.5 * u).cos() * v.sin() - (0.5 * u).sin() * (2.0 * v).sin()) * u.sin();
    let z = (0.5 * u).sin() * v.sin() + (0.5 * u).cos() * (2.0 * v).sin();    
    [x, y, z]   
}

pub fn klein_bottle3(u:f32, v:f32) -> [f32; 3] {
    let a = 8f32;
    let n = 3f32;
    let m = 1f32;
    
    let x = (a + (0.5 * u * n).cos() * v.sin() - (0.5 * u * n).sin() * (2.0 * v).sin()) * (0.5 * u * m).cos();        
    let y = (0.5 * u * n).sin() * v.sin() + (0.5 * u * n).cos() * (2.0 * v).sin();
    let z = (a + (0.5 * u * n).cos() * v.sin() - (0.5 * u * n).sin() * (2.0 * v).sin()) * (0.5 * u * m).sin();   
    [x, y, z]   
}

pub fn klein_bottle2(u:f32, v:f32) -> [f32; 3] {
    let (mut x, mut z) = (0f32, 0f32);
    let r = 4.0 * (1.0 - 0.5 * u.cos());
    if u >= 0.0 && u <= PI {
        x = 6.0 * u.cos() * (1.0 + u.sin()) + r * u.cos() * v.cos();
        z = 16.0 * u.sin() + r * u.sin() * v.cos();
    }  else if u > PI && u <= 2.0 * PI {
        x = 6.0 * u.cos() * (1.0 + u.sin()) + r *(v + PI).cos();
        z = 16.0 * u.sin();
    }
    let y = r * v.sin();
    [x, y, z]
}

pub fn klein_bottle(u:f32, v:f32) -> [f32; 3] {
    let x = 2.0/15.0*(3.0+5.0*u.cos()*u.sin())*v.sin(); 

    let y = -1.0/15.0*u.sin()*(3.0*v.cos()-3.0*(u.cos()).powf(2.0)*v.cos()-
    48.0*(u.cos()).powf(4.0)*v.cos()+48.0*(u.cos()).powf(6.0)*v.cos()-
    60.0*u.sin()+5.0*u.cos()*v.cos()*u.sin()-5.0*(u.cos()).powf(3.0)*v.cos()*u.sin()-
    80.0*(u.cos()).powf(5.0)*v.cos()*u.sin()+80.0*(u.cos()).powf(7.0)*v.cos()*u.sin());

    let z = -2.0/15.0*u.cos()*(3.0*v.cos()-30.0*u.sin() +
    90.0*(u.cos()).powf(4.0)*u.sin()-60.0*(u.cos()).powf(6.0)*u.sin() + 5.0*u.cos()*v.cos()*u.sin());

    [x, y, z]
}

pub fn astroid(u:f32, v:f32) -> [f32; 3] {
    let a = 1.5f32;
    let x = a * (u.cos()).powf(3.0) * (v.cos()).powf(3.0);
    let y = a * (u.sin()).powf(3.0);
    let z = a * (u.sin()).powf(3.0) * (v.cos()).powf(3.0);
    [x, y, z]
}

pub fn astroid2(u:f32, v:f32) -> [f32; 3] {
    let x = (u.sin()).powf(3.0) * v.cos();
    let y = (u.cos()).powf(3.0);
    let z = (u.sin()).powf(3.0) * v.sin();
    [x, y, z]
}

pub fn astroidal_torus(u:f32, v:f32) -> [f32; 3] {
    let a = 2.0;
    let b = 1.0;
    let c = 7854.0f32;
    let x = (a + b * (u.cos()).powf(3.0) * c.cos() - b * (u.sin()).powf(3.0) * c.sin()) * v.cos();
    let y = b * (u.cos()).powf(3.0) * c.sin() + b * (u.sin()).powf(3.0) * c.cos();
    let z = (a + b * (u.cos()).powf(3.0) * c.cos() - b * (u.sin()).powf(3.0) * c.sin()) * v.sin();
    [x, y, z]
}

pub fn bohemian_dome(u:f32, v:f32) -> [f32; 3] {
    let a = 0.7;
    let x = a * u.cos();
    let y = v.cos();
    let z = a * u.sin() + v.sin();
    [x, y, z]
}

pub fn boy_shape(u:f32, v:f32) -> [f32; 3] {
    let x = u.cos() * (1.0 / 3.0 * 2.0f32.sqrt() * u.cos() * (2.0 * v).cos() + 
        2.0 / 3.0 * u.sin() * v.cos()) / (1.0 - 2.0f32.sqrt() * u.sin() * u.cos() * (3.0 * v).sin());
    let y = u.cos() * u.cos() / (1.0 - 2.0f32.sqrt() * u.sin() * u.cos() * (3.0 * v).sin()) - 1.0;
    let z = u.cos() * (1.0 / 3.0 * 2.0f32.sqrt() * u.cos() * (2.0 * v).sin() - 
    2.0 / 3.0 * u.sin() * v.sin()) / (1.0 - 2.0f32.sqrt() * u.sin() * u.cos() * (3.0 * v).sin());
    [x, y, z]
}

pub fn enneper(u:f32, v:f32) -> [f32; 3] {
    let a = 1.0/3.0;
    let x = a * u * (1.0 - u * u / 3.0 + v * v);
    let y = a * (u * u - v * v);
    let z = a * v * (1.0 - v * v / 3.0 + u * u);
    [x, y, z]
}

pub fn henneberg(u:f32, v:f32) -> [f32; 3] {
    let x = u.sinh() * v.cos() - (3.0*u).sinh() * (3.0*v).cos()/3.0;
    let y = (2.0*u).cosh() * (2.0*v).cos();
    let z = u.sinh() * v.sin() - (3.0*u).sinh() * (3.0*v).sin()/3.0;
    [x, y, z]
}

pub fn kiss(u:f32, v:f32) -> [f32; 3] {
    let x = u * u * (1.0-u).sqrt() * v.cos();
    let y = u;
    let z = u * u * (1.0-u).sqrt() * v.sin();
    [x, y, z]
}

pub fn kuen(u:f32, v:f32) -> [f32; 3] {
    let x = 2.0 * u * v.cos();
    let y = 2.0 * (3.0 * v).cos();
    let z = 2.0 * u * v.sin();
    [x, y, z]
}

pub fn minimal(u:f32, v:f32) -> [f32; 3] {
    let x = u - (2.0*u).exp() * (2.0 *v).cos() /2.0 ;
    let y = 2.0 * u.exp() * v.cos();
    let z = -(v + (2.0*u).exp() * (2.0 * v).sin()/2.0);
    [x, y, z]
}

pub fn parabolic_cyclide(u:f32, v:f32) -> [f32; 3] {
    let x = u * (0.5 + v*v)/(1.0 + u*u + v*v);
    let y = 0.5 * (2.0*v*v + 0.5*(1.0 - u*u - v*v))/(1.0 + u*u + v*v);
    let z = v * (1.0 + u*u -0.5)/(1.0 + u*u + v*v);
    [x, y, z]
}

pub fn pear(u:f32, v:f32) -> [f32; 3] {
    let x = u * (u * (1.0 - u)).sqrt() * v.cos();
    let y = -u;
    let z = u * (u * (1.0 - u)).sqrt() * v.sin();
    [x, y, z]
}

pub fn plucker_conoid(u:f32, v:f32) -> [f32; 3] {
    let x = 2.0 * u * v.cos();
    let y = 2.0 * (3.0 * v).cos();
    let z = 2.0 * u * v.sin();
    [x, y, z]
}

pub fn steiner(u:f32, v:f32) -> [f32; 3] {
    let x = u.cos() * v.cos() * v.sin();
    let y = u.cos() * u.sin() * (v.cos()).powf(2.0);
    let z = u.sin() * v.cos() * v.sin();
    [x, y, z]
}

pub fn sinc(x:f32, z:f32, t:f32) -> [f32; 3] {
    let a = 1.01 + t.sin();
    let r = a * (x*x + z*z).sqrt();
    let y = if r == 0.0 { 1.0 } else { r.sin()/r };
    [x, y, z]
}

pub fn peaks(x:f32, z:f32, t:f32) -> [f32; 3] {
    let a = 1.00001 + t.sin();
    let b = 1.00001 + (1.5*t).sin();
    let c = 1.00001 + (2.0*t).sin();    
    let y = 3.0*(1.0-x)*(1.0-x)*(-a*(x*x)-a*(z+1.0)*(z+1.0)).exp()-
    10.0*(x/5.0-x*x*x-z*z*z*z*z)*(-b*x*x-b*z*z).exp() - 1.0/3.0*(-c*(x+1.0)*(x+1.0)-c*z*z).exp();
    [z, y, x]
}

pub fn poles(x:f32, z:f32, t:f32) -> [f32; 3] {
    let a = 1.5 * t.sin();
    let y =  x*z/(((x-a)*(x-a)*(x-a)).abs() + (z- 2.0*a)*(z- 2.0*a) + 2.0);
    [x, y, z]
}
