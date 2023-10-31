#![allow(dead_code)]
use cgmath::*;
use std::f32::consts::PI;
use std::collections::HashMap;
use super::colormap;
use super::math_func as mf;

#[derive(Default)]
pub struct ISurfaceOutput {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 3]>,
    pub colors2: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u16>,
    pub indices2: Vec<u16>,
} 

// region: parametric surface
pub struct IParametricSurface {
    pub surface_type: u32,
    pub surface_type_map: HashMap<u32, String>,
    pub umin: f32,
    pub umax: f32,
    pub vmin: f32,
    pub vmax: f32,
    pub u_resolution: u16,
    pub v_resolution: u16,
    pub scale: f32,
    pub aspect_ratio: f32,
    pub colormap_name: String,
    pub wireframe_color: String,
    pub colormap_direction: u32, // 0: x-direction, 1: y-direction, 2: z-direction
    pub uv_lens: [f32; 2],
}

fn surface_type_map() -> HashMap<u32,String> {
    let mut surface_type = HashMap::new();
    surface_type.insert(0, String::from("klein_bottle"));
    surface_type.insert(1, String::from("astroid"));
    surface_type.insert(2, String::from("astroid2"));
    surface_type.insert(3, String::from("astrodal_torus"));
    surface_type.insert(4, String::from("bohemian_dome"));
    surface_type.insert(5, String::from("boy_shape"));
    surface_type.insert(6, String::from("breather"));
    surface_type.insert(7, String::from("enneper"));
    surface_type.insert(8, String::from("figure8"));
    surface_type.insert(9, String::from("henneberg"));
    surface_type.insert(10, String::from("kiss"),);
    surface_type.insert(11, String::from("klein_bottle2"));
    surface_type.insert(12, String::from("klein_bottle3"));
    surface_type.insert(13, String::from("kuen"));
    surface_type.insert(14, String::from("minimal"));
    surface_type.insert(15, String::from("parabolic_cyclide"));
    surface_type.insert(16, String::from("pear"));
    surface_type.insert(17, String::from("plucker_conoid"));
    surface_type.insert(18, String::from("seashell"));
    surface_type.insert(19, String::from("sievert_enneper"));
    surface_type.insert(20, String::from("steiner"));
    surface_type.insert(21, String::from("torus"));
    surface_type.insert(22, String::from("wellenkugel"));
    surface_type
}

pub fn get_surface_type(key:u32) -> String {
    let map = surface_type_map();
    map.get(&key).map(|s| s.to_string()).unwrap_or_default()
}

impl Default for IParametricSurface {
    fn default() -> Self {
        Self {
            surface_type: 0,
            surface_type_map: surface_type_map(),
            umin: -1.0,
            umax: 1.0,
            vmin: -1.0,
            vmax: 1.0,
            u_resolution: 80,
            v_resolution: 80,
            scale: 1.0,
            aspect_ratio: 1.0,
            colormap_name: "jet".to_string(),
            wireframe_color: "white".to_string(),
            colormap_direction: 1,
            uv_lens: [1.0, 1.0],
        }
    }
}

/*pub fn get_key_from_value(map: &HashMap<u32, String>, target_value: String) -> Option<&u32> {
    for (key, value) in map {
        if *value == target_value {
            return Some(key);
        }
    }
    None
}*/

impl IParametricSurface {
    pub fn new(&mut self) -> ISurfaceOutput {
        if self.surface_type == 1 {
            (self.umin, self.umax, self.vmin, self.vmax) = (0.0, 2.0*PI, 0.0, 2.0*PI);
            self.parametric_surface_data(&mf::astroid)
        } else if self.surface_type == 2 {
            (self.umin, self.umax, self.vmin, self.vmax) = (0.0, 2.0*PI, 0.0, 2.0*PI);
            self.parametric_surface_data(&mf::astroid2)
        } else if self.surface_type == 3 {
            (self.umin, self.umax, self.vmin, self.vmax) = (-PI, PI, 0.0, 5.0);
            self.parametric_surface_data(&mf::astroidal_torus)
        } else if self.surface_type == 4 {
            (self.umin, self.umax, self.vmin, self.vmax) = (0.0, 2.0*PI, 0.0, 2.0*PI);
            self.parametric_surface_data(&mf::bohemian_dome)
        } else if self.surface_type == 5 {
            (self.umin, self.umax, self.vmin, self.vmax) = (0.0, PI, 0.0, PI);
            self.parametric_surface_data(&mf::boy_shape)
        } else if self.surface_type == 6 {
            (self.umin, self.umax, self.vmin, self.vmax) = (-14.0, 14.0, -12.0*PI, 12.0*PI);
            self.parametric_surface_data(&mf::breather)
        } else if self.surface_type == 7 {
            (self.umin, self.umax, self.vmin, self.vmax) = (-3.3, 3.3, -3.3, 3.3);
            self.parametric_surface_data(&mf::enneper)
        } else if self.surface_type == 8 {
            (self.umin, self.umax, self.vmin, self.vmax) = (0.0, 4.0*PI, 0.0, 2.0*PI);
            self.parametric_surface_data(&mf::figure8)
        } else if self.surface_type == 9 {
            (self.umin, self.umax, self.vmin, self.vmax) = (0.0, 1.0, 0.0, 2.0*PI);
            self.parametric_surface_data(&mf::henneberg)
        } else if self.surface_type == 10 {
            (self.umin, self.umax, self.vmin, self.vmax) = (-0.99999, 0.99999, 0.0, 2.0*PI);
            self.parametric_surface_data(&mf::kiss)
        } else if self.surface_type == 11 {
            (self.umin, self.umax, self.vmin, self.vmax) = (0.0, 2.0*PI, 0.0, 2.0*PI);
            self.parametric_surface_data(&mf::klein_bottle2)
        } else if self.surface_type == 12 {
            (self.umin, self.umax, self.vmin, self.vmax) = (0.0, 4.0*PI, 0.0, 2.0*PI);
            self.parametric_surface_data(&mf::klein_bottle3)
        } else if self.surface_type == 13 {
            (self.umin, self.umax, self.vmin, self.vmax) = (-4.5, 4.5, -5.0, 5.0);
            self.parametric_surface_data(&mf::kuen)
        } else if self.surface_type == 14 {
            (self.umin, self.umax, self.vmin, self.vmax) = (-3.0, 1.0, -3.0*PI, 3.0*PI);
            self.parametric_surface_data(&mf::minimal)
        } else if self.surface_type == 15 {
            (self.umin, self.umax, self.vmin, self.vmax) = (-5.0, 5.0, -5.0, 5.0);
            self.parametric_surface_data(&mf::parabolic_cyclide)
        } else if self.surface_type == 16 {
            (self.umin, self.umax, self.vmin, self.vmax) = (0.0, 1.0, 0.0, 2.0*PI);
            self.parametric_surface_data(&mf::pear)
        } else if self.surface_type == 17 {
            (self.umin, self.umax, self.vmin, self.vmax) = (-2.0, 2.0, 0.0, 2.0*PI);
            self.parametric_surface_data(&mf::plucker_conoid)
        } else if self.surface_type == 18 {
            (self.umin, self.umax, self.vmin, self.vmax) = (0.0, 6.0*PI, 0.0, 2.0*PI);
            self.parametric_surface_data(&mf::seashell)
        } else if self.surface_type == 19 {
            (self.umin, self.umax, self.vmin, self.vmax) = (-PI/2.1, PI/2.1, 0.001, PI/1.001);
            self.parametric_surface_data(&mf::sievert_enneper)
        } else if self.surface_type == 20 {
            (self.umin, self.umax, self.vmin, self.vmax) = (0.0, 1.999999*PI, 0.0, 0.999999*PI);
            self.parametric_surface_data(&mf::steiner)
        } else if self.surface_type == 21 {
            (self.umin, self.umax, self.vmin, self.vmax) = (0.0, 2.0*PI, 0.0, 2.0*PI);
            self.parametric_surface_data(&mf::torus)
        } else if self.surface_type == 22 {
            (self.umin, self.umax, self.vmin, self.vmax) = (0.0, 14.5, 0.0, 5.2);
            self.parametric_surface_data(&mf::wellenkugel)
        } else {
            (self.umin, self.umax, self.vmin, self.vmax) = (0.0, PI, 0.0, 2.0*PI);
            self.parametric_surface_data(&mf::klein_bottle)
        }
    }

    fn parametric_surface_data(&mut self, f:&dyn Fn(f32, f32) -> [f32; 3]) -> ISurfaceOutput {
        let mut positions: Vec<[f32; 3]> = vec![];
        let mut normals: Vec<[f32; 3]> = vec![];
        let mut colors: Vec<[f32; 3]> = vec![];
        let mut colors2: Vec<[f32; 3]> = vec![];
        let mut uvs: Vec<[f32; 2]> = vec![];

        let du = (self.umax - self.umin)/self.u_resolution as f32;
        let dv = (self.vmax - self.vmin)/self.v_resolution as f32;
        let (epsu, epsv) = (0.01 * du, 0.01 * dv);
        //let (mut p0, mut p1, mut p2, mut p3): (Vector3<f32>, Vector3<f32>, Vector3<f32>, Vector3<f32>);

        let (min_val, max_val, pts) = self.parametric_surface_range(f);
        let cdata = colormap::colormap_data(&self.colormap_name);
        let cdata2 = colormap::colormap_data(&self.wireframe_color);

        for i in 0..=self.u_resolution {
            let u = self.umin + du * i as f32;
            for j in 0..=self.v_resolution {
                let v = self.vmin + dv * j as f32;                
                positions.push(pts[i as usize][j as usize]);

                // calculate normals
                /*p0 = Vector3::from(f(u, v));
                if u - epsu >= 0.0 {
                    p1 = Vector3::from(f(u-epsu, v));
                    p2 = p0 - p1;
                } else {
                    p1 = Vector3::from(f(u+epsu, v));
                    p2 = p1 - p0;
                }
                if v - epsv >= 0.0 {
                    p1 = Vector3::from(f(u, v-epsv));
                    p3 = p0 - p1;
                } else {
                    p1 = Vector3::from(f(u, v+epsv));
                    p3 = p1 - p0;
                }
                let normal = p2.cross(p3).normalize();*/

                let nu = Vector3::from(f(u+epsu, v)) - Vector3::from(f(u-epsu, v));
                let nv = Vector3::from(f(u, v+epsv)) - Vector3::from(f(u, v-epsv));
                let normal = nu.cross(nv).normalize();
                normals.push(normal.into());

                // colormap
                let color = colormap::color_lerp(cdata, min_val, max_val, 
                    pts[i as usize][j as usize][self.colormap_direction as usize]);
                let color2 = colormap::color_lerp(cdata2, min_val, max_val, 
                    pts[i as usize][j as usize][self.colormap_direction as usize]);
                colors.push(color);
                colors2.push(color2);

                // uvs
                uvs.push([self.uv_lens[0]*(u-self.umin)/(self.umax-self.umin), 
                    self.uv_lens[1]*(v-self.vmin)/(self.vmax-self.vmin)
                ]);
            }
        }

        // calculate indices
        let mut indices: Vec<u16> = vec![];
        let mut indices2: Vec<u16> = vec![];
        let vertices_per_row = self.v_resolution + 1;

        for i in 0..self.u_resolution {
            for j in 0..self.v_resolution {
                let idx0 = j + i * vertices_per_row;
                let idx1 = j + 1 + i * vertices_per_row;
                let idx2 = j + 1 + (i + 1) * vertices_per_row;
                let idx3 = j + (i + 1) * vertices_per_row; 

                let values:Vec<u16> = vec![idx0, idx1, idx2, idx2, idx3, idx0];
                indices.extend(values);

                let values2:Vec<u16> = vec![idx0, idx1, idx0, idx3];
                indices2.extend(values2);
                if i == self.u_resolution - 1 || j == self.v_resolution - 1 {
                    let edge_values:Vec<u16> = vec![idx1, idx2, idx2, idx3];
                    indices2.extend(edge_values);
                }
            }
        }

        ISurfaceOutput { positions, normals, colors, colors2, uvs, indices, indices2 }
    }

    fn parametric_surface_range(&mut self, f:&dyn Fn(f32, f32) -> [f32; 3]) -> (f32, f32, Vec<Vec<[f32;3]>>) {
        let du = (self.umax - self.umin)/self.u_resolution as f32;
        let dv = (self.vmax - self.vmin)/self.v_resolution as f32;
        let (mut xmin, mut ymin, mut zmin) = (f32::MAX, f32::MAX, f32::MAX);
        let (mut xmax, mut ymax, mut zmax) = (f32::MIN, f32::MIN, f32::MIN);       

        let mut pts: Vec<Vec<[f32; 3]>> = vec![];
        for i in 0..=self.u_resolution {
            let u = self.umin + du * i as f32;
            let mut pt1: Vec<[f32; 3]> = vec![];
            for j in 0..=self.v_resolution {
                let v = self.vmin + dv * j as f32;
                let pt = f(u, v);
                xmin = if pt[0] < xmin { pt[0] } else { xmin };
                xmax = if pt[0] > xmax { pt[0] } else { xmax };
                ymin = if pt[1] < ymin { pt[1] } else { ymin };
                ymax = if pt[1] > ymax { pt[1] } else { ymax };
                zmin = if pt[2] < zmin { pt[2] } else { zmin };
                zmax = if pt[2] > zmax { pt[2] } else { zmax };
                pt1.push(pt);
            }
            pts.push(pt1);
        }

        let (mut min_val, mut max_val) = (f32::MAX, f32::MIN);
        let dist = (xmax - xmin).max(ymax - ymin).max(zmax - zmin);

        for i in 0..=self.u_resolution {
            for j in 0..=self.v_resolution {
                let mut pt = pts[i as usize][j as usize];
                pt[0] = self.scale * (pt[0] - 0.5 * (xmin + xmax)) / dist;
                pt[1] = self.scale * (pt[1] - 0.5 * (ymin + ymax)) / dist;
                pt[2] = self.scale * (pt[2] - 0.5 * (zmin + zmax)) / dist;
                let pt1 = pt[self.colormap_direction as usize];
                min_val = if pt1 < min_val { pt1 } else { min_val };
                max_val = if pt1 > max_val { pt1 } else { max_val };
                pts[i as usize][j as usize] = pt;
            }
        }
        (min_val, max_val, pts)
    }
}
// endregion: parametric surface

// region: simple surface
pub struct ISimpleSurface {
    pub surface_type: u32,
    pub xmin: f32,
    pub xmax: f32,
    pub zmin: f32,
    pub zmax: f32,
    pub x_resolution: u16,
    pub z_resolution: u16,
    pub scale: f32,
    pub aspect_ratio: f32,
    pub colormap_name: String,
    pub wireframe_color: String,
    pub colormap_direction: u32, // 0: x-direction, 1: y-direction, 2: z-direction
    pub t: f32,  // animation time parameter
    pub uv_lens: [f32; 2],
}

impl Default for ISimpleSurface {
    fn default() -> Self {
        Self {
            surface_type: 0,
            xmin: -1.0,
            xmax: 1.0,
            zmin: -1.0,
            zmax: 1.0,
            x_resolution: 30,
            z_resolution: 30,
            scale: 1.0,
            aspect_ratio: 1.0,
            colormap_name: "jet".to_string(),
            wireframe_color: "white".to_string(),
            colormap_direction: 1,
            t: 0.0,
            uv_lens: [1.0, 1.0],
        }
    }
}

impl ISimpleSurface { 
    pub fn new(&mut self) -> ISurfaceOutput {
        if self.surface_type == 0 {
            (self.xmin, self.xmax, self.zmin, self.zmax) = (-8.0, 8.0, -8.0, 8.0);
            self.aspect_ratio = 0.5;
            self.simple_surface_data(&mf::sinc)
        } else if self.surface_type == 1 {
            (self.xmin, self.xmax, self.zmin, self.zmax) = (-8.0, 8.0, -8.0, 8.0);
            self.aspect_ratio = 0.6;
            self.simple_surface_data(&mf::poles)
        } else {
            (self.xmin, self.xmax, self.zmin, self.zmax) = (-3.0, 3.0, -3.0, 3.0);
            self.aspect_ratio = 0.9;
            self.simple_surface_data(&mf::peaks)  
        }
    }

    fn simple_surface_data(&mut self, f:&dyn Fn(f32, f32, f32) -> [f32; 3]) -> ISurfaceOutput {
        let mut positions: Vec<[f32; 3]> = vec![];
        let mut normals: Vec<[f32; 3]> = vec![];
        let mut colors: Vec<[f32; 3]> = vec![];
        let mut colors2: Vec<[f32; 3]> = vec![];
        let mut uvs: Vec<[f32; 2]> = vec![];
        
        let dx = (self.xmax- self.xmin) / self.x_resolution as f32;
        let dz = (self.zmax - self.zmin) / self.z_resolution as f32;
        let (epsx, epsz) = (0.01 * dx, 0.01 * dz);
        
        let (ymin, ymax) = self.yrange(f);
        let cdata = colormap::colormap_data(&self.colormap_name);
        let cdata2 = colormap::colormap_data(&self.wireframe_color);

        for i in 0..=self.x_resolution {
            let x = self.xmin + dx * i as f32;
            for j in 0..=self.z_resolution {
                let z = self.zmin + dz * j as f32;
                let pos = self.normalize_data(f(x,z,self.t), ymin, ymax);
                positions.push(pos);

                // calculate normals
                let nx = Vector3::from(self.normalize_data(f(x+epsx, z, self.t), ymin, ymax)) - 
                         Vector3::from(self.normalize_data(f(x-epsx, z, self.t), ymin, ymax));
                let nz = Vector3::from(self.normalize_data(f(x, z+epsz, self.t), ymin, ymax)) - 
                         Vector3::from(self.normalize_data(f(x, z-epsz, self.t), ymin, ymax));
                let normal = nx.cross(nz).normalize();
                normals.push(normal.into());

                // colormap
                let range = if self.colormap_direction == 1 { self.scale * self.aspect_ratio} 
                    else {self.scale};
                let color = colormap::color_lerp(cdata, -range, range, 
                    pos[self.colormap_direction as usize]);
                let color2 = colormap::color_lerp(cdata2, -range, range, 
                    pos[self.colormap_direction as usize]);
                colors.push(color);
                colors2.push(color2);

                // uvs
                uvs.push([self.uv_lens[0]*(x-self.xmin)/(self.xmax-self.xmin), 
                    self.uv_lens[1]*(z-self.zmin)/(self.zmax-self.zmin)
                ]);
            }
        }

        // calculate indices
        let mut indices: Vec<u16> = vec![];
        let mut indices2: Vec<u16> = vec![];
        let vertices_per_row = self.z_resolution + 1;

        for i in 0..self.x_resolution {
            for j in 0..self.z_resolution {
                let idx0 = j + i * vertices_per_row;
                let idx1 = j + 1 + i * vertices_per_row;
                let idx2 = j + 1 + (i + 1) * vertices_per_row;
                let idx3 = j + (i + 1) * vertices_per_row; 

                let values:Vec<u16> = vec![idx0, idx1, idx2, idx2, idx3, idx0];
                indices.extend(values);

                let values2:Vec<u16> = vec![idx0, idx1, idx0, idx3];
                indices2.extend(values2);
                if i == self.x_resolution - 1 || j == self.z_resolution - 1 {
                    let edge_values:Vec<u16> = vec![idx1, idx2, idx2, idx3];
                    indices2.extend(edge_values);
                }
            }
        }

        ISurfaceOutput { positions, normals, colors, colors2, uvs, indices, indices2 }
    }

    fn normalize_data(&mut self, point:[f32; 3], ymin:f32, ymax:f32) -> [f32; 3] {
        let mut pt = point.clone();
        pt[0] = (-1.0 + 2.0 * (pt[0] - self.xmin) / (self.xmax - self.xmin)) * self.scale;
        pt[1] = (-1.0 + 2.0 * (pt[1] - ymin) / (ymax - ymin)) * self.scale * self.aspect_ratio;
        pt[2] = (-1.0 + 2.0 * (pt[2] - self.zmin) / (self.zmax - self.zmin)) * self.scale;
        pt
    }

    fn yrange(&mut self, f:&dyn Fn(f32, f32, f32) -> [f32; 3]) -> (f32, f32) {
        let dx = (self.xmax- self.xmin) / self.x_resolution as f32;
        let dz = (self.zmax - self.zmin) / self.z_resolution as f32;
        let mut ymin = f32::MAX;
        let mut ymax = f32::MIN;
    
        for i in 0..=self.x_resolution {
            let x = self.xmin + dx * i as f32;
            for j in 0..=self.z_resolution {
                let z = self.zmin + dz * j as f32;
                let pt = f(x, z, self.t);
                ymin = if pt[1] < ymin { pt[1] } else { ymin };
                ymax = if pt[1] > ymax { pt[1] } else { ymax };
            }
        }
        (ymin, ymax)
    }
}
// endregion: simple surface