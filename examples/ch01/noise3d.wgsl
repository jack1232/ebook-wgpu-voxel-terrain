// this wgsl noise function is based on the original
// webgl-noise library - https://github.com/ashima/webgl-noise

fn mod289_3d(x: vec3f) -> vec3f {
    return x - floor(x *(1.0/289.0))*289.0;
}

fn mod289_4d(x: vec4f) -> vec4f {
    return x - floor(x *(1.0/289.0))*289.0;
}

fn permute(x: vec4f) -> vec4f {
    return mod289_4d((x*34.0 + 10.0) *x); 
}

fn taylorInvSqrt(x: vec4f) -> vec4f {
    return 1.79284291400159 - x * 0.85373472095314; 
}

fn simplex3d(v: vec3f) -> f32 { 
    let C: vec2f = vec2(1.0/6.0, 1.0/3.0) ;
    let D: vec4f = vec4(0.0, 0.5, 1.0, 2.0);

    // First corner
    var i  = floor(v + dot(v, C.yyy) );
    let x0 =   v - i + dot(i, C.xxx) ;

    // Other corners
    let g = step(x0.yzx, x0.xyz);
    let l = 1.0 - g;
    let i1 = min(g.xyz, l.zxy);
    let i2 = max(g.xyz, l.zxy);

    let x1 = x0 - i1 + C.xxx;
    let x2 = x0 - i2 + C.yyy; 
    let x3 = x0 - D.yyy;      

    // Permutations
    i = mod289_3d(i); 
    let p = permute( permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
            + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
            + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    let n_ = 0.142857142857; // 1.0/7.0
    let ns = n_ * D.wyz - D.xzx;

    let j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

    let x_ = floor(j * ns.z);
    let y_ = floor(j - 7.0 * x_ );    // mod(j,N)

    let x = x_ *ns.x + ns.yyyy;
    let y = y_ *ns.x + ns.yyyy;
    let h = 1.0 - abs(x) - abs(y);

    let b0 = vec4( x.xy, y.xy );
    let b1 = vec4( x.zw, y.zw );

    let s0 = floor(b0)*2.0 + 1.0;
    let s1 = floor(b1)*2.0 + 1.0;
    let sh = -step(h, vec4(0.0));

    let a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    let a1 = b1.xzyw + s1.xzyw*sh.zzww ;

    var p0 = vec3(a0.xy,h.x);
    var p1 = vec3(a0.zw,h.y);
    var p2 = vec3(a1.xy,h.z);
    var p3 = vec3(a1.zw,h.w);

    // Normalise gradients
    let norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    var m = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), vec4(0.0));
    m = m * m;
    return 105.0 * dot( m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3) ));
}

fn rand2dTo1d(st: vec2f) -> f32 {
	var v = fract (st * vec2(5.3983, 5.4427));
	v += dot(v.yx, v.xy + vec2(21.5351, 14.3137));
	return fract(v.x * v.y * 95.4337);
}

fn rand1d(st: f32, mutator:f32) -> f32 {
    return -1.0 + 2.0 * fract(sin(st + mutator)*143758.5453);
}

fn rand2dTo2d(st: vec2f) -> vec2f {
    let rand = vec2(fract(sin(dot(st, vec2(12.9898, 78.233))) * 43758.5453),
                    fract(sin(dot(st, vec2(39.2944, 29.462))) * 23421.631));
    return rand * 2.0 - 1.0;
}

fn rand3d(st:f32) -> vec3f {
    return vec3(
        rand1d(st, 3.9812),
        rand1d(st, 7.1536),
        rand1d(st, 5.7241)
    );
}