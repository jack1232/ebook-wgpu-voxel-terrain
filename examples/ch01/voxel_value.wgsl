struct ValueBuffer{
    values: array<f32>,
}
@group(0) @binding(0) var<storage, read_write> valueBuffer : ValueBuffer;

struct IntParams {
    resolution: u32,
    octaves: u32,
    seed: u32,
}
@group(0) @binding(1) var<uniform> ips : IntParams;

struct FloatParams {     
    offset: vec3f,
    terrainSize: f32,  
    lacunarity: f32,
    persistence: f32,
    noiseScale: f32,
    noiseHeight: f32,    
    heightMultiplier: f32,
    floorOffset: f32,
}
@group(0) @binding(2) var<uniform> fps : FloatParams;

var<private> vmin: vec3f;
var<private> vmax: vec3f;

fn positionAt(index : vec3u) -> vec3f {
    vmin = vec3(-0.5 * fps.terrainSize);
    vmax = vec3(0.5 * fps.terrainSize);
    let vstep = (vmax-vmin)/(f32(ips.resolution) - 1.0);
    return vmin + (vstep * vec3<f32>(index.xyz));
}

fn getIdx(id: vec3u) -> u32 {
    return id.x + ips.resolution * ( id.y + id.z * ips.resolution);
}

fn noiseFunc(position: vec3f) -> f32 {  
    var noise = 0.0;
    var frequency = fps.noiseScale/100.0;
    var amplitude = 1.0;
    var height = 1.0;

    for(var i = 0u; i < ips.octaves; i = i + 1u){
        let r = rand3d(f32(i + ips.seed));        
        let offset = 1000.0 * r;
        let n = simplex3d((position  + fps.offset + offset)*frequency );
        var v = 1.0 - abs(n);
        v = v * v;
        v *= height;
        height = max(min(v * fps.heightMultiplier, 1.0), 0.0);
        noise += v * amplitude;
        amplitude *= fps.persistence;
        frequency *= fps.lacunarity;
    }

    let val = -(position.y + fps.floorOffset) + noise * fps.noiseHeight;
    return val;
}

@compute @workgroup_size(8, 8, 8)
fn cs_main(@builtin(global_invocation_id) id : vec3u) {
    var position = positionAt(id);
    var y = (position.y - vmin.y)/(vmax.y - vmin.y);
    if(y <= 0.0) {
        position.y = 0.0;
    }
    let idx = getIdx(id);
    valueBuffer.values[idx] = noiseFunc(position);
}