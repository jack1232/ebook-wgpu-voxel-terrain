// vertex shader
struct Uniforms {   
    viewProjectMat: mat4x4f,
    modelMat: mat4x4f,           
    normalMat: mat4x4f,            
};
@binding(0) @group(0) var<uniform> uniforms: Uniforms;

struct Input {
    @location(0) position: vec4f,    
    @location(1) color: vec4f,
    @location(2) boxPosition: vec4f,
    @location(3) boxNormal: vec4f, 
}

struct Output {
    @builtin(position) position : vec4f,
    @location(0) vPosition : vec4f,
    @location(1) vNormal : vec4f,
    @location(2) vColor: vec4f,
};

@vertex
fn vs_main(in:Input) -> Output {    
    var output: Output;            
    let mPosition = uniforms.modelMat * (in.position + in.boxPosition); 
    output.vPosition = mPosition;                  
    output.vNormal =  uniforms.normalMat * in.boxNormal;
    output.position = uniforms.viewProjectMat * mPosition; 
    output.vColor = in.color;              
    return output;
}