use bytemuck::{cast_slice, Pod, Zeroable};
use cgmath::{Matrix, Matrix4, SquareMatrix};
use std::{f32::consts::PI, iter};
use wgpu::{util::DeviceExt, VertexBufferLayout};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};
use wgpu_simplified as ws;
use wgpu_voxel_terrain::{marching_cubes_table, vertex_data as vd};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 4],
    normal: [f32; 4],
}

fn cube_vertices(side: f32) -> (Vec<Vertex>, Vec<u16>) {
    let (pos, _, norm, _, ind, _) = vd::create_cube_data(side);
    let mut data: Vec<Vertex> = Vec::with_capacity(pos.len());
    for i in 0..pos.len() {
        data.push(Vertex {
            position: [pos[i][0], pos[i][1], pos[i][2], 1.0],
            normal: [norm[i][0], norm[i][1], norm[i][2], 1.0],
        });
    }
    (data.to_vec(), ind)
}

struct State {
    init: ws::IWgpuInit,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniform_bind_groups: Vec<wgpu::BindGroup>,
    uniform_buffers: Vec<wgpu::Buffer>,
    indices_len: u32,

    cs_pipelines: Vec<wgpu::ComputePipeline>,
    cs_vertex_buffers: Vec<wgpu::Buffer>,
    cs_uniform_buffers: Vec<wgpu::Buffer>,
    cs_bind_groups: Vec<wgpu::BindGroup>,

    view_mat: Matrix4<f32>,
    project_mat: Matrix4<f32>,
    msaa_texture_view: wgpu::TextureView,
    depth_texture_view: wgpu::TextureView,
    animation_speed: f32,
    animation_direction: u32,

    resolution: u32,
    marching_cube_cells: u32,
    octaves: u32,
    seed: u32,
    isolevel: f32,
    terrain_size: f32,
    lacunarity: f32,
    persistence: f32,
    noise_scale: f32,
    noise_height: f32,
    height_multiplier: f32,
    floor_offset: f32,
    cube_side: f32,
    data_changed: bool,
    fps_counter: ws::FpsCounter,
}

impl State {
    async fn new(window: &Window, sample_count: u32, resolution: u32) -> Self {
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: 1024 * 1024 * 1024, //1024MB, defaulting to 128MB
            max_buffer_size: 1024 * 1024 * 1024,                 // 1024MB, defaulting to 256MB
            max_compute_invocations_per_workgroup: 512,          // dafaulting to 256
            ..Default::default()
        };
        let init = ws::IWgpuInit::new(&window, sample_count, Some(limits)).await;

        let resol = ws::round_to_multiple(resolution, 8);
        let marching_cube_cells = (resolution - 1) * (resolution - 1) * (resolution - 1);
        let vertex_count = 3 * 12 * marching_cube_cells;
        let vertex_buffer_size = 4 * vertex_count;
        println!("resolution = {}", resol);

        let vs_shader = init
            .device
            .create_shader_module(wgpu::include_wgsl!("minecraft_vert.wgsl"));
        let fs_shader = init
            .device
            .create_shader_module(wgpu::include_wgsl!("../ch01/shader_frag.wgsl"));

        let cs_value_file = include_str!("../ch01/voxel_value.wgsl");
        let cs_noise_file = include_str!("../ch01/noise3d.wgsl");
        let cs_value = init
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Compute Value Shader"),
                source: wgpu::ShaderSource::Wgsl([cs_noise_file, cs_value_file].join("\n").into()),
            });

        let cs_comp = init
            .device
            .create_shader_module(wgpu::include_wgsl!("minecraft_comp.wgsl"));

        // uniform data
        let camera_position = (12.0, 12.0, 12.0).into();
        let look_direction = (0.0, 0.0, 0.0).into();
        let up_direction = cgmath::Vector3::unit_y();
        let light_direction = [-0.5f32, -0.5, -0.5];

        let (view_mat, project_mat, vp_mat) = ws::create_vp_mat(
            camera_position,
            look_direction,
            up_direction,
            init.config.width as f32 / init.config.height as f32,
        );

        let model_mat =
            ws::create_model_mat([0.0, 5.0, 0.0], [0.0, 0.1 * PI, 0.0], [1.7, 1.7, 1.7]);
        let normal_mat = (model_mat.invert().unwrap()).transpose();

        // create vertex uniform buffers
        let vert_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Uniform Buffer"),
            size: 192,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        init.queue.write_buffer(
            &vert_uniform_buffer,
            0,
            bytemuck::cast_slice(vp_mat.as_ref() as &[f32; 16]),
        );
        init.queue.write_buffer(
            &vert_uniform_buffer,
            64,
            bytemuck::cast_slice(model_mat.as_ref() as &[f32; 16]),
        );
        init.queue.write_buffer(
            &vert_uniform_buffer,
            128,
            bytemuck::cast_slice(normal_mat.as_ref() as &[f32; 16]),
        );

        // create light uniform buffer. here we set eye_position = camera_position
        let light_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Uniform Buffer"),
            size: 48,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let eye_position: &[f32; 3] = camera_position.as_ref();
        init.queue.write_buffer(
            &light_uniform_buffer,
            0,
            cast_slice(light_direction.as_ref()),
        );
        init.queue
            .write_buffer(&light_uniform_buffer, 16, cast_slice(eye_position));

        // set specular light color to white
        let specular_color: [f32; 3] = [1.0, 1.0, 1.0];
        init.queue.write_buffer(
            &light_uniform_buffer,
            32,
            cast_slice(specular_color.as_ref()),
        );

        // material uniform buffer
        let material_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Material Uniform Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // set default material parameters
        let material = [0.1f32, 0.7, 0.4, 30.0];
        init.queue
            .write_buffer(&material_uniform_buffer, 0, cast_slice(material.as_ref()));

        // uniform bind group for vertex shader
        let (vert_bind_group_layout, vert_bind_group) = ws::create_bind_group(
            &init.device,
            vec![wgpu::ShaderStages::VERTEX],
            &[vert_uniform_buffer.as_entire_binding()],
        );

        // uniform bind group for fragment shader
        let (frag_bind_group_layout, frag_bind_group) = ws::create_bind_group(
            &init.device,
            vec![wgpu::ShaderStages::FRAGMENT, wgpu::ShaderStages::FRAGMENT],
            &[
                light_uniform_buffer.as_entire_binding(),
                material_uniform_buffer.as_entire_binding(),
            ],
        );

        let vertex_buffer_layouts = [
            VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![0 => Float32x4], // instance position
            },
            VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![1 => Float32x4], // instance color
            },
            VertexBufferLayout {
                array_stride: 32,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![2 => Float32x4, 3 => Float32x4], // pos, norm for cube
            },
        ];

        let pipeline_layout = init
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&vert_bind_group_layout, &frag_bind_group_layout],
                push_constant_ranges: &[],
            });

        let mut ppl = ws::IRenderPipeline {
            vs_shader: Some(&vs_shader),
            fs_shader: Some(&fs_shader),
            pipeline_layout: Some(&pipeline_layout),
            vertex_buffer_layout: &vertex_buffer_layouts,
            ..Default::default()
        };
        let pipeline = ppl.new(&init);
        let msaa_texture_view = ws::create_msaa_texture_view(&init);
        let depth_texture_view = ws::create_depth_view(&init);

        // create vertex and index buffers for cube
        let (cube_vertex_data, cube_index_data) = cube_vertices(0.5);
        let cube_vertex_buffer =
            init.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Cube Vertex Buffer"),
                    contents: cast_slice(&cube_vertex_data),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });

        let cube_index_buffer = init
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Cube Index Buffer"),
                contents: cast_slice(&cube_index_data),
                usage: wgpu::BufferUsages::INDEX,
            });

        // create compute pipeline for value
        let volume_elements = resol * resol * resol;
        let cs_value_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compuet Value Buffer"),
            size: 4 * volume_elements as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cs_value_int_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compuet Value Integer Uniform Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cs_value_float_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compuet Value Float Uniform Buffer"),
            size: 48,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (cs_value_bind_group_layout, cs_value_bind_group) = ws::create_bind_group_storage(
            &init.device,
            vec![
                wgpu::ShaderStages::COMPUTE,
                wgpu::ShaderStages::COMPUTE,
                wgpu::ShaderStages::COMPUTE,
            ],
            vec![
                wgpu::BufferBindingType::Storage { read_only: false },
                wgpu::BufferBindingType::Uniform,
                wgpu::BufferBindingType::Uniform,
            ],
            &[
                cs_value_buffer.as_entire_binding(),
                cs_value_int_buffer.as_entire_binding(),
                cs_value_float_buffer.as_entire_binding(),
            ],
        );

        let cs_value_pipeline_layout =
            init.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Compute Value Pipeline Layout"),
                    bind_group_layouts: &[&cs_value_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let cs_value_pipeline =
            init.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Compute Value Pipeline"),
                    layout: Some(&cs_value_pipeline_layout),
                    module: &cs_value,
                    entry_point: "cs_main",
                });

        // create compute pipeline for implicit surface
        let cs_table_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Table STorage Buffer"),
            size: (marching_cubes_table::EDGE_TABLE.len() + marching_cubes_table::TRI_TABLE.len())
                as u64
                * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        init.queue.write_buffer(
            &cs_table_buffer,
            0,
            cast_slice(marching_cubes_table::EDGE_TABLE),
        );
        init.queue.write_buffer(
            &cs_table_buffer,
            marching_cubes_table::EDGE_TABLE.len() as u64 * 4,
            cast_slice(marching_cubes_table::TRI_TABLE),
        );

        let cs_position_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Position Buffer"),
            size: vertex_buffer_size as u64,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cs_color_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Color Buffer"),
            size: vertex_buffer_size as u64,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        //let indirect_array = [500u32, 0, 0, 0];
        let cs_indirect_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Indirect Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cs_int_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Integer uniform Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cs_float_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Float uniform Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (cs_bind_group_layout, cs_bind_group) = ws::create_bind_group_storage(
            &init.device,
            vec![
                wgpu::ShaderStages::COMPUTE,
                wgpu::ShaderStages::COMPUTE,
                wgpu::ShaderStages::COMPUTE,
                wgpu::ShaderStages::COMPUTE,
                wgpu::ShaderStages::COMPUTE,
                wgpu::ShaderStages::COMPUTE,
                wgpu::ShaderStages::COMPUTE,
            ],
            vec![
                wgpu::BufferBindingType::Storage { read_only: true }, // marching table
                wgpu::BufferBindingType::Storage { read_only: true }, // value buffer
                wgpu::BufferBindingType::Storage { read_only: false }, // position
                wgpu::BufferBindingType::Storage { read_only: false }, // color
                wgpu::BufferBindingType::Storage { read_only: false }, // indirect params
                wgpu::BufferBindingType::Uniform,                     // int params
                wgpu::BufferBindingType::Uniform,                     // float params
            ],
            &[
                cs_table_buffer.as_entire_binding(),
                cs_value_buffer.as_entire_binding(),
                cs_position_buffer.as_entire_binding(),
                cs_color_buffer.as_entire_binding(),
                cs_indirect_buffer.as_entire_binding(),
                cs_int_buffer.as_entire_binding(),
                cs_float_buffer.as_entire_binding(),
            ],
        );

        let cs_pipeline_layout =
            init.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Compute Pipeline Layout"),
                    bind_group_layouts: &[&cs_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let cs_pipeline = init
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&cs_pipeline_layout),
                module: &cs_comp,
                entry_point: "cs_main",
            });

        Self {
            init,
            pipeline,
            vertex_buffer: cube_vertex_buffer,
            index_buffer: cube_index_buffer,
            uniform_bind_groups: vec![vert_bind_group, frag_bind_group],
            uniform_buffers: vec![
                vert_uniform_buffer,
                light_uniform_buffer,
                material_uniform_buffer,
            ],
            indices_len: cube_index_data.len() as u32,

            cs_pipelines: vec![cs_value_pipeline, cs_pipeline],
            cs_vertex_buffers: vec![cs_value_buffer, cs_position_buffer, cs_color_buffer],
            cs_uniform_buffers: vec![
                cs_value_int_buffer,
                cs_value_float_buffer,
                cs_int_buffer,
                cs_float_buffer,
                cs_indirect_buffer,
            ],
            cs_bind_groups: vec![cs_value_bind_group, cs_bind_group],

            view_mat,
            project_mat,
            msaa_texture_view,
            depth_texture_view,
            animation_speed: 1.0,
            animation_direction: 2,

            resolution: resol,
            marching_cube_cells,
            octaves: 10,
            seed: 1232,
            isolevel: 3.0,
            terrain_size: 40.0,
            lacunarity: 2.0,
            persistence: 0.5,
            noise_scale: 5.0,
            noise_height: 7.0,
            height_multiplier: 1.5,
            floor_offset: 10.0,
            cube_side: 0.5,
            data_changed: false,
            fps_counter: ws::FpsCounter::default(),
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.init.size = new_size;
            self.init.config.width = new_size.width;
            self.init.config.height = new_size.height;
            self.init
                .surface
                .configure(&self.init.device, &self.init.config);

            self.project_mat =
                ws::create_projection_mat(new_size.width as f32 / new_size.height as f32, true);
            let vp_mat = self.project_mat * self.view_mat;
            self.init.queue.write_buffer(
                &self.uniform_buffers[0],
                0,
                bytemuck::cast_slice(vp_mat.as_ref() as &[f32; 16]),
            );

            self.depth_texture_view = ws::create_depth_view(&self.init);
            if self.init.sample_count > 1 {
                self.msaa_texture_view = ws::create_msaa_texture_view(&self.init);
            }
        }
    }

    #[allow(unused_variables)]
    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode: Some(keycode),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match keycode {
                VirtualKeyCode::Space => {
                    self.animation_direction = (self.animation_direction + 1) % 3;
                    true
                }
                VirtualKeyCode::Q => {
                    self.cube_side += 0.02;
                    self.data_changed = true;
                    println!("cube size = {}", self.cube_side);
                    true
                }
                VirtualKeyCode::A => {
                    self.cube_side -= 0.02;
                    self.data_changed = true;
                    if self.cube_side < 0.1 {
                        self.cube_side = 0.1;
                    }
                    println!("cube size = {}", self.cube_side);
                    true
                }
                VirtualKeyCode::W => {
                    self.animation_speed += 0.1;
                    true
                }
                VirtualKeyCode::S => {
                    self.animation_speed -= 0.1;
                    if self.animation_speed < 0.0 {
                        self.animation_speed = 0.0;
                    }
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    fn update(&mut self, dt: std::time::Duration) {
        // update compute buffers for value
        let value_int_params = [self.resolution, self.octaves, self.seed, 0];
        self.init.queue.write_buffer(
            &self.cs_uniform_buffers[0],
            0,
            cast_slice(&value_int_params),
        );

        let dt1 = 10.0 * self.animation_speed * dt.as_secs_f32();
        let value_float_params = [
            if self.animation_direction == 0 {
                -dt1
            } else {
                0.0
            },
            if self.animation_direction == 1 {
                -dt1
            } else {
                0.0
            },
            if self.animation_direction == 2 {
                -dt1
            } else {
                0.0
            },
            self.terrain_size,
            self.lacunarity,
            self.persistence,
            self.noise_scale,
            self.noise_height,
            self.height_multiplier,
            self.floor_offset,
        ];
        self.init.queue.write_buffer(
            &self.cs_uniform_buffers[1],
            0,
            cast_slice(&value_float_params),
        );

        // update compute buffers for terrain
        let int_params = [self.resolution, 0, 0, 0];
        self.init.queue.write_buffer(
            &self.cs_uniform_buffers[2],
            0,
            bytemuck::cast_slice(&int_params),
        );

        let float_params = [self.terrain_size, self.isolevel, 0.0, 0.0];
        self.init.queue.write_buffer(
            &self.cs_uniform_buffers[3],
            0,
            bytemuck::cast_slice(&float_params),
        );

        let indirect_array = [500u32, 0, 0, 0];
        self.init.queue.write_buffer(
            &self.cs_uniform_buffers[4],
            0,
            bytemuck::cast_slice(&indirect_array),
        );

        if self.data_changed {
            let (cube_vertex_data, _) = cube_vertices(self.cube_side);
            self.init.queue.write_buffer(
                &&self.vertex_buffer,
                0,
                bytemuck::cast_slice(&cube_vertex_data),
            );
            self.data_changed = false;
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.init.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.init
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

        // compute pass for value
        {
            let mut cs_index_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute value Pass"),
            });
            cs_index_pass.set_pipeline(&self.cs_pipelines[0]);
            cs_index_pass.set_bind_group(0, &self.cs_bind_groups[0], &[]);
            cs_index_pass.dispatch_workgroups(
                self.resolution / 8,
                self.resolution / 8,
                self.resolution / 8,
            );
        }

        // compute pass for vertices
        {
            let mut cs_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            });
            cs_pass.set_pipeline(&self.cs_pipelines[1]);
            cs_pass.set_bind_group(0, &self.cs_bind_groups[1], &[]);
            cs_pass.dispatch_workgroups(
                self.resolution / 8,
                self.resolution / 8,
                self.resolution / 8,
            );
        }

        // render pass
        {
            let color_attach = ws::create_color_attachment(&view);
            let msaa_attach = ws::create_msaa_color_attachment(&view, &self.msaa_texture_view);
            let color_attachment = if self.init.sample_count == 1 {
                color_attach
            } else {
                msaa_attach
            };
            let depth_attachment = ws::create_depth_stencil_attachment(&self.depth_texture_view);

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: Some(depth_attachment),
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_vertex_buffer(0, self.cs_vertex_buffers[1].slice(..));
            render_pass.set_vertex_buffer(1, self.cs_vertex_buffers[2].slice(..));
            render_pass.set_vertex_buffer(2, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.set_bind_group(0, &self.uniform_bind_groups[0], &[]);
            render_pass.set_bind_group(1, &self.uniform_bind_groups[1], &[]);
            render_pass.draw_indexed(0..self.indices_len, 0, 0..self.marching_cube_cells);
        }
        self.fps_counter.print_fps(5);
        self.init.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn main() {
    let mut sample_count = 1u32;
    let mut resolution = 64u32;

    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        sample_count = args[1].parse::<u32>().unwrap();
    }
    if args.len() > 2 {
        resolution = args[2].parse::<u32>().unwrap();
    }

    env_logger::init();
    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .build(&event_loop)
        .unwrap();
    window.set_title(&*format!("{}", "minecraft"));

    let mut state = pollster::block_on(State::new(&window, sample_count, resolution));
    let render_start_time = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => {
            if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
        }
        Event::RedrawRequested(_) => {
            let now = std::time::Instant::now();
            let dt = now - render_start_time;
            state.update(dt);

            match state.render() {
                Ok(_) => {}
                Err(wgpu::SurfaceError::Lost) => state.resize(state.init.size),
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        _ => {}
    });
}