//! GPU (CUDA Driver API PTX) and CPU multithreaded thermal load actuators.

use std::ffi::c_void;
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use libloading::{Library, Symbol};

const CUDA_FMA_PTX_NULL_TERM: &[u8] = b"\
.version 7.0
.target sm_75
.address_size 64

.visible .entry burn_kernel(
    .param .u64 out_ptr,
    .param .u32 iters
)
{
    .reg .f32 %f<8>;
    .reg .u32 %r<5>;
    .reg .u64 %rd<2>;
    .reg .pred %p1;

    ld.param.u64 %rd1, [out_ptr];
    ld.param.u32 %r4, [iters];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mad.lo.u32 %r3, %r2, 256, %r1;
    cvt.rn.f32.u32 %f1, %r3;
    mov.f32 %f2, 0f3F800001;
    mov.f32 %f3, 0f3F000001;
    mov.f32 %f4, 0f3F400001;
    mov.f32 %f5, 0f3F200001;

LOOP:
    setp.eq.u32 %p1, %r4, 0;
    @%p1 bra DONE;
    fma.rn.f32 %f1, %f1, %f2, %f3;
    fma.rn.f32 %f2, %f2, %f3, %f4;
    fma.rn.f32 %f3, %f3, %f4, %f5;
    fma.rn.f32 %f4, %f4, %f5, %f1;
    fma.rn.f32 %f5, %f5, %f1, %f2;
    sub.u32 %r4, %r4, 1;
    bra LOOP;

DONE:
    st.global.f32 [%rd1], %f1;
    ret;
}
\0";

/// Interface for modulating hardware compute load to generate heat.
pub trait ThermalLoadActuator {
    /// Sets the active duty cycle `[0.0, 1.0]` and active hardware targets.
    fn apply_duty_cycle(&mut self, duty_cycle: f64, enable_gpu: bool, enable_cpu: bool);

    /// Stops all background compute threads cleanly.
    fn shutdown_workers(&mut self);
}

/// Splits a time window into `(busy_seconds, sleep_seconds)` for a duty cycle.
///
/// # Examples
/// ```
/// use heater::load_workers::compute_slice_timings;
/// let (busy, sleep) = compute_slice_timings(0.5, 0.1).unwrap();
/// assert!((busy - 0.05).abs() < 1e-6);
/// assert!((sleep - 0.05).abs() < 1e-6);
/// ```
pub fn compute_slice_timings(duty_cycle: f64, window_sec: f64) -> Result<(f64, f64), String> {
    if !(0.0..=1.0).contains(&duty_cycle) {
        return Err(format!(
            "Expected duty_cycle in [0.0, 1.0], got {duty_cycle}"
        ));
    }
    let busy_sec = duty_cycle * window_sec;
    Ok((busy_sec, window_sec - busy_sec))
}

fn cpu_burn_slice(duration_sec: f64) {
    let deadline = Instant::now() + Duration::from_secs_f64(duration_sec);
    let mut acc: f64 = 1.000_001;
    while Instant::now() < deadline {
        for _ in 0..4000 {
            acc = acc.mul_add(1.000_000_1, 0.000_000_3).sin().abs() + 1.0;
        }
        black_box(acc);
    }
}

fn cpu_worker_loop(duty_bits: Arc<AtomicU64>, stop_flag: Arc<AtomicBool>) {
    while !stop_flag.load(Ordering::Relaxed) {
        let duty = f64::from_bits(duty_bits.load(Ordering::Relaxed)).clamp(0.0, 1.0);
        if duty <= 0.005 {
            thread::sleep(Duration::from_millis(40));
            continue;
        }
        if let Ok((busy_sec, sleep_sec)) = compute_slice_timings(duty, 0.06) {
            cpu_burn_slice(busy_sec);
            if sleep_sec > 0.001 {
                thread::sleep(Duration::from_secs_f64(sleep_sec));
            }
        }
    }
}

struct CudaDriverSession {
    lib: Library,
    ctx: *mut c_void,
    func: *mut c_void,
    dptr: u64,
}

impl CudaDriverSession {
    fn open_session() -> Result<Self, String> {
        unsafe {
            let lib = Library::new("libcuda.so.1").map_err(|e| e.to_string())?;
            let cu_init: Symbol<unsafe extern "C" fn(u32) -> i32> =
                lib.get(b"cuInit\0").map_err(|e| e.to_string())?;
            if cu_init(0) != 0 {
                return Err("Expected cuInit(0) == 0".to_string());
            }
            Self::create_context_and_module(lib)
        }
    }

    unsafe fn create_context_and_module(lib: Library) -> Result<Self, String> {
        unsafe {
            let mut dev: i32 = 0;
            let cu_dev: Symbol<unsafe extern "C" fn(*mut i32, i32) -> i32> =
                lib.get(b"cuDeviceGet\0").map_err(|e| e.to_string())?;
            cu_dev(&mut dev, 0);
            let mut ctx: *mut c_void = std::ptr::null_mut();
            let cu_ctx: Symbol<unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32> =
                lib.get(b"cuCtxCreate_v2\0").map_err(|e| e.to_string())?;
            cu_ctx(&mut ctx, 0, dev);
            let mut module: *mut c_void = std::ptr::null_mut();
            let cu_mod: Symbol<unsafe extern "C" fn(*mut *mut c_void, *const u8) -> i32> =
                lib.get(b"cuModuleLoadData\0").map_err(|e| e.to_string())?;
            cu_mod(&mut module, CUDA_FMA_PTX_NULL_TERM.as_ptr());
            let mut func: *mut c_void = std::ptr::null_mut();
            let cu_fn: Symbol<
                unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const u8) -> i32,
            > = lib
                .get(b"cuModuleGetFunction\0")
                .map_err(|e| e.to_string())?;
            cu_fn(&mut func, module, b"burn_kernel\0".as_ptr());
            let mut dptr: u64 = 0;
            let cu_alloc: Symbol<unsafe extern "C" fn(*mut u64, usize) -> i32> =
                lib.get(b"cuMemAlloc_v2\0").map_err(|e| e.to_string())?;
            cu_alloc(&mut dptr, 64);
            Ok(Self {
                lib,
                ctx,
                func,
                dptr,
            })
        }
    }

    fn launch_burn_step(&self, iterations: u32) {
        unsafe {
            let Ok(launch) = self.lib.get::<unsafe extern "C" fn(
                *mut c_void,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                *mut c_void,
                *mut *mut c_void,
                *mut *mut c_void,
            ) -> i32>(b"cuLaunchKernel\0") else {
                return;
            };
            let mut dptr_arg = self.dptr;
            let mut iters_arg = iterations;
            let mut args: [*mut c_void; 2] = [
                (&mut dptr_arg as *mut u64).cast::<c_void>(),
                (&mut iters_arg as *mut u32).cast::<c_void>(),
            ];
            let _ = launch(
                self.func,
                1024,
                1,
                1,
                256,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if let Ok(sync) = self
                .lib
                .get::<unsafe extern "C" fn() -> i32>(b"cuCtxSynchronize\0")
            {
                let _ = sync();
            }
        }
    }

    fn close_session(&self) {
        unsafe {
            if let Ok(free_fn) = self
                .lib
                .get::<unsafe extern "C" fn(u64) -> i32>(b"cuMemFree_v2\0")
            {
                let _ = free_fn(self.dptr);
            }
            if let Ok(destroy_fn) = self
                .lib
                .get::<unsafe extern "C" fn(*mut c_void) -> i32>(b"cuCtxDestroy_v2\0")
            {
                let _ = destroy_fn(self.ctx);
            }
        }
    }
}

fn gpu_worker_loop(duty_bits: Arc<AtomicU64>, stop_flag: Arc<AtomicBool>) {
    let Ok(session) = CudaDriverSession::open_session() else {
        return;
    };
    while !stop_flag.load(Ordering::Relaxed) {
        let duty = f64::from_bits(duty_bits.load(Ordering::Relaxed)).clamp(0.0, 1.0);
        if duty <= 0.005 {
            thread::sleep(Duration::from_millis(40));
            continue;
        }
        let t0 = Instant::now();
        session.launch_burn_step(120_000);
        let elapsed = t0.elapsed().as_secs_f64();
        let sleep_sec = elapsed * (1.0 - duty) / duty;
        if sleep_sec > 0.001 {
            thread::sleep(Duration::from_secs_f64(sleep_sec.min(0.25)));
        }
    }
    session.close_session();
}

/// Spawns and coordinates 1 GPU worker thread and N native CPU worker threads.
pub struct ThreadedLoadActuator {
    gpu_duty_bits: Arc<AtomicU64>,
    cpu_duty_bits: Arc<AtomicU64>,
    stop_flag: Arc<AtomicBool>,
    handles: Vec<JoinHandle<()>>,
}

impl ThreadedLoadActuator {
    /// Spawns the GPU CUDA worker thread and `cpu_threads` CPU worker threads.
    pub fn spawn_workers(cpu_threads: Option<usize>) -> Self {
        let available = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let count = cpu_threads.unwrap_or_else(|| available.saturating_sub(2).max(1));
        let gpu_duty_bits = Arc::new(AtomicU64::new(0.0_f64.to_bits()));
        let cpu_duty_bits = Arc::new(AtomicU64::new(0.0_f64.to_bits()));
        let stop_flag = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::with_capacity(count + 1);
        let g_duty = Arc::clone(&gpu_duty_bits);
        let g_stop = Arc::clone(&stop_flag);
        handles.push(thread::spawn(move || gpu_worker_loop(g_duty, g_stop)));
        for _ in 0..count {
            let c_duty = Arc::clone(&cpu_duty_bits);
            let c_stop = Arc::clone(&stop_flag);
            handles.push(thread::spawn(move || cpu_worker_loop(c_duty, c_stop)));
        }
        Self {
            gpu_duty_bits,
            cpu_duty_bits,
            stop_flag,
            handles,
        }
    }
}

impl ThermalLoadActuator for ThreadedLoadActuator {
    fn apply_duty_cycle(&mut self, duty_cycle: f64, enable_gpu: bool, enable_cpu: bool) {
        let clamped = duty_cycle.clamp(0.0, 1.0);
        let g_val = if enable_gpu { clamped } else { 0.0 };
        let c_val = if enable_cpu { clamped } else { 0.0 };
        self.gpu_duty_bits.store(g_val.to_bits(), Ordering::Relaxed);
        self.cpu_duty_bits.store(c_val.to_bits(), Ordering::Relaxed);
    }

    fn shutdown_workers(&mut self) {
        self.gpu_duty_bits
            .store(0.0_f64.to_bits(), Ordering::Relaxed);
        self.cpu_duty_bits
            .store(0.0_f64.to_bits(), Ordering::Relaxed);
        self.stop_flag.store(true, Ordering::Relaxed);
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}
