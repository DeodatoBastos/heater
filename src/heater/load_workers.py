"""GPU and CPU thermal load actuators with duty-cycle modulation."""

import ctypes
import math
import multiprocessing
from multiprocessing.sharedctypes import Synchronized
import os
import time
from typing import Protocol

CUDA_FMA_PTX_SOURCE: bytes = b"""
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
"""


class ThermalLoadActuator(Protocol):
    """Interface for modulating hardware compute load to generate heat."""

    def apply_duty_cycle(
        self,
        duty_cycle: float,
        enable_gpu: bool,
        enable_cpu: bool,
    ) -> None:
        """Sets the active duty cycle [0.0, 1.0] and active hardware targets."""
        ...

    def shutdown_workers(self) -> None:
        """Stops all background compute processes cleanly."""
        ...


def compute_slice_timings(
    duty_cycle: float, window_sec: float = 0.08
) -> tuple[float, float]:
    """Splits a time window into (busy_seconds, sleep_seconds) for a duty cycle.

    Example:
        >>> compute_slice_timings(0.5, 0.1)
        (0.05, 0.05)
    """
    if not 0.0 <= duty_cycle <= 1.0:
        raise ValueError(
            f"Expected duty_cycle in [0.0, 1.0], got {duty_cycle}"
        )
    busy_sec = duty_cycle * window_sec
    return busy_sec, window_sec - busy_sec


def _cpu_burn_slice(duration_sec: float) -> None:
    """Executes tight floating-point math on one CPU core for duration_sec."""
    deadline = time.monotonic() + duration_sec
    acc = 1.000001
    while time.monotonic() < deadline:
        for _ in range(2500):
            acc = math.sin(acc) * math.cos(acc) + 1.000001


def _cpu_worker_loop(
    shared_duty: "Synchronized[float]",
    shared_stop: "Synchronized[int]",
) -> None:
    """Worker process loop that modulates CPU burn vs sleep based on shared_duty."""
    while shared_stop.value == 0:
        current_duty = max(0.0, min(1.0, float(shared_duty.value)))
        if current_duty <= 0.005:
            time.sleep(0.05)
            continue
        busy_sec, sleep_sec = compute_slice_timings(current_duty, 0.08)
        _cpu_burn_slice(busy_sec)
        if sleep_sec > 0.001:
            time.sleep(sleep_sec)


class CudaDriverSession:
    """Manages a CUDA Driver API context and PTX FMA kernel for GPU heating."""

    def __init__(self, library_name: str = "libcuda.so.1") -> None:
        self._cuda = ctypes.CDLL(library_name)
        self._ctx = ctypes.c_void_p()
        self._mod = ctypes.c_void_p()
        self._func = ctypes.c_void_p()
        self._dptr = ctypes.c_uint64()
        self._setup_context()

    def _setup_context(self) -> None:
        """Initializes CUDA driver, context, PTX module, and output buffer."""
        if self._cuda.cuInit(0) != 0:
            raise RuntimeError("Expected cuInit(0) == 0")
        dev = ctypes.c_int()
        self._cuda.cuDeviceGet(ctypes.byref(dev), 0)
        self._cuda.cuCtxCreate_v2(ctypes.byref(self._ctx), 0, dev)
        self._cuda.cuModuleLoadData(ctypes.byref(self._mod), CUDA_FMA_PTX_SOURCE)
        self._cuda.cuModuleGetFunction(
            ctypes.byref(self._func), self._mod, b"burn_kernel"
        )
        self._cuda.cuMemAlloc_v2(ctypes.byref(self._dptr), 64)

    def launch_burn_step(self, iterations: int = 120000) -> None:
        """Launches one ~20ms FMA kernel burst and synchronizes."""
        iters = ctypes.c_uint32(iterations)
        args = (ctypes.c_void_p * 2)(
            ctypes.cast(ctypes.byref(self._dptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(iters), ctypes.c_void_p),
        )
        self._cuda.cuLaunchKernel(
            self._func, 1024, 1, 1, 256, 1, 1, 0, None, args, None
        )
        self._cuda.cuCtxSynchronize()

    def close_session(self) -> None:
        """Frees device memory and destroys the CUDA context."""
        self._cuda.cuMemFree_v2(self._dptr)
        self._cuda.cuCtxDestroy_v2(self._ctx)


def _gpu_worker_loop(
    shared_duty: "Synchronized[float]",
    shared_stop: "Synchronized[int]",
) -> None:
    """Dedicated process that modulates CUDA FMA kernel bursts by duty cycle."""
    try:
        session = CudaDriverSession()
    except OSError:
        return
    try:
        _run_gpu_duty_cycles(session, shared_duty, shared_stop)
    finally:
        session.close_session()


def _run_gpu_duty_cycles(
    session: CudaDriverSession,
    shared_duty: "Synchronized[float]",
    shared_stop: "Synchronized[int]",
) -> None:
    """Executes timed CUDA bursts and sleeps according to shared_duty."""
    while shared_stop.value == 0:
        duty = max(0.0, min(1.0, float(shared_duty.value)))
        if duty <= 0.005:
            time.sleep(0.05)
            continue
        t0 = time.monotonic()
        session.launch_burn_step()
        elapsed = time.monotonic() - t0
        sleep_sec = elapsed * (1.0 - duty) / duty
        if sleep_sec > 0.001:
            time.sleep(min(0.25, sleep_sec))


class MultiprocessLoadActuator:
    """Spawns and coordinates 1 GPU worker and N CPU worker processes."""

    def __init__(self, cpu_worker_count: int | None = None) -> None:
        cores = os.cpu_count() or 4
        self._cpu_count: int = cpu_worker_count or max(1, cores - 2)
        self._gpu_duty: "Synchronized[float]" = multiprocessing.Value("d", 0.0)
        self._cpu_duty: "Synchronized[float]" = multiprocessing.Value("d", 0.0)
        self._stop_flag: "Synchronized[int]" = multiprocessing.Value("i", 0)
        self._workers: list[multiprocessing.Process] = []
        self._spawn_all_workers()

    def _spawn_all_workers(self) -> None:
        """Starts the GPU worker process and CPU worker pool."""
        gpu_proc = multiprocessing.Process(
            target=_gpu_worker_loop,
            args=(self._gpu_duty, self._stop_flag),
            daemon=True,
        )
        gpu_proc.start()
        self._workers.append(gpu_proc)
        for _ in range(self._cpu_count):
            cpu_proc = multiprocessing.Process(
                target=_cpu_worker_loop,
                args=(self._cpu_duty, self._stop_flag),
                daemon=True,
            )
            cpu_proc.start()
            self._workers.append(cpu_proc)

    def apply_duty_cycle(
        self,
        duty_cycle: float,
        enable_gpu: bool,
        enable_cpu: bool,
    ) -> None:
        """Updates shared memory duty targets for GPU and CPU workers.

        Example:
            >>> # actuator.apply_duty_cycle(0.75, True, True)
        """
        clamped = max(0.0, min(1.0, duty_cycle))
        self._gpu_duty.value = clamped if enable_gpu else 0.0
        self._cpu_duty.value = clamped if enable_cpu else 0.0

    def shutdown_workers(self) -> None:
        """Signals all workers to stop and joins them cleanly."""
        self._gpu_duty.value = 0.0
        self._cpu_duty.value = 0.0
        self._stop_flag.value = 1
        for proc in self._workers:
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()
