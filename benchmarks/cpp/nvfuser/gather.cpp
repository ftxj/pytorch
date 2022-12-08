#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <benchmarks/cpp/nvfuser/utils.h>

using namespace torch::jit::fuser::cuda;

static void setupTorchGather(Fusion* fusion, int select_dim, DataType dtype) {
  FusionGuard fg(fusion);
  constexpr int nDim = 3;
  // set up input tensor views
  auto t0 = makeContigTensor(nDim, dtype);
  auto t_idx = makeContigTensor(nDim, DataType::Int);
  auto t1 = makeContigTensor(nDim, dtype);

  fusion->addInput(t1);
  fusion->addInput(t0);
  fusion->addInput(t_idx);

  auto t2 = torch_gather(t0, select_dim, t_idx);
  auto t3 = mul(t1, t2);
  auto t4 = add(t3, IrBuilder::create<Double>(17.0));

  fusion->addOutput(t4);
}

static void NvFuserScheduler_TorchGather(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    int select_dim,
    DataType dtype) {

  at::manual_seed(0);
  auto input_dims = benchmark_state.range(0);
  auto index_dims = benchmark_state.range(1);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto options_i = 
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn(input_dims, options); 
  at::Tensor t_idx = at::randint(0, input_dims[dim], index_dims, options_i);
  at::Tensor t1 = at::zeros(index_dims, options);

  fusion_executor_cache->profile(true);
  fusion_executor_cache->runFusionWithInputs({t0, t_idx, t1});

  auto compile_log = fusion_executor_cache->getMostRecentExecutorInfo();
  auto executor_instance = compile_log.fusion_executor;
  auto params = toString(compile_log.params);
  auto lparams = toString(compile_log.fusion_executor->lastLaunchParams());

  benchmark_state.SetLabel(params + lparams);

  fusion_executor_cache->profile(false);
  executor_instance->setMeasureKernelTimeFlag(true);
  // Sync everything up before we start
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    clearL2Cache();
    auto cg_outputs = fusion_executor_cache->runFusionWithInputs({t0, t_idx, t1});
    benchmark_state.SetIterationTime(
        executor_instance->kernelTimeMs() / 1000.0);
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (iter_size * bcast_size * 2 + iter_size) * int64_t(dataTypeSize(dtype)));
}

static void Baseline_TorchGather(
    benchmark::State& benchmark_state,
    int select_dim,
    DataType dtype) {
  
  at::manual_seed(0);
  auto input_dims = benchmark_state.range(0);
  auto index_dims = benchmark_state.range(1);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto options_i = 
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn(input_dims, options); 
  at::Tensor t_idx = at::randint(0, input_dims[dim], index_dims, options_i);
  at::Tensor t1 = at::zeros(index_dims, options);

  // Sync everything up before we start
  clearL2Cache();
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    auto t2 = at::gather(t0, select_dim, t_idx);
    auto t3 = at::mul(t1, t2);
    auto t4 = at::add(t3, IrBuilder::create<Double>(17.0));
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    clearL2Cache();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
  }

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (nFeat * select_size * 2 /*2 elemwise ops*/ + select_size + nFeat * select_size/*index select op*/)
      * int64_t(dataTypeSize(dtype)));
}
//------------------------------------------------------------------------------
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_TorchGather_Input_fp32,
    setupTorchGather,
    NvFuserScheduler_TorchGather,
    0,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_TorchGather_Input_fp32)
    ->Ranges({{1024 * 1024, 32}, {1024, 8}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();