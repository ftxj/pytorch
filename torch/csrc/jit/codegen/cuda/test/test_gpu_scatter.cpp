#if defined(USE_CUDA)
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/codegen/cuda/test/test_gpu_validator.h>
#include <torch/csrc/jit/codegen/cuda/test/test_utils.h>

#include <torch/torch.h>

namespace torch {
namespace jit {

using namespace torch::jit::fuser::cuda;
using namespace at::indexing;

TEST_F(NVFuserTest, FusionScatterSameSize_CUDA) {
  const int max_dim_size = 64;
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv_input = makeContigTensor(2);
  TensorView* tv_idx = makeContigTensor(2, DataType::Int);
  TensorView* tv_src = makeContigTensor(2);

  fusion.addInput(tv_input);
  fusion.addInput(tv_idx);
  fusion.addInput(tv_src);
  auto tv_out = scatter_add(tv_input, 0, tv_idx, tv_src);
  fusion.addOutput(tv_out); // init as zero ?

  std::vector<int64_t> same_dims {4, 4};
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

  at::Tensor src = at::randn(same_dims, options); // lookup
  at::Tensor index = at::randint(0, 3, same_dims, options_i);
  at::Tensor input = at::randn(same_dims, options);

  std::vector<IValue> aten_inputs = {input, index, src};

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
  
  auto out_ref = input.scatter_add_(0, index, src);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {out_ref}, __LINE__, __FILE__);
}

// Test correctness of our generated code, will delete later.
TEST_F(NVFuserTest, FusionScatterCompileRtc_CUDA) {
  FusionExecutor fe;
  std::string kernel = R"(
__global__ void kernel1(Tensor<float, 2> T2, Tensor<int64_t, 2> T1, Tensor<float, 2> T0, Tensor<float, 2> T3) {
  int i55;
  i55 = (((nvfuser_index_t)blockIdx.x) * 128) + ((nvfuser_index_t)threadIdx.x);
  if ((i55 < (T0.size[0] * T0.size[1]))) {
    atomicAdd(&T3[((T1[i55] * T0.size[1]) + (i55 % T0.size[1]))], T0[i55]);
     atomicAdd(&T3[i55], T2[i55]);
   }
}
    )";
  fe.compileRtc(kernel, "CudaCodeGen::kernel1");
  LaunchParams lp(
      256, // gdimx
      1, // gdimy
      1, // gdimz
      128, // bdimx
      1, // bdimy
      1 // bdimz
  );
  lp.setSmem(0);
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const auto options_i =
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

  const std::vector<int64_t> tensor_dims = {10, 10};
  auto input = at::randn(tensor_dims, options);
  auto src = at::randn(tensor_dims, options);
  auto index = at::randint(0, 8, tensor_dims, options_i);

  auto out0 = at::empty_like(input);

  fe.runRtc(lp, {input, index, src, out0});

  auto out_ref = input.scatter_add_(0, index, src);

  TORCH_CHECK(out_ref.allclose(out0));
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)