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

TEST_F(NVFuserTest, FusionScatterCodeGen_CUDA) {
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
  auto tv_out = scatter(tv_input, 0, tv_idx, tv_src);
  fusion.addOutput(tv_out); // init as zero ?

  tv_out->axis(1)->parallelize(ParallelType::BIDy);
  tv_out->merge(0);
  tv_out->split(0, 4);
  tv_out->axis(0)->parallelize(ParallelType::BIDx);
  tv_out->reorder({{-1, 0}});

  std::vector<int64_t> same_dims{4, 4};
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

  at::Tensor src = at::randn(same_dims, options); // lookup
  at::Tensor index = at::randint(0, 3, same_dims, options_i);
  at::Tensor input = at::randn(same_dims, options);

  std::vector<IValue> aten_inputs = {input, index, src};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto out_ref = input.scatter_(0, index, src);

  TORCH_CHECK(out_ref.equal(outputs[0]));
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
