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

TEST_F(NVFuserTest, FusionScatter1DIndexTvFusion_CUDA) {
  const int input_dim = 256;
  const int src_dim = 256;
  const int idx_dim = 256;

  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv_input = makeContigTensor(1);
  TensorView* tv_idx_1 = makeContigTensor(1, DataType::Int);
  TensorView* tv_idx_2 = makeContigTensor(1, DataType::Int);
  TensorView* tv_src = makeContigTensor(1);

  fusion.addInput(tv_input);
  fusion.addInput(tv_idx_1);
  fusion.addInput(tv_idx_2);
  fusion.addInput(tv_src);
  auto tv_idx = add(tv_idx_1, tv_idx_2);
  auto tv_out = scatter(tv_input, 0, tv_idx, tv_src);
  fusion.addOutput(tv_out); // init as zero ?

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i =
      torch::TensorOptions().dtype(torch::kLong).device(at::kCUDA, 0);
  at::Tensor idx_1 = at::randperm(idx_dim, options_i);
  at::Tensor idx_2 = at::zeros(idx_dim, options_i);
  at::Tensor input = at::randn({input_dim}, options);
  at::Tensor src = at::randn({src_dim}, options);

  auto t_index = at::add(idx_1, idx_2);
  auto out_ref = at::scatter(input, 0, t_index, src);

  std::vector<IValue> aten_inputs = {input, idx_1, idx_2, src};

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
  testValidate(&fusion, cg_outputs, aten_inputs, {out_ref}, __LINE__, __FILE__);
}

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
  auto tv_out = scatter(tv_input, 1, tv_idx, tv_src);
  fusion.addOutput(tv_out); // init as zero ?

  tv_out->axis(1)->parallelize(ParallelType::BIDy);
  tv_out->merge(0);
  tv_out->split(0, 4);
  tv_out->axis(0)->parallelize(ParallelType::BIDx);
  tv_out->reorder({{-1, 0}});

  std::vector<int64_t> same_dims{4, 4};
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor src = at::randn(same_dims, options); // lookup
  std::vector<int64_t> index_vec = {
      0, 1, 2, 3, 2, 1, 0, 3, 3, 1, 2, 0, 3, 2, 1, 0};
  auto opts = torch::TensorOptions().dtype(torch::kLong);
  at::Tensor input_idx_cpu =
      torch::from_blob(index_vec.data(), same_dims, opts).clone();
  auto index = input_idx_cpu.to(torch::kCUDA);
  at::Tensor input = at::randn(same_dims, options);

  std::vector<IValue> aten_inputs = {input, index, src};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);

  auto outputs = fe.runFusion(aten_inputs);

  auto out_ref = input.scatter_(1, index, src);
  TORCH_CHECK(out_ref.equal(outputs[0]));
}

TEST_F(NVFuserTest, FusionScatterSrcTvFusion_CUDA) {
  const int max_dim_size = 64;
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv_input = makeContigTensor(2);
  TensorView* tv_idx = makeContigTensor(2, DataType::Int);
  TensorView* tv_src_1 = makeContigTensor(2);
  TensorView* tv_src_2 = makeContigTensor(2);

  fusion.addInput(tv_input);
  fusion.addInput(tv_idx);
  fusion.addInput(tv_src_1);
  fusion.addInput(tv_src_2);
  auto tv_src = add(tv_src_1, tv_src_2);
  auto tv_out = scatter(tv_input, 1, tv_idx, tv_src);
  fusion.addOutput(tv_out); // init as zero ?

  std::vector<int64_t> same_dims{4, 4};
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor src_1 = at::randn(same_dims, options); // lookup
  at::Tensor src_2 = at::randn(same_dims, options); // lookup

  std::vector<int64_t> index_vec = {
      0, 1, 2, 3, 2, 1, 0, 3, 3, 1, 2, 0, 3, 2, 1, 0};
  auto opts = torch::TensorOptions().dtype(torch::kLong);
  at::Tensor input_idx_cpu =
      torch::from_blob(index_vec.data(), same_dims, opts).clone();
  auto index = input_idx_cpu.to(torch::kCUDA);
  at::Tensor input = at::randn(same_dims, options);

  auto at_src = at::add(src_1, src_2);
  auto out_ref = at::scatter(input, 1, index, at_src);

  std::vector<IValue> aten_inputs = {input, index, src_1, src_2};

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
  testValidate(&fusion, cg_outputs, aten_inputs, {out_ref}, __LINE__, __FILE__);
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
