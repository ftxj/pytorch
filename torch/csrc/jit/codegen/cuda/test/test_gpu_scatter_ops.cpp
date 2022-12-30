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
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/test/test_gpu_validator.h>
#include <torch/csrc/jit/codegen/cuda/test/test_utils.h>
#include <torch/torch.h>

namespace torch {
namespace jit {

using namespace torch::jit::fuser::cuda;
using namespace at::indexing;

auto randomVector(int64_t low, int64_t high, int rank) {
  std::vector<int64_t> out(rank, 0);
  for (int idim = 0; idim < rank; ++idim) {
    out[idim] = (std::rand() % (high - low)) + low;
  }
  return out;
}

auto randomIndexVector(
    const std::vector<int64_t>& input_dims,
    int64_t low,
    int rank) {
  std::vector<int64_t> index_dims(rank, 0);
  for (int idim = 0; idim < rank; ++idim) {
    index_dims[idim] = (std::rand() % (input_dims[idim] - low)) + low;
  }
  return index_dims;
}

at::Tensor generateScatter1DIndex(int64_t min, int64_t extent) {
  auto options_i =
      torch::TensorOptions().dtype(torch::kLong).device(at::kCUDA, 0);

  at::Tensor idx = at::randperm(extent, options_i) + min;
  return idx;
}

at::Tensor generateScatter2DIndex(
    int64_t min,
    int64_t extent_1d,
    int64_t extent_2d,
    int select_id) {
  auto options_i =
      torch::TensorOptions().dtype(torch::kLong).device(at::kCUDA, 0);
  if (select_id == 0) {
    auto idx = at::randint(0, extent_2d, {extent_1d, extent_2d}, options_i);
    for (size_t i = 0; i < extent_1d; ++i) {
      idx[i] = at::randperm(extent_2d, options_i) + min;
    }
    return idx.transpose(0, 1).contiguous();
  } else {
    auto idx = at::randint(0, extent_1d, {extent_2d, extent_1d}, options_i);
    for (size_t i = 0; i < extent_2d; ++i) {
      idx[i] = at::randperm(extent_1d, options_i) + min;
    }
    return idx;
  }
}

TEST_F(NVFuserTest, FusionScatter1DIndexTvFusion_CUDA) {
  const int input_dim = 128;
  const int src_dim = 128;
  const int idx_dim = 128;

  at::manual_seed(0);
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
  fusion.addOutput(tv_out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i =
      torch::TensorOptions().dtype(torch::kLong).device(at::kCUDA, 0);

  at::Tensor idx = generateScatter1DIndex(0, idx_dim);
  at::Tensor idx_1 = at::randint(0, idx_dim, {idx_dim}, options_i);
  at::Tensor idx_2 = idx - idx_1;
  at::Tensor input = at::randn({input_dim}, options);
  at::Tensor src = at::randn({src_dim}, options);

  auto t_index = at::add(idx_1, idx_2);
  auto out_ref = at::scatter(input, 0, t_index, src);

  std::vector<IValue> aten_inputs = {input, idx_1, idx_2, src};

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
  testValidate(&fusion, cg_outputs, aten_inputs, {out_ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionScatter2DSelectOneIndexTvFusion_CUDA) {
  const std::vector<std::vector<int64_t>> input_dims = {
      {2, 2}, {3, 3}, {2048, 2048}, {1024, 2048}, {1024, 2048}};

  const std::vector<std::vector<int64_t>> src_dims = {
      {3, 3}, {2, 2}, {1024, 1024}, {512, 512}, {512, 512}};

  const std::vector<std::vector<int64_t>> idx_dims = {
      {2, 2}, {2, 2}, {512, 256}, {1, 256}, {512, 1}};

  at::manual_seed(0);
  for (size_t test_id = 0; test_id < idx_dims.size(); ++test_id) {
    auto fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    TensorView* tv_input = makeContigTensor(2);
    TensorView* tv_idx_1 = makeContigTensor(2, DataType::Int);
    TensorView* tv_idx_2 = makeContigTensor(2, DataType::Int);
    TensorView* tv_src = makeContigTensor(2);

    fusion.addInput(tv_input);
    fusion.addInput(tv_idx_1);
    fusion.addInput(tv_idx_2);
    fusion.addInput(tv_src);

    auto tv_idx = add(tv_idx_1, tv_idx_2);
    auto tv_out = scatter(tv_input, 0, tv_idx, tv_src);
    fusion.addOutput(tv_out);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto options_i =
        torch::TensorOptions().dtype(torch::kLong).device(at::kCUDA, 0);

    at::Tensor idx = generateScatter2DIndex(
        0, idx_dims[test_id][1], idx_dims[test_id][0], 0);

    at::Tensor idx_1 = at::randint(0, 1024, idx_dims[test_id], options_i);
    at::Tensor idx_2 = idx - idx_1;
    at::Tensor input = at::randn(input_dims[test_id], options);
    at::Tensor src = at::randn(src_dims[test_id], options);
    auto t_index = at::add(idx_1, idx_2);
    auto out_ref = at::scatter(input, 0, t_index, src);

    std::vector<IValue> aten_inputs = {input, idx_1, idx_2, src};

    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
    testValidate(
        &fusion, cg_outputs, aten_inputs, {out_ref}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionScatter2DOneHot_Fusion) {
  const std::vector<std::vector<int64_t>> self_dims = {
      {1, 1, 4}, {3, 3, 8}, {16, 16, 16}, {16, 512, 1024}};

  const std::vector<std::vector<int64_t>> idx_base_dims = {
      {1, 1}, {3, 3}, {16, 16}, {16, 512}};

  at::manual_seed(0);
  for (size_t test_id = 0; test_id < self_dims.size(); ++test_id) {
    auto fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    TensorView* tv_self = makeContigTensor(3);
    TensorView* tv_idx_base = makeContigTensor(2, DataType::Int);
    TensorView* tv_src = makeContigTensor(3);

    fusion.addInput(tv_self);
    fusion.addInput(tv_idx_base);
    fusion.addInput(tv_src);

    auto tv_idx = unsqueeze(tv_idx_base, -1);
    auto tv_out = scatter(tv_self, -1, tv_idx, tv_src);

    fusion.addOutput(tv_out);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto options_i =
        torch::TensorOptions().dtype(torch::kLong).device(at::kCUDA, 0);

    at::Tensor idx_base = generateScatter2DIndex(
        0, idx_base_dims[test_id][1], idx_base_dims[test_id][0], 1);
    at::Tensor idx = idx_base.unsqueeze(-1);

    at::Tensor src = at::ones(self_dims[test_id], options);
    at::Tensor self = at::zeros(self_dims[test_id], options);

    auto out_ref = at::one_hot(idx_base, self_dims[test_id][2]);

    std::vector<IValue> aten_inputs = {self, idx_base, src};

    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
    testValidate(
        &fusion, cg_outputs, aten_inputs, {out_ref}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionScatterOnehot_CUDA) {
  const std::vector<std::vector<int64_t>> x_size = {
      {1, 1}, {3, 3}, {16, 16}, {16, 512}};

  const std::vector<int64_t> classes = {4, 8, 16, 1024};

  at::manual_seed(0);
  for (size_t test_id = 0; test_id < classes.size(); ++test_id) {
    auto fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    TensorView* tv_idx_base = makeContigTensor(2, DataType::Int);

    fusion.addInput(tv_idx_base);

    auto tv_out = onehot(tv_idx_base, classes[test_id]);

    fusion.addOutput(tv_out);

    at::Tensor idx_base =
        generateScatter2DIndex(0, x_size[test_id][1], x_size[test_id][0], 1);

    auto out_ref = at::one_hot(idx_base, classes[test_id]);

    std::vector<IValue> aten_inputs = {idx_base};

    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
    testValidate(
        &fusion, cg_outputs, aten_inputs, {out_ref}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionScatter2DSelectOneSrcTvFusion_CUDA) {
  const std::vector<std::vector<int64_t>> input_dims = {
      {2, 2}, {3, 3}, {2048, 2048}, {1024, 2048}, {1024, 2048}};

  const std::vector<std::vector<int64_t>> src_dims = {
      {3, 3}, {2, 2}, {1024, 1024}, {512, 512}, {512, 512}};

  const std::vector<std::vector<int64_t>> idx_dims = {
      {2, 2}, {2, 2}, {512, 256}, {1, 256}, {512, 1}};

  at::manual_seed(0);
  for (size_t test_id = 0; test_id < idx_dims.size(); ++test_id) {
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
    auto tv_out = scatter(tv_input, 0, tv_idx, tv_src);
    fusion.addOutput(tv_out);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto options_i =
        torch::TensorOptions().dtype(torch::kLong).device(at::kCUDA, 0);

    at::Tensor t_input = at::randn(input_dims[test_id], options);
    at::Tensor t_idx = generateScatter2DIndex(
        0, idx_dims[test_id][1], idx_dims[test_id][0], 0);
    at::Tensor t_src_1 = at::randn(src_dims[test_id], options);
    at::Tensor t_src_2 = at::randn(src_dims[test_id], options);

    auto t_src = at::add(t_src_1, t_src_2);
    auto out_ref = at::scatter(t_input, 0, t_idx, t_src);

    std::vector<IValue> aten_inputs = {t_input, t_idx, t_src_1, t_src_2};

    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
    testValidate(
        &fusion, cg_outputs, aten_inputs, {out_ref}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionScatterAddOpAllDim_CUDA) {
  const std::vector<std::vector<int64_t>> inp_dims = {
      {2, 2}, {3, 3}, {2048, 2048}, {1024, 2048}, {1024, 2048}};

  const std::vector<std::vector<int64_t>> src_dims = {
      {3, 3}, {2, 2}, {1024, 1024}, {512, 512}, {512, 512}};

  const std::vector<std::vector<int64_t>> idx_dims = {
      {2, 2}, {2, 2}, {512, 256}, {1, 256}, {512, 1}};
  at::manual_seed(0);
  for (size_t test_id = 0; test_id < idx_dims.size(); ++test_id) {
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
    auto tv_out = scatter_add(tv_input, 0, tv_idx, tv_src);
    fusion.addOutput(tv_out);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto options_i =
        torch::TensorOptions().dtype(torch::kLong).device(at::kCUDA, 0);

    at::Tensor t_input = at::randn(inp_dims[test_id], options);
    at::Tensor t_idx =
        at::randint(0, inp_dims[test_id][0], idx_dims[test_id], options_i);
    at::Tensor t_src_1 = at::randn(src_dims[test_id], options);
    at::Tensor t_src_2 = at::randn(src_dims[test_id], options);

    auto t_src = at::add(t_src_1, t_src_2);
    auto out_ref = at::scatter_add(t_input, 0, t_idx, t_src);

    std::vector<IValue> aten_inputs = {t_input, t_idx, t_src_1, t_src_2};

    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

    testValidate(
        &fusion, cg_outputs, aten_inputs, {out_ref}, __LINE__, __FILE__);
  }
}
TEST_F(NVFuserTest, FusionScatterOpCompileRtc_CUDA) {
  FusionExecutor fe;
  std::string kernel = R"(
__global__ void kernel1(Tensor<float, 2> T0, Tensor<int64_t, 2> T1, Tensor<float, 2> T2, Tensor<float, 2> T3, Tensor<float, 2> T5) {
  int i99;
  i99 = (((nvfuser_index_t)blockIdx.x) * 128) + ((nvfuser_index_t)threadIdx.x);
  int i106;
  i106 = i99 % T0.size[1];
  int i107;
  i107 = ((i99 / T0.size[1]) * T2.size[1]) + i106;
  if ((i99 < (T0.size[0] * T0.size[1]))) {
    float T9[1];
    T9[0] = 0;
    T9[0]
       = T3[i107];
    float T8[1];
    T8[0] = 0;
    T8[0]
       = T2[i107];
    int64_t T7[1];
    T7[0]
       = T1[i99];
    float T6[1];
    T6[0]
       = T0[i99];
    float T4[1];
    T4[0]
      = T8[0]
      + T9[0];
    printf("self[%ld, %ld] += src[%ld, %ld]\n", T7[0], i106, i99 / T0.size[1], i106);
    atomicAdd(&T5[(i106 + (T0.size[1] * T7[0]))], T4[0]);
  }
}
)";
  fe.compileRtc(kernel, "CudaCodeGen::kernel1");
  LaunchParams lp(
      1, // gdimx
      1, // gdimy
      1, // gdimz
      128, // bdimx
      1, // bdimy
      1 // bdimz
  );
  const std::vector<int64_t> inp_dims = {3, 3};
  const std::vector<int64_t> src_dims = {2, 2};
  const std::vector<int64_t> idx_dims = {2, 2};
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i =
      torch::TensorOptions().dtype(torch::kLong).device(at::kCUDA, 0);

  at::Tensor t_input = at::randn(inp_dims, options);
  at::Tensor t_idx = at::randint(0, inp_dims[0], idx_dims, options_i);
  at::Tensor t_src_1 = at::randn(src_dims, options);
  at::Tensor t_src_2 = at::randn(src_dims, options);
  at::Tensor t_out = t_input.clone().detach();

  lp.setSmem(0);
  fe.runRtc(lp, {t_input, t_idx, t_src_1, t_src_2, t_out});

  auto t_src = at::add(t_src_1, t_src_2);
  auto out_ref = at::scatter_add(t_input, 0, t_idx, t_src);

  std::cout << "inp = " << std::endl;
  std::cout << t_input << std::endl;

  std::cout << "t_src_1 = " << std::endl;
  std::cout << t_src_1 << std::endl;

  std::cout << "t_src_2 = " << std::endl;
  std::cout << t_src_2 << std::endl;

  std::cout << "t_src = " << std::endl;
  std::cout << t_src << std::endl;

  std::cout << "t_idx = " << std::endl;
  std::cout << t_idx << std::endl;

  std::cout << "out_ref = " << std::endl;
  std::cout << out_ref << std::endl;

  std::cout << "out = " << std::endl;
  std::cout << t_out << std::endl;
  TORCH_CHECK(out_ref.allclose(t_out));
}
} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
