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

// pass
TEST_F(NVFuserTest, TorchGatherOpAllDim_CUDA) {
  const int max_dim_size = 64;
  std::srand(std::time(nullptr));
  for (int rank = 2; rank <= 5; ++rank) {
    for (int dim = 0; dim < rank; ++dim) {
      auto fusion_ptr = std::make_unique<Fusion>();
      Fusion& fusion = *fusion_ptr.get();
      FusionGuard fg(&fusion);

      TensorView* tv1 = makeContigTensor(rank);
      TensorView* tv_idx = makeContigTensor(rank, DataType::Int);
      fusion.addInput(tv1);
      fusion.addInput(tv_idx);
      auto tv_out = torch_gather(tv1, dim, tv_idx);
      fusion.addOutput(tv_out);

      std::cout << fusion << std::endl;

      std::vector<int64_t> input_dims(rank, 0);
      for (int idim = 0; idim < rank; ++idim) {
        input_dims[idim] = (std::rand() % max_dim_size) + 2;
      }

      std::vector<int64_t> index_dims(rank, 0);
      for (int idim = 0; idim < rank; ++idim) {
        index_dims[idim] = (std::rand() % input_dims[idim]) + 1;
      }

      at::manual_seed(0);
      auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
      auto options_i =
          at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

      at::Tensor input = at::randn(input_dims, options); // lookup
      at::Tensor input_idx =
          at::randint(0, input_dims[dim], index_dims, options_i);
      at::Tensor output = at::zeros(index_dims, options);

      auto tv_out_ref = at::gather(input, dim, input_idx);

      std::vector<IValue> aten_inputs = {input, input_idx};

      FusionExecutorCache executor_cache(std::move(fusion_ptr));
      auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
      testValidate(
          &fusion, cg_outputs, aten_inputs, {tv_out_ref}, __LINE__, __FILE__);
    }
  }
}

// pass
TEST_F(NVFuserTest, FusionGatherAddMulSmallSize_CUDA) {
  const int max_dim_size = 64;
  std::srand(std::time(nullptr));
  for (int rank = 1; rank <= 5; ++rank) {
    for (int dim = 0; dim < rank; ++dim) {
      auto fusion_ptr = std::make_unique<Fusion>();
      Fusion& fusion = *fusion_ptr.get();
      FusionGuard fg(&fusion);

      TensorView* tv1 = makeContigTensor(rank);
      TensorView* tv_idx = makeContigTensor(rank, DataType::Int);
      fusion.addInput(tv1);
      fusion.addInput(tv_idx);
      auto tv_gather = torch_gather(tv1, dim, tv_idx);
      auto tv_add = add(tv_gather, tv_gather);
      auto tv_out = mul(tv_gather, tv_add);
      fusion.addOutput(tv_out);

      std::vector<int64_t> input_dims(rank, 0);
      for (int idim = 0; idim < rank; ++idim) {
        input_dims[idim] = (std::rand() % max_dim_size) + 2;
      }

      std::vector<int64_t> index_dims(rank, 0);
      for (int idim = 0; idim < rank; ++idim) {
        index_dims[idim] = (std::rand() % input_dims[idim]) + 1;
      }

      at::manual_seed(0);
      auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
      auto options_i =
          at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

      at::Tensor input = at::randn(input_dims, options); // lookup
      at::Tensor input_idx =
          at::randint(0, input_dims[dim], index_dims, options_i);
      at::Tensor output = at::zeros(index_dims, options);

      auto t_gather = at::gather(input, dim, input_idx);
      auto t_add = at::add(t_gather, t_gather);
      auto tv_out_ref = at::mul(t_gather, t_add);

      std::vector<IValue> aten_inputs = {input, input_idx};

      FusionExecutorCache executor_cache(std::move(fusion_ptr));
      auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
      testValidate(
          &fusion, cg_outputs, aten_inputs, {tv_out_ref}, __LINE__, __FILE__);
    }
  }
}

// pass
TEST_F(NVFuserTest, FusionAddGatherSumAdd_CUDA) {
  const int max_dim_size = 4;
  std::srand(std::time(nullptr));
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

  for (int rank = 2; rank <= 5; ++rank) {
    for (int dim = 0; dim < rank; ++dim) {
      auto fusion_ptr = std::make_unique<Fusion>();
      Fusion& fusion = *fusion_ptr.get();
      FusionGuard fg(&fusion);

      TensorView* tv_lookup = makeContigTensor(rank);
      TensorView* tv_idx_1 = makeContigTensor(rank, DataType::Int);
      TensorView* tv_idx_2 = makeContigTensor(rank, DataType::Int);
      TensorView* tv_add_1 = makeContigTensor(rank - 1);

      fusion.addInput(tv_lookup);
      fusion.addInput(tv_idx_1);
      fusion.addInput(tv_idx_2);
      fusion.addInput(tv_add_1);

      auto tv_index = add(tv_idx_1, tv_idx_2);
      auto tv_out = torch_gather(tv_lookup, dim, tv_index);
      // auto tv_sum = sum(tv_gather, {0}, true);
      // auto tv_out = add(tv_sum, tv_add_1);

      fusion.addOutput(tv_out);

      std::vector<int64_t> input_dims(rank, 0);
      for (int idim = 0; idim < rank; ++idim) {
        input_dims[idim] = (std::rand() % max_dim_size) + 2;
      }

      std::vector<int64_t> index_dims(rank, 0);
      for (int idim = 0; idim < rank; ++idim) {
        index_dims[idim] = (std::rand() % input_dims[idim]) + 1;
      }

      std::vector<int64_t> add_1_dims(rank - 1, 0);
      for (int idim = 0; idim < rank - 1; ++idim) {
        add_1_dims[idim] = index_dims[idim + 1];
      }

      at::Tensor t_lookup = at::randn(input_dims, options); // lookup
      at::Tensor t_idx_1 =
          at::randint(0, input_dims[dim] / 2, index_dims, options_i);
      at::Tensor t_idx_2 =
          at::randint(0, input_dims[dim] / 2, index_dims, options_i);
      at::Tensor t_add_1 = at::randn(add_1_dims, options); // lookup

      auto t_index = at::add(t_idx_1, t_idx_2);
      auto t_out = at::gather(t_lookup, dim, t_index);
      // auto t_sum = at::sum(t_gather, {0}, true);
      // auto t_out = at::add(t_sum, t_add_1);

      std::vector<IValue> aten_inputs = {t_lookup, t_idx_1, t_idx_2, t_add_1};
      FusionExecutorCache executor_cache(std::move(fusion_ptr));
      auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
      testValidate(
          &fusion, cg_outputs, aten_inputs, {t_out}, __LINE__, __FILE__);
    }
  }
}

// pass
TEST_F(NVFuserTest, FusionGatherSumAdd_CUDA) {
  const int max_dim_size = 64;
  std::srand(std::time(nullptr));
  at::manual_seed(0);
  for (int rank = 2; rank <= 5; ++rank) {
    for (int dim = 0; dim < rank; ++dim) {
      auto fusion_ptr = std::make_unique<Fusion>();
      Fusion& fusion = *fusion_ptr.get();
      FusionGuard fg(&fusion);

      TensorView* tv1 = makeContigTensor(rank);
      TensorView* tv_idx = makeContigTensor(rank, DataType::Int);
      TensorView* tv2 = makeContigTensor(rank - 1);

      fusion.addInput(tv1);
      fusion.addInput(tv_idx);
      fusion.addInput(tv2);

      auto tv_gather = torch_gather(tv1, dim, tv_idx);
      auto tv_sum = sum(tv_gather, {0}, true);
      auto tv_out = add(tv_sum, tv2);

      fusion.addOutput(tv_out);

      std::vector<int64_t> input_dims(rank, 0);
      for (int idim = 0; idim < rank; ++idim) {
        input_dims[idim] = (std::rand() % max_dim_size) + 2;
      }

      std::vector<int64_t> index_dims(rank, 0);
      for (int idim = 0; idim < rank; ++idim) {
        index_dims[idim] = (std::rand() % input_dims[idim]) + 1;
      }

      std::vector<int64_t> input2_dims(rank - 1, 0);
      for (int idim = 0; idim < rank - 1; ++idim) {
        input2_dims[idim] = index_dims[idim + 1];
      }

      auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
      auto options_i =
          at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

      at::Tensor input = at::randn(input_dims, options); // lookup
      at::Tensor input2 = at::randn(input2_dims, options); // lookup
      at::Tensor input_idx =
          at::randint(0, input_dims[dim], index_dims, options_i);
      at::Tensor output = at::zeros(index_dims, options);

      auto t_gather = at::gather(input, dim, input_idx);
      auto t_sum = at::sum(t_gather, {0}, true);
      auto tv_out_ref = at::add(input2, t_sum);

      std::vector<IValue> aten_inputs = {input, input_idx, input2};

      FusionExecutorCache executor_cache(std::move(fusion_ptr));
      auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
      testValidate(
          &fusion, cg_outputs, aten_inputs, {tv_out_ref}, __LINE__, __FILE__);
    }
  }
}

// pass
TEST_F(NVFuserTest, FusionGatherAddMulHugeSize_CUDA) {
  const int max_dim_size = 45536;
  std::srand(std::time(nullptr));
  for (int rank = 1; rank <= 2; ++rank) {
    for (int dim = 0; dim < rank; ++dim) {
      auto fusion_ptr = std::make_unique<Fusion>();
      Fusion& fusion = *fusion_ptr.get();
      FusionGuard fg(&fusion);

      TensorView* tv1 = makeContigTensor(rank);
      TensorView* tv_idx = makeContigTensor(rank, DataType::Int);
      fusion.addInput(tv1);
      fusion.addInput(tv_idx);
      auto tv_gather = torch_gather(tv1, dim, tv_idx);
      auto tv_add = add(tv_gather, tv_gather);
      auto tv_out = mul(tv_gather, tv_add);
      fusion.addOutput(tv_out);

      std::vector<int64_t> input_dims(rank, 0);
      for (int idim = 0; idim < rank; ++idim) {
        input_dims[idim] = (std::rand() % max_dim_size) + 2;
      }

      std::vector<int64_t> index_dims(rank, 0);
      for (int idim = 0; idim < rank; ++idim) {
        index_dims[idim] = (std::rand() % input_dims[idim]) + 1;
      }

      at::manual_seed(0);
      auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
      auto options_i =
          at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

      at::Tensor input = at::randn(input_dims, options); // lookup
      at::Tensor input_idx =
          at::randint(0, input_dims[dim], index_dims, options_i);
      at::Tensor output = at::zeros(index_dims, options);

      auto t_gather = at::gather(input, dim, input_idx);
      auto t_add = at::add(t_gather, t_gather);
      auto tv_out_ref = at::mul(t_gather, t_add);

      std::vector<IValue> aten_inputs = {input, input_idx};

      FusionExecutorCache executor_cache(std::move(fusion_ptr));
      auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
      testValidate(
          &fusion, cg_outputs, aten_inputs, {tv_out_ref}, __LINE__, __FILE__);
    }
  }
}

// will generate two kernels
TEST_F(NVFuserTest, GatherCannotFusion_CUDA) {
  const int max_dim_size = 45536;
  const int rank = 2;

  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv1 = makeContigTensor(rank);
  TensorView* tv_idx = makeContigTensor(rank, DataType::Int);
  fusion.addInput(tv1);
  fusion.addInput(tv_idx);

  auto tv_inp = add(tv1, tv1);
  auto tv_gather = torch_gather(tv_inp, 0, tv_idx);
  fusion.addOutput(tv_gather);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t1 = at::randn({4, 4}, options);
  at::Tensor t_idx = at::randint(0, 4, {4, 4}, options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t1, t_idx});
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)