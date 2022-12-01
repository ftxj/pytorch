#if defined(USE_CUDA)
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/codegen.h>
#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/executor_launch_params.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/fusion_segmenter.h>
#include <torch/csrc/jit/codegen/cuda/grouped_reduction.h>
#include <torch/csrc/jit/codegen/cuda/inlining.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ir_graphviz.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_magic_zero.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/reduction_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>
#include <torch/csrc/jit/codegen/cuda/test/test_gpu_validator.h>
#include <torch/csrc/jit/codegen/cuda/test/test_utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/transform_rfactor.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/codegen/cuda/parser.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/torch.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <thread>
#include <ctime>
#include <cstdlib>
  // Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser::cuda;

// pass
TEST_F(NVFuserTest, TorchGatherOpAllDim_CUDA) {
  const int max_dim_size = 64;
  std::srand(std::time(nullptr));
  for(int rank = 1; rank <= 5; ++rank) {
    for(int dim = 0; dim < rank; ++dim) {
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
      for(int idim = 0; idim < rank; ++idim) {
        input_dims[idim] = (std::rand() % max_dim_size) + 2;
      }

      std::vector<int64_t> index_dims(rank, 0);
      for(int idim = 0; idim < rank; ++idim) {
        index_dims[idim] = (std::rand() % input_dims[idim]) + 1;
      }

      at::manual_seed(0);
      auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
      auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

      at::Tensor input = at::randn(input_dims, options); // lookup
      at::Tensor input_idx = at::randint(0, input_dims[dim], index_dims, options_i);
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
TEST_F(NVFuserTest, TorchGatherElementwiseFusion_CUDA) {
  const int max_dim_size = 64;
  std::srand(std::time(nullptr));
  for(int rank = 1; rank <= 5; ++rank) {
    for(int dim = 0; dim < rank; ++dim) {
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
      for(int idim = 0; idim < rank; ++idim) {
        input_dims[idim] = (std::rand() % max_dim_size) + 2;
      }

      std::vector<int64_t> index_dims(rank, 0);
      for(int idim = 0; idim < rank; ++idim) {
        index_dims[idim] = (std::rand() % input_dims[idim]) + 1;
      }

      at::manual_seed(0);
      auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
      auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

      at::Tensor input = at::randn(input_dims, options); // lookup
      at::Tensor input_idx = at::randint(0, input_dims[dim], index_dims, options_i);
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
TEST_F(NVFuserTest, TorchGatherReduceFusion_CUDA) {
  const int max_dim_size = 64;
  std::srand(std::time(nullptr));
  at::manual_seed(0);
  for(int rank = 2; rank <= 5; ++rank) {
    for(int dim = 0; dim < rank; ++dim) {
      auto fusion_ptr = std::make_unique<Fusion>();
      Fusion& fusion = *fusion_ptr.get();
      FusionGuard fg(&fusion);

      TensorView* tv1 = makeContigTensor(rank);
      TensorView* tv_idx = makeContigTensor(rank, DataType::Int);
      TensorView* tv2 = makeContigTensor(rank -1);

      fusion.addInput(tv1);
      fusion.addInput(tv_idx);
      fusion.addInput(tv2);

      auto tv_gather = torch_gather(tv1, dim, tv_idx);
      auto tv_sum = sum(tv_gather, {0}, true);
      auto tv_out = add(tv_sum, tv2);

      fusion.addOutput(tv_out);

      std::vector<int64_t> input_dims(rank, 0);
      for(int idim = 0; idim < rank; ++idim) {
        input_dims[idim] = (std::rand() % max_dim_size) + 2;
      }

      std::vector<int64_t> index_dims(rank, 0);
      for(int idim = 0; idim < rank; ++idim) {
        index_dims[idim] = (std::rand() % input_dims[idim]) + 1;
      }

      std::vector<int64_t> input2_dims(rank - 1, 0);
      for(int idim = 0; idim < rank - 1; ++idim) {
        input2_dims[idim] = index_dims[idim + 1];
      }


      auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
      auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

      at::Tensor input = at::randn(input_dims, options); // lookup
      at::Tensor input2 = at::randn(input2_dims, options); // lookup
      at::Tensor input_idx = at::randint(0, input_dims[dim], index_dims, options_i);
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
TEST_F(NVFuserTest, TorchGatherElementwiseFusionBigSize_CUDA) {
  const int max_dim_size = 45536;
  std::srand(std::time(nullptr));
  for(int rank = 1; rank <= 2; ++rank) {
    for(int dim = 0; dim < rank; ++dim) {
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
      for(int idim = 0; idim < rank; ++idim) {
        input_dims[idim] = (std::rand() % max_dim_size) + 2;
      }

      std::vector<int64_t> index_dims(rank, 0);
      for(int idim = 0; idim < rank; ++idim) {
        index_dims[idim] = (std::rand() % input_dims[idim]) + 1;
      }

      at::manual_seed(0);
      auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
      auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

      at::Tensor input = at::randn(input_dims, options); // lookup
      at::Tensor input_idx = at::randint(0, input_dims[dim], index_dims, options_i);
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

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)