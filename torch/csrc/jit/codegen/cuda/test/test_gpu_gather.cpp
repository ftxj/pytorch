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
using namespace at::indexing;

// sh build.sh;
// build/bin/test_jit --gtest_filter='NVFuserTest.FusionIndexSelect_CUDA*'

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
      std::cout << "success on rank = " << rank << ", dim = " << dim << std::endl;
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
      std::cout << "success on rank = " << rank << ", dim = " << dim << std::endl;
    } 
  }
}

// pass
TEST_F(NVFuserTest, TorchGatherReduceFusion_CUDA) {
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
      auto tv_sum = sum(tv_gather, {0}, true);
      auto tv_out = add(tv_sum, tv_sum);
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
      auto t_sum = at::sum(t_gather, {0}, true);
      auto tv_out_ref = at::add(t_sum, t_sum);
    

      std::vector<IValue> aten_inputs = {input, input_idx};

      FusionExecutorCache executor_cache(std::move(fusion_ptr));
      auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
      testValidate(
          &fusion, cg_outputs, aten_inputs, {tv_out_ref}, __LINE__, __FILE__);
      std::cout << "success on rank = " << rank << ", dim = " << dim << std::endl;
    } 
  }
}

// pass
TEST_F(NVFuserTest, GatherHandsOnFusion_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int nDims = 3;
  int x = 4, y = 4, z = 2;
  int ix = 2, iy = 2, iz = 2;
  int min_elm = 2;
  const int select_dim = 2;

  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv2 = makeContigTensor(nDims, DataType::Int);

  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  TensorView* tv_dim_0_4 = torch_gather(tv0, select_dim, tv2);
  TensorView* tv_dim_0_5 = add(tv1, tv_dim_0_4);
  TensorView* tv_dim_0_6 = mul(tv_dim_0_4, tv_dim_0_5);


  fusion.addOutput(tv_dim_0_6);
  tv0->computeAt(tv_dim_0_6, -1);
  // tv_dim_0_4->axis(-1)->parallelize(ParallelType::TIDx);


  std::vector<int64_t> storage_x(ix * iy * iz, 0);
  for (int i = 0; i < ix; ++i) {
    for (int j = 0; j < iy; ++j) {
      for (int k = 0; k < iz; ++k) {
        storage_x[i * (iy * iz) + j * iz + k] = std::abs(std::rand()) % min_elm;
      }
    }
  }
  auto opts = torch::TensorOptions().dtype(torch::kLong);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto input_index = torch::from_blob(storage_x.data(), {ix, iy, iz}, opts).clone().to(torch::kCUDA);
  at::Tensor input_lookup = at::randn({x, y, z}, options);
  at::Tensor input_tensor = at::randn({ix, iy, iz}, options);
  auto output1 = at::randn({ix, iy, iz}, options);
  auto output2 = at::randn({ix, iy, iz}, options);
  auto output3 = at::randn({ix, iy, iz}, options);

  auto torch_gather_res_0 = at::gather(input_lookup, select_dim, input_index);
  auto torch_add_res_0 = at::add(torch_gather_res_0, input_tensor);
  auto torch_mul_res_0 = at::mul(torch_gather_res_0, torch_add_res_0);


  std::vector<IValue> aten_inputs = {input_lookup, input_tensor, input_index};
  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  std::cout << fe.kernelString() << std::endl;

  fe.runFusion(aten_inputs, {output1});


  TORCH_CHECK(torch_mul_res_0.allclose(output1));
}

// pass
TEST_F(NVFuserTest, TorchGatherReduceAutoFusionCode_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 3;
  int x = 6, y = 6, z = 7;
  int ix = 5, iy = 3, iz = 4;
  int min_elm = 3;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(nDims, DataType::Int);

  fusion.addInput(tv0);
  fusion.addInput(tv_idx);

  auto tv1 = torch_gather(tv0, 0, tv_idx);
  auto tv2 = torch_gather(tv0, 1, tv_idx);
  auto tv3 = torch_gather(tv0, 2, tv_idx);
  auto tv4 = sum(tv1, {0}, true);
  auto tv5 = sum(tv2, {1}, true);
  auto tv6 = sum(tv3, {0, 1}, true);
  auto tv7 = add(tv1, tv4);
  auto tv8 = add(tv2, tv5);
  auto tv9 = add(tv3, tv6);
  // Register your outputs
  fusion.addOutput(tv7);
  fusion.addOutput(tv8);
  fusion.addOutput(tv9);

  std::vector<int64_t> storage_x(ix * iy * iz, 0);
  for (int i = 0; i < ix; ++i) {
    for (int j = 0; j < iy; ++j) {
      for (int k = 0; k < iz; ++k) {
        storage_x[i * (iy * iz) + j * iz + k] = std::abs(std::rand()) % min_elm;
      }
    }
  }
  auto opts = torch::TensorOptions().dtype(torch::kLong);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({x, y, z}, options);
  auto idx = torch::from_blob(storage_x.data(), {ix, iy, iz}, opts).clone().to(torch::kCUDA);

  auto t1 = at::gather(t0, 0, idx);
  auto t2 = at::gather(t0, 1, idx);
  auto t3 = at::gather(t0, 2, idx);
  auto t4 = at::sum(t1, {0}, true);
  auto t5 = at::sum(t2, {1}, true);
  auto t6 = at::sum(t3, IntArrayRef{0, 1}, true);
  auto t7 = t1 + t4;
  auto t8 = t2 + t5;
  auto t9 = t3 + t6;

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, idx});
  
  TORCH_CHECK(t7.allclose(cg_outputs[0]));
  TORCH_CHECK(t8.allclose(cg_outputs[1]));
  TORCH_CHECK(t9.allclose(cg_outputs[2]));
}


} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)