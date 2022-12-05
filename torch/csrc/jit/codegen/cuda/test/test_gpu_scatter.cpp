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
TEST_F(NVFuserTest, ScatterAddOpAllDim_CUDA) {
  const int max_dim_size = 64;
  std::srand(std::time(nullptr));
  for(int rank = 1; rank <= 5; ++rank) {
    for(int dim = 0; dim < rank; ++dim) {
      auto fusion_ptr = std::make_unique<Fusion>();
      Fusion& fusion = *fusion_ptr.get();
      FusionGuard fg(&fusion);

      TensorView* tv1 = makeContigTensor(rank);
      TensorView* tv_idx = makeContigTensor(rank, DataType::Int);
      TensorView* tv_out = makeContigTensor(rank);

      fusion.addInput(tv1);
      fusion.addInput(tv_idx);
      scatter_add(tv_out, tv1, dim, tv_idx);
      fusion.addOutput(tv_out);
    

      std::vector<int64_t> input_dims(rank, 0);
      for(int idim = 0; idim < rank; ++idim) {
        input_dims[idim] = (std::rand() % max_dim_size) + 2;
      }

      std::vector<int64_t> output_dims(rank, 0);
      for(int idim = 0; idim < rank; ++idim) {
        output_dims[idim] = (std::rand() % input_dims[idim]) + 2;
      }
      
      std::vector<int64_t> index_dims(rank, 0);
      for(int idim = 0; idim < rank; ++idim) {
        index_dims[idim] = (std::rand() % input_dims[idim]) + 1;
        index_dims[idim] = (index_dims[idim] % output_dims[idim]) + 1;
      }



      at::manual_seed(0);
      auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
      auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

      at::Tensor input = at::randn(input_dims, options); // lookup
      at::Tensor input_idx = at::randint(0, output_dims[dim], index_dims, options_i);
      at::Tensor output = at::randn(output_dims, options);

      auto tv_out_ref = output.scatter_add_(dim, input_idx, input);
        
      std::vector<IValue> aten_inputs = {input, input_idx, output};

      FusionExecutorCache executor_cache(std::move(fusion_ptr));
      auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
      testValidate(
          &fusion, cg_outputs, aten_inputs, {tv_out_ref}, __LINE__, __FILE__);
      std::cout << "success on rank = " << rank << ", dim = " << dim << std::endl;

    } 
  }
}

TEST_F(NVFuserTest, ScatterAddElementwiseFusion_CUDA) {
  const int max_dim_size = 64;
  std::srand(std::time(nullptr));
  int rank = 2;
  int dim = 0;
  // for(int rank = 1; rank <= 5; ++rank) {
    // for(int dim = 0; dim < rank; ++dim) {
      auto fusion_ptr = std::make_unique<Fusion>();
      Fusion& fusion = *fusion_ptr.get();
      FusionGuard fg(&fusion);

      TensorView* tv1 = makeContigTensor(rank);
      TensorView* tv_idx = makeContigTensor(rank, DataType::Int);
      TensorView* tv_out = makeContigTensor(rank);

      fusion.addInput(tv1);
      fusion.addInput(tv_idx);
      fusion.addInput(tv_out);
      auto new_out = scatter_add(tv_out, tv1, dim, tv_idx);
      fusion.addOutput(new_out);
      std::cout << fusion << std::endl;

      std::vector<int64_t> input_dims(rank, 0);
      for(int idim = 0; idim < rank; ++idim) {
        input_dims[idim] = (std::rand() % max_dim_size) + 2;
      }

      std::vector<int64_t> output_dims(rank, 0);
      for(int idim = 0; idim < rank; ++idim) {
        output_dims[idim] = (std::rand() % input_dims[idim]) + 2;
      }
      
      std::vector<int64_t> index_dims(rank, 0);
      for(int idim = 0; idim < rank; ++idim) {
        index_dims[idim] = (std::rand() % input_dims[idim]) + 1;
        index_dims[idim] = (index_dims[idim] % output_dims[idim]) + 1;
      }



      at::manual_seed(0);
      auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
      auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

      at::Tensor input = at::randn(input_dims, options); // lookup
      at::Tensor input_idx = at::randint(0, output_dims[dim], index_dims, options_i);
      at::Tensor output = at::randn(output_dims, options);

      auto tv_out_ref = output.scatter_add_(dim, input_idx, input);
    
      std::vector<IValue> aten_inputs = {input, input_idx, output};

      FusionExecutor fe;
      fe.compileFusion(&fusion, aten_inputs);
      std::cout << fe.kernelString() << std::endl;

    // } 
  // }
}

TEST_F(NVFuserTest, FusionScatterCompileRtc_CUDA) {
  FusionExecutor fe;
  std::string kernel = R"(
__global__ void kernel1(Tensor<float, 1> T0, Tensor<int64_t, 1> T1, Tensor<float, 1> T2, Tensor<float, 1> T3) {
  int i41;
  i41 = (((nvfuser_index_t)blockIdx.x) * 128) + ((nvfuser_index_t)threadIdx.x);
  if ((i41 < T2.size[0])) {
    atomicAdd(&T2[(T1[i41] * 1)], T0[i41]);
     T3[(T1[i41] * 1)] = T2[(T1[i41] * 1)];
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
  
  const std::vector<int64_t> tensor_dims = {8};
  auto in0 = at::randn(tensor_dims, options);
  auto in_idx = at::randint(0, 8, {8}, options_i);
  auto in_out = at::randn(tensor_dims, options);
  auto in_out2 = in_out;
  auto out0 = at::empty_like(in_out);

  fe.runRtc(lp, {in0, in_idx, in_out, out0});

  std::cout << "input = " << std::endl;
  std::cout << in0 << std::endl;
  std::cout << "index = " << std::endl;
  std::cout << in_idx << std::endl;
  std::cout << "output init = " << std::endl;
  std::cout << in_out << std::endl;

  std::cout << "output res = " << std::endl;
  std::cout << out0 << std::endl;

  std::cout << "output init = " << std::endl;
  std::cout << in_out2 << std::endl;
  auto out_ref = in_out2.scatter_add_(0, in_idx, in0);
  std::cout << "output ref = " << std::endl;
  std::cout << out_ref << std::endl;
  
  TORCH_CHECK(out_ref.allclose(out0));
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)