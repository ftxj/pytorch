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
  // Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser::cuda;
using namespace at::indexing;

// sh build.sh;
// build/bin/test_jit --gtest_filter='NVFuserTest.FusionIndexSelect_CUDA*'

TEST_F(NVFuserTest, DebugAllocate_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int nDims = 3;

  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv2 = makeContigTensor(nDims);

  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);
  
  TensorView* tv4 = add(tv0, tv1);
  TensorView* tv5 = add(tv2, tv4);

  fusion.addOutput(tv5);


  FusionExecutor fe;
  fe.compileFusion(&fusion);
  std::cout << fe.kernelString() << std::endl;

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

  std::cout << fusion << std::endl;

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
TEST_F(NVFuserTest, TorchGatherAutoFusionCode_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 3;
  int x = 6, y = 6, z = 7;
  int ix = 5, iy = 3, iz = 4;
  int min_elm = 3;
  const int dim = 2;
  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims, DataType::Int);
  // TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv2 = makeContigTensor(nDims);

  // Register your inputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  TensorView* tv3 = torch_gather(tv0, dim, tv1);
  TensorView* tv4 = mul(tv2, tv3);
  TensorView* tv5 = add(tv3, tv4);
  // Register your outputs
  fusion.addOutput(tv5);


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

  at::Tensor input_lookup = at::randn({x, y, z}, options);
  // at::Tensor input_index = at::randn({ix, iy, iz}, options);
  auto input_index = torch::from_blob(storage_x.data(), {ix, iy, iz}, opts).clone().to(torch::kCUDA);
  at::Tensor input_tensor = at::randn({ix, iy, iz}, options);
  auto output1 = at::randn({ix, iy, iz}, options);

  auto torch_gather_res = at::gather(input_lookup, dim, input_index);
  auto torch_mul_res = at::mul(input_tensor, torch_gather_res);
  auto torch_add_res = at::add(torch_gather_res, torch_mul_res);

  std::vector<IValue> aten_inputs = {input_lookup, input_index, input_tensor};
  
  auto lparams = schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  std::cout << fe.kernelString() << std::endl;

  fe.runFusion(aten_inputs, {output1}, lparams);

  std::cout << output1 << std::endl;
  std::cout << "ref = " << std::endl;
  std::cout << torch_add_res << std::endl;
  
  TORCH_CHECK(torch_add_res.allclose(output1));
}


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



TEST_F(NVFuserTest, TorchGatherKernel_CUDA) {
  FusionExecutor fe;
  std::string kernel = R"(
__global__ void kernel1(Tensor<float, 3> T0, Tensor<int64_t, 3> T1, Tensor<float, 3> T2) {
  int i64;
  i64 = (((nvfuser_index_t)blockIdx.x) * 128) + ((nvfuser_index_t)threadIdx.x);
  if ((i64 < (T1.size[0] * (T1.size[1] * T1.size[2])))) {
    if(i64 == 0) {
      printf("index dim 0 = %d, 1 = %d, 2= %d\n", (int)T1.size[0], (int)T1.size[1], (int)T1.size[2]);
      printf("input dim 0 = %d, 1 = %d, 2= %d\n", (int)T0.size[0], (int)T0.size[1], (int)T0.size[2]);
    }

    float T3[1];
    T3[0]
       = T0[(T1[i64] * (T0.size[2] * T0.size[1])) + (((i64 / T1.size[0]) % T1.size[1]) * T0.size[2]) + (((i64 / 1) % T1.size[2]) * 1)];
    T2[i64]
       = T3[0];

    printf("%d = [%ld][%ld][%ld] = %f\n", i64, T1[i64], (i64 / T1.size[0]) % T1.size[1], ((i64 / 1) % T1.size[2]), T3[0]);

  }
}
    )";
  fe.compileRtc(kernel, "CudaCodeGen::kernel1");
  LaunchParams lp(
      16, // gdimx
      1, // gdimy
      1, // gdimz
      16, // bdimx
      1, // bdimy
      1 // bdimz
  );
  lp.setSmem(0);

  int nDims = 3;
  int x = 14, y = 51, z = 15;
  int ix = 10, iy = 23, iz = 10;
  int min_elm = 10;

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
  at::Tensor input_lookup = at::randn({ix, iy, iz}, options);
  auto output1 = at::randn({ix, iy, iz}, options);

  fe.runRtc(lp, {input_lookup, input_index, output1});
  auto out_ref = at::gather(input_lookup, 0, input_index);
  
  std::cout << "input = " << std::endl;
  std::cout << input_lookup << std::endl;

  std::cout << "index = " << std::endl;
  std::cout << input_index << std::endl;

  std::cout << "ref out = " << std::endl;
  std::cout << out_ref << std::endl;

  std::cout << "out = " << std::endl;
  std::cout << output1 << std::endl;

  TORCH_CHECK(out_ref.allclose(output1));
}


} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)