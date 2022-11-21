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
// build/bin/test_jit
// --gtest_filter='NVFuserTest*FusionIndexSelectExplicitBroadcast_CUDA*'
// build/bin/test_jit --gtest_filter='NVFuserTest*FusionIndexSelect_CUDA*'

TEST_F(NVFuserTest, GatherNodeOutDim_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 2;

  // Set up your input tensor views
  TensorView* tv_inp = makeContigTensor(nDims + 1);
  TensorView* tv_idx = makeContigTensor(nDims);
  // Register your inputs
  fusion.addInput(tv_inp);
  fusion.addInput(tv_idx);

  TensorView* tv1 = add(tv_inp, tv_idx);

  TensorView* gather_node = torch_gather(tv1, 1, tv_idx);

  fusion.addOutput(gather_node);

  std::cout << gather_node << std::endl;
  std::cout << fusion << std::endl;
}

// pass
TEST_F(NVFuserTest, GatherCodeCheck_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 2;
  int nElem = 4;

  // Set up your input tensor views
  TensorView* tv_inp = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(nDims, DataType::Int);
  // Register your inputs
  fusion.addInput(tv_inp);
  fusion.addInput(tv_idx);

  TensorView* gather_node = torch_gather(tv_inp, 0, tv_idx);

  fusion.addOutput(gather_node);

  std::cout << fusion << std::endl;

  std::vector<int64_t> storage(nElem * nElem ,0);
  for (int i = 0; i < nElem; ++i) {
    for (int j = 0; j < nElem; ++j) {
      storage[i * nElem + j] = std::abs(std::rand()) % nElem;
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input0 = at::randn({nElem, nElem}, options);

  auto opts = torch::TensorOptions().dtype(torch::kLong);
  auto input1 = torch::from_blob(storage.data(), {nElem, nElem}, opts).clone().to(torch::kCUDA);

  auto output = at::empty_like(input0);

  std::cout << "input :" << std::endl;
  std::cout << input0 << std::endl;

  std::cout << "index :" << std::endl;
  std::cout << input1 << std::endl;

  auto tv0_ref = at::gather(input0, 0, input1);

  std::cout << "ref output :" << std::endl;
  std::cout << tv0_ref << std::endl;

  std::vector<IValue> aten_inputs = {input0, input1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  std::cout << fe.kernelString() << std::endl;

  fe.runFusion(aten_inputs, {output});


  std::cout << "output :" << std::endl;
  std::cout << output << std::endl;

  TORCH_CHECK(tv0_ref.allclose(output));
}

// pass
TEST_F(NVFuserTest, GatherFusionCheckCode_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int nDims = 2;
  int nElem = 4;

  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv2 = makeContigTensor(nDims, DataType::Int);
  // TensorView* tv2 = makeContigTensor(nDims);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);
  
  TensorView* tv3 = add(tv0, tv1);
  TensorView* tv4 = torch_gather(tv3, 0, tv2);
  TensorView* tv5 = add(tv3, tv4);
  TensorView* tv6 = mul(tv4, tv5);

  fusion.addOutput(tv6);

  tv6->split(-1, 8);
  tv4->computeAt(tv6, -1);
  // can not parallelize on Index axis ?
  // tv6->axis(0)->parallelize(ParallelType::BIDx); 
  // tv6->axis(-1)->parallelize(ParallelType::TIDx);


  std::cout << fusion << std::endl;
  FusionExecutor fe;
  fe.compileFusion(&fusion);
  std::cout << fe.kernelString() << std::endl;
}


TEST_F(NVFuserTest, FusionGatherOpPointwise_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int nDims = 3;
  int x = 31, y = 65, z = 103;
  int ix = 20, iy = 32, iz = 50;
  int min_elm = 31;

  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* index = makeContigTensor(nDims, DataType::Int);
  

  fusion.addInput(tv0);
  fusion.addInput(index);

  auto tv1 = torch_gather(tv0, 0, index);
  auto tv2 = torch_gather(tv0, 1, index);
  auto tv3 = torch_gather(tv0, 2, index);
  fusion.addOutput(tv1);
  fusion.addOutput(tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);


  at::Tensor t0 = at::randn({x, y, z}, options);
  
  auto output1 = at::randn({ix, iy, iz}, options);
  auto output2 = at::randn({ix, iy, iz}, options);
  auto output3 = at::randn({ix, iy, iz}, options);
  
  std::vector<int64_t> storage_x(ix * iy * iz, 0);
  // std::vector<int64_t> storage_y(ix * iy * iz, 0);
  // std::vector<int64_t> storage_z(ix * iy * iz, 0);
  for (int i = 0; i < ix; ++i) {
    for (int j = 0; j < iy; ++j) {
      for (int k = 0; k < iz; ++k) {
        storage_x[i * (iy * iz) + j * iz + k] = std::abs(std::rand()) % min_elm;
        // storage_y[i * (iy * iz) + j * iz + k] = std::abs(std::rand()) % y;
        // storage_z[i * (iy * iz) + j * iz + k] = std::abs(std::rand()) % z;
      }
    }
  }
  auto opts = torch::TensorOptions().dtype(torch::kLong);
  auto input1 = torch::from_blob(storage_x.data(), {ix, iy, iz}, opts).clone().to(torch::kCUDA);
  // auto input2 = torch::from_blob(storage_y.data(), {ix, iy, iz}, opts).clone().to(torch::kCUDA);
  // auto input3 = torch::from_blob(storage_z.data(), {ix, iy, iz}, opts).clone().to(torch::kCUDA);
  
  auto t1 = at::gather(t0, 0, input1);
  auto t2 = at::gather(t0, 1, input1);
  auto t3 = at::gather(t0, 2, input1);

  std::vector<IValue> aten_inputs = {t0, input1};
  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  std::cout << fe.kernelString() << std::endl;

  fe.runFusion(aten_inputs, {output1, output2, output3});

  TORCH_CHECK(t1.allclose(output1));
  TORCH_CHECK(t2.allclose(output2));
  TORCH_CHECK(t3.allclose(output3));
  
  // testValidate(
  //     &fusion, cg_outputs, {t0, index}, {t1, t2, t3}, __LINE__, __FILE__);
}

// error
TEST_F(NVFuserTest, TorchGatherHandsOnFusion_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 2;
  int nElem = 4;
  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims, DataType::Int);
  TensorView* tv2 = makeContigTensor(nDims);

  // Register your inputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  TensorView* tv3 = torch_gather(tv0, 0, tv1);
  TensorView* tv4 = mul(tv2, tv3);
  // TensorView* tv3 = add(IrBuilder::create<Double>(17.0), tv2);
  // Register your outputs
  fusion.addOutput(tv4);

  std::cout << fusion << std::endl;

  std::vector<int64_t> storage(nElem * nElem ,0);
  for (int i = 0; i < nElem; ++i) {
    for (int j = 0; j < nElem; ++j) {
      storage[i * nElem + j] = std::abs(std::rand()) % nElem;
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input0 = at::randn({nElem, nElem}, options);
  auto input2 = at::randn({nElem, nElem}, options);

  auto opts = torch::TensorOptions().dtype(torch::kLong);
  auto input1 = torch::from_blob(storage.data(), {nElem, nElem}, opts).clone().to(torch::kCUDA);

  auto output = at::empty_like(input0);

  std::cout << "input :" << std::endl;
  std::cout << input0 << std::endl;

  std::cout << "index :" << std::endl;
  std::cout << input1 << std::endl;

  auto tmp = at::gather(input0, 0, input1);
  auto tv0_ref = at::mul(tmp, input2);

  std::cout << "ref output :" << std::endl;
  std::cout << tv0_ref << std::endl;

  std::vector<IValue> aten_inputs = {input0, input1, input2};

  auto lparams = schedulePointwise(&fusion, aten_inputs);
  std::cout << "schedulePointwise success" << std::endl;

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  std::cout << fe.kernelString() << std::endl;

  fe.runFusion(aten_inputs, {output}, lparams);


  std::cout << "output :" << std::endl;
  std::cout << output << std::endl;

  TORCH_CHECK(tv0_ref.allclose(output));
}



} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)