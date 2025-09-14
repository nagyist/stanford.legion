/*
 * Copyright 2025 Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "realm/realm_c.h"
#include "test_mock.h"
#include "test_common.h"
#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <assert.h>
#include <map>
#include <set>
#include <gtest/gtest.h>

using namespace Realm;

namespace Realm {
  extern bool enable_unit_tests;
};

class CAffinityBaseTest {
protected:
  void initialize(int num_nodes)
  {
    Realm::enable_unit_tests = true;
    runtime_impl = std::make_unique<MockRuntimeImplMachineModel>();
    runtime_impl->init(num_nodes);
  }

  void finalize(void) { runtime_impl->finalize(); }

protected:
  std::unique_ptr<MockRuntimeImplMachineModel> runtime_impl{nullptr};
};

// test failed cases

class CAffinityFailedTest : public CAffinityBaseTest, public ::testing::Test {
protected:
  void SetUp() override { CAffinityBaseTest::initialize(1); }

  void TearDown() override { CAffinityBaseTest::finalize(); }
};

TEST_F(CAffinityFailedTest, InvalidRuntime)
{
  realm_memory_t mem1 = ID::make_memory(0, 0).convert<Memory>();
  realm_memory_t mem2 = ID::make_memory(0, 1).convert<Memory>();
  realm_affinity_details_t details;
  realm_status_t status =
      realm_runtime_get_memory_memory_affinity(nullptr, mem1, mem2, &details);
  EXPECT_EQ(status, REALM_RUNTIME_ERROR_NOT_INITIALIZED);
}

TEST_F(CAffinityFailedTest, InvalidMemory)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_memory_t mem1 = REALM_NO_MEM;
  realm_memory_t mem2 = ID::make_memory(0, 1).convert<Memory>();
  realm_affinity_details_t details;
  realm_status_t status =
      realm_runtime_get_memory_memory_affinity(runtime, mem1, mem2, &details);
  EXPECT_EQ(status, REALM_MEMORY_ERROR_INVALID_MEMORY);
}

TEST_F(CAffinityFailedTest, InvalidProcessor)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_processor_t proc = REALM_NO_PROC;
  realm_memory_t mem = ID::make_memory(0, 1).convert<Memory>();
  realm_affinity_details_t details;
  realm_status_t status =
      realm_runtime_get_processor_memory_affinity(runtime, proc, mem, &details);
  EXPECT_EQ(status, REALM_PROCESSOR_ERROR_INVALID_PROCESSOR);
}

TEST_F(CAffinityFailedTest, InvalidDetails)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_memory_t mem1 = ID::make_memory(0, 0).convert<Memory>();
  realm_memory_t mem2 = ID::make_memory(0, 1).convert<Memory>();
  realm_affinity_details_t details;
  realm_status_t status =
      realm_runtime_get_memory_memory_affinity(runtime, mem1, mem2, nullptr);
  EXPECT_EQ(status, REALM_ERROR_INVALID_PARAMETER);
}

class CProcessorMemoryAffinityTest : public CAffinityBaseTest, public ::testing::Test {
protected:
  void SetUp() override
  {
    CAffinityBaseTest::initialize(1);
    // there is a proc-mem affinity between proc0 and mem0, but not proc0 and mem1
    MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded procs_mems = {
        {proc_info1}, {mem_info1, mem_info2}, {proc1_mem1_affinity}, {}};
    runtime_impl->setup_mock_proc_mems(procs_mems);
  }

  void TearDown() override { CAffinityBaseTest::finalize(); }

  static constexpr MockRuntimeImplMachineModel::MockProcessorInfo proc_info1{
      0, Processor::Kind::LOC_PROC, 0};
  static constexpr MockRuntimeImplMachineModel::MockMemoryInfo mem_info1{
      0, Memory::Kind::SYSTEM_MEM, 1024, 0};
  static constexpr MockRuntimeImplMachineModel::MockMemoryInfo mem_info2{
      1, Memory::Kind::SYSTEM_MEM, 1024, 0};
  static constexpr MockRuntimeImplMachineModel::MockProcessorMemoryAffinity
      proc1_mem1_affinity{0, 0, 1000, 1};
};

TEST_F(CProcessorMemoryAffinityTest, HasProcessorMemoryAffinity)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_processor_t proc =
      ID::make_processor(proc_info1.address_space, proc_info1.idx).convert<Processor>();
  realm_memory_t mem =
      ID::make_memory(mem_info1.address_space, mem_info1.idx).convert<Memory>();
  realm_affinity_details_t details;
  realm_status_t status =
      realm_runtime_get_processor_memory_affinity(runtime, proc, mem, &details);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(details.bandwidth, proc1_mem1_affinity.bandwidth);
  EXPECT_EQ(details.latency, proc1_mem1_affinity.latency);
}

TEST_F(CProcessorMemoryAffinityTest, NoProcessorMemoryAffinity)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_processor_t proc =
      ID::make_processor(proc_info1.address_space, proc_info1.idx).convert<Processor>();
  realm_memory_t mem =
      ID::make_memory(mem_info2.address_space, mem_info2.idx).convert<Memory>();
  realm_affinity_details_t details;
  realm_status_t status =
      realm_runtime_get_processor_memory_affinity(runtime, proc, mem, &details);
  // no proc-mem affinity between proc1 and mem2
  EXPECT_EQ(status, REALM_RUNTIME_ERROR_INVALID_AFFINITY);
}

class CMemoryMemoryAffinityTest : public CAffinityBaseTest, public ::testing::Test {
protected:
  void SetUp() override
  {
    CAffinityBaseTest::initialize(1);
    // there is a proc-mem affinity between proc0 and mem0, but not proc0 and mem1
    // there is a mem-mem affinity between mem0 and mem1, but not mem0 and mem2 and mem1
    // and mem2
    MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded procs_mems = {
        {proc_info1},
        {mem_info1, mem_info2, mem_info3},
        {proc1_mem1_affinity},
        {mem1_mem2_affinity}};
    runtime_impl->setup_mock_proc_mems(procs_mems);
  }

  void TearDown() override { CAffinityBaseTest::finalize(); }

  static constexpr MockRuntimeImplMachineModel::MockProcessorInfo proc_info1{
      0, Processor::Kind::LOC_PROC, 0};
  static constexpr MockRuntimeImplMachineModel::MockMemoryInfo mem_info1{
      0, Memory::Kind::SYSTEM_MEM, 1024, 0};
  static constexpr MockRuntimeImplMachineModel::MockMemoryInfo mem_info2{
      1, Memory::Kind::SYSTEM_MEM, 1024, 0};
  static constexpr MockRuntimeImplMachineModel::MockMemoryInfo mem_info3{
      2, Memory::Kind::GPU_FB_MEM, 1024, 0};
  static constexpr MockRuntimeImplMachineModel::MockProcessorMemoryAffinity
      proc1_mem1_affinity{0, 0, 1000, 1};
  static constexpr MockRuntimeImplMachineModel::MockMemoryMemoryAffinity
      mem1_mem2_affinity{0, 1, 5000, 10};
};

TEST_F(CMemoryMemoryAffinityTest, HasMemoryMemoryAffinity)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_memory_t mem1 =
      ID::make_memory(mem_info1.address_space, mem_info1.idx).convert<Memory>();
  realm_memory_t mem2 =
      ID::make_memory(mem_info2.address_space, mem_info2.idx).convert<Memory>();
  realm_affinity_details_t details;
  realm_status_t status =
      realm_runtime_get_memory_memory_affinity(runtime, mem1, mem2, &details);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(details.bandwidth, mem1_mem2_affinity.bandwidth);
  EXPECT_EQ(details.latency, mem1_mem2_affinity.latency);
}

TEST_F(CMemoryMemoryAffinityTest, NoMemoryMemoryAffinity)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_memory_t mem1 =
      ID::make_memory(mem_info1.address_space, mem_info1.idx).convert<Memory>();
  realm_memory_t mem2 =
      ID::make_memory(mem_info3.address_space, mem_info3.idx).convert<Memory>();
  realm_affinity_details_t details;
  realm_status_t status =
      realm_runtime_get_memory_memory_affinity(runtime, mem1, mem2, &details);
  EXPECT_EQ(status, REALM_RUNTIME_ERROR_INVALID_AFFINITY);
}
