// Copyright (C) 2021 lusyu1986@icloud.com

#include <cstdio>
#include <string>

#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/time/clock.h"
#include "ComputationalAdvertising/src/util/os/thread_cpu.h"
#include "ComputationalAdvertising/src/util/os/vpopen.h"
#include "ComputationalAdvertising/src/util/os/network.h"
#include "ComputationalAdvertising/src/util/os/resource_used.h"
#include "ComputationalAdvertising/src/util/os/semaphore.h"
#include "ComputationalAdvertising/src/util/io.h"
#include "ComputationalAdvertising/src/util/comm.h"
#include "ComputationalAdvertising/src/util/process/process_status.h"
#include "ComputationalAdvertising/src/util/process/process_initiator.h"

using std::string;

#ifdef __APPLE__
#elif defined(__linux__)
TEST(UTIL_OS_THREAD_CPU, GET_THREAD_CPU_INIT_VALUE) {
  int64_t get_cpu_id = -1;
  ASSERT_TRUE(get_thread_cpu(&get_cpu_id));
  ASSERT_EQ(get_cpu_id, 0);
}

TEST(UTIL_OS_THREAD_CPU, SET_THREAD_CPU) {
  int64_t set_cpu_id = 5;
  ASSERT_TRUE(set_thread_cpu(set_cpu_id));

  int64_t get_cpu_id = -1;
  ASSERT_TRUE(get_thread_cpu(&get_cpu_id));
  ASSERT_EQ(get_cpu_id, set_cpu_id);
}

TEST(UTIL_OS_NETWORK, GET_HOST_IP) {
  string ip;
  ASSERT_TRUE(get_host_ip(&ip));
  LOG(INFO) << "IP: " << ip;
}
#endif

TEST(UTIL_OS_RESOURCE_USED, PROCESS) {
  struct ResourceUsed used;
  ASSERT_TRUE(get_process_resource_used(&used));
  LOG(INFO) << "resident_mb: " << used.resident_mb
            << ", user_time: "   << used.user_time
            << ", system_time: " << used.system_time;
}

TEST(UTIL_OS_RESOURCE_USED, SYSTEM) {
  struct ResourceUsed used;
  ASSERT_TRUE(get_system_resource_used(&used));
  LOG(INFO) << "resident_mb: " << used.resident_mb
            << ", user_time: "   << used.user_time
            << ", system_time: " << used.system_time;
}

TEST(UTIL_OS_VPOPEN, VPOPEN) {
  FILE *fp = vpopen("pwd", "r");
  ASSERT_NE(fp, nullptr);
  ASSERT_NE(vpclose(fp), -1);
}

TEST(UTIL_IO, READ_LINE) {
  FILE *fp = fopen("data/files/hello", "r");
  ASSERT_NE(fp, nullptr);

  string line;
  ASSERT_TRUE(computational_advertising::read_line(fp, &line));
  ASSERT_STREQ(line.c_str(), "Hello world!");

  ASSERT_EQ(fclose(fp), 0);
}

TEST(UTIL_IO, READ_FILE) {
  FILE *fp = fopen("data/files/hello", "r");
  ASSERT_NE(fp, nullptr);

  string file;
  ASSERT_TRUE(computational_advertising::read_file(fp, &file));
  ASSERT_STREQ(file.c_str(), "Hello world!\nAnd c++!\nbazel!\n");

  ASSERT_EQ(fclose(fp), 0);
}

TEST(UTIL_COMM, EXECUTE_VFORK) {
  ASSERT_TRUE(computational_advertising::execute_vfork("echo \"hello world\""));

  string out;
  ASSERT_TRUE(computational_advertising::execute_vfork("echo \"hello world\"", &out));
  ASSERT_STREQ(out.c_str(), "hello world\n");
}

TEST(UTIL_COMM, EXECUTE_SPAWN) {
  ASSERT_TRUE(computational_advertising::execute("echo \"hello world\""));

  string out;
  ASSERT_TRUE(computational_advertising::execute("echo \"hello world\"", &out));
  ASSERT_STREQ(out.c_str(), "hello world\n");
}

TEST(UTIL_PROCESS, STATUS) {
  computational_advertising::ProcessStatus process_status;
  absl::SleepFor(absl::Seconds(10));
}

TEST(UTIL_FUNCTIONAL, SEMAPHORE) {
  Semaphore s(5);
  s.wait();
  s.post();
}

int32_t main(int32_t argc, char *argv[]) {
  computational_advertising::init(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
