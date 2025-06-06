#include <nexus.h>

#include <iostream>
#include <fstream>
#include <numeric>

std::vector<std::string_view> nexusArgs;

int main() {

  auto sys = nexus::getSystem();
  auto rt = sys.getRuntime(1);

  auto count = rt.getDevices().size();

  std::cout << "RUNTIME: " << rt.getProperty<std::string>(NP_Name) << " - " << count << std::endl;

  for (int i = 0; i < count; ++i) {
    auto dev = rt.getDevice(i);
    std::cout << "  Device: " << dev.getProperty<std::string>(NP_Name) << " - " << dev.getProperty<std::string>(NP_Architecture) << std::endl;
  }
  std::vector<char> data(1024, 1);
  std::vector<float> vecA(1024, 1.0);
  std::vector<float> vecB(1024, 2.0);
  std::vector<float> vecResult_GPU(1024, 0.0); // For GPU result

  size_t size = 1024 * sizeof(float);

  auto dev0 = rt.getDevice(0);

  // std::ifstream f("kernel.so", std::ios::binary);
  // std::vector<char> soData;
  // soData.insert(soData.begin(), std::istream_iterator<char>(f), std::istream_iterator<char>());
  
  auto nlib = dev0.createLibrary("kernel.so");

  auto kern = nlib.getKernel("add_vectors");
  std::cout << "   Kernel: " << kern.getId() << std::endl;

  auto buf0 = sys.createBuffer(size, vecA.data());
  auto buf1 = sys.createBuffer(size, vecB.data());
  auto buf2 = sys.createBuffer(size, vecResult_GPU.data());

  auto cpv0 = sys.copyBuffer(buf0, dev0);
  auto cpv1 = sys.copyBuffer(buf1, dev0);
  auto cpv2 = sys.copyBuffer(buf2, dev0);
  //std::cout << "    CopyBuffer: " << cpv0 << std::endl;

  auto sched = dev0.createSchedule();
  std::cout << "    CList: " << sched.getId() << std::endl;

  auto cmd = sched.createCommand(kern);
  cmd.setArgument(0, cpv0);
  cmd.setArgument(1, cpv1);
  cmd.setArgument(2, cpv2);

  cmd.finalize(32, 1024);

  sched.run();

  cpv2.copy(vecResult_GPU.data());

  return 0;
}
