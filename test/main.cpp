#include <nexus.h>

#include <iostream>
#include <fstream>
#include <numeric>

std::vector<std::string_view> nexusArgs;

int main() {

  auto sys = nexus::getSystem();
  auto rt = sys.getRuntime(1);

  auto count = rt.getDeviceCount();

  std::cout << "RUNTIME: " << rt.getProperty<std::string>(NP_Name) << " - " << count << std::endl;

  for (int i = 0; i < count; ++i) {
    std::cout << "  Device: " << rt.getProperty<std::string>(i, NP_Name) << " - " << rt.getProperty<std::string>(i, NP_Architecture) << std::endl;
  }
  std::vector<char> data(1024, 1);

  auto dev0 = rt.getDevice(0);

  std::ifstream f("kernel.so", std::ios::binary);
  std::vector<char> soData;
  soData.insert(soData.begin(), std::istream_iterator<char>(f), std::istream_iterator<char>());
  
  auto nlib = dev0.createLibrary("kernel.so");

  auto kern = nlib.getKernel("add_vectors");
  std::cout << "   Kernel: " << kern.getId() << std::endl;

  auto buf = sys.createBuffer(data.size(), data.data());

  auto cpv = sys.copyBuffer(buf, dev0);
  std::cout << "    CopyBuffer: " << cpv << std::endl;

  auto sched = dev0.createSchedule();
  std::cout << "    CList: " << sched.getId() << std::endl;

  auto cmd = sched.createCommand(kern);

  return 0;
}
