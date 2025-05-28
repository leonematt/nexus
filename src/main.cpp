#include <nexus.h>

#include <iostream>

std::vector<std::string_view> nexusArgs;

int main() {

  auto sys = nexus::getSystem();
  auto rt = sys.getRuntime(1);

  std::cout << "RUNTIME: " << rt->getProperty<std::string>(NP_Name) << " - " << rt->getDeviceCount() << std::endl;

  for (int i = 0; i < rt->getDeviceCount(); ++i) {
    std::cout << "  Device: " << rt->getProperty<std::string>(i, NP_Name) << " - " << rt->getProperty<std::string>(i, NP_Architecture) << std::endl;
  }
  auto *dev0 = rt->getDevice(0);
  auto bufId = dev0->createBuffer(100);
  std::cout << "    Buffer: " << bufId << std::endl;

  auto queId = dev0->createCommandList();
  std::cout << "    CList: " << queId << std::endl;

  auto dev = nexus::lookupDevice("amd-gpu-gfx942");
  if (dev) {
    {
      const char *key = "name";
      auto pval = dev->getProperty<std::string>(key);
      std::cout << "PROP(" << key << "): ";
      if (pval)
        std::cout << *pval;
      else
        std::cout << "NOT FOUND";
      std::cout << std::endl;
    }
    {
      std::vector<std::string> prop_path = {"coreSubsystem", "maxPerUnit"};
      auto pval = dev->getProperty<int64_t>(prop_path);

      // make slash path
      std::string path = std::accumulate(std::begin(prop_path), std::end(prop_path), std::string(),
                                [](std::string &ss, std::string &s)
                                {
                                    return ss.empty() ? s : ss + "/" + s;
                                });
      std::cout << "PROP(" << path << "): ";
      if (pval)
        std::cout << *pval;
      else
        std::cout << "NOT FOUND";
      std::cout << std::endl;
    }

    std::vector<NXSAPI_PropertyEnum> prop_epath = {NP_CoreSubsystem, NP_Count};
    auto eval = dev->getProperty<int64_t>(prop_epath);
  }
  return 0;
}
