#include <nexus.h>

#include <iostream>

std::vector<std::string_view> nexusArgs;

int main() {

  auto sys = nexus::getSystem();
  auto rt = sys.getRuntime(0);

  std::cout << "RUNTIME: " << rt->getName() << std::endl;

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
  }
  return 0;
}
