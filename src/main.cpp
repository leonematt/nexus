#include <nexus.h>

#include <iostream>
#include <optional>

std::vector<std::string_view> nexusArgs;

int main() {

  auto dev = nexus::lookupDevice("amd-gpu-gfx942");
  if (dev) {
    auto pval = dev->getProperty<std::string>("name");
    if (pval)
      std::cout << "PROP: " << *pval << std::endl;
    else
      std::cout << "PROP: NOT FOUND" << std::endl;
  }
  return 0;
}
