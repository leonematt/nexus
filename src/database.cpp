#include <dirent.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include <nexus.h>

using namespace nexus;

DevicePropMap::DevicePropMap(const char *filename) {
  // Load json from file
  // TODO: defer until requested
  // TODO: catch errors
  std::ifstream f(filename);
  propertyMap = json::parse(f);
  std::cout << "DevicePropMap: " << filename << " - " << propertyMap.size() << std::endl;
}

#if 0
Prop DevicePropMap::getProperty(const std::string &name) const {
  auto ii = propertyMap.find(name);
  if (ii != propertyMap.end())
    return ii->second;
  return Prop();
}
#endif

std::vector<std::string> splitPaths(const std::string& paths, char delimiter) {
  std::vector<std::string> result;
  std::stringstream ss(paths);
  std::string path;
  while (std::getline(ss, path, delimiter)) {
      result.push_back(path);
  }
  return result;
}

static bool initDevices(DeviceMap &devs) {
  // Load from NEXUS_DEVICE_PATH
  const char* env = std::getenv("NEXUS_DEVICE_PATH");
  if (!env) {
      std::cerr << "NEXUS_DEVICE_PATH environment variable is not set." << std::endl;
      env = "../device_lib";
  }

  std::vector<std::string> directories = splitPaths(env, ':');
  for (const auto& directory : directories) {
    DIR* dir = opendir(directory.c_str());
    if (!dir) {
        std::cerr << "Failed to open directory: " << directory << std::endl;
        continue;
    }

    struct dirent* entry;
    std::cout << "Files in " << directory << ":" << std::endl;
    while ((entry = readdir(dir)) != nullptr) {
      if (entry->d_type == DT_REG) {
        std::string filename(entry->d_name);
        std::cout << "  " << filename << std::endl;
        std::string filepath = directory + '/' + filename;
        std::string::size_type const p(filename.find_last_of('.'));
        std::string basename = filename.substr(0, p);

        devs.emplace(basename, nexus::DevicePropMap(filepath.c_str()));
      }
    }
    closedir(dir);
  }
  return true;
}

std::optional<const DevicePropMap> nexus::lookupDevice(const char *archName) {
  static DeviceMap s_devices;
  static bool init = initDevices(s_devices);
  auto ii = s_devices.find(archName);
  if (ii != s_devices.end())
    return ii->second;
  return std::nullopt;
}

