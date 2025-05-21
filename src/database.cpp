#include <dirent.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include <nexus/database.h>
#include <nexus/log.h>

using namespace nexus;

#define NEXUS_LOG_MODULE "database"

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
    NEXUS_LOG(NEXUS_STATUS_WARN, "NEXUS_DEVICE_PATH environment variable is not set.");
    env = "../device_lib";
  }

  std::vector<std::string> directories = splitPaths(env, ':');
  for (const auto& directory : directories) {
    DIR* dir = opendir(directory.c_str());
    if (!dir) {
      NEXUS_LOG(NEXUS_STATUS_WARN, "Failed to open directory: " << directory);
      continue;
    }

    struct dirent* entry;
    NEXUS_LOG(NEXUS_STATUS_NOTE, "Reading directory: " << directory);
    while ((entry = readdir(dir)) != nullptr) {
      if (entry->d_type == DT_REG) {
        std::string filename(entry->d_name);
        NEXUS_LOG(NEXUS_STATUS_NOTE, "  Reading file: " << filename);
        std::string filepath = directory + '/' + filename;
        std::string::size_type const p(filename.find_last_of('.'));
        std::string basename = filename.substr(0, p);

        devs.emplace(basename, filepath);
      }
    }
    closedir(dir);
  }
  return true;
}

std::optional<Device> nexus::lookupDevice(const std::string &archName) {
  static DeviceMap s_devices;
  static bool init = initDevices(s_devices);
  auto ii = s_devices.find(archName);
  if (ii != s_devices.end())
    return ii->second;
  return std::nullopt;
}

