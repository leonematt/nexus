#include <dirent.h>
#include <nexus/device_db.h>
#include <nexus/log.h>
#include <nexus/utility.h>

#include <fstream>
#include <iostream>
#include <sstream>

using namespace nexus;

#define NEXUS_LOG_MODULE "device_db"

std::vector<std::string> splitPaths(const std::string &paths, char delimiter) {
  std::vector<std::string> result;
  std::stringstream ss(paths);
  std::string path;
  while (std::getline(ss, path, delimiter)) {
    result.push_back(path);
  }
  return result;
}

static bool initDevices(DeviceMap &devs) {
  iterateEnvPaths("NEXUS_DEVICE_PATH", "../device_lib",
                  [&](const std::string &path, const std::string &name) {
                    NEXUS_LOG(NEXUS_STATUS_NOTE, "  File: " << name);
                    std::string::size_type const p(name.find_last_of('.'));
                    std::string basename = name.substr(0, p);
                    devs.emplace(basename, path);
                  });
  return true;
}

const DeviceMap *nexus::getDeviceDB() {
  static DeviceMap s_devices;
  static bool init = initDevices(s_devices);
  return &s_devices;
}

Properties nexus::lookupDevice(const std::string &archName) {
  const DeviceMap *devmap = getDeviceDB();
  auto ii = devmap->find(archName);
  if (ii != devmap->end()) return ii->second;
  return Properties();
}
