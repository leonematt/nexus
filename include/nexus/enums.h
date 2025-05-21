#ifndef NEXUS_ENUMS_H
#define NEXUS_ENUMS_H

#include <string>

namespace nexus {

enum class DeviceProp {
    NAME,
    ARCHITECTURE,
    TARGET,
    TRIPLE,

    CORE_HIERARCHY,
    CORE_COUNT,
    SIMD_COUNT_PER_CORE,
    SIMD_SIZE,

    MEMORY_HIERARCHY,
    MEMORY_TYPE,
    MEMORY_BUS_WIDTH,
    MEMORY_BANDWIDTH,
    
    L2_CACHE_SIZE,
    VECTOR_REGISTER_FILE_SIZE,
    SCALAR_REGISTER_FILE_SIZE,
    REGISTER_WIDTH,
    SHARED_MEMORY_SIZE,
    L1_DATA_CACHE_SIZE,
    INSTRUCTION_CACHE_SIZE,
    BASE_CLOCK,
    BOOST_CLOCK,
    VRAM,
    PEAK_POWER,
    INSTRUCTION_ISSUE_LATENCY,
    DATA_TYPES,
};

std::string getPropName(DeviceProp);

DeviceProp getProp(std::string propName);

} // namespace nexus

#endif // NEXUS_ENUMS_H