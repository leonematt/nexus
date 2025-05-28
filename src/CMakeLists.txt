project(nexus-test)

set(CMAKE_CXX_VISIBILITY_PRESET default)

add_library(nexus-api SHARED
    properties.cpp
    device_db.cpp
    utility.cpp
    system.cpp
    runtime.cpp
    ${CMAKE_SOURCE_DIR}/include/nexus-api/src/nxs_propertys.cpp
    ${CMAKE_SOURCE_DIR}/include/nexus-api/src/nxs_functions.cpp
)

target_link_libraries(nexus-api PRIVATE nlohmann_json::nlohmann_json)

add_executable(nexus-test 
    main.cpp
)

target_link_libraries(nexus-test PRIVATE nexus-api nlohmann_json::nlohmann_json)
