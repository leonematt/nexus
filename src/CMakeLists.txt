project(nexus-test)

add_executable(nexus-test main.cpp device.cpp device_db.cpp utility.cpp system.cpp runtime.cpp)
target_link_libraries(nexus-test PRIVATE nlohmann_json::nlohmann_json)

