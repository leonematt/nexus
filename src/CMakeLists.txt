project(nexus-test)

add_executable(nexus-test main.cpp database.cpp)
target_link_libraries(nexus-test PRIVATE nlohmann_json::nlohmann_json)

