project(nexus-test)

add_executable(nexus-test 
    main.cpp
)

target_link_libraries(nexus-test PRIVATE nexus-api)
