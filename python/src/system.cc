#include "pynexus.h"
#include <iostream>

#include <nexus.h>
#include <nexus-api.h>

namespace py = pybind11;

using namespace nexus;

void pynexus::init_system_bindings(py::module &&m) {
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    // generate enums for Nexus properties and status

    py::class_<Buffer>(m, "buffer", py::module_local())
        .def("get_property_str", 
            [](Buffer &self, const std::string &name) {
                //auto prop = nxsGetPropEnum(name.c_str());
                //return self.getProperty<std::string>(prop);
                return "";
            });

    py::class_<Kernel>(m, "kernel", py::module_local())
        .def("get_property_str", 
            [](Kernel &self, const std::string &name) {
                //auto prop = nxsGetPropEnum(name.c_str());
                //return self.getProperty<std::string>(prop);
                return "";
            });
    
    py::class_<Library>(m, "library", py::module_local())
        .def("get_property_str", 
            [](Library &self, const std::string &name) {
                //auto prop = nxsGetPropEnum(name.c_str());
                //return self.getProperty<std::string>(prop);
                return "";
            })
        .def("get_kernel", 
            [](Library &self, const std::string &name) {
                return self.getKernel(name);
            });

    py::class_<Command>(m, "command", py::module_local())
        .def("get_property_str", 
            [](Command &self, const std::string &name) {
                //auto prop = nxsGetPropEnum(name.c_str());
                //return self.getProperty<std::string>(prop);
                return "";
            })
        .def("set_buffer", 
            [](Command &self, int index, Buffer buf) {
                return self.setArgument(index, buf);
            })
        .def("finalize", 
            [](Command &self, int groupSize, int gridSize) {
                return self.finalize(groupSize, gridSize);
            });
        

    py::class_<Schedule>(m, "schedule", py::module_local())
        .def("get_property_str", 
            [](Schedule &self, const std::string &name) {
                //auto prop = nxsGetPropEnum(name.c_str());
                //return self.getProperty<std::string>(prop);
                return "";
            })
        .def("create_command", 
            [](Schedule &self, Kernel kernel) {
                return self.createCommand(kernel);
            });

    
    py::class_<Device>(m, "device", py::module_local())
        .def("get_property_str", 
            [](Device &self, const std::string &name) {
                auto prop = nxsGetPropEnum(name.c_str());
                return self.getProperty<std::string>(prop);
            })
        .def("create_library", 
            [](Device &self, const std::string &filepath) {
                return self.createLibrary(filepath);
            })
        .def("create_schedule", 
            [](Device &self) {
                return self.createSchedule();
            });
    
    
    py::class_<Devices>(m, "devices", py::module_local())
        .def("size", 
            [](Devices &self) {
                return self.size();
            })
        .def("get", 
            [](Devices &self, int idx) {
                return self.get(idx);
            });
        
    
    py::class_<Runtime>(m, "runtime", py::module_local())
        .def("get_device",
            [](Runtime &self, nxs_int id) { return self.getDevice(id); })
        .def("get_devices",
            [](Runtime &self) { return self.getDevices(); })
        .def("get_property_str",
            [](Runtime &self, const std::string &name) {
                auto prop = nxsGetPropEnum(name.c_str());
                return self.getProperty<std::string>(prop);
            })
        .def("get_property_int",
            [](Runtime &self, const std::string &name) {
                auto prop = nxsGetPropEnum(name.c_str());
                return self.getProperty<int64_t>(prop);
            });

    py::class_<Runtimes>(m, "runtimes", py::module_local())
        .def("size", 
            [](Runtimes &self) {
                return self.size();
            })
        .def("get", 
            [](Runtimes &self, int idx) {
                return self.get(idx);
            });
  
    // query
    m.def("get_runtimes", []() { return nexus::getSystem().getRuntimes(); });

    m.def("create_buffer", [](size_t size) { return nexus::getSystem().createBuffer(size); });
}

// Function to get a value from the dictionary
py::object pynexus::create_buffer(nxs_int size, const std::string& key) {
    return py::none();  // Return None if key is not found
}

