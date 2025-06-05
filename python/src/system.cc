#include "pynexus.h"
#include <iostream>

#include <nexus.h>
#include <nexus-api.h>

namespace py = pybind11;

using namespace nexus;

void pynexus::init_system_bindings(py::module &&m) {
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    py::class_<Device>(m, "device", py::module_local())
        .def("get_property_str", 
            [](Device &self, const std::string &name) {
                std::string pname = "NP_";
                pname += name;
                auto prop = nxsGetPropEnum(pname.c_str());
                return self.getProperty<std::string>(prop);
            });

    py::class_<Runtime>(m, "runtime", py::module_local())
        .def("device_count",
            [](Runtime &self) { return self.getDeviceCount(); })
        .def("get_device",
            [](Runtime &self, nxs_int id) { return self.getDevice(id); })
        .def("get_property_str",
            [](Runtime &self, const std::string &name) {
                std::string pname = "NP_";
                pname += name;
                auto prop = nxsGetPropEnum(pname.c_str());
                return self.getProperty<std::string>(prop);
            })
        .def("get_property_int",
            [](Runtime &self, const std::string &name) {
                auto prop = nxsGetPropEnum(name.c_str());
                return self.getProperty<int64_t>(prop);
            });
    
  
    // query
    m.def("runtime_count", []() { return nexus::getSystem().getRuntimeCount(); });

    m.def("get_runtime", [](int idx) { return nexus::getSystem().getRuntime(idx); });

}

// Function to get a value from the dictionary
py::object pynexus::create_buffer(nxs_int size, const std::string& key) {
    return py::none();  // Return None if key is not found
}

