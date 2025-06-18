#include "pynexus.h"
#include <iostream>

#include <pybind11/stl.h>

#include <nexus.h>
#include <nexus-api.h>

namespace py = pybind11;

using namespace nexus;


struct DevPtr {
    void *ptr;
    size_t size;
};

static DevPtr getPointer(PyObject *obj) {
    DevPtr result = { nullptr, 0 };
    if (obj == Py_None) {
        return result;
    }
    PyObject *data_ptr_m = PyObject_GetAttrString(obj, "data_ptr");
    PyObject *nbytes_ret = PyObject_GetAttrString(obj, "nbytes");
    if (data_ptr_m && nbytes_ret) {
        PyObject *empty_tuple = PyTuple_New(0);
        PyObject *data_ret = PyObject_Call(data_ptr_m, empty_tuple, NULL);
        Py_DECREF(empty_tuple);
        Py_DECREF(data_ptr_m);
        if (!PyLong_Check(data_ret)) {
            PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
            return result;
        }
        result.ptr = (void *)PyLong_AsUnsignedLongLong(data_ret);
        result.size = PyLong_AsUnsignedLongLong(nbytes_ret);
    }
    return result;
}

#if 0
static DevPtr getPointer(py::object obj) {
    DevPtr result = { nullptr, 0 };
    if (obj.is_none()) {
        return result;
    }
    py::object data_ptr_m = obj.attr("data_ptr");
    py::object nbytes_ret = obj.attr("nbytes");
    if (!data_ptr_m.is_none() && !nbytes_res.is_none()) {
        py::object data_ret = data_ptr_m();
        if (!PyLong_Check(data_ret)) {
            PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
            return result;
        }
        result.ptr = (void *)data_ret.cast<int64_t>();
        result.size = PyLong_AsUnsignedLongLong(nbytes_ret);
    }
    return result;
}
#endif

void pynexus::init_system_bindings(py::module &m) {
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    auto mstatus = m.def_submodule("status");

    auto statusEnum = py::enum_<nxs_status>(mstatus, "nxs_status", py::module_local());
    for (nxs_int i = NXS_STATUS_MIN; i <= NXS_STATUS_MAX; ++i) {
        nxs_status status = (nxs_status)i;
        const char *str = nxsGetStatusName(i);
        if (str && *str)
            statusEnum.value(str, status);
    }
    statusEnum.export_values();

    auto mprop = m.def_submodule("property");
    auto propEnum = py::enum_<nxs_property>(mprop, "nxs_property", py::module_local());
    for (nxs_int i = 0; i <= NXS_PROPERTY_CNT; ++i) {
        nxs_property prop = (nxs_property)i;
        const char *str = nxsGetPropName(i);
        if (str && *str)
            propEnum.value(str, prop);
    }
    propEnum.export_values();

    // generate enums for Nexus properties and status

    py::class_<Properties>(m, "_properties", py::module_local())
        .def("__bool__", 
            [](Properties &self) {
                return (bool)self;
            })
        .def("get_property_str", 
            [](Properties &self, const std::string &name) {
                if (auto pval = self.getProperty(name))
                    return nexus::getPropertyValue<std::string>(*pval);
                return std::string();
            })
        .def("get_property_str", 
            [](Properties &self, nxs_property prop) {
                return self.getProp<std::string>(prop);
            })
        .def("get_property_int",
            [](Properties &self, const std::string &name) {
                if (auto pval = self.getProperty(name))
                    return nexus::getPropertyValue<nxs_long>(*pval);
                return (nxs_long)NXS_InvalidDevice;
            })
        .def("get_property_int",
            [](Properties &self, nxs_property prop) {
                return self.getProp<nxs_long>(prop);
            })
        .def("get_property_str", 
            [](Properties &self, const std::vector<std::string> &path) {
                if (auto pval = self.getProperty(path))
                    return nexus::getPropertyValue<std::string>(*pval);
                return std::string();
            })
        .def("get_property_str", 
            [](Properties &self, const std::vector<nxs_int> &path) {
                if (auto pval = self.getProperty(path))
                    return nexus::getPropertyValue<std::string>(*pval);
                return std::string();
            })
        .def("get_property_int", 
            [](Properties &self, const std::vector<std::string> &path) {
                if (auto pval = self.getProperty(path))
                    return nexus::getPropertyValue<nxs_long>(*pval);
                return (nxs_long)NXS_InvalidDevice;
            })
        .def("get_property_int", 
            [](Properties &self, const std::vector<nxs_int> &path) {
                if (auto pval = self.getProperty(path))
                    return nexus::getPropertyValue<nxs_long>(*pval);
                return (nxs_long)NXS_InvalidDevice;
            });
    
    py::class_<Buffer>(m, "_buffer", py::module_local())
        .def("__bool__", 
            [](Buffer &self) {
                return (bool)self;
            })
        .def("get_property_str", 
            [](Buffer &self, const std::string &name) {
                //auto prop = nxsGetPropEnum(name.c_str());
                //return self.getProperty<std::string>(prop);
                return std::string();
            })
        .def("copy", 
            [](Buffer &self, py::object tensor) {
                auto local = self.getLocal();
                auto devp = getPointer(tensor.ptr());
                if (devp.ptr != nullptr && local.getHostData() != nullptr && devp.size == self.getSize()) {
                    return local.copy(devp.ptr);
                }
                return NXS_InvalidDevice;
            });

    py::class_<Kernel>(m, "_kernel", py::module_local())
        .def("__bool__", 
            [](Kernel &self) {
                return (bool)self;
            })
        .def("get_property_str", 
            [](Kernel &self, const std::string &name) {
                //auto prop = nxsGetPropEnum(name.c_str());
                //return self.getProperty<std::string>(prop);
                return std::string();
            });
    
    py::class_<Library>(m, "_library", py::module_local())
        .def("__bool__", 
            [](Library &self) {
                return (bool)self;
            })
        .def("get_property_str", 
            [](Library &self, const std::string &name) {
                //auto prop = nxsGetPropEnum(name.c_str());
                //return self.getProperty<std::string>(prop);
                return std::string();
            })
        .def("get_kernel", 
            [](Library &self, const std::string &name) {
                return self.getKernel(name);
            });

    py::class_<Command>(m, "_command", py::module_local())
        .def("__bool__", 
            [](Command &self) {
                return (bool)self;
            })
        .def("get_property_str", 
            [](Command &self, const std::string &name) {
                //auto prop = nxsGetPropEnum(name.c_str());
                //return self.getProperty<std::string>(prop);
                return std::string();
            })
        .def("set_buffer", 
            [](Command &self, int index, Buffer buf) {
                return self.setArgument(index, buf);
            })
        .def("finalize", 
            [](Command &self, int groupSize, int gridSize) {
                return self.finalize(groupSize, gridSize);
            });
        

    py::class_<Schedule>(m, "_schedule", py::module_local())
        .def("__bool__", 
            [](Schedule &self) {
                return (bool)self;
            })
        .def("get_property_str", 
            [](Schedule &self, const std::string &name) {
                //auto prop = nxsGetPropEnum(name.c_str());
                //return self.getProperty<std::string>(prop);
                return std::string();
            })
        .def("create_command", 
            [](Schedule &self, Kernel kernel) {
                return self.createCommand(kernel);
            })
        .def("run", 
            [](Schedule &self) {
                return self.run();
            });
    
    
    py::class_<Device>(m, "_device", py::module_local())
        .def("__bool__", 
            [](Device &self) {
                return (bool)self;
            })
        .def("get_property_str",
            [](Device &self, const std::string &name) {
                auto prop = nxsGetPropEnum(name.c_str());
                return self.getProp<std::string>(prop);
            })
        .def("get_device_info",
            [](Device &self) {
                return self.getProperties();
            })
        .def("create_buffer", 
            [](Device &self, py::object tensor) {
                auto devp = getPointer(tensor.ptr());
                return self.createBuffer(devp.size, devp.ptr);
            })
        .def("create_buffer", 
            [](Device &self, size_t size) {
                return self.createBuffer(size);
            })
        .def("copy_buffer", 
            [](Device &self, Buffer buf) {
                return self.copyBuffer(buf);
            })
        .def("load_library", 
            [](Device &self, const std::string &filepath) {
                return self.createLibrary(filepath);
            })
        .def("create_schedule", 
            [](Device &self) {
                return self.createSchedule();
            });
    
    
    py::class_<Devices>(m, "_devices", py::module_local())
        .def("__iter__",
            [](const Devices &rts) {
                return py::make_iterator(rts.begin(), rts.end());
            }, py::keep_alive<0, 1>() /* Essential: keep object alive */)
        .def("size", 
            [](Devices &self) {
                return self.size();
            })
        .def("__getitem__", 
            [](Devices &self, int idx) {
                return self.get(idx);
            });
        
    
    py::class_<Runtime>(m, "_runtime", py::module_local())
        .def("__bool__", 
            [](Runtime &self) {
                return (bool)self;
            })
        .def("get_device",
            [](Runtime &self, nxs_int id) { return self.getDevice(id); })
        .def("get_devices",
            [](Runtime &self) { return self.getDevices(); })
        .def("get_property_str",
            [](Runtime &self, const std::string &name) {
                auto prop = nxsGetPropEnum(name.c_str());
                return self.getProp<std::string>(prop);
            })
        .def("get_property_int",
            [](Runtime &self, const std::string &name) {
                auto prop = nxsGetPropEnum(name.c_str());
                return self.getProp<nxs_long>(prop);
            })
        .def("get_property_str",
            [](Runtime &self, nxs_property prop) {
                return self.getProp<std::string>(prop);
            })
        .def("get_property_int",
            [](Runtime &self, nxs_property prop) {
                return self.getProp<nxs_long>(prop);
            });

    py::class_<Runtimes>(m, "_runtimes", py::module_local())
        .def("__iter__",
            [](const Runtimes &rts) {
                return py::make_iterator(rts.begin(), rts.end());
            }, py::keep_alive<0, 1>() /* Essential: keep object alive */)
        .def("size", 
            [](Runtimes &self) {
                return self.size();
            })
        .def("__getitem__", 
            [](Runtimes &self, int idx) {
                return self.get(idx);
            });
  
    // query
    m.def("get_runtimes", []() { return nexus::getSystem().getRuntimes(); });

    m.def("create_buffer", [](size_t size) { return nexus::getSystem().createBuffer(size); });
    m.def("create_buffer", [](py::object tens) {
        auto devp = getPointer(tens.ptr());
        return nexus::getSystem().createBuffer(devp.size, devp.ptr);
    });

    m.def("get_chip_info", []() {
        return *nexus::getDeviceDB();
    });
    m.def("lookup_chip_info", [](const std::string &name) {
        return nexus::lookupDevice(name);
    });

}
