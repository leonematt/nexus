/*
 */

#if defined(NEXUS_API_GENERATE_FUNC_DECL)
/************************************************************************
 * Generate the Function declarations
 ***********************************************************************/
/* Generate the Function extern */
#define NEXUS_API_FUNC(RETURN_TYPE, NAME, ...) \
    extern NXS_API_EXTERN_C NXS_API_ENTRY RETURN_TYPE NXS_API_CALL nxs##NAME(__VA_ARGS__);

#else
#if defined(NEXUS_API_GENERATE_FUNC_ENUM)
/************************************************************************
 * Generate the Function Enum
 ***********************************************************************/
/* Generate the Enum name */
#define NEXUS_API_FUNC(RETURN_TYPE, NAME, ...) \
        NF_nxs##NAME,
    
/* Declare the Enumeration */
enum _nxs_function {

#else
#if defined(NEXUS_API_GENERATE_FUNC_TYPE)
/************************************************************************
 * Generate the Function typedefs
 ***********************************************************************/
 /* Generate enum lookup of Function type */
#ifdef __cplusplus
template <nxs_function Tfn>
struct nxsFunctionType { typedef void *type; };
#endif


 /* Generate the Function typedefs */
#define _NEXUS_API_FUNC(RETURN_TYPE, NAME, ...) \
    typedef RETURN_TYPE NXS_API_CALL NXS_CONCAT(nxs##NAME, _t)(__VA_ARGS__); \
    typedef NXS_CONCAT(nxs##NAME, _t) * NXS_CONCAT(nxs##NAME, _fn);

#ifdef __cplusplus
#define NEXUS_API_FUNC(RETURN_TYPE, NAME, ...) \
    _NEXUS_API_FUNC(RETURN_TYPE, NAME, __VA_ARGS__) \
    template <> struct nxsFunctionType<NF_nxs##NAME> { typedef NXS_CONCAT(nxs##NAME, _fn) type; };
#else
#define NEXUS_API_FUNC(RETURN_TYPE, NAME, ...) \
    _NEXUS_API_FUNC(RETURN_TYPE, NAME, __VA_ARGS__)
#endif

#endif
#endif
#endif


/************************************************************************
 * Define API Functions
 ***********************************************************************/

/************************************************************************
 * @def GetRuntimeProperty
 * @brief Lookup 
 ***********************************************************************/
NEXUS_API_FUNC(nxs_status, GetRuntimeProperty,
    nxs_uint runtime_property_id,
    void *property_value,
    size_t* property_value_size
)

/************************************************************************
 * @def GetRuntimeProperty
 * @brief Lookup 
 ***********************************************************************/
NEXUS_API_FUNC(nxs_int, GetDeviceCount)

/************************************************************************
 * @def GetDeviceProperty
 * @brief Lookup 
 ***********************************************************************/
NEXUS_API_FUNC(nxs_status, GetDeviceProperty,
    nxs_int device_id,
    nxs_uint property_id,
    void *property_value,
    size_t* property_value_size
)

/************************************************************************
 * @def GetDevicePropertyFromPath
 * @brief Lookup 
 ***********************************************************************/
NEXUS_API_FUNC(nxs_status, GetDevicePropertyFromPath,
    nxs_int device_id,
    nxs_uint property_path_count,
    nxs_uint *property_path_ids,
    void *property_value,
    size_t* property_value_size
)

/************************************************************************
 * @def CreateBuffer
 * @brief Create buffer on the device
  * @return Negative value is an error status.
  *         Non-negative is the bufferId.
***********************************************************************/
NEXUS_API_FUNC(nxs_int, CreateBuffer,
    nxs_int device_id,
    size_t size,
    nxs_mem_flags flags,
    void* host_ptr
)
/************************************************************************
 * @def CreateBuffer
 * @brief Create buffer on the device
  * @return Negative value is an error status.
  *         Non-negative is the bufferId.
***********************************************************************/
NEXUS_API_FUNC(nxs_status, CopyBuffer,
    nxs_int buffer_id,
    void* host_ptr
)
/************************************************************************
 * @def ReleaseBuffer
 * @brief Release the buffer on the device
  * @return Error status or Succes.
***********************************************************************/
NEXUS_API_FUNC(nxs_status, ReleaseBuffer,
    nxs_int buffer_id
)


/************************************************************************
 * @def CreateLibrary
 * @brief Create command buffer on the device
 * @return Negative value is an error status.
 *         Non-negative is the bufferId.
 ***********************************************************************/
NEXUS_API_FUNC(nxs_int, CreateLibrary,
    nxs_int device_id,
    void *library_data,
    nxs_uint data_size
)
/************************************************************************
 * @def CreateLibrary
 * @brief Create command buffer on the device
 * @return Negative value is an error status.
 *         Non-negative is the bufferId.
 ***********************************************************************/
NEXUS_API_FUNC(nxs_int, CreateLibraryFromFile,
    nxs_int device_id,
    const char *library_data
)
/************************************************************************
 * @def ReleaseLibrary
 * @brief Release the buffer on the device
  * @return Error status or Succes.
***********************************************************************/
NEXUS_API_FUNC(nxs_status, ReleaseLibrary,
    nxs_int library_id
)

/************************************************************************
 * @def CreateLibrary
 * @brief Create command buffer on the device
 * @return Negative value is an error status.
 *         Non-negative is the bufferId.
 ***********************************************************************/
NEXUS_API_FUNC(nxs_int, GetKernel,
    nxs_int library_id,
    const char *kernel_name
)

/************************************************************************
 * @def CreateCommandBuffer
 * @brief Create command buffer on the device
 * @return Negative value is an error status.
 *         Non-negative is the bufferId.
 ***********************************************************************/
NEXUS_API_FUNC(nxs_int, CreateSchedule,
    nxs_int device_id,
    nxs_command_queue_properties properties
)
/************************************************************************
 * @def ReleaseCommandList
 * @brief Release the buffer on the device
  * @return Error status or Succes.
***********************************************************************/
NEXUS_API_FUNC(nxs_status, RunSchedule,
    nxs_int schedule_id,
    nxs_bool blocking
)
/************************************************************************
 * @def ReleaseCommandList
 * @brief Release the buffer on the device
  * @return Error status or Succes.
***********************************************************************/
NEXUS_API_FUNC(nxs_status, ReleaseSchedule,
    nxs_int schedule_id
)

/************************************************************************
 * @def CreateCommand
 * @brief Create command buffer on the device
 * @return Negative value is an error status.
 *         Non-negative is the bufferId.
 ***********************************************************************/
NEXUS_API_FUNC(nxs_int, CreateCommand,
    nxs_int schedule_id,
    nxs_int kernel_id
)
/************************************************************************
 * @def ReleaseCommandList
 * @brief Release the buffer on the device
  * @return Error status or Succes.
***********************************************************************/
NEXUS_API_FUNC(nxs_status, ReleaseCommand,
    nxs_int command_id
)

/************************************************************************
 * @def CreateCommand
 * @brief Create command buffer on the device
 * @return Negative value is an error status.
 *         Non-negative is the bufferId.
 ***********************************************************************/
NEXUS_API_FUNC(nxs_status, SetCommandArgument,
    nxs_int command_id,
    nxs_int argument_index,
    nxs_int buffer_id
)
/************************************************************************
 * @def CreateCommand
 * @brief Create command buffer on the device
 * @return Negative value is an error status.
 *         Non-negative is the bufferId.
 ***********************************************************************/
NEXUS_API_FUNC(nxs_status, FinalizeCommand,
    nxs_int command_id,
    nxs_int group_size,
    nxs_int grid_size
)



#ifdef NEXUS_API_GENERATE_FUNC_ENUM
    NXS_FUNCTION_CNT,
    NXS_FUNCTION_PREFIX_LEN = 3
}; /* close _nxs_function */

typedef enum _nxs_function nxs_function;

const char *nxsGetFuncName(nxs_function funcEnum);
nxs_function nxsGetFuncEnum(const char *funcName);

#endif


#undef NEXUS_API_GENERATE_FUNC_DECL
#undef NEXUS_API_GENERATE_FUNC_ENUM
#undef NEXUS_API_GENERATE_FUNC_TYPE

#undef NEXUS_API_FUNC
