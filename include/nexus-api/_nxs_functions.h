/*
 */

#if defined(NEXUS_API_GENERATE_FUNC_DECL)
/************************************************************************
 * Generate the Function declarations
 ***********************************************************************/
// Generate the Function extern
#define NEXUS_API_FUNC(RETURN_TYPE, NAME, ...) \
    extern NXS_API_ENTRY RETURN_TYPE NXS_API_CALL nxs##NAME(__VA_ARGS__);

#else
#if defined(NEXUS_API_GENERATE_FUNC_ENUM)
/************************************************************************
 * Generate the Function Enum
 ***********************************************************************/
// Generate the Enum name
#define NEXUS_API_FUNC(RETURN_TYPE, NAME, ...) \
        FN_nxs##NAME,
    
// Declare the Enumeration
enum NXSAPI_FunctionEnum {

#else
#if defined(NEXUS_API_GENERATE_FUNC_TYPE)
/************************************************************************
 * Generate the Function typedefs
 ***********************************************************************/
        
// Generate the Function typedefs
#define NEXUS_API_FUNC(RETURN_TYPE, NAME, ...) \
    typedef RETURN_TYPE NXS_API_CALL NXS_CONCAT(nxs##NAME, _t)(__VA_ARGS__); \
    typedef NXS_CONCAT(nxs##NAME, _t) * NXS_CONCAT(nxs##NAME, _fn);

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
NEXUS_API_FUNC(nxs_int, GetRuntimeProperty,
    nxs_uint runtime_property_id,
    void *property_value,
    size_t* property_value_size
)

/************************************************************************
 * @def GetRuntimeProperty
 * @brief Lookup 
 ***********************************************************************/
NEXUS_API_FUNC(nxs_int, GetDeviceCount,
    nxs_uint* num_devices
)

/************************************************************************
 * @def GetDeviceProperty
 * @brief Lookup 
 ***********************************************************************/
NEXUS_API_FUNC(nxs_int, GetDeviceProperty,
    nxs_uint device_id,
    nxs_uint property_id,
    void *property_value,
    size_t* property_value_size
)

/************************************************************************
 * @def GetDevicePropertyFromPath
 * @brief Lookup 
 ***********************************************************************/
NEXUS_API_FUNC(nxs_int, GetDevicePropertyFromPath,
    nxs_uint device_id,
    nxs_uint property_path_count,
    nxs_uint *property_id,
    void *property_value,
    size_t* property_value_size
)

/************************************************************************
 * @def CreateBuffer
 * @brief Create buffer in the context
 ***********************************************************************/
NEXUS_API_FUNC(nxs_uint, CreateBuffer,
    nxs_uint device_id,
    size_t size,
    nxs_mem_flags flags,
    void* host_ptr,
    nxs_int* errcode_ret
)


#if 0

typedef nxs_int NXS_API_CALL nxsGetDeviceIDs_t(
    nxs_platform_id platform,
    nxs_device_type device_type,
    nxs_uint num_entries,
    nxs_device_id* devices,
    nxs_uint* num_devices);

typedef nxsGetDeviceIDs_t *
nxsGetDeviceIDs_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsGetDeviceInfo_t(
    nxs_device_id device,
    nxs_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

typedef nxsGetDeviceInfo_t *
nxsGetDeviceInfo_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_context NXS_API_CALL nxsCreateContext_t(
    const nxs_context_properties* properties,
    nxs_uint num_devices,
    const nxs_device_id* devices,
    void (NXS_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
    void* user_data,
    nxs_int* errcode_ret);

typedef nxsCreateContext_t *
nxsCreateContext_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_context NXS_API_CALL nxsCreateContextFromType_t(
    const nxs_context_properties* properties,
    nxs_device_type device_type,
    void (NXS_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
    void* user_data,
    nxs_int* errcode_ret);

typedef nxsCreateContextFromType_t *
nxsCreateContextFromType_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsRetainContext_t(
    nxs_context context);

typedef nxsRetainContext_t *
nxsRetainContext_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsReleaseContext_t(
    nxs_context context);

typedef nxsReleaseContext_t *
nxsReleaseContext_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsGetContextInfo_t(
    nxs_context context,
    nxs_context_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

typedef nxsGetContextInfo_t *
nxsGetContextInfo_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsRetainCommandQueue_t(
    nxs_command_queue command_queue);

typedef nxsRetainCommandQueue_t *
nxsRetainCommandQueue_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsReleaseCommandQueue_t(
    nxs_command_queue command_queue);

typedef nxsReleaseCommandQueue_t *
nxsReleaseCommandQueue_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsGetCommandQueueInfo_t(
    nxs_command_queue command_queue,
    nxs_command_queue_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

typedef nxsGetCommandQueueInfo_t *
nxsGetCommandQueueInfo_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_mem NXS_API_CALL nxsCreateBuffer_t(
    nxs_context context,
    nxs_mem_flags flags,
    size_t size,
    void* host_ptr,
    nxs_int* errcode_ret);

typedef nxsCreateBuffer_t *
nxsCreateBuffer_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsRetainMemObject_t(
    nxs_mem memobj);

typedef nxsRetainMemObject_t *
nxsRetainMemObject_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsReleaseMemObject_t(
    nxs_mem memobj);

typedef nxsReleaseMemObject_t *
nxsReleaseMemObject_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsGetSupportedImageFormats_t(
    nxs_context context,
    nxs_mem_flags flags,
    nxs_mem_object_type image_type,
    nxs_uint num_entries,
    nxs_image_format* image_formats,
    nxs_uint* num_image_formats);

typedef nxsGetSupportedImageFormats_t *
nxsGetSupportedImageFormats_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsGetMemObjectInfo_t(
    nxs_mem memobj,
    nxs_mem_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

typedef nxsGetMemObjectInfo_t *
nxsGetMemObjectInfo_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsGetImageInfo_t(
    nxs_mem image,
    nxs_image_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

typedef nxsGetImageInfo_t *
nxsGetImageInfo_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsRetainSampler_t(
    nxs_sampler sampler);

typedef nxsRetainSampler_t *
nxsRetainSampler_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsReleaseSampler_t(
    nxs_sampler sampler);

typedef nxsReleaseSampler_t *
nxsReleaseSampler_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsGetSamplerInfo_t(
    nxs_sampler sampler,
    nxs_sampler_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

typedef nxsGetSamplerInfo_t *
nxsGetSamplerInfo_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_program NXS_API_CALL nxsCreateProgramWithSource_t(
    nxs_context context,
    nxs_uint count,
    const char** strings,
    const size_t* lengths,
    nxs_int* errcode_ret);

typedef nxsCreateProgramWithSource_t *
nxsCreateProgramWithSource_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_program NXS_API_CALL nxsCreateProgramWithBinary_t(
    nxs_context context,
    nxs_uint num_devices,
    const nxs_device_id* device_list,
    const size_t* lengths,
    const unsigned char** binaries,
    nxs_int* binary_status,
    nxs_int* errcode_ret);

typedef nxsCreateProgramWithBinary_t *
nxsCreateProgramWithBinary_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsRetainProgram_t(
    nxs_program program);

typedef nxsRetainProgram_t *
nxsRetainProgram_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsReleaseProgram_t(
    nxs_program program);

typedef nxsReleaseProgram_t *
nxsReleaseProgram_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsBuildProgram_t(
    nxs_program program,
    nxs_uint num_devices,
    const nxs_device_id* device_list,
    const char* options,
    void (NXS_CALLBACK* pfn_notify)(nxs_program program, void* user_data),
    void* user_data);

typedef nxsBuildProgram_t *
nxsBuildProgram_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsGetProgramInfo_t(
    nxs_program program,
    nxs_program_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

typedef nxsGetProgramInfo_t *
nxsGetProgramInfo_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsGetProgramBuildInfo_t(
    nxs_program program,
    nxs_device_id device,
    nxs_program_build_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

typedef nxsGetProgramBuildInfo_t *
nxsGetProgramBuildInfo_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_kernel NXS_API_CALL nxsCreateKernel_t(
    nxs_program program,
    const char* kernel_name,
    nxs_int* errcode_ret);

typedef nxsCreateKernel_t *
nxsCreateKernel_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsCreateKernelsInProgram_t(
    nxs_program program,
    nxs_uint num_kernels,
    nxs_kernel* kernels,
    nxs_uint* num_kernels_ret);

typedef nxsCreateKernelsInProgram_t *
nxsCreateKernelsInProgram_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsRetainKernel_t(
    nxs_kernel kernel);

typedef nxsRetainKernel_t *
nxsRetainKernel_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsReleaseKernel_t(
    nxs_kernel kernel);

typedef nxsReleaseKernel_t *
nxsReleaseKernel_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsSetKernelArg_t(
    nxs_kernel kernel,
    nxs_uint arg_index,
    size_t arg_size,
    const void* arg_value);

typedef nxsSetKernelArg_t *
nxsSetKernelArg_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsGetKernelInfo_t(
    nxs_kernel kernel,
    nxs_kernel_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

typedef nxsGetKernelInfo_t *
nxsGetKernelInfo_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsGetKernelWorkGroupInfo_t(
    nxs_kernel kernel,
    nxs_device_id device,
    nxs_kernel_work_group_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

typedef nxsGetKernelWorkGroupInfo_t *
nxsGetKernelWorkGroupInfo_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsWaitForEvents_t(
    nxs_uint num_events,
    const nxs_event* event_list);

typedef nxsWaitForEvents_t *
nxsWaitForEvents_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsGetEventInfo_t(
    nxs_event event,
    nxs_event_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

typedef nxsGetEventInfo_t *
nxsGetEventInfo_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsRetainEvent_t(
    nxs_event event);

typedef nxsRetainEvent_t *
nxsRetainEvent_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsReleaseEvent_t(
    nxs_event event);

typedef nxsReleaseEvent_t *
nxsReleaseEvent_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsGetEventProfilingInfo_t(
    nxs_event event,
    nxs_profiling_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

typedef nxsGetEventProfilingInfo_t *
nxsGetEventProfilingInfo_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsFlush_t(
    nxs_command_queue command_queue);

typedef nxsFlush_t *
nxsFlush_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsFinish_t(
    nxs_command_queue command_queue);

typedef nxsFinish_t *
nxsFinish_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsEnqueueReadBuffer_t(
    nxs_command_queue command_queue,
    nxs_mem buffer,
    nxs_bool blocking_read,
    size_t offset,
    size_t size,
    void* ptr,
    nxs_uint num_events_in_wait_list,
    const nxs_event* event_wait_list,
    nxs_event* event);

typedef nxsEnqueueReadBuffer_t *
nxsEnqueueReadBuffer_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsEnqueueWriteBuffer_t(
    nxs_command_queue command_queue,
    nxs_mem buffer,
    nxs_bool blocking_write,
    size_t offset,
    size_t size,
    const void* ptr,
    nxs_uint num_events_in_wait_list,
    const nxs_event* event_wait_list,
    nxs_event* event);

typedef nxsEnqueueWriteBuffer_t *
nxsEnqueueWriteBuffer_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsEnqueueCopyBuffer_t(
    nxs_command_queue command_queue,
    nxs_mem src_buffer,
    nxs_mem dst_buffer,
    size_t src_offset,
    size_t dst_offset,
    size_t size,
    nxs_uint num_events_in_wait_list,
    const nxs_event* event_wait_list,
    nxs_event* event);

typedef nxsEnqueueCopyBuffer_t *
nxsEnqueueCopyBuffer_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsEnqueueReadImage_t(
    nxs_command_queue command_queue,
    nxs_mem image,
    nxs_bool blocking_read,
    const size_t* origin,
    const size_t* region,
    size_t row_pitch,
    size_t slice_pitch,
    void* ptr,
    nxs_uint num_events_in_wait_list,
    const nxs_event* event_wait_list,
    nxs_event* event);

typedef nxsEnqueueReadImage_t *
nxsEnqueueReadImage_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsEnqueueWriteImage_t(
    nxs_command_queue command_queue,
    nxs_mem image,
    nxs_bool blocking_write,
    const size_t* origin,
    const size_t* region,
    size_t input_row_pitch,
    size_t input_slice_pitch,
    const void* ptr,
    nxs_uint num_events_in_wait_list,
    const nxs_event* event_wait_list,
    nxs_event* event);

typedef nxsEnqueueWriteImage_t *
nxsEnqueueWriteImage_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsEnqueueCopyImage_t(
    nxs_command_queue command_queue,
    nxs_mem src_image,
    nxs_mem dst_image,
    const size_t* src_origin,
    const size_t* dst_origin,
    const size_t* region,
    nxs_uint num_events_in_wait_list,
    const nxs_event* event_wait_list,
    nxs_event* event);

typedef nxsEnqueueCopyImage_t *
nxsEnqueueCopyImage_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsEnqueueCopyImageToBuffer_t(
    nxs_command_queue command_queue,
    nxs_mem src_image,
    nxs_mem dst_buffer,
    const size_t* src_origin,
    const size_t* region,
    size_t dst_offset,
    nxs_uint num_events_in_wait_list,
    const nxs_event* event_wait_list,
    nxs_event* event);

typedef nxsEnqueueCopyImageToBuffer_t *
nxsEnqueueCopyImageToBuffer_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsEnqueueCopyBufferToImage_t(
    nxs_command_queue command_queue,
    nxs_mem src_buffer,
    nxs_mem dst_image,
    size_t src_offset,
    const size_t* dst_origin,
    const size_t* region,
    nxs_uint num_events_in_wait_list,
    const nxs_event* event_wait_list,
    nxs_event* event);

typedef nxsEnqueueCopyBufferToImage_t *
nxsEnqueueCopyBufferToImage_fn NXS_API_SUFFIX__VERSION_1_0;

typedef void* NXS_API_CALL nxsEnqueueMapBuffer_t(
    nxs_command_queue command_queue,
    nxs_mem buffer,
    nxs_bool blocking_map,
    nxs_map_flags map_flags,
    size_t offset,
    size_t size,
    nxs_uint num_events_in_wait_list,
    const nxs_event* event_wait_list,
    nxs_event* event,
    nxs_int* errcode_ret);

typedef nxsEnqueueMapBuffer_t *
nxsEnqueueMapBuffer_fn NXS_API_SUFFIX__VERSION_1_0;

typedef void* NXS_API_CALL nxsEnqueueMapImage_t(
    nxs_command_queue command_queue,
    nxs_mem image,
    nxs_bool blocking_map,
    nxs_map_flags map_flags,
    const size_t* origin,
    const size_t* region,
    size_t* image_row_pitch,
    size_t* image_slice_pitch,
    nxs_uint num_events_in_wait_list,
    const nxs_event* event_wait_list,
    nxs_event* event,
    nxs_int* errcode_ret);

typedef nxsEnqueueMapImage_t *
nxsEnqueueMapImage_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsEnqueueUnmapMemObject_t(
    nxs_command_queue command_queue,
    nxs_mem memobj,
    void* mapped_ptr,
    nxs_uint num_events_in_wait_list,
    const nxs_event* event_wait_list,
    nxs_event* event);

typedef nxsEnqueueUnmapMemObject_t *
nxsEnqueueUnmapMemObject_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsEnqueueNDRangeKernel_t(
    nxs_command_queue command_queue,
    nxs_kernel kernel,
    nxs_uint work_dim,
    const size_t* global_work_offset,
    const size_t* global_work_size,
    const size_t* local_work_size,
    nxs_uint num_events_in_wait_list,
    const nxs_event* event_wait_list,
    nxs_event* event);

typedef nxsEnqueueNDRangeKernel_t *
nxsEnqueueNDRangeKernel_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsEnqueueNativeKernel_t(
    nxs_command_queue command_queue,
    void (NXS_CALLBACK* user_func)(void*),
    void* args,
    size_t cb_args,
    nxs_uint num_mem_objects,
    const nxs_mem* mem_list,
    const void** args_mem_loc,
    nxs_uint num_events_in_wait_list,
    const nxs_event* event_wait_list,
    nxs_event* event);

typedef nxsEnqueueNativeKernel_t *
nxsEnqueueNativeKernel_fn NXS_API_SUFFIX__VERSION_1_0;

typedef nxs_int NXS_API_CALL nxsSetCommandQueueProperty_t(
    nxs_command_queue command_queue,
    nxs_command_queue_properties properties,
    nxs_bool enable,
    nxs_command_queue_properties* old_properties);

typedef nxsSetCommandQueueProperty_t *
nxsSetCommandQueueProperty_fn NXS_API_SUFFIX__VERSION_1_0_DEPRECATED;

typedef nxs_mem NXS_API_CALL nxsCreateImage2D_t(
    nxs_context context,
    nxs_mem_flags flags,
    const nxs_image_format* image_format,
    size_t image_width,
    size_t image_height,
    size_t image_row_pitch,
    void* host_ptr,
    nxs_int* errcode_ret);

typedef nxsCreateImage2D_t *
nxsCreateImage2D_fn NXS_API_SUFFIX__VERSION_1_1_DEPRECATED;

typedef nxs_mem NXS_API_CALL nxsCreateImage3D_t(
    nxs_context context,
    nxs_mem_flags flags,
    const nxs_image_format* image_format,
    size_t image_width,
    size_t image_height,
    size_t image_depth,
    size_t image_row_pitch,
    size_t image_slice_pitch,
    void* host_ptr,
    nxs_int* errcode_ret);

typedef nxsCreateImage3D_t *
nxsCreateImage3D_fn NXS_API_SUFFIX__VERSION_1_1_DEPRECATED;

typedef nxs_int NXS_API_CALL nxsEnqueueMarker_t(
    nxs_command_queue command_queue,
    nxs_event* event);

typedef nxsEnqueueMarker_t *
nxsEnqueueMarker_fn NXS_API_SUFFIX__VERSION_1_1_DEPRECATED;

typedef nxs_int NXS_API_CALL nxsEnqueueWaitForEvents_t(
    nxs_command_queue command_queue,
    nxs_uint num_events,
    const nxs_event* event_list);

typedef nxsEnqueueWaitForEvents_t *
nxsEnqueueWaitForEvents_fn NXS_API_SUFFIX__VERSION_1_1_DEPRECATED;

typedef nxs_int NXS_API_CALL nxsEnqueueBarrier_t(
    nxs_command_queue command_queue);

typedef nxsEnqueueBarrier_t *
nxsEnqueueBarrier_fn NXS_API_SUFFIX__VERSION_1_1_DEPRECATED;

typedef nxs_int NXS_API_CALL nxsUnloadCompiler_t(
    void );

typedef nxsUnloadCompiler_t *
nxsUnloadCompiler_fn NXS_API_SUFFIX__VERSION_1_1_DEPRECATED;

typedef void* NXS_API_CALL nxsGetExtensionFunctionAddress_t(
    const char* func_name);

typedef nxsGetExtensionFunctionAddress_t *
nxsGetExtensionFunctionAddress_fn NXS_API_SUFFIX__VERSION_1_1_DEPRECATED;

typedef nxs_command_queue NXS_API_CALL nxsCreateCommandQueue_t(
    nxs_context context,
    nxs_device_id device,
    nxs_command_queue_properties properties,
    nxs_int* errcode_ret);

typedef nxsCreateCommandQueue_t *
nxsCreateCommandQueue_fn NXS_API_SUFFIX__VERSION_1_2_DEPRECATED;

typedef nxs_sampler NXS_API_CALL nxsCreateSampler_t(
    nxs_context context,
    nxs_bool normalized_coords,
    nxs_addressing_mode addressing_mode,
    nxs_filter_mode filter_mode,
    nxs_int* errcode_ret);

typedef nxsCreateSampler_t *
nxsCreateSampler_fn NXS_API_SUFFIX__VERSION_1_2_DEPRECATED;

typedef nxs_int NXS_API_CALL nxsEnqueueTask_t(
    nxs_command_queue command_queue,
    nxs_kernel kernel,
    nxs_uint num_events_in_wait_list,
    const nxs_event* event_wait_list,
    nxs_event* event);

typedef nxsEnqueueTask_t *
nxsEnqueueTask_fn NXS_API_SUFFIX__VERSION_1_2_DEPRECATED;

#ifdef NXS_VERSION_1_1

typedef nxs_mem NXS_API_CALL nxsCreateSubBuffer_t(
    nxs_mem buffer,
    nxs_mem_flags flags,
    nxs_buffer_create_type buffer_create_type,
    const void* buffer_create_info,
    nxs_int* errcode_ret);

typedef nxsCreateSubBuffer_t *
nxsCreateSubBuffer_fn NXS_API_SUFFIX__VERSION_1_1;

typedef nxs_int NXS_API_CALL nxsSetMemObjectDestructorCallback_t(
    nxs_mem memobj,
    void (NXS_CALLBACK* pfn_notify)(nxs_mem memobj, void* user_data),
    void* user_data);

typedef nxsSetMemObjectDestructorCallback_t *
nxsSetMemObjectDestructorCallback_fn NXS_API_SUFFIX__VERSION_1_1;

typedef nxs_event NXS_API_CALL nxsCreateUserEvent_t(
    nxs_context context,
    nxs_int* errcode_ret);

typedef nxsCreateUserEvent_t *
nxsCreateUserEvent_fn NXS_API_SUFFIX__VERSION_1_1;

typedef nxs_int NXS_API_CALL nxsSetUserEventStatus_t(
    nxs_event event,
    nxs_int execution_status);

typedef nxsSetUserEventStatus_t *
nxsSetUserEventStatus_fn NXS_API_SUFFIX__VERSION_1_1;

typedef nxs_int NXS_API_CALL nxsSetEventCallback_t(
    nxs_event event,
    nxs_int command_exec_callback_type,
    void (NXS_CALLBACK* pfn_notify)(nxs_event event, nxs_int event_command_status, void *user_data),
    void* user_data);

typedef nxsSetEventCallback_t *
nxsSetEventCallback_fn NXS_API_SUFFIX__VERSION_1_1;

typedef nxs_int NXS_API_CALL nxsEnqueueReadBufferRect_t(
    nxs_command_queue command_queue,
    nxs_mem buffer,
    nxs_bool blocking_read,
    const size_t* buffer_origin,
    const size_t* host_origin,
    const size_t* region,
    size_t buffer_row_pitch,
    size_t buffer_slice_pitch,
    size_t host_row_pitch,
    size_t host_slice_pitch,
    void* ptr,
    nxs_uint num_events_in_wait_list,
    const nxs_event* event_wait_list,
    nxs_event* event);

typedef nxsEnqueueReadBufferRect_t *
nxsEnqueueReadBufferRect_fn NXS_API_SUFFIX__VERSION_1_1;

typedef nxs_int NXS_API_CALL nxsEnqueueWriteBufferRect_t(
    nxs_command_queue command_queue,
    nxs_mem buffer,
    nxs_bool blocking_write,
    const size_t* buffer_origin,
    const size_t* host_origin,
    const size_t* region,
    size_t buffer_row_pitch,
    size_t buffer_slice_pitch,
    size_t host_row_pitch,
    size_t host_slice_pitch,
    const void* ptr,
    nxs_uint num_events_in_wait_list,
    const nxs_event* event_wait_list,
    nxs_event* event);

typedef nxsEnqueueWriteBufferRect_t *
nxsEnqueueWriteBufferRect_fn NXS_API_SUFFIX__VERSION_1_1;

typedef nxs_int NXS_API_CALL nxsEnqueueCopyBufferRect_t(
    nxs_command_queue command_queue,
    nxs_mem src_buffer,
    nxs_mem dst_buffer,
    const size_t* src_origin,
    const size_t* dst_origin,
    const size_t* region,
    size_t src_row_pitch,
    size_t src_slice_pitch,
    size_t dst_row_pitch,
    size_t dst_slice_pitch,
    nxs_uint num_events_in_wait_list,
    const nxs_event* event_wait_list,
    nxs_event* event);

typedef nxsEnqueueCopyBufferRect_t *
nxsEnqueueCopyBufferRect_fn NXS_API_SUFFIX__VERSION_1_1;

#endif /* NXS_VERSION_1_1 */

#ifdef NXS_VERSION_1_2

typedef nxs_int NXS_API_CALL nxsCreateSubDevices_t(
    nxs_device_id in_device,
    const nxs_device_partition_property* properties,
    nxs_uint num_devices,
    nxs_device_id* out_devices,
    nxs_uint* num_devices_ret);

typedef nxsCreateSubDevices_t *
nxsCreateSubDevices_fn NXS_API_SUFFIX__VERSION_1_2;

typedef nxs_int NXS_API_CALL nxsRetainDevice_t(
    nxs_device_id device);

typedef nxsRetainDevice_t *
nxsRetainDevice_fn NXS_API_SUFFIX__VERSION_1_2;

typedef nxs_int NXS_API_CALL nxsReleaseDevice_t(
    nxs_device_id device);

typedef nxsReleaseDevice_t *
nxsReleaseDevice_fn NXS_API_SUFFIX__VERSION_1_2;

typedef nxs_mem NXS_API_CALL nxsCreateImage_t(
    nxs_context context,
    nxs_mem_flags flags,
    const nxs_image_format* image_format,
    const nxs_image_desc* image_desc,
    void* host_ptr,
    nxs_int* errcode_ret);

typedef nxsCreateImage_t *
nxsCreateImage_fn NXS_API_SUFFIX__VERSION_1_2;

typedef nxs_program NXS_API_CALL nxsCreateProgramWithBuiltInKernels_t(
    nxs_context context,
    nxs_uint num_devices,
    const nxs_device_id* device_list,
    const char* kernel_names,
    nxs_int* errcode_ret);

typedef nxsCreateProgramWithBuiltInKernels_t *
nxsCreateProgramWithBuiltInKernels_fn NXS_API_SUFFIX__VERSION_1_2;

typedef nxs_int NXS_API_CALL nxsCompileProgram_t(
    nxs_program program,
    nxs_uint num_devices,
    const nxs_device_id* device_list,
    const char* options,
    nxs_uint num_input_headers,
    const nxs_program* input_headers,
    const char** header_include_names,
    void (NXS_CALLBACK* pfn_notify)(nxs_program program, void* user_data),
    void* user_data);

typedef nxsCompileProgram_t *
nxsCompileProgram_fn NXS_API_SUFFIX__VERSION_1_2;

typedef nxs_program NXS_API_CALL nxsLinkProgram_t(
    nxs_context context,
    nxs_uint num_devices,
    const nxs_device_id* device_list,
    const char* options,
    nxs_uint num_input_programs,
    const nxs_program* input_programs,
    void (NXS_CALLBACK* pfn_notify)(nxs_program program, void* user_data),
    void* user_data,
    nxs_int* errcode_ret);

typedef nxsLinkProgram_t *
nxsLinkProgram_fn NXS_API_SUFFIX__VERSION_1_2;

typedef nxs_int NXS_API_CALL nxsUnloadPlatformCompiler_t(
    nxs_platform_id platform);

typedef nxsUnloadPlatformCompiler_t *
nxsUnloadPlatformCompiler_fn NXS_API_SUFFIX__VERSION_1_2;

typedef nxs_int NXS_API_CALL nxsGetKernelArgInfo_t(
    nxs_kernel kernel,
    nxs_uint arg_index,
    nxs_kernel_arg_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

typedef nxsGetKernelArgInfo_t *
nxsGetKernelArgInfo_fn NXS_API_SUFFIX__VERSION_1_2;

typedef nxs_int NXS_API_CALL nxsEnqueueFillBuffer_t(
    nxs_command_queue command_queue,
    nxs_mem buffer,
    const void* pattern,
    size_t pattern_size,
    size_t offset,
    size_t size,
    nxs_uint num_events_in_wait_list,
    const nxs_event* event_wait_list,
    nxs_event* event);

typedef nxsEnqueueFillBuffer_t *
nxsEnqueueFillBuffer_fn NXS_API_SUFFIX__VERSION_1_2;

typedef nxs_int NXS_API_CALL nxsEnqueueFillImage_t(
    nxs_command_queue command_queue,
    nxs_mem image,
    const void* fill_color,
    const size_t* origin,
    const size_t* region,
    nxs_uint num_events_in_wait_list,
    const nxs_event* event_wait_list,
    nxs_event* event);

typedef nxsEnqueueFillImage_t *
nxsEnqueueFillImage_fn NXS_API_SUFFIX__VERSION_1_2;

typedef nxs_int NXS_API_CALL nxsEnqueueMigrateMemObjects_t(
    nxs_command_queue command_queue,
    nxs_uint num_mem_objects,
    const nxs_mem* mem_objects,
    nxs_mem_migration_flags flags,
    nxs_uint num_events_in_wait_list,
    const nxs_event* event_wait_list,
    nxs_event* event);

typedef nxsEnqueueMigrateMemObjects_t *
nxsEnqueueMigrateMemObjects_fn NXS_API_SUFFIX__VERSION_1_2;

typedef nxs_int NXS_API_CALL nxsEnqueueMarkerWithWaitList_t(
    nxs_command_queue command_queue,
    nxs_uint num_events_in_wait_list,
    const nxs_event* event_wait_list,
    nxs_event* event);

typedef nxsEnqueueMarkerWithWaitList_t *
nxsEnqueueMarkerWithWaitList_fn NXS_API_SUFFIX__VERSION_1_2;

typedef nxs_int NXS_API_CALL nxsEnqueueBarrierWithWaitList_t(
    nxs_command_queue command_queue,
    nxs_uint num_events_in_wait_list,
    const nxs_event* event_wait_list,
    nxs_event* event);

typedef nxsEnqueueBarrierWithWaitList_t *
nxsEnqueueBarrierWithWaitList_fn NXS_API_SUFFIX__VERSION_1_2;

typedef void* NXS_API_CALL nxsGetExtensionFunctionAddressForPlatform_t(
    nxs_platform_id platform,
    const char* func_name);

typedef nxsGetExtensionFunctionAddressForPlatform_t *
nxsGetExtensionFunctionAddressForPlatform_fn NXS_API_SUFFIX__VERSION_1_2;

#endif /* NXS_VERSION_1_2 */

#ifdef NXS_VERSION_2_0

typedef nxs_command_queue NXS_API_CALL nxsCreateCommandQueueWithProperties_t(
    nxs_context context,
    nxs_device_id device,
    const nxs_queue_properties* properties,
    nxs_int* errcode_ret);

typedef nxsCreateCommandQueueWithProperties_t *
nxsCreateCommandQueueWithProperties_fn NXS_API_SUFFIX__VERSION_2_0;

#endif /* NXS_VERSION_2_0 */

#ifdef NXS_VERSION_2_1

typedef nxs_int NXS_API_CALL nxsSetDefaultDeviceCommandQueue_t(
    nxs_context context,
    nxs_device_id device,
    nxs_command_queue command_queue);

typedef nxsSetDefaultDeviceCommandQueue_t *
nxsSetDefaultDeviceCommandQueue_fn NXS_API_SUFFIX__VERSION_2_1;

typedef nxs_int NXS_API_CALL nxsGetDeviceAndHostTimer_t(
    nxs_device_id device,
    nxs_ulong* device_timestamp,
    nxs_ulong* host_timestamp);

typedef nxsGetDeviceAndHostTimer_t *
nxsGetDeviceAndHostTimer_fn NXS_API_SUFFIX__VERSION_2_1;

typedef nxs_int NXS_API_CALL nxsGetHostTimer_t(
    nxs_device_id device,
    nxs_ulong* host_timestamp);

typedef nxsGetHostTimer_t *
nxsGetHostTimer_fn NXS_API_SUFFIX__VERSION_2_1;

typedef nxs_program NXS_API_CALL nxsCreateProgramWithIL_t(
    nxs_context context,
    const void* il,
    size_t length,
    nxs_int* errcode_ret);

typedef nxsCreateProgramWithIL_t *
nxsCreateProgramWithIL_fn NXS_API_SUFFIX__VERSION_2_1;

typedef nxs_kernel NXS_API_CALL nxsCloneKernel_t(
    nxs_kernel source_kernel,
    nxs_int* errcode_ret);

typedef nxsCloneKernel_t *
nxsCloneKernel_fn NXS_API_SUFFIX__VERSION_2_1;

typedef nxs_int NXS_API_CALL nxsGetKernelSubGroupInfo_t(
    nxs_kernel kernel,
    nxs_device_id device,
    nxs_kernel_sub_group_info param_name,
    size_t input_value_size,
    const void* input_value,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

typedef nxsGetKernelSubGroupInfo_t *
nxsGetKernelSubGroupInfo_fn NXS_API_SUFFIX__VERSION_2_1;

typedef nxs_int NXS_API_CALL nxsEnqueueSVMMigrateMem_t(
    nxs_command_queue command_queue,
    nxs_uint num_svm_pointers,
    const void** svm_pointers,
    const size_t* sizes,
    nxs_mem_migration_flags flags,
    nxs_uint num_events_in_wait_list,
    const nxs_event* event_wait_list,
    nxs_event* event);

typedef nxsEnqueueSVMMigrateMem_t *
nxsEnqueueSVMMigrateMem_fn NXS_API_SUFFIX__VERSION_2_1;

#endif /* NXS_VERSION_2_1 */

#ifdef NXS_VERSION_3_0

typedef nxs_mem NXS_API_CALL nxsCreateBufferWithProperties_t(
    nxs_context context,
    const nxs_mem_properties* properties,
    nxs_mem_flags flags,
    size_t size,
    void* host_ptr,
    nxs_int* errcode_ret);

typedef nxsCreateBufferWithProperties_t *
nxsCreateBufferWithProperties_fn NXS_API_SUFFIX__VERSION_3_0;

typedef nxs_mem NXS_API_CALL nxsCreateImageWithProperties_t(
    nxs_context context,
    const nxs_mem_properties* properties,
    nxs_mem_flags flags,
    const nxs_image_format* image_format,
    const nxs_image_desc* image_desc,
    void* host_ptr,
    nxs_int* errcode_ret);

typedef nxsCreateImageWithProperties_t *
nxsCreateImageWithProperties_fn NXS_API_SUFFIX__VERSION_3_0;

#endif /* NXS_VERSION_3_0 */
#endif /* diabled */

#ifdef NEXUS_API_GENERATE_FUNC_ENUM
    NXSAPI_FUNCTION_COUNT,
    NXSAPI_FUNCTION_PREFIX_LEN = 3
}; /* close NXSAPI_FunctionEnum */

const char *nxsGetFuncName(enum NXSAPI_FunctionEnum funcEnum);

enum NXSAPI_FunctionEnum nxsGetFuncEnum(const char *funcName);

#endif


#undef NEXUS_API_GENERATE_FUNC_DECL
#undef NEXUS_API_GENERATE_FUNC_ENUM
#undef NEXUS_API_GENERATE_FUNC_TYPE

#undef NEXUS_API_FUNC
