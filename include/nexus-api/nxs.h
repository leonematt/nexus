/*******************************************************************************
 * Copyright (c) 2008-2020 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef __NEXUSAPI_NXS_H
#define __NEXUSAPI_NXS_H

#include <nexus-api/nxs_version.h>
#include <nexus-api/nxs_platform.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************/

typedef struct _nxs_platform_id *    nxs_platform_id;
typedef struct _nxs_device_id *      nxs_device_id;
typedef struct _nxs_context *        nxs_context;
typedef struct _nxs_command_queue *  nxs_command_queue;
typedef struct _nxs_mem *            nxs_mem;
typedef struct _nxs_program *        nxs_program;
typedef struct _nxs_kernel *         nxs_kernel;
typedef struct _nxs_event *          nxs_event;
typedef struct _nxs_sampler *        nxs_sampler;

typedef nxs_uint             nxs_bool;                     /* WARNING!  Unlike nxs_ types in nxs_platform.h, nxs_bool is not guaranteed to be the same size as the bool in kernels. */
typedef nxs_ulong            nxs_bitfield;
typedef nxs_ulong            nxs_properties;
typedef nxs_bitfield         nxs_device_type;
typedef nxs_uint             nxs_platform_info;
typedef nxs_uint             nxs_device_info;
typedef nxs_bitfield         nxs_device_fp_config;
typedef nxs_uint             nxs_device_mem_cache_type;
typedef nxs_uint             nxs_device_local_mem_type;
typedef nxs_bitfield         nxs_device_exec_capabilities;
#ifdef NXS_VERSION_2_0
typedef nxs_bitfield         nxs_device_svm_capabilities;
#endif
typedef nxs_bitfield         nxs_command_queue_properties;
#ifdef NXS_VERSION_1_2
typedef intptr_t            nxs_device_partition_property;
typedef nxs_bitfield         nxs_device_affinity_domain;
#endif

typedef intptr_t            nxs_context_properties;
typedef nxs_uint             nxs_context_info;
#ifdef NXS_VERSION_2_0
typedef nxs_properties       nxs_queue_properties;
#endif
typedef nxs_uint             nxs_command_queue_info;
typedef nxs_uint             nxs_channel_order;
typedef nxs_uint             nxs_channel_type;
typedef nxs_bitfield         nxs_mem_flags;
#ifdef NXS_VERSION_2_0
typedef nxs_bitfield         nxs_svm_mem_flags;
#endif
typedef nxs_uint             nxs_mem_object_type;
typedef nxs_uint             nxs_mem_info;
#ifdef NXS_VERSION_1_2
typedef nxs_bitfield         nxs_mem_migration_flags;
#endif
typedef nxs_uint             nxs_image_info;
#ifdef NXS_VERSION_1_1
typedef nxs_uint             nxs_buffer_create_type;
#endif
typedef nxs_uint             nxs_addressing_mode;
typedef nxs_uint             nxs_filter_mode;
typedef nxs_uint             nxs_sampler_info;
typedef nxs_bitfield         nxs_map_flags;
#ifdef NXS_VERSION_2_0
typedef intptr_t            nxs_pipe_properties;
typedef nxs_uint             nxs_pipe_info;
#endif
typedef nxs_uint             nxs_program_info;
typedef nxs_uint             nxs_program_build_info;
#ifdef NXS_VERSION_1_2
typedef nxs_uint             nxs_program_binary_type;
#endif
typedef nxs_int              nxs_build_status;
typedef nxs_uint             nxs_kernel_info;
#ifdef NXS_VERSION_1_2
typedef nxs_uint             nxs_kernel_arg_info;
typedef nxs_uint             nxs_kernel_arg_address_qualifier;
typedef nxs_uint             nxs_kernel_arg_access_qualifier;
typedef nxs_bitfield         nxs_kernel_arg_type_qualifier;
#endif
typedef nxs_uint             nxs_kernel_work_group_info;
#ifdef NXS_VERSION_2_1
typedef nxs_uint             nxs_kernel_sub_group_info;
#endif
typedef nxs_uint             nxs_event_info;
typedef nxs_uint             nxs_command_type;
typedef nxs_uint             nxs_profiling_info;
#ifdef NXS_VERSION_2_0
typedef nxs_properties       nxs_sampler_properties;
typedef nxs_uint             nxs_kernel_exec_info;
#endif
#ifdef NXS_VERSION_3_0
typedef nxs_bitfield         nxs_device_atomic_capabilities;
typedef nxs_bitfield         nxs_device_device_enqueue_capabilities;
typedef nxs_uint             nxs_khronos_vendor_id;
typedef nxs_properties nxs_mem_properties;
#endif
typedef nxs_uint nxs_version;

typedef struct _nxs_image_format {
    nxs_channel_order        image_channel_order;
    nxs_channel_type         image_channel_data_type;
} nxs_image_format;

#ifdef NXS_VERSION_1_2

typedef struct _nxs_image_desc {
    nxs_mem_object_type      image_type;
    size_t                  image_width;
    size_t                  image_height;
    size_t                  image_depth;
    size_t                  image_array_size;
    size_t                  image_row_pitch;
    size_t                  image_slice_pitch;
    nxs_uint                 num_mip_levels;
    nxs_uint                 num_samples;
#ifdef NXS_VERSION_2_0
#if defined(__GNUC__)
    __extension__                   /* Prevents warnings about anonymous union in -pedantic builds */
#endif
#if defined(_MSC_VER) && !defined(__STDC__)
#pragma warning( push )
#pragma warning( disable : 4201 )   /* Prevents warning about nameless struct/union in /W4 builds */
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc11-extensions" /* Prevents warning about nameless union being C11 extension*/
#endif
#if defined(_MSC_VER) && defined(__STDC__)
    /* Anonymous unions are not supported in /Za builds */
#else
    union {
#endif
#endif
      nxs_mem                  buffer;
#ifdef NXS_VERSION_2_0
#if defined(_MSC_VER) && defined(__STDC__)
    /* Anonymous unions are not supported in /Za builds */
#else
      nxs_mem                  mem_object;
    };
#endif
#if defined(_MSC_VER) && !defined(__STDC__)
#pragma warning( pop )
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#endif
} nxs_image_desc;

#endif

#ifdef NXS_VERSION_1_1

typedef struct _nxs_buffer_region {
    size_t                  origin;
    size_t                  size;
} nxs_buffer_region;

#endif

#ifdef NXS_VERSION_3_0

#define NXS_NAME_VERSION_MAX_NAME_SIZE 64

typedef struct _nxs_name_version {
    nxs_version              version;
    char                    name[NXS_NAME_VERSION_MAX_NAME_SIZE];
} nxs_name_version;

#endif

/******************************************************************************/

/* Error Codes */
enum _nxs_status {
    NXS_Success                              = 0,
    NXS_DeviceNotFound                       = -1,
    NXS_DeviceNotAvailable                   = -2,
    NXS_CompilerNotAvailable                 = -3,
    NXS_MemObjectAllocationFailure           = -4,
    NXS_OutOfResources                       = -5,
    NXS_OutOfHostMemory                      = -6,
    NXS_ProfilingInfoNotAvailable            = -7,
    NXS_MemCopyOverlap                       = -8,
    NXS_ImageFormatMismatch                  = -9,
    NXS_ImageFormatNotSupported              = -10,
    NXS_BuildProgramFailure                  = -11,
    NXS_MapFailure                           = -12,
    NXS_MisalignedSubBufferOffset            = -13,
    NXS_ExecStatusErrorForEventsInWaitList   = -14,
    NXS_CompileProgramFailure                = -15,
    NXS_LinkerNotAvailable                   = -16,
    NXS_LinkProgramFailure                   = -17,
    NXS_DevicePartitionFailed                = -18,
    NXS_KernelArgInfoNotAvailable            = -19,
    NXS_InvalidValue                         = -30,
    NXS_InvalidDeviceType                    = -31,
    NXS_InvalidRuntime                       = -32,
    NXS_InvalidDevice                        = -33,
    NXS_InvalidContext                       = -34,
    NXS_InvalidQueueProperties               = -35,
    NXS_InvalidCommandQueue                  = -36,
    NXS_InvalidHostPtr                       = -37,
    NXS_InvalidMemObject                     = -38,
    NXS_InvalidImageFormatDescriptor         = -39,
    NXS_InvalidImageSize                     = -40,
    NXS_InvalidSampler                       = -41,
    NXS_InvalidBinary                        = -42,
    NXS_InvalidBuildOptions                  = -43,
    NXS_InvalidProgram                       = -44,
    NXS_InvalidProgramExecutable             = -45,
    NXS_InvalidKernelName                    = -46,
    NXS_InvalidKernelDefinition              = -47,
    NXS_InvalidKernel                        = -48,
    NXS_InvalidArgIndex                      = -49,
    NXS_InvalidArgValue                      = -50,
    NXS_InvalidArgSize                       = -51,
    NXS_InvalidKernelArgs                    = -52,
    NXS_InvalidWorkDimension                 = -53,
    NXS_InvalidWorkGroupSize                 = -54,
    NXS_InvalidWorkItemSize                  = -55,
    NXS_InvalidGlobalOffset                  = -56,
    NXS_InvalidEventWaitList                 = -57,
    NXS_InvalidEvent                         = -58,
    NXS_InvalidOperation                     = -59,
    NXS_InvalidGlObject                      = -60,
    NXS_InvalidBufferSize                    = -61,
    NXS_InvalidMipLevel                      = -62,
    NXS_InvalidGlobalWorkSize                = -63,
    NXS_InvalidProperty                      = -64,
    NXS_InvalidImageDescriptor               = -65,
    NXS_InvalidCompilerOptions               = -66,
    NXS_InvalidDeviceQueue                   = -70,
    NXS_InvalidSpecId                        = -71,
    NXS_MaxSizeRestrictionExceeded           = -72,

    NXS_STATUS_MIN                           = -72,
    NXS_STATUS_MAX                           = 0,
    NXS_STATUS_PREFIX_LEN                    = 4
};

typedef enum _nxs_status nxs_status;

inline nxs_bool nxs_success(nxs_int result) { return result >= 0; }
inline nxs_bool nxs_failed(nxs_int result) { return result < 0; }

inline nxs_bool nxs_valid_id(nxs_int id) { return id >= 0; }

/* nxs_bool */
#define NXS_FALSE                                    0
#define NXS_TRUE                                     1
#ifdef NXS_VERSION_1_2
#define NXS_BLOCKING                                 NXS_TRUE
#define NXS_NON_BLOCKING                             NXS_FALSE
#endif

/* nxs_platform_info */
#define NXS_RUNTIME_PROFILE                         0x0900
#define NXS_RUNTIME_VERSION                         0x0901
#define NXS_RUNTIME_NAME                            0x0902
#define NXS_RUNTIME_VENDOR                          0x0903
#define NXS_RUNTIME_EXTENSIONS                      0x0904
#ifdef NXS_VERSION_2_1
#define NXS_RUNTIME_HOST_TIMER_RESOLUTION           0x0905
#endif
#ifdef NXS_VERSION_3_0
#define NXS_RUNTIME_NUMERIC_VERSION                 0x0906
#define NXS_RUNTIME_EXTENSIONS_WITH_VERSION         0x0907
#endif

/* nxs_device_type - bitfield */
#define NXS_DEVICE_TYPE_DEFAULT                      (1 << 0)
#define NXS_DEVICE_TYPE_CPU                          (1 << 1)
#define NXS_DEVICE_TYPE_GPU                          (1 << 2)
#define NXS_DEVICE_TYPE_ACCELERATOR                  (1 << 3)
#ifdef NXS_VERSION_1_2
#define NXS_DEVICE_TYPE_CUSTOM                       (1 << 4)
#endif
#define NXS_DEVICE_TYPE_ALL                          0xFFFFFFFF

/* nxs_device_info */
#define NXS_DEVICE_TYPE                                   0x1000
#define NXS_DEVICE_VENDOR_ID                              0x1001
#define NXS_DEVICE_MAX_COMPUTE_UNITS                      0x1002
#define NXS_DEVICE_MAX_WORK_ITEM_DIMENSIONS               0x1003
#define NXS_DEVICE_MAX_WORK_GROUP_SIZE                    0x1004
#define NXS_DEVICE_MAX_WORK_ITEM_SIZES                    0x1005
#define NXS_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR            0x1006
#define NXS_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT           0x1007
#define NXS_DEVICE_PREFERRED_VECTOR_WIDTH_INT             0x1008
#define NXS_DEVICE_PREFERRED_VECTOR_WIDTH_LONG            0x1009
#define NXS_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT           0x100A
#define NXS_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE          0x100B
#define NXS_DEVICE_MAX_CLOCK_FREQUENCY                    0x100C
#define NXS_DEVICE_ADDRESS_BITS                           0x100D
#define NXS_DEVICE_MAX_READ_IMAGE_ARGS                    0x100E
#define NXS_DEVICE_MAX_WRITE_IMAGE_ARGS                   0x100F
#define NXS_DEVICE_MAX_MEM_ALLOC_SIZE                     0x1010
#define NXS_DEVICE_IMAGE2D_MAX_WIDTH                      0x1011
#define NXS_DEVICE_IMAGE2D_MAX_HEIGHT                     0x1012
#define NXS_DEVICE_IMAGE3D_MAX_WIDTH                      0x1013
#define NXS_DEVICE_IMAGE3D_MAX_HEIGHT                     0x1014
#define NXS_DEVICE_IMAGE3D_MAX_DEPTH                      0x1015
#define NXS_DEVICE_IMAGE_SUPPORT                          0x1016
#define NXS_DEVICE_MAX_PARAMETER_SIZE                     0x1017
#define NXS_DEVICE_MAX_SAMPLERS                           0x1018
#define NXS_DEVICE_MEM_BASE_ADDR_ALIGN                    0x1019
#define NXS_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE               0x101A
#define NXS_DEVICE_SINGLE_FP_CONFIG                       0x101B
#define NXS_DEVICE_GLOBAL_MEM_CACHE_TYPE                  0x101C
#define NXS_DEVICE_GLOBAL_MEM_CACHELINE_SIZE              0x101D
#define NXS_DEVICE_GLOBAL_MEM_CACHE_SIZE                  0x101E
#define NXS_DEVICE_GLOBAL_MEM_SIZE                        0x101F
#define NXS_DEVICE_MAX_CONSTANT_BUFFER_SIZE               0x1020
#define NXS_DEVICE_MAX_CONSTANT_ARGS                      0x1021
#define NXS_DEVICE_LOCAL_MEM_TYPE                         0x1022
#define NXS_DEVICE_LOCAL_MEM_SIZE                         0x1023
#define NXS_DEVICE_ERROR_CORRECTION_SUPPORT               0x1024
#define NXS_DEVICE_PROFILING_TIMER_RESOLUTION             0x1025
#define NXS_DEVICE_ENDIAN_LITTLE                          0x1026
#define NXS_DEVICE_AVAILABLE                              0x1027
#define NXS_DEVICE_COMPILER_AVAILABLE                     0x1028
#define NXS_DEVICE_EXECUTION_CAPABILITIES                 0x1029
#define NXS_DEVICE_QUEUE_PROPERTIES                       0x102A    /* deprecated */
#ifdef NXS_VERSION_2_0
#define NXS_DEVICE_QUEUE_ON_HOST_PROPERTIES               0x102A
#endif
#define NXS_DEVICE_NAME                                   0x102B
#define NXS_DEVICE_VENDOR                                 0x102C
#define NXS_DRIVER_VERSION                                0x102D
#define NXS_DEVICE_PROFILE                                0x102E
#define NXS_DEVICE_VERSION                                0x102F
#define NXS_DEVICE_EXTENSIONS                             0x1030
#define NXS_DEVICE_RUNTIME                               0x1031
#ifdef NXS_VERSION_1_2
#define NXS_DEVICE_DOUBLE_FP_CONFIG                       0x1032
#endif
/* 0x1033 reserved for NXS_DEVICE_HALF_FP_CONFIG which is already defined in "nxs_ext.h" */
#ifdef NXS_VERSION_1_1
#define NXS_DEVICE_PREFERRED_VECTOR_WIDTH_HALF            0x1034
#define NXS_DEVICE_HOST_UNIFIED_MEMORY                    0x1035   /* deprecated */
#define NXS_DEVICE_NATIVE_VECTOR_WIDTH_CHAR               0x1036
#define NXS_DEVICE_NATIVE_VECTOR_WIDTH_SHORT              0x1037
#define NXS_DEVICE_NATIVE_VECTOR_WIDTH_INT                0x1038
#define NXS_DEVICE_NATIVE_VECTOR_WIDTH_LONG               0x1039
#define NXS_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT              0x103A
#define NXS_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE             0x103B
#define NXS_DEVICE_NATIVE_VECTOR_WIDTH_HALF               0x103C
#define NXS_DEVICE_NEXUSAPI_C_VERSION                       0x103D
#endif
#ifdef NXS_VERSION_1_2
#define NXS_DEVICE_LINKER_AVAILABLE                       0x103E
#define NXS_DEVICE_BUILT_IN_KERNELS                       0x103F
#define NXS_DEVICE_IMAGE_MAX_BUFFER_SIZE                  0x1040
#define NXS_DEVICE_IMAGE_MAX_ARRAY_SIZE                   0x1041
#define NXS_DEVICE_PARENT_DEVICE                          0x1042
#define NXS_DEVICE_PARTITION_MAX_SUB_DEVICES              0x1043
#define NXS_DEVICE_PARTITION_PROPERTIES                   0x1044
#define NXS_DEVICE_PARTITION_AFFINITY_DOMAIN              0x1045
#define NXS_DEVICE_PARTITION_TYPE                         0x1046
#define NXS_DEVICE_REFERENCE_COUNT                        0x1047
#define NXS_DEVICE_PREFERRED_INTEROP_USER_SYNC            0x1048
#define NXS_DEVICE_PRINTF_BUFFER_SIZE                     0x1049
#endif
#ifdef NXS_VERSION_2_0
#define NXS_DEVICE_IMAGE_PITCH_ALIGNMENT                  0x104A
#define NXS_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT           0x104B
#define NXS_DEVICE_MAX_READ_WRITE_IMAGE_ARGS              0x104C
#define NXS_DEVICE_MAX_GLOBAL_VARIABLE_SIZE               0x104D
#define NXS_DEVICE_QUEUE_ON_DEVICE_PROPERTIES             0x104E
#define NXS_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE         0x104F
#define NXS_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE               0x1050
#define NXS_DEVICE_MAX_ON_DEVICE_QUEUES                   0x1051
#define NXS_DEVICE_MAX_ON_DEVICE_EVENTS                   0x1052
#define NXS_DEVICE_SVM_CAPABILITIES                       0x1053
#define NXS_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE   0x1054
#define NXS_DEVICE_MAX_PIPE_ARGS                          0x1055
#define NXS_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS           0x1056
#define NXS_DEVICE_PIPE_MAX_PACKET_SIZE                   0x1057
#define NXS_DEVICE_PREFERRED_RUNTIME_ATOMIC_ALIGNMENT    0x1058
#define NXS_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT      0x1059
#define NXS_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT       0x105A
#endif
#ifdef NXS_VERSION_2_1
#define NXS_DEVICE_IL_VERSION                             0x105B
#define NXS_DEVICE_MAX_NUM_SUB_GROUPS                     0x105C
#define NXS_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS 0x105D
#endif
#ifdef NXS_VERSION_3_0
#define NXS_DEVICE_NUMERIC_VERSION                        0x105E
#define NXS_DEVICE_EXTENSIONS_WITH_VERSION                0x1060
#define NXS_DEVICE_ILS_WITH_VERSION                       0x1061
#define NXS_DEVICE_BUILT_IN_KERNELS_WITH_VERSION          0x1062
#define NXS_DEVICE_ATOMIC_MEMORY_CAPABILITIES             0x1063
#define NXS_DEVICE_ATOMIC_FENCE_CAPABILITIES              0x1064
#define NXS_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT         0x1065
#define NXS_DEVICE_NEXUSAPI_C_ALL_VERSIONS                  0x1066
#define NXS_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE     0x1067
#define NXS_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT 0x1068
#define NXS_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT          0x1069
/* 0x106A to 0x106E - Reserved for upcoming KHR extension */
#define NXS_DEVICE_NEXUSAPI_C_FEATURES                      0x106F
#define NXS_DEVICE_DEVICE_ENQUEUE_CAPABILITIES            0x1070
#define NXS_DEVICE_PIPE_SUPPORT                           0x1071
#define NXS_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED      0x1072
#endif

/* nxs_device_fp_config - bitfield */
#define NXS_FP_DENORM                                (1 << 0)
#define NXS_FP_INF_NAN                               (1 << 1)
#define NXS_FP_ROUND_TO_NEAREST                      (1 << 2)
#define NXS_FP_ROUND_TO_ZERO                         (1 << 3)
#define NXS_FP_ROUND_TO_INF                          (1 << 4)
#define NXS_FP_FMA                                   (1 << 5)
#ifdef NXS_VERSION_1_1
#define NXS_FP_SOFT_FLOAT                            (1 << 6)
#endif
#ifdef NXS_VERSION_1_2
#define NXS_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT         (1 << 7)
#endif

/* nxs_device_mem_cache_type */
#define NXS_NONE                                     0x0
#define NXS_READ_ONLY_CACHE                          0x1
#define NXS_READ_WRITE_CACHE                         0x2

/* nxs_device_local_mem_type */
#define NXS_LOCAL                                    0x1
#define NXS_GLOBAL                                   0x2

/* nxs_device_exec_capabilities - bitfield */
#define NXS_EXEC_KERNEL                              (1 << 0)
#define NXS_EXEC_NATIVE_KERNEL                       (1 << 1)

/* nxs_command_queue_properties - bitfield */
#define NXS_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE      (1 << 0)
#define NXS_QUEUE_PROFILING_ENABLE                   (1 << 1)
#ifdef NXS_VERSION_2_0
#define NXS_QUEUE_ON_DEVICE                          (1 << 2)
#define NXS_QUEUE_ON_DEVICE_DEFAULT                  (1 << 3)
#endif

/* nxs_context_info */
#define NXS_CONTEXT_REFERENCE_COUNT                  0x1080
#define NXS_CONTEXT_DEVICES                          0x1081
#define NXS_CONTEXT_PROPERTIES                       0x1082
#ifdef NXS_VERSION_1_1
#define NXS_CONTEXT_NUM_DEVICES                      0x1083
#endif

/* nxs_context_properties */
#define NXS_CONTEXT_RUNTIME                         0x1084
#ifdef NXS_VERSION_1_2
#define NXS_CONTEXT_INTEROP_USER_SYNC                0x1085
#endif

#ifdef NXS_VERSION_1_2

/* nxs_device_partition_property */
#define NXS_DEVICE_PARTITION_EQUALLY                 0x1086
#define NXS_DEVICE_PARTITION_BY_COUNTS               0x1087
#define NXS_DEVICE_PARTITION_BY_COUNTS_LIST_END      0x0
#define NXS_DEVICE_PARTITION_BY_AFFINITY_DOMAIN      0x1088

#endif

#ifdef NXS_VERSION_1_2

/* nxs_device_affinity_domain */
#define NXS_DEVICE_AFFINITY_DOMAIN_NUMA               (1 << 0)
#define NXS_DEVICE_AFFINITY_DOMAIN_L4_CACHE           (1 << 1)
#define NXS_DEVICE_AFFINITY_DOMAIN_L3_CACHE           (1 << 2)
#define NXS_DEVICE_AFFINITY_DOMAIN_L2_CACHE           (1 << 3)
#define NXS_DEVICE_AFFINITY_DOMAIN_L1_CACHE           (1 << 4)
#define NXS_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE (1 << 5)

#endif

#ifdef NXS_VERSION_2_0

/* nxs_device_svm_capabilities */
#define NXS_DEVICE_SVM_COARSE_GRAIN_BUFFER           (1 << 0)
#define NXS_DEVICE_SVM_FINE_GRAIN_BUFFER             (1 << 1)
#define NXS_DEVICE_SVM_FINE_GRAIN_SYSTEM             (1 << 2)
#define NXS_DEVICE_SVM_ATOMICS                       (1 << 3)

#endif

/* nxs_command_queue_info */
#define NXS_QUEUE_CONTEXT                            0x1090
#define NXS_QUEUE_DEVICE                             0x1091
#define NXS_QUEUE_REFERENCE_COUNT                    0x1092
#define NXS_QUEUE_PROPERTIES                         0x1093
#ifdef NXS_VERSION_2_0
#define NXS_QUEUE_SIZE                               0x1094
#endif
#ifdef NXS_VERSION_2_1
#define NXS_QUEUE_DEVICE_DEFAULT                     0x1095
#endif
#ifdef NXS_VERSION_3_0
#define NXS_QUEUE_PROPERTIES_ARRAY                   0x1098
#endif

/* nxs_mem_flags and nxs_svm_mem_flags - bitfield */
#define NXS_MEM_READ_WRITE                           (1 << 0)
#define NXS_MEM_WRITE_ONLY                           (1 << 1)
#define NXS_MEM_READ_ONLY                            (1 << 2)
#define NXS_MEM_USE_HOST_PTR                         (1 << 3)
#define NXS_MEM_ALLOC_HOST_PTR                       (1 << 4)
#define NXS_MEM_COPY_HOST_PTR                        (1 << 5)
/* reserved                                         (1 << 6)    */
#ifdef NXS_VERSION_1_2
#define NXS_MEM_HOST_WRITE_ONLY                      (1 << 7)
#define NXS_MEM_HOST_READ_ONLY                       (1 << 8)
#define NXS_MEM_HOST_NO_ACCESS                       (1 << 9)
#endif
#ifdef NXS_VERSION_2_0
#define NXS_MEM_SVM_FINE_GRAIN_BUFFER                (1 << 10)   /* used by nxs_svm_mem_flags only */
#define NXS_MEM_SVM_ATOMICS                          (1 << 11)   /* used by nxs_svm_mem_flags only */
#define NXS_MEM_KERNEL_READ_AND_WRITE                (1 << 12)
#endif

#ifdef NXS_VERSION_1_2

/* nxs_mem_migration_flags - bitfield */
#define NXS_MIGRATE_MEM_OBJECT_HOST                  (1 << 0)
#define NXS_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED     (1 << 1)

#endif

/* nxs_channel_order */
#define NXS_R                                        0x10B0
#define NXS_A                                        0x10B1
#define NXS_RG                                       0x10B2
#define NXS_RA                                       0x10B3
#define NXS_RGB                                      0x10B4
#define NXS_RGBA                                     0x10B5
#define NXS_BGRA                                     0x10B6
#define NXS_ARGB                                     0x10B7
#define NXS_INTENSITY                                0x10B8
#define NXS_LUMINANCE                                0x10B9
#ifdef NXS_VERSION_1_1
#define NXS_Rx                                       0x10BA
#define NXS_RGx                                      0x10BB
#define NXS_RGBx                                     0x10BC
#endif
#ifdef NXS_VERSION_2_0
#define NXS_DEPTH                                    0x10BD
#define NXS_sRGB                                     0x10BF
#define NXS_sRGBx                                    0x10C0
#define NXS_sRGBA                                    0x10C1
#define NXS_sBGRA                                    0x10C2
#define NXS_ABGR                                     0x10C3
#endif

/* nxs_channel_type */
#define NXS_SNORM_INT8                               0x10D0
#define NXS_SNORM_INT16                              0x10D1
#define NXS_UNORM_INT8                               0x10D2
#define NXS_UNORM_INT16                              0x10D3
#define NXS_UNORM_SHORT_565                          0x10D4
#define NXS_UNORM_SHORT_555                          0x10D5
#define NXS_UNORM_INT_101010                         0x10D6
#define NXS_SIGNED_INT8                              0x10D7
#define NXS_SIGNED_INT16                             0x10D8
#define NXS_SIGNED_INT32                             0x10D9
#define NXS_UNSIGNED_INT8                            0x10DA
#define NXS_UNSIGNED_INT16                           0x10DB
#define NXS_UNSIGNED_INT32                           0x10DC
#define NXS_HALF_FLOAT                               0x10DD
#define NXS_FLOAT                                    0x10DE
#ifdef NXS_VERSION_2_1
#define NXS_UNORM_INT_101010_2                       0x10E0
#endif

/* nxs_mem_object_type */
#define NXS_MEM_OBJECT_BUFFER                        0x10F0
#define NXS_MEM_OBJECT_IMAGE2D                       0x10F1
#define NXS_MEM_OBJECT_IMAGE3D                       0x10F2
#ifdef NXS_VERSION_1_2
#define NXS_MEM_OBJECT_IMAGE2D_ARRAY                 0x10F3
#define NXS_MEM_OBJECT_IMAGE1D                       0x10F4
#define NXS_MEM_OBJECT_IMAGE1D_ARRAY                 0x10F5
#define NXS_MEM_OBJECT_IMAGE1D_BUFFER                0x10F6
#endif
#ifdef NXS_VERSION_2_0
#define NXS_MEM_OBJECT_PIPE                          0x10F7
#endif

/* nxs_mem_info */
#define NXS_MEM_TYPE                                 0x1100
#define NXS_MEM_FLAGS                                0x1101
#define NXS_MEM_SIZE                                 0x1102
#define NXS_MEM_HOST_PTR                             0x1103
#define NXS_MEM_MAP_COUNT                            0x1104
#define NXS_MEM_REFERENCE_COUNT                      0x1105
#define NXS_MEM_CONTEXT                              0x1106
#ifdef NXS_VERSION_1_1
#define NXS_MEM_ASSOCIATED_MEMOBJECT                 0x1107
#define NXS_MEM_OFFSET                               0x1108
#endif
#ifdef NXS_VERSION_2_0
#define NXS_MEM_USES_SVM_POINTER                     0x1109
#endif
#ifdef NXS_VERSION_3_0
#define NXS_MEM_PROPERTIES                           0x110A
#endif

/* nxs_image_info */
#define NXS_IMAGE_FORMAT                             0x1110
#define NXS_IMAGE_ELEMENT_SIZE                       0x1111
#define NXS_IMAGE_ROW_PITCH                          0x1112
#define NXS_IMAGE_SLICE_PITCH                        0x1113
#define NXS_IMAGE_WIDTH                              0x1114
#define NXS_IMAGE_HEIGHT                             0x1115
#define NXS_IMAGE_DEPTH                              0x1116
#ifdef NXS_VERSION_1_2
#define NXS_IMAGE_ARRAY_SIZE                         0x1117
#define NXS_IMAGE_BUFFER                             0x1118
#define NXS_IMAGE_NUM_MIP_LEVELS                     0x1119
#define NXS_IMAGE_NUM_SAMPLES                        0x111A
#endif


/* nxs_pipe_info */
#ifdef NXS_VERSION_2_0
#define NXS_PIPE_PACKET_SIZE                         0x1120
#define NXS_PIPE_MAX_PACKETS                         0x1121
#endif
#ifdef NXS_VERSION_3_0
#define NXS_PIPE_PROPERTIES                          0x1122
#endif

/* nxs_addressing_mode */
#define NXS_ADDRESS_NONE                             0x1130
#define NXS_ADDRESS_CLAMP_TO_EDGE                    0x1131
#define NXS_ADDRESS_CLAMP                            0x1132
#define NXS_ADDRESS_REPEAT                           0x1133
#ifdef NXS_VERSION_1_1
#define NXS_ADDRESS_MIRRORED_REPEAT                  0x1134
#endif

/* nxs_filter_mode */
#define NXS_FILTER_NEAREST                           0x1140
#define NXS_FILTER_LINEAR                            0x1141

/* nxs_sampler_info */
#define NXS_SAMPLER_REFERENCE_COUNT                  0x1150
#define NXS_SAMPLER_CONTEXT                          0x1151
#define NXS_SAMPLER_NORMALIZED_COORDS                0x1152
#define NXS_SAMPLER_ADDRESSING_MODE                  0x1153
#define NXS_SAMPLER_FILTER_MODE                      0x1154
#ifdef NXS_VERSION_2_0
/* These enumerants are for the nxs_khr_mipmap_image extension.
   They have since been added to nxs_ext.h with an appropriate
   KHR suffix, but are left here for backwards compatibility. */
#define NXS_SAMPLER_MIP_FILTER_MODE                  0x1155
#define NXS_SAMPLER_LOD_MIN                          0x1156
#define NXS_SAMPLER_LOD_MAX                          0x1157
#endif
#ifdef NXS_VERSION_3_0
#define NXS_SAMPLER_PROPERTIES                       0x1158
#endif

/* nxs_map_flags - bitfield */
#define NXS_MAP_READ                                 (1 << 0)
#define NXS_MAP_WRITE                                (1 << 1)
#ifdef NXS_VERSION_1_2
#define NXS_MAP_WRITE_INVALIDATE_REGION              (1 << 2)
#endif

/* nxs_program_info */
#define NXS_PROGRAM_REFERENCE_COUNT                  0x1160
#define NXS_PROGRAM_CONTEXT                          0x1161
#define NXS_PROGRAM_NUM_DEVICES                      0x1162
#define NXS_PROGRAM_DEVICES                          0x1163
#define NXS_PROGRAM_SOURCE                           0x1164
#define NXS_PROGRAM_BINARY_SIZES                     0x1165
#define NXS_PROGRAM_BINARIES                         0x1166
#ifdef NXS_VERSION_1_2
#define NXS_PROGRAM_NUM_KERNELS                      0x1167
#define NXS_PROGRAM_KERNEL_NAMES                     0x1168
#endif
#ifdef NXS_VERSION_2_1
#define NXS_PROGRAM_IL                               0x1169
#endif
#ifdef NXS_VERSION_2_2
#define NXS_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT       0x116A
#define NXS_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT       0x116B
#endif

/* nxs_program_build_info */
#define NXS_PROGRAM_BUILD_STATUS                     0x1181
#define NXS_PROGRAM_BUILD_OPTIONS                    0x1182
#define NXS_PROGRAM_BUILD_LOG                        0x1183
#ifdef NXS_VERSION_1_2
#define NXS_PROGRAM_BINARY_TYPE                      0x1184
#endif
#ifdef NXS_VERSION_2_0
#define NXS_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE 0x1185
#endif

#ifdef NXS_VERSION_1_2

/* nxs_program_binary_type */
#define NXS_PROGRAM_BINARY_TYPE_NONE                 0x0
#define NXS_PROGRAM_BINARY_TYPE_COMPILED_OBJECT      0x1
#define NXS_PROGRAM_BINARY_TYPE_LIBRARY              0x2
#define NXS_PROGRAM_BINARY_TYPE_EXECUTABLE           0x4

#endif

/* nxs_build_status */
#define NXS_BUILD_SUCCESS                            0
#define NXS_BUILD_NONE                               -1
#define NXS_BUILD_ERROR                              -2
#define NXS_BUILD_IN_PROGRESS                        -3

/* nxs_kernel_info */
#define NXS_KERNEL_FUNCTION_NAME                     0x1190
#define NXS_KERNEL_NUM_ARGS                          0x1191
#define NXS_KERNEL_REFERENCE_COUNT                   0x1192
#define NXS_KERNEL_CONTEXT                           0x1193
#define NXS_KERNEL_PROGRAM                           0x1194
#ifdef NXS_VERSION_1_2
#define NXS_KERNEL_ATTRIBUTES                        0x1195
#endif

#ifdef NXS_VERSION_1_2

/* nxs_kernel_arg_info */
#define NXS_KERNEL_ARG_ADDRESS_QUALIFIER             0x1196
#define NXS_KERNEL_ARG_ACCESS_QUALIFIER              0x1197
#define NXS_KERNEL_ARG_TYPE_NAME                     0x1198
#define NXS_KERNEL_ARG_TYPE_QUALIFIER                0x1199
#define NXS_KERNEL_ARG_NAME                          0x119A

#endif

#ifdef NXS_VERSION_1_2

/* nxs_kernel_arg_address_qualifier */
#define NXS_KERNEL_ARG_ADDRESS_GLOBAL                0x119B
#define NXS_KERNEL_ARG_ADDRESS_LOCAL                 0x119C
#define NXS_KERNEL_ARG_ADDRESS_CONSTANT              0x119D
#define NXS_KERNEL_ARG_ADDRESS_PRIVATE               0x119E

#endif

#ifdef NXS_VERSION_1_2

/* nxs_kernel_arg_access_qualifier */
#define NXS_KERNEL_ARG_ACCESS_READ_ONLY              0x11A0
#define NXS_KERNEL_ARG_ACCESS_WRITE_ONLY             0x11A1
#define NXS_KERNEL_ARG_ACCESS_READ_WRITE             0x11A2
#define NXS_KERNEL_ARG_ACCESS_NONE                   0x11A3

#endif

#ifdef NXS_VERSION_1_2

/* nxs_kernel_arg_type_qualifier */
#define NXS_KERNEL_ARG_TYPE_NONE                     0
#define NXS_KERNEL_ARG_TYPE_CONST                    (1 << 0)
#define NXS_KERNEL_ARG_TYPE_RESTRICT                 (1 << 1)
#define NXS_KERNEL_ARG_TYPE_VOLATILE                 (1 << 2)
#ifdef NXS_VERSION_2_0
#define NXS_KERNEL_ARG_TYPE_PIPE                     (1 << 3)
#endif

#endif

/* nxs_kernel_work_group_info */
#define NXS_KERNEL_WORK_GROUP_SIZE                   0x11B0
#define NXS_KERNEL_COMPILE_WORK_GROUP_SIZE           0x11B1
#define NXS_KERNEL_LOCAL_MEM_SIZE                    0x11B2
#define NXS_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE 0x11B3
#define NXS_KERNEL_PRIVATE_MEM_SIZE                  0x11B4
#ifdef NXS_VERSION_1_2
#define NXS_KERNEL_GLOBAL_WORK_SIZE                  0x11B5
#endif

#ifdef NXS_VERSION_2_1

/* nxs_kernel_sub_group_info */
#define NXS_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE    0x2033
#define NXS_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE       0x2034
#define NXS_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT    0x11B8
#define NXS_KERNEL_MAX_NUM_SUB_GROUPS                0x11B9
#define NXS_KERNEL_COMPILE_NUM_SUB_GROUPS            0x11BA

#endif

#ifdef NXS_VERSION_2_0

/* nxs_kernel_exec_info */
#define NXS_KERNEL_EXEC_INFO_SVM_PTRS                0x11B6
#define NXS_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM   0x11B7

#endif

/* nxs_event_info */
#define NXS_EVENT_COMMAND_QUEUE                      0x11D0
#define NXS_EVENT_COMMAND_TYPE                       0x11D1
#define NXS_EVENT_REFERENCE_COUNT                    0x11D2
#define NXS_EVENT_COMMAND_EXECUTION_STATUS           0x11D3
#ifdef NXS_VERSION_1_1
#define NXS_EVENT_CONTEXT                            0x11D4
#endif

/* nxs_command_type */
#define NXS_COMMAND_NDRANGE_KERNEL                   0x11F0
#define NXS_COMMAND_TASK                             0x11F1
#define NXS_COMMAND_NATIVE_KERNEL                    0x11F2
#define NXS_COMMAND_READ_BUFFER                      0x11F3
#define NXS_COMMAND_WRITE_BUFFER                     0x11F4
#define NXS_COMMAND_COPY_BUFFER                      0x11F5
#define NXS_COMMAND_READ_IMAGE                       0x11F6
#define NXS_COMMAND_WRITE_IMAGE                      0x11F7
#define NXS_COMMAND_COPY_IMAGE                       0x11F8
#define NXS_COMMAND_COPY_IMAGE_TO_BUFFER             0x11F9
#define NXS_COMMAND_COPY_BUFFER_TO_IMAGE             0x11FA
#define NXS_COMMAND_MAP_BUFFER                       0x11FB
#define NXS_COMMAND_MAP_IMAGE                        0x11FC
#define NXS_COMMAND_UNMAP_MEM_OBJECT                 0x11FD
#define NXS_COMMAND_MARKER                           0x11FE
#define NXS_COMMAND_ACQUIRE_GL_OBJECTS               0x11FF
#define NXS_COMMAND_RELEASE_GL_OBJECTS               0x1200
#ifdef NXS_VERSION_1_1
#define NXS_COMMAND_READ_BUFFER_RECT                 0x1201
#define NXS_COMMAND_WRITE_BUFFER_RECT                0x1202
#define NXS_COMMAND_COPY_BUFFER_RECT                 0x1203
#define NXS_COMMAND_USER                             0x1204
#endif
#ifdef NXS_VERSION_1_2
#define NXS_COMMAND_BARRIER                          0x1205
#define NXS_COMMAND_MIGRATE_MEM_OBJECTS              0x1206
#define NXS_COMMAND_FILL_BUFFER                      0x1207
#define NXS_COMMAND_FILL_IMAGE                       0x1208
#endif
#ifdef NXS_VERSION_2_0
#define NXS_COMMAND_SVM_FREE                         0x1209
#define NXS_COMMAND_SVM_MEMCPY                       0x120A
#define NXS_COMMAND_SVM_MEMFILL                      0x120B
#define NXS_COMMAND_SVM_MAP                          0x120C
#define NXS_COMMAND_SVM_UNMAP                        0x120D
#endif
#ifdef NXS_VERSION_3_0
#define NXS_COMMAND_SVM_MIGRATE_MEM                  0x120E
#endif

/* command execution status */
#define NXS_COMPLETE                                 0x0
#define NXS_RUNNING                                  0x1
#define NXS_SUBMITTED                                0x2
#define NXS_QUEUED                                   0x3

/* nxs_buffer_create_type */
#ifdef NXS_VERSION_1_1
#define NXS_BUFFER_CREATE_TYPE_REGION                0x1220
#endif

/* nxs_profiling_info */
#define NXS_PROFILING_COMMAND_QUEUED                 0x1280
#define NXS_PROFILING_COMMAND_SUBMIT                 0x1281
#define NXS_PROFILING_COMMAND_START                  0x1282
#define NXS_PROFILING_COMMAND_END                    0x1283
#ifdef NXS_VERSION_2_0
#define NXS_PROFILING_COMMAND_COMPLETE               0x1284
#endif

/* nxs_device_atomic_capabilities - bitfield */
#ifdef NXS_VERSION_3_0
#define NXS_DEVICE_ATOMIC_ORDER_RELAXED          (1 << 0)
#define NXS_DEVICE_ATOMIC_ORDER_ACQ_REL          (1 << 1)
#define NXS_DEVICE_ATOMIC_ORDER_SEQ_CST          (1 << 2)
#define NXS_DEVICE_ATOMIC_SCOPE_WORK_ITEM        (1 << 3)
#define NXS_DEVICE_ATOMIC_SCOPE_WORK_GROUP       (1 << 4)
#define NXS_DEVICE_ATOMIC_SCOPE_DEVICE           (1 << 5)
#define NXS_DEVICE_ATOMIC_SCOPE_ALL_DEVICES      (1 << 6)
#endif

/* nxs_device_device_enqueue_capabilities - bitfield */
#ifdef NXS_VERSION_3_0
#define NXS_DEVICE_QUEUE_SUPPORTED               (1 << 0)
#define NXS_DEVICE_QUEUE_REPLACEABLE_DEFAULT     (1 << 1)
#endif

/* nxs_khronos_vendor_id */
#define NXS_KHRONOS_VENDOR_ID_CODEPLAY               0x10004

/* nxs_version */
#define NXS_VERSION_MAJOR_BITS (10)
#define NXS_VERSION_MINOR_BITS (10)
#define NXS_VERSION_PATCH_BITS (12)

#define NXS_VERSION_MAJOR_MASK ((1 << NXS_VERSION_MAJOR_BITS) - 1)
#define NXS_VERSION_MINOR_MASK ((1 << NXS_VERSION_MINOR_BITS) - 1)
#define NXS_VERSION_PATCH_MASK ((1 << NXS_VERSION_PATCH_BITS) - 1)

#define NXS_VERSION_MAJOR(version) \
  ((version) >> (NXS_VERSION_MINOR_BITS + NXS_VERSION_PATCH_BITS))

#define NXS_VERSION_MINOR(version) \
  (((version) >> NXS_VERSION_PATCH_BITS) & NXS_VERSION_MINOR_MASK)

#define NXS_VERSION_PATCH(version) ((version) & NXS_VERSION_PATCH_MASK)

#define NXS_MAKE_VERSION(major, minor, patch)                      \
  ((((major) & NXS_VERSION_MAJOR_MASK)                             \
       << (NXS_VERSION_MINOR_BITS + NXS_VERSION_PATCH_BITS)) |      \
   (((minor) & NXS_VERSION_MINOR_MASK) << NXS_VERSION_PATCH_BITS) | \
   ((patch) & NXS_VERSION_PATCH_MASK))

/********************************************************************************************************/

#ifdef __cplusplus
}
#endif

#endif  /* __NEXUSAPI_NXS_H */
