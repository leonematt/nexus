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
#define NXS_SUCCESS                                  0
#define NXS_DEVICE_NOT_FOUND                         -1
#define NXS_DEVICE_NOT_AVAILABLE                     -2
#define NXS_COMPILER_NOT_AVAILABLE                   -3
#define NXS_MEM_OBJECT_ALLOCATION_FAILURE            -4
#define NXS_OUT_OF_RESOURCES                         -5
#define NXS_OUT_OF_HOST_MEMORY                       -6
#define NXS_PROFILING_INFO_NOT_AVAILABLE             -7
#define NXS_MEM_COPY_OVERLAP                         -8
#define NXS_IMAGE_FORMAT_MISMATCH                    -9
#define NXS_IMAGE_FORMAT_NOT_SUPPORTED               -10
#define NXS_BUILD_PROGRAM_FAILURE                    -11
#define NXS_MAP_FAILURE                              -12
#ifdef NXS_VERSION_1_1
#define NXS_MISALIGNED_SUB_BUFFER_OFFSET             -13
#define NXS_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST -14
#endif
#ifdef NXS_VERSION_1_2
#define NXS_COMPILE_PROGRAM_FAILURE                  -15
#define NXS_LINKER_NOT_AVAILABLE                     -16
#define NXS_LINK_PROGRAM_FAILURE                     -17
#define NXS_DEVICE_PARTITION_FAILED                  -18
#define NXS_KERNEL_ARG_INFO_NOT_AVAILABLE            -19
#endif

#define NXS_INVALID_VALUE                            -30
#define NXS_INVALID_DEVICE_TYPE                      -31
#define NXS_INVALID_RUNTIME                         -32
#define NXS_INVALID_DEVICE                           -33
#define NXS_INVALID_CONTEXT                          -34
#define NXS_INVALID_QUEUE_PROPERTIES                 -35
#define NXS_INVALID_COMMAND_QUEUE                    -36
#define NXS_INVALID_HOST_PTR                         -37
#define NXS_INVALID_MEM_OBJECT                       -38
#define NXS_INVALID_IMAGE_FORMAT_DESCRIPTOR          -39
#define NXS_INVALID_IMAGE_SIZE                       -40
#define NXS_INVALID_SAMPLER                          -41
#define NXS_INVALID_BINARY                           -42
#define NXS_INVALID_BUILD_OPTIONS                    -43
#define NXS_INVALID_PROGRAM                          -44
#define NXS_INVALID_PROGRAM_EXECUTABLE               -45
#define NXS_INVALID_KERNEL_NAME                      -46
#define NXS_INVALID_KERNEL_DEFINITION                -47
#define NXS_INVALID_KERNEL                           -48
#define NXS_INVALID_ARG_INDEX                        -49
#define NXS_INVALID_ARG_VALUE                        -50
#define NXS_INVALID_ARG_SIZE                         -51
#define NXS_INVALID_KERNEL_ARGS                      -52
#define NXS_INVALID_WORK_DIMENSION                   -53
#define NXS_INVALID_WORK_GROUP_SIZE                  -54
#define NXS_INVALID_WORK_ITEM_SIZE                   -55
#define NXS_INVALID_GLOBAL_OFFSET                    -56
#define NXS_INVALID_EVENT_WAIT_LIST                  -57
#define NXS_INVALID_EVENT                            -58
#define NXS_INVALID_OPERATION                        -59
#define NXS_INVALID_GL_OBJECT                        -60
#define NXS_INVALID_BUFFER_SIZE                      -61
#define NXS_INVALID_MIP_LEVEL                        -62
#define NXS_INVALID_GLOBAL_WORK_SIZE                 -63
#ifdef NXS_VERSION_1_1
#define NXS_INVALID_PROPERTY                         -64
#endif
#ifdef NXS_VERSION_1_2
#define NXS_INVALID_IMAGE_DESCRIPTOR                 -65
#define NXS_INVALID_COMPILER_OPTIONS                 -66
#define NXS_INVALID_LINKER_OPTIONS                   -67
#define NXS_INVALID_DEVICE_PARTITION_COUNT           -68
#endif
#ifdef NXS_VERSION_2_0
#define NXS_INVALID_PIPE_SIZE                        -69
#define NXS_INVALID_DEVICE_QUEUE                     -70
#endif
#ifdef NXS_VERSION_2_2
#define NXS_INVALID_SPEC_ID                          -71
#define NXS_MAX_SIZE_RESTRICTION_EXCEEDED            -72
#endif


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

/* NXS_NO_PROTOTYPES implies NXS_NO_CORE_PROTOTYPES: */
#if defined(NXS_NO_PROTOTYPES) && !defined(NXS_NO_CORE_PROTOTYPES)
#define NXS_NO_CORE_PROTOTYPES
#endif

#if !defined(NXS_NO_CORE_PROTOTYPES)

/* Platform API */
extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsGetPlatformIDs(nxs_uint          num_entries,
                 nxs_platform_id * platforms,
                 nxs_uint *        num_platforms) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsGetPlatformInfo(nxs_platform_id   platform,
                  nxs_platform_info param_name,
                  size_t           param_value_size,
                  void *           param_value,
                  size_t *         param_value_size_ret) NXS_API_SUFFIX__VERSION_1_0;

/* Device APIs */
extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsGetDeviceIDs(nxs_platform_id   platform,
               nxs_device_type   device_type,
               nxs_uint          num_entries,
               nxs_device_id *   devices,
               nxs_uint *        num_devices) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsGetDeviceInfo(nxs_device_id    device,
                nxs_device_info  param_name,
                size_t          param_value_size,
                void *          param_value,
                size_t *        param_value_size_ret) NXS_API_SUFFIX__VERSION_1_0;

#ifdef NXS_VERSION_1_2

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsCreateSubDevices(nxs_device_id                         in_device,
                   const nxs_device_partition_property * properties,
                   nxs_uint                              num_devices,
                   nxs_device_id *                       out_devices,
                   nxs_uint *                            num_devices_ret) NXS_API_SUFFIX__VERSION_1_2;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsRetainDevice(nxs_device_id device) NXS_API_SUFFIX__VERSION_1_2;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsReleaseDevice(nxs_device_id device) NXS_API_SUFFIX__VERSION_1_2;

#endif

#ifdef NXS_VERSION_2_1

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsSetDefaultDeviceCommandQueue(nxs_context           context,
                               nxs_device_id         device,
                               nxs_command_queue     command_queue) NXS_API_SUFFIX__VERSION_2_1;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsGetDeviceAndHostTimer(nxs_device_id    device,
                        nxs_ulong*       device_timestamp,
                        nxs_ulong*       host_timestamp) NXS_API_SUFFIX__VERSION_2_1;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsGetHostTimer(nxs_device_id device,
               nxs_ulong *   host_timestamp) NXS_API_SUFFIX__VERSION_2_1;

#endif

/* Context APIs */
extern NXS_API_ENTRY nxs_context NXS_API_CALL
nxsCreateContext(const nxs_context_properties * properties,
                nxs_uint              num_devices,
                const nxs_device_id * devices,
                void (NXS_CALLBACK * pfn_notify)(const char * errinfo,
                                                const void * private_info,
                                                size_t       cb,
                                                void *       user_data),
                void *               user_data,
                nxs_int *             errcode_ret) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_context NXS_API_CALL
nxsCreateContextFromType(const nxs_context_properties * properties,
                        nxs_device_type      device_type,
                        void (NXS_CALLBACK * pfn_notify)(const char * errinfo,
                                                        const void * private_info,
                                                        size_t       cb,
                                                        void *       user_data),
                        void *              user_data,
                        nxs_int *            errcode_ret) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsRetainContext(nxs_context context) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsReleaseContext(nxs_context context) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsGetContextInfo(nxs_context         context,
                 nxs_context_info    param_name,
                 size_t             param_value_size,
                 void *             param_value,
                 size_t *           param_value_size_ret) NXS_API_SUFFIX__VERSION_1_0;

#ifdef NXS_VERSION_3_0

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsSetContextDestructorCallback(nxs_context         context,
                               void (NXS_CALLBACK* pfn_notify)(nxs_context context,
                                                              void* user_data),
                               void*              user_data) NXS_API_SUFFIX__VERSION_3_0;

#endif

/* Command Queue APIs */

#ifdef NXS_VERSION_2_0

extern NXS_API_ENTRY nxs_command_queue NXS_API_CALL
nxsCreateCommandQueueWithProperties(nxs_context               context,
                                   nxs_device_id             device,
                                   const nxs_queue_properties *    properties,
                                   nxs_int *                 errcode_ret) NXS_API_SUFFIX__VERSION_2_0;

#endif

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsRetainCommandQueue(nxs_command_queue command_queue) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsReleaseCommandQueue(nxs_command_queue command_queue) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsGetCommandQueueInfo(nxs_command_queue      command_queue,
                      nxs_command_queue_info param_name,
                      size_t                param_value_size,
                      void *                param_value,
                      size_t *              param_value_size_ret) NXS_API_SUFFIX__VERSION_1_0;

/* Memory Object APIs */
extern NXS_API_ENTRY nxs_mem NXS_API_CALL
nxsCreateBuffer(nxs_context   context,
               nxs_mem_flags flags,
               size_t       size,
               void *       host_ptr,
               nxs_int *     errcode_ret) NXS_API_SUFFIX__VERSION_1_0;

#ifdef NXS_VERSION_1_1

extern NXS_API_ENTRY nxs_mem NXS_API_CALL
nxsCreateSubBuffer(nxs_mem                   buffer,
                  nxs_mem_flags             flags,
                  nxs_buffer_create_type    buffer_create_type,
                  const void *             buffer_create_info,
                  nxs_int *                 errcode_ret) NXS_API_SUFFIX__VERSION_1_1;

#endif

#ifdef NXS_VERSION_1_2

extern NXS_API_ENTRY nxs_mem NXS_API_CALL
nxsCreateImage(nxs_context              context,
              nxs_mem_flags            flags,
              const nxs_image_format * image_format,
              const nxs_image_desc *   image_desc,
              void *                  host_ptr,
              nxs_int *                errcode_ret) NXS_API_SUFFIX__VERSION_1_2;

#endif

#ifdef NXS_VERSION_2_0

extern NXS_API_ENTRY nxs_mem NXS_API_CALL
nxsCreatePipe(nxs_context                 context,
             nxs_mem_flags               flags,
             nxs_uint                    pipe_packet_size,
             nxs_uint                    pipe_max_packets,
             const nxs_pipe_properties * properties,
             nxs_int *                   errcode_ret) NXS_API_SUFFIX__VERSION_2_0;

#endif

#ifdef NXS_VERSION_3_0

extern NXS_API_ENTRY nxs_mem NXS_API_CALL
nxsCreateBufferWithProperties(nxs_context                context,
                             const nxs_mem_properties * properties,
                             nxs_mem_flags              flags,
                             size_t                    size,
                             void *                    host_ptr,
                             nxs_int *                  errcode_ret) NXS_API_SUFFIX__VERSION_3_0;

extern NXS_API_ENTRY nxs_mem NXS_API_CALL
nxsCreateImageWithProperties(nxs_context                context,
                            const nxs_mem_properties * properties,
                            nxs_mem_flags              flags,
                            const nxs_image_format *   image_format,
                            const nxs_image_desc *     image_desc,
                            void *                    host_ptr,
                            nxs_int *                  errcode_ret) NXS_API_SUFFIX__VERSION_3_0;

#endif

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsRetainMemObject(nxs_mem memobj) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsReleaseMemObject(nxs_mem memobj) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsGetSupportedImageFormats(nxs_context           context,
                           nxs_mem_flags         flags,
                           nxs_mem_object_type   image_type,
                           nxs_uint              num_entries,
                           nxs_image_format *    image_formats,
                           nxs_uint *            num_image_formats) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsGetMemObjectInfo(nxs_mem           memobj,
                   nxs_mem_info      param_name,
                   size_t           param_value_size,
                   void *           param_value,
                   size_t *         param_value_size_ret) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsGetImageInfo(nxs_mem           image,
               nxs_image_info    param_name,
               size_t           param_value_size,
               void *           param_value,
               size_t *         param_value_size_ret) NXS_API_SUFFIX__VERSION_1_0;

#ifdef NXS_VERSION_2_0

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsGetPipeInfo(nxs_mem           pipe,
              nxs_pipe_info     param_name,
              size_t           param_value_size,
              void *           param_value,
              size_t *         param_value_size_ret) NXS_API_SUFFIX__VERSION_2_0;

#endif

#ifdef NXS_VERSION_1_1

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsSetMemObjectDestructorCallback(nxs_mem memobj,
                                 void (NXS_CALLBACK * pfn_notify)(nxs_mem memobj,
                                                                 void * user_data),
                                 void * user_data) NXS_API_SUFFIX__VERSION_1_1;

#endif

/* SVM Allocation APIs */

#ifdef NXS_VERSION_2_0

extern NXS_API_ENTRY void * NXS_API_CALL
nxsSVMAlloc(nxs_context       context,
           nxs_svm_mem_flags flags,
           size_t           size,
           nxs_uint          alignment) NXS_API_SUFFIX__VERSION_2_0;

extern NXS_API_ENTRY void NXS_API_CALL
nxsSVMFree(nxs_context        context,
          void *            svm_pointer) NXS_API_SUFFIX__VERSION_2_0;

#endif

/* Sampler APIs */

#ifdef NXS_VERSION_2_0

extern NXS_API_ENTRY nxs_sampler NXS_API_CALL
nxsCreateSamplerWithProperties(nxs_context                     context,
                              const nxs_sampler_properties *  sampler_properties,
                              nxs_int *                       errcode_ret) NXS_API_SUFFIX__VERSION_2_0;

#endif

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsRetainSampler(nxs_sampler sampler) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsReleaseSampler(nxs_sampler sampler) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsGetSamplerInfo(nxs_sampler         sampler,
                 nxs_sampler_info    param_name,
                 size_t             param_value_size,
                 void *             param_value,
                 size_t *           param_value_size_ret) NXS_API_SUFFIX__VERSION_1_0;

/* Program Object APIs */
extern NXS_API_ENTRY nxs_program NXS_API_CALL
nxsCreateProgramWithSource(nxs_context        context,
                          nxs_uint           count,
                          const char **     strings,
                          const size_t *    lengths,
                          nxs_int *          errcode_ret) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_program NXS_API_CALL
nxsCreateProgramWithBinary(nxs_context                     context,
                          nxs_uint                        num_devices,
                          const nxs_device_id *           device_list,
                          const size_t *                 lengths,
                          const unsigned char **         binaries,
                          nxs_int *                       binary_status,
                          nxs_int *                       errcode_ret) NXS_API_SUFFIX__VERSION_1_0;

#ifdef NXS_VERSION_1_2

extern NXS_API_ENTRY nxs_program NXS_API_CALL
nxsCreateProgramWithBuiltInKernels(nxs_context            context,
                                  nxs_uint               num_devices,
                                  const nxs_device_id *  device_list,
                                  const char *          kernel_names,
                                  nxs_int *              errcode_ret) NXS_API_SUFFIX__VERSION_1_2;

#endif

#ifdef NXS_VERSION_2_1

extern NXS_API_ENTRY nxs_program NXS_API_CALL
nxsCreateProgramWithIL(nxs_context    context,
                     const void*    il,
                     size_t         length,
                     nxs_int*        errcode_ret) NXS_API_SUFFIX__VERSION_2_1;

#endif

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsRetainProgram(nxs_program program) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsReleaseProgram(nxs_program program) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsBuildProgram(nxs_program           program,
               nxs_uint              num_devices,
               const nxs_device_id * device_list,
               const char *         options,
               void (NXS_CALLBACK *  pfn_notify)(nxs_program program,
                                                void * user_data),
               void *               user_data) NXS_API_SUFFIX__VERSION_1_0;

#ifdef NXS_VERSION_1_2

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsCompileProgram(nxs_program           program,
                 nxs_uint              num_devices,
                 const nxs_device_id * device_list,
                 const char *         options,
                 nxs_uint              num_input_headers,
                 const nxs_program *   input_headers,
                 const char **        header_include_names,
                 void (NXS_CALLBACK *  pfn_notify)(nxs_program program,
                                                  void * user_data),
                 void *               user_data) NXS_API_SUFFIX__VERSION_1_2;

extern NXS_API_ENTRY nxs_program NXS_API_CALL
nxsLinkProgram(nxs_context           context,
              nxs_uint              num_devices,
              const nxs_device_id * device_list,
              const char *         options,
              nxs_uint              num_input_programs,
              const nxs_program *   input_programs,
              void (NXS_CALLBACK *  pfn_notify)(nxs_program program,
                                               void * user_data),
              void *               user_data,
              nxs_int *             errcode_ret) NXS_API_SUFFIX__VERSION_1_2;

#endif

#ifdef NXS_VERSION_2_2

extern NXS_API_ENTRY NXS_API_PREFIX__VERSION_2_2_DEPRECATED nxs_int NXS_API_CALL
nxsSetProgramReleaseCallback(nxs_program          program,
                            void (NXS_CALLBACK * pfn_notify)(nxs_program program,
                                                            void * user_data),
                            void *              user_data) NXS_API_SUFFIX__VERSION_2_2_DEPRECATED;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsSetProgramSpecializationConstant(nxs_program  program,
                                   nxs_uint     spec_id,
                                   size_t      spec_size,
                                   const void* spec_value) NXS_API_SUFFIX__VERSION_2_2;

#endif

#ifdef NXS_VERSION_1_2

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsUnloadPlatformCompiler(nxs_platform_id platform) NXS_API_SUFFIX__VERSION_1_2;

#endif

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsGetProgramInfo(nxs_program         program,
                 nxs_program_info    param_name,
                 size_t             param_value_size,
                 void *             param_value,
                 size_t *           param_value_size_ret) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsGetProgramBuildInfo(nxs_program            program,
                      nxs_device_id          device,
                      nxs_program_build_info param_name,
                      size_t                param_value_size,
                      void *                param_value,
                      size_t *              param_value_size_ret) NXS_API_SUFFIX__VERSION_1_0;

/* Kernel Object APIs */
extern NXS_API_ENTRY nxs_kernel NXS_API_CALL
nxsCreateKernel(nxs_program      program,
               const char *    kernel_name,
               nxs_int *        errcode_ret) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsCreateKernelsInProgram(nxs_program     program,
                         nxs_uint        num_kernels,
                         nxs_kernel *    kernels,
                         nxs_uint *      num_kernels_ret) NXS_API_SUFFIX__VERSION_1_0;

#ifdef NXS_VERSION_2_1

extern NXS_API_ENTRY nxs_kernel NXS_API_CALL
nxsCloneKernel(nxs_kernel     source_kernel,
              nxs_int*       errcode_ret) NXS_API_SUFFIX__VERSION_2_1;

#endif

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsRetainKernel(nxs_kernel    kernel) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsReleaseKernel(nxs_kernel   kernel) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsSetKernelArg(nxs_kernel    kernel,
               nxs_uint      arg_index,
               size_t       arg_size,
               const void * arg_value) NXS_API_SUFFIX__VERSION_1_0;

#ifdef NXS_VERSION_2_0

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsSetKernelArgSVMPointer(nxs_kernel    kernel,
                         nxs_uint      arg_index,
                         const void * arg_value) NXS_API_SUFFIX__VERSION_2_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsSetKernelExecInfo(nxs_kernel            kernel,
                    nxs_kernel_exec_info  param_name,
                    size_t               param_value_size,
                    const void *         param_value) NXS_API_SUFFIX__VERSION_2_0;

#endif

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsGetKernelInfo(nxs_kernel       kernel,
                nxs_kernel_info  param_name,
                size_t          param_value_size,
                void *          param_value,
                size_t *        param_value_size_ret) NXS_API_SUFFIX__VERSION_1_0;

#ifdef NXS_VERSION_1_2

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsGetKernelArgInfo(nxs_kernel       kernel,
                   nxs_uint         arg_indx,
                   nxs_kernel_arg_info  param_name,
                   size_t          param_value_size,
                   void *          param_value,
                   size_t *        param_value_size_ret) NXS_API_SUFFIX__VERSION_1_2;

#endif

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsGetKernelWorkGroupInfo(nxs_kernel                  kernel,
                         nxs_device_id               device,
                         nxs_kernel_work_group_info  param_name,
                         size_t                     param_value_size,
                         void *                     param_value,
                         size_t *                   param_value_size_ret) NXS_API_SUFFIX__VERSION_1_0;

#ifdef NXS_VERSION_2_1

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsGetKernelSubGroupInfo(nxs_kernel                   kernel,
                        nxs_device_id                device,
                        nxs_kernel_sub_group_info    param_name,
                        size_t                      input_value_size,
                        const void*                 input_value,
                        size_t                      param_value_size,
                        void*                       param_value,
                        size_t*                     param_value_size_ret) NXS_API_SUFFIX__VERSION_2_1;

#endif

/* Event Object APIs */
extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsWaitForEvents(nxs_uint             num_events,
                const nxs_event *    event_list) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsGetEventInfo(nxs_event         event,
               nxs_event_info    param_name,
               size_t           param_value_size,
               void *           param_value,
               size_t *         param_value_size_ret) NXS_API_SUFFIX__VERSION_1_0;

#ifdef NXS_VERSION_1_1

extern NXS_API_ENTRY nxs_event NXS_API_CALL
nxsCreateUserEvent(nxs_context    context,
                  nxs_int *      errcode_ret) NXS_API_SUFFIX__VERSION_1_1;

#endif

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsRetainEvent(nxs_event event) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsReleaseEvent(nxs_event event) NXS_API_SUFFIX__VERSION_1_0;

#ifdef NXS_VERSION_1_1

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsSetUserEventStatus(nxs_event   event,
                     nxs_int     execution_status) NXS_API_SUFFIX__VERSION_1_1;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsSetEventCallback(nxs_event    event,
                   nxs_int      command_exec_callback_type,
                   void (NXS_CALLBACK * pfn_notify)(nxs_event event,
                                                   nxs_int   event_command_status,
                                                   void *   user_data),
                   void *      user_data) NXS_API_SUFFIX__VERSION_1_1;

#endif

/* Profiling APIs */
extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsGetEventProfilingInfo(nxs_event            event,
                        nxs_profiling_info   param_name,
                        size_t              param_value_size,
                        void *              param_value,
                        size_t *            param_value_size_ret) NXS_API_SUFFIX__VERSION_1_0;

/* Flush and Finish APIs */
extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsFlush(nxs_command_queue command_queue) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsFinish(nxs_command_queue command_queue) NXS_API_SUFFIX__VERSION_1_0;

/* Enqueued Commands APIs */
extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueReadBuffer(nxs_command_queue    command_queue,
                    nxs_mem              buffer,
                    nxs_bool             blocking_read,
                    size_t              offset,
                    size_t              size,
                    void *              ptr,
                    nxs_uint             num_events_in_wait_list,
                    const nxs_event *    event_wait_list,
                    nxs_event *          event) NXS_API_SUFFIX__VERSION_1_0;

#ifdef NXS_VERSION_1_1

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueReadBufferRect(nxs_command_queue    command_queue,
                        nxs_mem              buffer,
                        nxs_bool             blocking_read,
                        const size_t *      buffer_origin,
                        const size_t *      host_origin,
                        const size_t *      region,
                        size_t              buffer_row_pitch,
                        size_t              buffer_slice_pitch,
                        size_t              host_row_pitch,
                        size_t              host_slice_pitch,
                        void *              ptr,
                        nxs_uint             num_events_in_wait_list,
                        const nxs_event *    event_wait_list,
                        nxs_event *          event) NXS_API_SUFFIX__VERSION_1_1;

#endif

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueWriteBuffer(nxs_command_queue   command_queue,
                     nxs_mem             buffer,
                     nxs_bool            blocking_write,
                     size_t             offset,
                     size_t             size,
                     const void *       ptr,
                     nxs_uint            num_events_in_wait_list,
                     const nxs_event *   event_wait_list,
                     nxs_event *         event) NXS_API_SUFFIX__VERSION_1_0;

#ifdef NXS_VERSION_1_1

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueWriteBufferRect(nxs_command_queue    command_queue,
                         nxs_mem              buffer,
                         nxs_bool             blocking_write,
                         const size_t *      buffer_origin,
                         const size_t *      host_origin,
                         const size_t *      region,
                         size_t              buffer_row_pitch,
                         size_t              buffer_slice_pitch,
                         size_t              host_row_pitch,
                         size_t              host_slice_pitch,
                         const void *        ptr,
                         nxs_uint             num_events_in_wait_list,
                         const nxs_event *    event_wait_list,
                         nxs_event *          event) NXS_API_SUFFIX__VERSION_1_1;

#endif

#ifdef NXS_VERSION_1_2

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueFillBuffer(nxs_command_queue   command_queue,
                    nxs_mem             buffer,
                    const void *       pattern,
                    size_t             pattern_size,
                    size_t             offset,
                    size_t             size,
                    nxs_uint            num_events_in_wait_list,
                    const nxs_event *   event_wait_list,
                    nxs_event *         event) NXS_API_SUFFIX__VERSION_1_2;

#endif

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueCopyBuffer(nxs_command_queue    command_queue,
                    nxs_mem              src_buffer,
                    nxs_mem              dst_buffer,
                    size_t              src_offset,
                    size_t              dst_offset,
                    size_t              size,
                    nxs_uint             num_events_in_wait_list,
                    const nxs_event *    event_wait_list,
                    nxs_event *          event) NXS_API_SUFFIX__VERSION_1_0;

#ifdef NXS_VERSION_1_1

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueCopyBufferRect(nxs_command_queue    command_queue,
                        nxs_mem              src_buffer,
                        nxs_mem              dst_buffer,
                        const size_t *      src_origin,
                        const size_t *      dst_origin,
                        const size_t *      region,
                        size_t              src_row_pitch,
                        size_t              src_slice_pitch,
                        size_t              dst_row_pitch,
                        size_t              dst_slice_pitch,
                        nxs_uint             num_events_in_wait_list,
                        const nxs_event *    event_wait_list,
                        nxs_event *          event) NXS_API_SUFFIX__VERSION_1_1;

#endif

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueReadImage(nxs_command_queue     command_queue,
                   nxs_mem               image,
                   nxs_bool              blocking_read,
                   const size_t *       origin,
                   const size_t *       region,
                   size_t               row_pitch,
                   size_t               slice_pitch,
                   void *               ptr,
                   nxs_uint              num_events_in_wait_list,
                   const nxs_event *     event_wait_list,
                   nxs_event *           event) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueWriteImage(nxs_command_queue    command_queue,
                    nxs_mem              image,
                    nxs_bool             blocking_write,
                    const size_t *      origin,
                    const size_t *      region,
                    size_t              input_row_pitch,
                    size_t              input_slice_pitch,
                    const void *        ptr,
                    nxs_uint             num_events_in_wait_list,
                    const nxs_event *    event_wait_list,
                    nxs_event *          event) NXS_API_SUFFIX__VERSION_1_0;

#ifdef NXS_VERSION_1_2

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueFillImage(nxs_command_queue   command_queue,
                   nxs_mem             image,
                   const void *       fill_color,
                   const size_t *     origin,
                   const size_t *     region,
                   nxs_uint            num_events_in_wait_list,
                   const nxs_event *   event_wait_list,
                   nxs_event *         event) NXS_API_SUFFIX__VERSION_1_2;

#endif

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueCopyImage(nxs_command_queue     command_queue,
                   nxs_mem               src_image,
                   nxs_mem               dst_image,
                   const size_t *       src_origin,
                   const size_t *       dst_origin,
                   const size_t *       region,
                   nxs_uint              num_events_in_wait_list,
                   const nxs_event *     event_wait_list,
                   nxs_event *           event) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueCopyImageToBuffer(nxs_command_queue command_queue,
                           nxs_mem           src_image,
                           nxs_mem           dst_buffer,
                           const size_t *   src_origin,
                           const size_t *   region,
                           size_t           dst_offset,
                           nxs_uint          num_events_in_wait_list,
                           const nxs_event * event_wait_list,
                           nxs_event *       event) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueCopyBufferToImage(nxs_command_queue command_queue,
                           nxs_mem           src_buffer,
                           nxs_mem           dst_image,
                           size_t           src_offset,
                           const size_t *   dst_origin,
                           const size_t *   region,
                           nxs_uint          num_events_in_wait_list,
                           const nxs_event * event_wait_list,
                           nxs_event *       event) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY void * NXS_API_CALL
nxsEnqueueMapBuffer(nxs_command_queue command_queue,
                   nxs_mem           buffer,
                   nxs_bool          blocking_map,
                   nxs_map_flags     map_flags,
                   size_t           offset,
                   size_t           size,
                   nxs_uint          num_events_in_wait_list,
                   const nxs_event * event_wait_list,
                   nxs_event *       event,
                   nxs_int *         errcode_ret) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY void * NXS_API_CALL
nxsEnqueueMapImage(nxs_command_queue  command_queue,
                  nxs_mem            image,
                  nxs_bool           blocking_map,
                  nxs_map_flags      map_flags,
                  const size_t *    origin,
                  const size_t *    region,
                  size_t *          image_row_pitch,
                  size_t *          image_slice_pitch,
                  nxs_uint           num_events_in_wait_list,
                  const nxs_event *  event_wait_list,
                  nxs_event *        event,
                  nxs_int *          errcode_ret) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueUnmapMemObject(nxs_command_queue command_queue,
                        nxs_mem           memobj,
                        void *           mapped_ptr,
                        nxs_uint          num_events_in_wait_list,
                        const nxs_event * event_wait_list,
                        nxs_event *       event) NXS_API_SUFFIX__VERSION_1_0;

#ifdef NXS_VERSION_1_2

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueMigrateMemObjects(nxs_command_queue       command_queue,
                           nxs_uint                num_mem_objects,
                           const nxs_mem *         mem_objects,
                           nxs_mem_migration_flags flags,
                           nxs_uint                num_events_in_wait_list,
                           const nxs_event *       event_wait_list,
                           nxs_event *             event) NXS_API_SUFFIX__VERSION_1_2;

#endif

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueNDRangeKernel(nxs_command_queue command_queue,
                       nxs_kernel        kernel,
                       nxs_uint          work_dim,
                       const size_t *   global_work_offset,
                       const size_t *   global_work_size,
                       const size_t *   local_work_size,
                       nxs_uint          num_events_in_wait_list,
                       const nxs_event * event_wait_list,
                       nxs_event *       event) NXS_API_SUFFIX__VERSION_1_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueNativeKernel(nxs_command_queue  command_queue,
                      void (NXS_CALLBACK * user_func)(void *),
                      void *            args,
                      size_t            cb_args,
                      nxs_uint           num_mem_objects,
                      const nxs_mem *    mem_list,
                      const void **     args_mem_loc,
                      nxs_uint           num_events_in_wait_list,
                      const nxs_event *  event_wait_list,
                      nxs_event *        event) NXS_API_SUFFIX__VERSION_1_0;

#ifdef NXS_VERSION_1_2

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueMarkerWithWaitList(nxs_command_queue  command_queue,
                            nxs_uint           num_events_in_wait_list,
                            const nxs_event *  event_wait_list,
                            nxs_event *        event) NXS_API_SUFFIX__VERSION_1_2;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueBarrierWithWaitList(nxs_command_queue  command_queue,
                             nxs_uint           num_events_in_wait_list,
                             const nxs_event *  event_wait_list,
                             nxs_event *        event) NXS_API_SUFFIX__VERSION_1_2;

#endif

#ifdef NXS_VERSION_2_0

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueSVMFree(nxs_command_queue  command_queue,
                 nxs_uint           num_svm_pointers,
                 void *            svm_pointers[],
                 void (NXS_CALLBACK * pfn_free_func)(nxs_command_queue queue,
                                                    nxs_uint          num_svm_pointers,
                                                    void *           svm_pointers[],
                                                    void *           user_data),
                 void *            user_data,
                 nxs_uint           num_events_in_wait_list,
                 const nxs_event *  event_wait_list,
                 nxs_event *        event) NXS_API_SUFFIX__VERSION_2_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueSVMMemcpy(nxs_command_queue  command_queue,
                   nxs_bool           blocking_copy,
                   void *            dst_ptr,
                   const void *      src_ptr,
                   size_t            size,
                   nxs_uint           num_events_in_wait_list,
                   const nxs_event *  event_wait_list,
                   nxs_event *        event) NXS_API_SUFFIX__VERSION_2_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueSVMMemFill(nxs_command_queue  command_queue,
                    void *            svm_ptr,
                    const void *      pattern,
                    size_t            pattern_size,
                    size_t            size,
                    nxs_uint           num_events_in_wait_list,
                    const nxs_event *  event_wait_list,
                    nxs_event *        event) NXS_API_SUFFIX__VERSION_2_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueSVMMap(nxs_command_queue  command_queue,
                nxs_bool           blocking_map,
                nxs_map_flags      flags,
                void *            svm_ptr,
                size_t            size,
                nxs_uint           num_events_in_wait_list,
                const nxs_event *  event_wait_list,
                nxs_event *        event) NXS_API_SUFFIX__VERSION_2_0;

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueSVMUnmap(nxs_command_queue  command_queue,
                  void *            svm_ptr,
                  nxs_uint           num_events_in_wait_list,
                  const nxs_event *  event_wait_list,
                  nxs_event *        event) NXS_API_SUFFIX__VERSION_2_0;

#endif

#ifdef NXS_VERSION_2_1

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsEnqueueSVMMigrateMem(nxs_command_queue         command_queue,
                       nxs_uint                  num_svm_pointers,
                       const void **            svm_pointers,
                       const size_t *           sizes,
                       nxs_mem_migration_flags   flags,
                       nxs_uint                  num_events_in_wait_list,
                       const nxs_event *         event_wait_list,
                       nxs_event *               event) NXS_API_SUFFIX__VERSION_2_1;

#endif

#ifdef NXS_VERSION_1_2

/* Extension function access
 *
 * Returns the extension function address for the given function name,
 * or NULL if a valid function can not be found.  The client must
 * check to make sure the address is not NULL, before using or
 * calling the returned function address.
 */
extern NXS_API_ENTRY void * NXS_API_CALL
nxsGetExtensionFunctionAddressForPlatform(nxs_platform_id platform,
                                         const char *   func_name) NXS_API_SUFFIX__VERSION_1_2;

#endif

#ifdef NXS_USE_DEPRECATED_NEXUSAPI_1_0_APIS
    /*
     *  WARNING:
     *     This API introduces mutable state into the OpenCL implementation. It has been REMOVED
     *  to better facilitate thread safety.  The 1.0 API is not thread safe. It is not tested by the
     *  OpenCL 1.1 conformance test, and consequently may not work or may not work dependably.
     *  It is likely to be non-performant. Use of this API is not advised. Use at your own risk.
     *
     *  Software developers previously relying on this API are instructed to set the command queue
     *  properties when creating the queue, instead.
     */
    extern NXS_API_ENTRY nxs_int NXS_API_CALL
    nxsSetCommandQueueProperty(nxs_command_queue              command_queue,
                              nxs_command_queue_properties   properties,
                              nxs_bool                       enable,
                              nxs_command_queue_properties * old_properties) NXS_API_SUFFIX__VERSION_1_0_DEPRECATED;
#endif /* NXS_USE_DEPRECATED_NEXUSAPI_1_0_APIS */

/* Deprecated OpenCL 1.1 APIs */
extern NXS_API_ENTRY NXS_API_PREFIX__VERSION_1_1_DEPRECATED nxs_mem NXS_API_CALL
nxsCreateImage2D(nxs_context              context,
                nxs_mem_flags            flags,
                const nxs_image_format * image_format,
                size_t                  image_width,
                size_t                  image_height,
                size_t                  image_row_pitch,
                void *                  host_ptr,
                nxs_int *                errcode_ret) NXS_API_SUFFIX__VERSION_1_1_DEPRECATED;

extern NXS_API_ENTRY NXS_API_PREFIX__VERSION_1_1_DEPRECATED nxs_mem NXS_API_CALL
nxsCreateImage3D(nxs_context              context,
                nxs_mem_flags            flags,
                const nxs_image_format * image_format,
                size_t                  image_width,
                size_t                  image_height,
                size_t                  image_depth,
                size_t                  image_row_pitch,
                size_t                  image_slice_pitch,
                void *                  host_ptr,
                nxs_int *                errcode_ret) NXS_API_SUFFIX__VERSION_1_1_DEPRECATED;

extern NXS_API_ENTRY NXS_API_PREFIX__VERSION_1_1_DEPRECATED nxs_int NXS_API_CALL
nxsEnqueueMarker(nxs_command_queue    command_queue,
                nxs_event *          event) NXS_API_SUFFIX__VERSION_1_1_DEPRECATED;

extern NXS_API_ENTRY NXS_API_PREFIX__VERSION_1_1_DEPRECATED nxs_int NXS_API_CALL
nxsEnqueueWaitForEvents(nxs_command_queue  command_queue,
                        nxs_uint          num_events,
                        const nxs_event * event_list) NXS_API_SUFFIX__VERSION_1_1_DEPRECATED;

extern NXS_API_ENTRY NXS_API_PREFIX__VERSION_1_1_DEPRECATED nxs_int NXS_API_CALL
nxsEnqueueBarrier(nxs_command_queue command_queue) NXS_API_SUFFIX__VERSION_1_1_DEPRECATED;

extern NXS_API_ENTRY NXS_API_PREFIX__VERSION_1_1_DEPRECATED nxs_int NXS_API_CALL
nxsUnloadCompiler(void) NXS_API_SUFFIX__VERSION_1_1_DEPRECATED;

extern NXS_API_ENTRY NXS_API_PREFIX__VERSION_1_1_DEPRECATED void * NXS_API_CALL
nxsGetExtensionFunctionAddress(const char * func_name) NXS_API_SUFFIX__VERSION_1_1_DEPRECATED;

/* Deprecated OpenCL 2.0 APIs */
extern NXS_API_ENTRY NXS_API_PREFIX__VERSION_1_2_DEPRECATED nxs_command_queue NXS_API_CALL
nxsCreateCommandQueue(nxs_context                     context,
                     nxs_device_id                   device,
                     nxs_command_queue_properties    properties,
                     nxs_int *                       errcode_ret) NXS_API_SUFFIX__VERSION_1_2_DEPRECATED;

extern NXS_API_ENTRY NXS_API_PREFIX__VERSION_1_2_DEPRECATED nxs_sampler NXS_API_CALL
nxsCreateSampler(nxs_context          context,
                nxs_bool             normalized_coords,
                nxs_addressing_mode  addressing_mode,
                nxs_filter_mode      filter_mode,
                nxs_int *            errcode_ret) NXS_API_SUFFIX__VERSION_1_2_DEPRECATED;

extern NXS_API_ENTRY NXS_API_PREFIX__VERSION_1_2_DEPRECATED nxs_int NXS_API_CALL
nxsEnqueueTask(nxs_command_queue  command_queue,
              nxs_kernel         kernel,
              nxs_uint           num_events_in_wait_list,
              const nxs_event *  event_wait_list,
              nxs_event *        event) NXS_API_SUFFIX__VERSION_1_2_DEPRECATED;

#endif /* !defined(NXS_NO_CORE_PROTOTYPES) */

#ifdef __cplusplus
}
#endif

#endif  /* __NEXUSAPI_NXS_H */
