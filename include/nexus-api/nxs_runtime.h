
#include <nexus-api/nxs.h>


#ifdef __cplusplus
extern "C" {
#endif

extern NXS_API_ENTRY nxs_int NXS_API_CALL
nxsGetRuntimeInfo(nxs_platform_id   platform,
                  nxs_platform_info param_name,
                  size_t           param_value_size,
                  void *           param_value,
                  size_t *         param_value_size_ret) NXS_API_SUFFIX__VERSION_1_0;


#ifdef __cplusplus
}
#endif