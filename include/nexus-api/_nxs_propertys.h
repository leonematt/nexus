/*
 */


#if defined(NEXUS_API_GENERATE_PROP_ENUM)
/************************************************************************
 * Generate the Function declarations
 ***********************************************************************/
// Generate the Function extern

#define NEXUS_API_PROP(NAME, TYPE, DESC) \
    NP_##NAME,

enum NXSAPI_PropertyEnum {

#else
#if defined(NEXUS_API_GENERATE_PROP_TYPE)
/************************************************************************
 * Generate the Function typedefs
 ***********************************************************************/
        
// Generate the Function typedefs
#define NEXUS_API_PROP(NAME, TYPE, DESC) \
    typedef TYPE NXS_CONCAT(nxs##NAME, _t);

#endif
#endif


/************************************************************************
 * Define API Functions
 ***********************************************************************/

/************************************************************************
 * @def Name
 * @brief Object Name 
 ***********************************************************************/
NEXUS_API_PROP(Name,            char *,      "Unit Name")
NEXUS_API_PROP(Type,            char *,      "Unit Type")
NEXUS_API_PROP(Description,     char *,      "Unit Description")

/* Property Hierarchy */
NEXUS_API_PROP(Count,           nxs_uint,    "Number of Units")
NEXUS_API_PROP(Size,            nxs_uint,    "Number of Sub-Units")
NEXUS_API_PROP(SubUnits,        void *,      "Sub-Unit Vector")
NEXUS_API_PROP(SubUnitType,     char *,      "Sub-Unit Type")

/* Device Properties */
NEXUS_API_PROP(Vendor,          char *,      "Vendor Name")
NEXUS_API_PROP(Architecture,    char *,      "Architecture Designation")
NEXUS_API_PROP(Version,         char *,      "Version String")
NEXUS_API_PROP(MajorVersion,    nxs_uint,    "Major Version")
NEXUS_API_PROP(MinorVersion,    nxs_uint,    "Minor Version")

NEXUS_API_PROP(CoreSubsystem,   void *,      "Core Subsystem Hierarchy")
NEXUS_API_PROP(MemorySubsystem, void *,      "Memory Subsystem Hierarchy")

NEXUS_API_PROP(ClockModes,      void *,      "Clock Modes")
NEXUS_API_PROP(BaseClock,       nxs_uint,    "Max Power")
NEXUS_API_PROP(PowerModes,      void *,      "Power Modes")
NEXUS_API_PROP(MaxPower,        nxs_double,  "Max Power")

/* THIS SHOULD BE GENERATED FROM THE SCHEMA */


/************************************************************************
 * Cleanup
 ***********************************************************************/
#ifdef NEXUS_API_GENERATE_PROP_ENUM
    NP_PROPERTY_COUNT,
    NP_PROPERTY_PREFIX_SIZE = 3
}; /* close RuntimeFuncEnum */

/* Translation functions */
const char *nxsGetPropName(enum NXSAPI_PropertyEnum propEnum);
enum NXSAPI_PropertyEnum nxsGetPropEnum(const char *propName);

const char *nxsGetStatusName(enum NXSAPI_StatusEnum statusEnum);
enum NXSAPI_StatusEnum nxsGetStatusEnum(const char *statusName);

#endif

#undef NEXUS_API_GENERATE_PROP_ENUM
#undef NEXUS_API_GENERATE_PROP_TYPE

#undef NEXUS_API_PROP
