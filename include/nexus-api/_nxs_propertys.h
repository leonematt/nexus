/*
 */


#if defined(NEXUS_API_GENERATE_PROP_ENUM)
/************************************************************************
 * Generate the Function declarations
 ***********************************************************************/
// Generate the Function extern

#define NEXUS_API_PROP(NAME, TYPE, DESC) \
    NP_##NAME,

enum _nxs_property {

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
    NXS_PROPERTY_CNT,
    NXS_PROPERTY_PREFIX_LEN        = 3
}; /* close _nxs_property */

typedef enum _nxs_property nxs_property;

/* Translation functions */
nxs_int nxsGetPropCount();
const char *nxsGetPropName(nxs_property propEnum);
nxs_property nxsGetPropEnum(const char *propName);

const char *nxsGetStatusName(nxs_status statusEnum);
nxs_status nxsGetStatusEnum(const char *statusName);

#endif

#undef NEXUS_API_GENERATE_PROP_ENUM
#undef NEXUS_API_GENERATE_PROP_TYPE

#undef NEXUS_API_PROP
