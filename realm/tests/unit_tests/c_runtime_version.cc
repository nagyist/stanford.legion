#include "realm/realm_c.h"
#include <gtest/gtest.h>

namespace Realm {
  extern const char *realm_library_version;
}

using namespace Realm;

TEST(RuntimeVersionTest, GetLibraryVersionSuccess)
{
  const char *version = nullptr;
  realm_status_t status = realm_get_library_version(&version);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_NE(version, nullptr);
  EXPECT_EQ(strcmp(version, Realm::realm_library_version), 0);
}

TEST(RuntimeVersionTest, GetLibraryVersionNullVersion)
{
  realm_status_t status = realm_get_library_version(nullptr);
  EXPECT_EQ(status, REALM_ERROR_INVALID_PARAMETER);
}