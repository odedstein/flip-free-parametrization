#ifndef PARAMETRIZATIONS_ASSERT_H
#define PARAMETRIZATIONS_ASSERT_H

// Custom assert macro that can be enabled / disabled separately from NDEBUG.
// If PARAMETRIZATIONS_DEBUG is defined, it will assert even though assertions
//  are turned off.
// If PARAMETRIZATIONS_DEBUG is undefined, this will behave like standard
//  assert.
//


#if defined(PARAMETRIZATIONS_DEBUG) || !defined(NDEBUG)
#   include <cstdlib>
#   include <iostream>
namespace parametrization {
inline void custom_assert(const char* condition,
                          const char* function,
                          const char* file,
                          const int line) {
    std::cerr << "Assertion failed: (" << condition << "), function " <<
    function << ", file " << file << ", line " << line << "." << std::endl;
    abort();
}
}
#   define parametrization_assert(X) \
        ((void) \
            (!!(X) || (parametrization::custom_assert \
                (#X, __func__, __FILE__, __LINE__) \
            , 0)) \
        )
#else
#   define parametrization_assert(X) ((void) 0)
#endif


#endif
