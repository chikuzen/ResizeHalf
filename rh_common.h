#ifndef RH_COMMON_H
#define RH_COMMON_H

#include <cstdint>

#pragma warning(disable: 4100)

#if defined(_M_IX86) || defined(_M_AMD64) || defined(__i686) || defined(__x86_64)
    #if !defined(__GNUC__)
        #define __SSE2__
        #define __SSSE3__
    #endif
#endif

//#undef __SSE2__

#if defined(__SSSE3__)
    #include <tmmintrin.h>
#elif defined(__SSE2__)
    #include <emmintrin.h>
#else
    #include <cstdlib>
#endif

#if defined(__GNUC__)
    #define F_INLINE inline __attribute__((always_inline))
#else
    #define F_INLINE __forceinline
#endif


struct RGBA {
    uint8_t r, g, b, a;
};


#if defined(__SSE2__)
template <bool ALIGNED>
static F_INLINE __m128i load(const uint8_t* _s)
{
    auto s = reinterpret_cast<const __m128i*>(_s);
    if (ALIGNED) {
        return _mm_load_si128(s);
    } else {
        return _mm_loadu_si128(s);
    }
}
#endif

#endif

