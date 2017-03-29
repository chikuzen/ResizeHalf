#ifndef RH_COMMON_H
#define RH_COMMON_H

#include <cstdint>
#include <algorithm>

#pragma warning(disable: 4505)

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


struct RGB24 {
    uint8_t r, g, b;
};


struct RGBA {
    uint8_t r, g, b, a;
};


struct RGBAi {
    int r, g, b, a;
    RGBAi(const RGB24& x, const int m=1) :
        r(x.r * m), g(x.g * m), b(x.b * m), a(0) {}
    RGBAi(const RGBA& x, const int m=1) :
        r(x.r * m), g(x.g * m), b(x.b * m), a(x.a * m) {}
    RGBAi& operator +(const RGBAi& x)
    {
        r += x.r; g += x.g; b += x.b; a += x.a;
        return *this;
    }
    RGBAi& operator +(const int x)
    {
        r += x; g += x; b += x; a += x;
        return *this;
    }
    RGBAi& operator /(const int x)
    {
        r /= x; g /= x; b /= x; a /= x;
        return *this;
    }
    operator RGB24()
    {
        return RGB24{
            static_cast<uint8_t>(r),
            static_cast<uint8_t>(g),
            static_cast<uint8_t>(b),
        };
    }
    operator RGBA()
    {
        return RGBA{
            static_cast<uint8_t>(r),
            static_cast<uint8_t>(g),
            static_cast<uint8_t>(b),
            static_cast<uint8_t>(a),
        };
    }
    template <typename T> T div2()
    {
        return T((*this + 1) / 2);
    }
    template <typename T> T div4()
    {
        return T((*this + 2) / 4);
    }
    template <typename T> T div16()
    {
        return T((*this + 8) / 16);
    }
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

static F_INLINE void stream(void* d, const __m128i& v)
{
    _mm_stream_si128(reinterpret_cast<__m128i*>(d), v);
}

static F_INLINE void storeu(void* d, const __m128i& v)
{
    _mm_storeu_si128(reinterpret_cast<__m128i*>(d), v);
}
#endif

#endif

