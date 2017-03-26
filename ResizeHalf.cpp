#include <cstring>

#include "ResizeHalf.h"

#if defined(_M_IX86) || defined(_M_AMD64) || defined(__i686) || defined(__x86_64)
    #if !defined(__GNUC__)
        #define __SSE2__
    #endif
#endif

//#undef __SSE2__

#if defined(__SSE2__)

#include <emmintrin.h>

#if defined(__GNUC__)
    #define F_INLINE inline __attribute__((always_inline))
#else
    #define F_INLINE __forceinline
#endif


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


static F_INLINE __m128i resize_h(const __m128i& a, const __m128i& b)
{
    __m128i s0 = _mm_shuffle_epi32(a, _MM_SHUFFLE(3, 1, 2, 0));
    __m128i s1 = _mm_shuffle_epi32(b, _MM_SHUFFLE(3, 1, 2, 0));
    return _mm_avg_epu8(_mm_unpacklo_epi64(s0, s1), _mm_unpackhi_epi64(s0, s1));
}


template <bool ALIGNED>
static void proc_hv(const uint8_t* srcp, uint8_t* dstp, const size_t width,
                    const size_t height, const size_t sstride,
                    const size_t dstride) noexcept
{
    auto s = srcp;
    auto d = dstp;
    auto w = width & ~1;
    auto h = height & ~1;

    const __m128i one = _mm_set1_epi8(1);

    for (size_t y = 0; y < h; y += 2) {
        for (size_t x = 0; x < w; x += 8) {
            __m128i s0 = load<ALIGNED>(s + 4 * x);
            s0 = resize_h(s0, load<ALIGNED>(s + 4 * x + 16));
            __m128i s1 = load<ALIGNED>(s + 4 * x + sstride);
            s1 = resize_h(s1, load<ALIGNED>(s + 4 * x + 16 + sstride));
            s0 = _mm_avg_epu8(s0, _mm_subs_epu8(s1, one));
            _mm_store_si128(reinterpret_cast<__m128i*>(d + 2 * x), s0);
        }
        s += 2 * sstride;
        d += dstride;
    }

    if (h != height) {
        for (size_t x = 0; x < w; x += 8) {
            __m128i s0 = load<ALIGNED>(s + 4 * x);
            s0 = resize_h(s0, load<ALIGNED>(s + 4 * x + 16));
            _mm_store_si128(reinterpret_cast<__m128i*>(d + 2 * x), s0);
        }
    }

    if (w != width) {
        auto ss = srcp + w * 4;
        auto dd = dstp + ((width + 1) / 2 - 1) * 4;

        for (size_t y = 0; y < h; y += 2) {
            dd[0] = (ss[0] + ss[sstride + 0] + 1) / 2;
            dd[1] = (ss[1] + ss[sstride + 1] + 1) / 2;
            dd[2] = (ss[2] + ss[sstride + 2] + 1) / 2;
            dd[3] = (ss[3] + ss[sstride + 3] + 1) / 2;
            ss += 2 * sstride;
            dd += dstride;
        }

        if (h != height) {
            *reinterpret_cast<uint32_t*>(dd) =
                *reinterpret_cast<const uint32_t*>(ss);
        }
    }
}


template <bool ALIGNED>
static void proc_h(const uint8_t* srcp, uint8_t* dstp, const size_t width,
                   const size_t height, const size_t sstride,
                   const size_t dstride) noexcept
{
    auto s = srcp;
    auto d = dstp;
    auto w = width & ~1;

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < w; x += 8) {
            __m128i s0 = load<ALIGNED>(s + 4 * x);
            __m128i s1 = load<ALIGNED>(s + 4 * x + 16);
            s0 = resize_h(s0, s1);
            _mm_store_si128(reinterpret_cast<__m128i*>(d + 2 * x), s0);
        }
        s += sstride;
        d += dstride;
    }

    if (w != width) {
        auto ss = srcp + w * 4;
        auto dd = dstp + ((width + 1) / 2 - 1) * 4;
        for (size_t y = 0; y < height; ++y) {
            *reinterpret_cast<uint32_t*>(dd) =
                *reinterpret_cast<const uint32_t*>(ss);
            ss += sstride;
            dd += dstride;
        }
    }

}


template <bool ALIGNED>
static void proc_v(const uint8_t* srcp, uint8_t* dstp, const size_t width,
                   const size_t height, const size_t sstride,
                   const size_t dstride) noexcept
{
    auto h = height & ~1;

    for (size_t y = 0; y < h; y += 2) {
        for (size_t x = 0; x < width; x += 4) {
            __m128i s0 = load<ALIGNED>(srcp + 4 * x);
            __m128i s1 = load<ALIGNED>(srcp + 4 * x + sstride);
            s0 = _mm_avg_epu8(s0, s1);
            _mm_store_si128(reinterpret_cast<__m128i*>(dstp + 4 * x), s0);
        }
        srcp += 2 * sstride;
        dstp += dstride;
    }

    if (h != height) {
        std::memcpy(dstp, srcp, width * 4);
    }
}
#else // __SSE2__

#include <cstdlib>

struct RGBA {
    uint8_t r, g, b, a;
};

static void proc_hv(const uint8_t* srcp, uint8_t* dstp, const size_t width,
                    const size_t height, const size_t sstride,
                    const size_t dstride) noexcept
{
    auto s = reinterpret_cast<const RGBA*>(srcp);
    auto d = reinterpret_cast<RGBA*>(dstp);
    auto w = width & ~1;
    auto h = height & ~1;
    auto ss = sstride / sizeof(RGBA);
    auto ds = dstride / sizeof(RGBA);

    for (size_t y = 0; y < h; y += 2) {
        auto sb = s + ss;
        for (size_t x = 0; x < w; x += 2) {
            auto dx = x / 2;
            auto foo = s[x];
            d[dx].r = (s[x].r + s[x + 1].r + sb[x].r + sb[x + 1].r + 2) / 4;
            d[dx].g = (s[x].g + s[x + 1].g + sb[x].g + sb[x + 1].g + 2) / 4;
            d[dx].b = (s[x].b + s[x + 1].b + sb[x].b + sb[x + 1].b + 2) / 4;
            d[dx].a = (s[x].a + s[x + 1].a + sb[x].a + sb[x + 1].a + 2) / 4;
        }
        if (w != width) {
            auto dx = (width + 1) / 2 - 1;
            d[dx].r = (s[w].r + sb[w].r + 1) / 2;
            d[dx].g = (s[w].g + sb[w].g + 1) / 2;
            d[dx].b = (s[w].b + sb[w].b + 1) / 2;
            d[dx].a = (s[w].a + sb[w].a + 1) / 2;
        }
        s += 2 * ss;
        d += ds;
    }

    if (h != height) {
        for (size_t x = 0; x < w; x += 2) {
            auto dx = x / 2;
            d[dx].r = (s[x].r + s[x + 1].r + 1) / 2;
            d[dx].g = (s[x].g + s[x + 1].g + 1) / 2;
            d[dx].b = (s[x].b + s[x + 1].b + 1) / 2;
            d[dx].a = (s[x].a + s[x + 1].a + 1) / 2;
        }
        if (w != width) {
            d[(width + 1) / 2 - 1] = s[w];
        }
    }
}


static void proc_h(const uint8_t* srcp, uint8_t* dstp, const size_t width,
                   const size_t height, const size_t sstride,
                   const size_t dstride) noexcept
{
    auto s = reinterpret_cast<const RGBA*>(srcp);
    auto d = reinterpret_cast<RGBA*>(dstp);
    auto w = width & ~1;
    auto ss = sstride / sizeof(RGBA);
    auto ds = dstride / sizeof(RGBA);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < w; x += 2) {
            auto dx = x / 2;
            d[dx].r = (s[x].r + s[x + 1].r + 1) / 2;
            d[dx].g = (s[x].g + s[x + 1].g + 1) / 2;
            d[dx].b = (s[x].b + s[x + 1].b + 1) / 2;
            d[dx].a = (s[x].a + s[x + 1].a + 1) / 2;
        }
        if (w != width) {
            d[(width + 1) / 2 - 1] = s[w];
        }
        s += ss;
        d += ds;
    }
}


static void proc_v(const uint8_t* srcp, uint8_t* dstp, const size_t width,
                   const size_t height, const size_t sstride,
                   const size_t dstride) noexcept
{
    auto s = reinterpret_cast<const RGBA*>(srcp);
    auto d = reinterpret_cast<RGBA*>(dstp);
    auto h = height & ~1;
    auto ss = sstride / sizeof(RGBA);
    auto ds = dstride / sizeof(RGBA);

    for (size_t y = 0; y < h; y += 2) {
        auto sb = s + ss;
        for (size_t x = 0; x < width; ++x) {
            d[x].r = (s[x].r + sb[x].r + 1) / 2;
            d[x].g = (s[x].g + sb[x].g + 1) / 2;
            d[x].b = (s[x].b + sb[x].b + 1) / 2;
            d[x].a = (s[x].a + sb[x].a + 1) / 2;
        }
        s += 2 * ss;
        d += ds;
    }

    if (h != height) {
        std::memcpy(d, s, width * 4);
    }
}

#endif


ResizeHalf::
ResizeHalf(const size_t mw, const size_t mh, const size_t a) : image(nullptr)
{
    if ((a & 15) != 0) {
        throw std::runtime_error("invalid align was specified");
    }
    align = a - 1;
    width = mw;
    height = mh;
    stride = (mw * 4 + align) & ~align;
    alloc();
}


ResizeHalf::~ResizeHalf()
{
#if defined(__SSE2__)
    _mm_free(image);
#else
    std::free(image);
#endif
    image = nullptr;
}


void ResizeHalf::alloc()
{
#if defined(__SSE2__)
    _mm_free(image);
    image = static_cast<uint8_t*>(_mm_malloc(stride * height, align + 1));
#else
    std::free(image);
    image = static_cast<uint8_t*>(std::malloc(stride * height));
#endif
    if (!image) {
        throw std::runtime_error("failed to allocate buffer.");
    }
    buffsize = stride * height;
}


const size_t ResizeHalf::
prepare(const size_t sw, const size_t sh, const size_t ss, ProcType pt)
{
    size_t sstride = ss == 0 ? sw * 4 : ss;
    if (sstride < sw * 4) {
        throw std::runtime_error("inavlid stride was specified.");
    }

    width = pt == PROC_V ? sw : (sw + 1) / 2;
    height = pt == PROC_H ? sh : (sh + 1) / 2;
    stride = (width * 4 + align) & ~align;
    if (height * stride > buffsize) {
        alloc();
    }

    return sstride;
}


template <ResizeHalf::ProcType TYPE>
void ResizeHalf::
resize(const uint8_t* srcp, const size_t sw, const size_t sh,
       const size_t ss)
{
    auto sstride = prepare(sw, sh, ss, TYPE);

#if defined(__SSE2__)
    if (((reinterpret_cast<uintptr_t>(srcp) | sstride) & align) == 0) {
        switch (TYPE) {
        case PROC_HV:
            proc_hv<true>(srcp, image, sw, sh, sstride, stride);
            return;
        case PROC_H:
            proc_h<true>(srcp, image, sw, sh, sstride, stride);
            return;
        default:
            proc_v<true>(srcp, image, sw, sh, sstride, stride);
            return;
        }
    } else {
        switch (TYPE) {
        case PROC_HV:
            proc_hv<false>(srcp, image, sw, sh, sstride, stride);
            return;
        case PROC_H:
            proc_h<false>(srcp, image, sw, sh, sstride, stride);
            return;
        default:
            proc_v<false>(srcp, image, sw, sh, sstride, stride);
        }
    }
#else
    switch (TYPE) {
    case PROC_HV:
        proc_hv(srcp, image, sw, sh, sstride, stride);
        return;
    case PROC_H:
        proc_h(srcp, image, sw, sh, sstride, stride);
        return;
    default:
        proc_v(srcp, image, sw, sh, sstride, stride);
    }
#endif
}


void ResizeHalf::
resizeHV(const uint8_t* srcp, const size_t sw, const size_t sh,
         const size_t ss)
{
    resize<PROC_HV>(srcp, sw, sh, ss);
}


void ResizeHalf::
resizeHorizontal(const uint8_t* srcp, const size_t sw, const size_t sh,
                 const size_t ss)
{
    resize<PROC_H>(srcp, sw, sh, ss);
}


void ResizeHalf::
resizeVertical(const uint8_t* srcp, const size_t sw, const size_t sh,
               const size_t ss)
{
    resize<PROC_V>(srcp, sw, sh, ss);
}

