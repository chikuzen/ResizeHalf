#ifndef BILINEAR_FUNCTIONS_H
#define BILINEAR_FUNCTIONS_H

#include "rh_common.h"

#if defined(__SSE2__)

// Bilinear Resize for RGBA
static F_INLINE __m128i bl_h_rgba(const __m128i& _a, const __m128i& _b)
{
    __m128i a = _mm_shuffle_epi32(_a, _MM_SHUFFLE(3, 1, 2, 0));
    __m128i b = _mm_shuffle_epi32(_b, _MM_SHUFFLE(3, 1, 2, 0));
    return _mm_avg_epu8(_mm_unpacklo_epi64(a, b), _mm_unpackhi_epi64(a, b));
}


template <bool ALIGNED>
static void bilinear_hv_rgba(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    auto s = srcp;
    auto d = dstp;
    auto w = width & ~1;
    auto h = height & ~1;

    const __m128i one = _mm_set1_epi8(1);

    for (size_t y = 0; y < h; y += 2) {
        for (size_t x = 0; x < w; x += 8) {
            __m128i s0 = bl_h_rgba(
                load<ALIGNED>(s + 4 * x),
                load<ALIGNED>(s + 4 * x + 16));

            __m128i s1 = bl_h_rgba(
                load<ALIGNED>(s + 4 * x + sstride),
                load<ALIGNED>(s + 4 * x + 16 + sstride));

            s0 = _mm_avg_epu8(s0, _mm_subs_epu8(s1, one));
            _mm_stream_si128(reinterpret_cast<__m128i*>(d + 2 * x), s0);
        }
        s += 2 * sstride;
        d += dstride;
    }
}


template <bool ALIGNED>
static void bilinear_h_rgba(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    auto w = width & ~1;

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < w; x += 8) {
            __m128i ret = bl_h_rgba(
                load<ALIGNED>(srcp + 4 * x),
                load<ALIGNED>(srcp + 4 * x + 16));
            _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + 2 * x), ret);
        }
        srcp += sstride;
        dstp += dstride;
    }
}


template <bool ALIGNED>
static void bilinear_v_rgba(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    auto h = height & ~1;

    for (size_t y = 0; y < h; y += 2) {
        for (size_t x = 0; x < width; x += 4) {
            __m128i ret = _mm_avg_epu8(
                load<ALIGNED>(srcp + 4 * x),
                load<ALIGNED>(srcp + 4 * x + sstride));
            _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + 4 * x), ret);
        }
        srcp += 2 * sstride;
        dstp += dstride;
    }
}


// Bilinear Resize for GREY8
static F_INLINE __m128i bl_h_grey(
    const __m128i& l0, const __m128i& l1, const __m128i& mask)
{
    return _mm_avg_epu8(
        _mm_packus_epi16(_mm_and_si128(l0, mask), _mm_and_si128(l1, mask)),
        _mm_packus_epi16(_mm_srli_epi16(l0, 8), _mm_srli_epi16(l1, 8)));
}


template <bool ALIGNED>
static void bilinear_hv_grey(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    auto w = width & ~1;
    auto h = height & ~1;
    const __m128i mask = _mm_set1_epi16(0x00FF);
    const __m128i one = _mm_set1_epi8(1);

    for (size_t y = 0; y < h; y += 2) {
        for (size_t x = 0; x < w; x += 32) {
            __m128i s0 = bl_h_grey(
                load<ALIGNED>(srcp + x),
                load<ALIGNED>(srcp + x + 16), mask);

            __m128i s1 = bl_h_grey(
                load<ALIGNED>(srcp + x + sstride),
                load<ALIGNED>(srcp + x + 16 + sstride), mask);

            s0 = _mm_avg_epu8(s0, _mm_subs_epu8(s1, one));
            _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + x / 2), s0);
        }
        srcp += 2 * sstride;
        dstp += dstride;
    }
}


template <bool ALIGNED>
static void bilinear_h_grey(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    auto w = width & ~1;
    const __m128i mask = _mm_set1_epi16(0x00FF);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < w; x += 32) {
            __m128i ret = bl_h_grey(
                load<ALIGNED>(srcp + x),
                load<ALIGNED>(srcp + x + 16), mask);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + x / 2), ret);
        }
        srcp += sstride;
        dstp += dstride;
    }
}


template <bool ALIGNED>
static void bilinear_v_grey(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    auto h = height & ~1;

    for (size_t y = 0; y < h; y += 2) {
        for (size_t x = 0; x < width; x += 16) {
            __m128i ret = _mm_avg_epu8(
                load<ALIGNED>(srcp + x),
                load<ALIGNED>(srcp + x + sstride));
            _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + x), ret);
        }
        srcp += 2 * sstride;
        dstp += dstride;
    }
}


// Bilinear Resize for RGB888
static void bilinear_hv_rgb(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    // not implemented yet...
}


static void bilinear_h_rgb(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    // not implemented yet...
}


static void bilinear_v_rgb(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    // not implemented yet...
}

#endif // __SSE2__

// Bilinear Resize for RGBA (no SIMD)
static void bilinear_hv_rgba_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
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
        s += 2 * ss;
        d += ds;
    }
}


static void bilinear_h_rgba_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
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
        s += ss;
        d += ds;
    }
}


static void bilinear_v_rgba_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
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
}


// Bilinear Resize for GREY8 (no SIMD)
static void bilinear_hv_grey_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    auto w = width & ~1;
    auto h = height & ~1;

    for (size_t y = 0; y < h; y += 2) {
        auto sb = srcp + sstride;
        for (size_t x = 0; x < w; x += 2) {
            dstp[x / 2] = (srcp[x] + srcp[x + 1] + sb[x] + sb[x + 1] + 2) / 4;
        }
        srcp += 2 * sstride;
        dstp += dstride;
    }
}


static void bilinear_v_grey_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    auto h = height & ~1;

    for (size_t y = 0; y < h; y += 2) {
        auto sb = srcp + sstride;
        for (size_t x = 0; x < width; ++x) {
            dstp[x] = (srcp[x] + sb[x] + 1) / 2;
        }
        srcp += 2 * sstride;
        dstp += dstride;
    }
}


static void bilinear_h_grey_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    auto w = width & ~1;

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < w; x += 2) {
            dstp[x / 2] = (srcp[x] + srcp[x + 1] + 1) / 2;
        }
        srcp += sstride;
        dstp += dstride;
    }
}


// Bilinear Resize for RGB888 (no SIMD)
static void bilinear_hv_rgb_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    // not implemented yet...
}


static void bilinear_h_rgb_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    // not implemented yet...
}


static void bilinear_v_rgb_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    // not implemented yet...
}


#endif  // BILINEAR_FUNCTIONS_H

