#ifndef BILINEAR_FUNCTIONS_H
#define BILINEAR_FUNCTIONS_H

#include "rh_common.h"

#if defined(__SSSE3__)

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
    auto w = width & ~1;
    auto h = height & ~1;

    const __m128i one = _mm_set1_epi8(1);

    for (size_t y = 0; y < h; y += 2) {
        auto sb = srcp + sstride;
        for (size_t x = 0; x < w; x += 8) {
            __m128i s0 = _mm_avg_epu8(
                load<ALIGNED>(srcp + 4 * x), load<ALIGNED>(sb + 4 * x));
            __m128i s1 = _mm_avg_epu8(
                load<ALIGNED>(srcp + 4 * x + 16), load<ALIGNED>(sb + 4 * x + 16));
            s0 = bl_h_rgba(s0, _mm_subs_epu8(s1, one));
            stream(dstp + 2 * x, s0);
        }
        srcp += 2 * sstride;
        dstp += dstride;
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
            stream(dstp + 2 * x, ret);
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
            stream(dstp + 4 * x, ret);
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
        auto sb = srcp + sstride;
        for (size_t x = 0; x < w; x += 32) {
            __m128i s0 = _mm_avg_epu8(
                load<ALIGNED>(srcp + x), load<ALIGNED>(sb + x));
            __m128i s1 = _mm_avg_epu8(
                load<ALIGNED>(srcp + x + 16), load<ALIGNED>(sb + x + 16));
            s0 = bl_h_grey(s0, _mm_subs_epu8(s1, one), mask);
            stream(dstp + x / 2, s0);
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
            stream(dstp + x / 2, ret);
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
            stream(dstp + x, ret);
        }
        srcp += 2 * sstride;
        dstp += dstride;
    }
}


// Bilinear Resize for RGB888
static void bilinear_hv_rgb888(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    auto w = width & ~1;
    auto h = height & ~1;

    const __m128i smask0 = _mm_setr_epi8(
        0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
    const __m128i smask1 = _mm_setr_epi8(
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);

    for (size_t y = 0; y < h; y += 2) {
        auto sb = srcp + sstride;
        for (size_t x = 0; x < w; x += 8) {
            __m128i s0 = _mm_avg_epu8(
                load<false>(srcp + 3 * x), load<false>(sb + 3 * x));
            __m128i s1 = _mm_avg_epu8(
                load<false>(srcp + 3 * x + 12), load<false>(sb + 3 * x + 12));
            s0 = _mm_shuffle_epi8(s0, smask0);
            s1 = _mm_shuffle_epi8(s1, smask0);
            s0 = _mm_shuffle_epi8(bl_h_rgba(s0, s1), smask1);
            storeu(dstp + 3 * x / 2, s0);
        }
        srcp += 2 * sstride;
        dstp += dstride;
    }
}


static void bilinear_h_rgb888(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    auto w = width & ~1;
    const __m128i smask0 = _mm_setr_epi8(
        0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
    const __m128i smask1 = _mm_setr_epi8(
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < w; x += 8) {
            __m128i s0 = _mm_shuffle_epi8(load<false>(srcp + 3 * x), smask0);
            __m128i s1 = _mm_shuffle_epi8(load<false>(srcp + 3 * x + 12), smask0);
            s0 = _mm_shuffle_epi8(bl_h_rgba(s0, s1), smask1);
            storeu(dstp + 3 * x / 2, s0);
        }
        srcp += sstride;
        dstp += dstride;
    }
}


static void bilinear_v_rgb888(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    bilinear_v_grey<false>(srcp, dstp, width * 3, height, sstride, dstride);
}

#endif // __SSSE3__

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
            d[x / 2] = (
                RGBAi(s[x], 1) + RGBAi(s[x + 1], 1) +
                RGBAi(sb[x], 1) + RGBAi(sb[x + 1], 1)).div4<RGBA>();
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
            d[x / 2] = (RGBAi(s[x], 1) + RGBAi(s[x + 1], 1)).div2<RGBA>();
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
            d[x] = (RGBAi(s[x], 1) + RGBAi(sb[x], 1)).div2<RGBA>();
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
static void bilinear_hv_rgb888_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    auto w = width & ~1;
    auto h = height & ~1;

    for (size_t y = 0; y < h; y += 2) {
        auto sa = reinterpret_cast<const RGB24*>(srcp);
        auto sb = reinterpret_cast<const RGB24*>(srcp + sstride);
        auto d = reinterpret_cast<RGB24*>(dstp);
        for (size_t x = 0; x < w; x += 2) {
            d[x / 2] = (
                RGBAi(sa[x], 1) + RGBAi(sa[x + 1], 1) +
                RGBAi(sb[x], 1) + RGBAi(sb[x + 1], 1)).div4<RGB24>();
        }
        srcp += 2 * sstride;
        dstp += dstride;
    }
}


static void bilinear_h_rgb888_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    auto w = width & ~1;

    for (size_t y = 0; y < height; ++y) {
        auto s = reinterpret_cast<const RGB24*>(srcp);
        auto d = reinterpret_cast<RGB24*>(dstp);
        for (size_t x = 0; x < w; x += 2) {
            d[x / 2] = (RGBAi(s[x], 1) + RGBAi(s[x + 1], 1)).div2<RGB24>();
        }
        srcp += sstride;
        dstp += dstride;
    }
}


static void bilinear_v_rgb888_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    bilinear_v_grey_c(srcp, dstp, width * 3, height, sstride, dstride);
}


#endif  // BILINEAR_FUNCTIONS_H

