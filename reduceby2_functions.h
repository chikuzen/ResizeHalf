#ifndef REDUCE_BY_2_FUNCTIONS_H
#define REDUCE_BY_2_FUNCTIONS_H

#include "rh_common.h"

#if defined(__SSE2__)

// ReduceBy2 helper
static F_INLINE __m128i red_by_2(
    const __m128i& avobe, const __m128i& middle, const __m128i& bellow,
    const __m128i& one)
{
    return _mm_avg_epu8(
        _mm_avg_epu8(avobe, middle),
        _mm_subs_epu8(_mm_avg_epu8(bellow, middle), one));
}


// ReduceBy2 for RGBA
static F_INLINE __m128i red_by_2_h_rgba(
    const __m128i& _a, const __m128i& _b, const __m128i& _c, const __m128i& one)
{
    __m128i t0 = _mm_shuffle_epi32(_a, _MM_SHUFFLE(3, 1, 2, 0));
    __m128i t1 = _mm_shuffle_epi32(_b, _MM_SHUFFLE(3, 1, 2, 0));
    __m128i a = _mm_unpacklo_epi64(t0, t1);
    __m128i b = _mm_unpackhi_epi64(t0, t1);
    __m128i c = _mm_or_si128(_mm_srli_si128(a, 4), _mm_slli_si128(_c, 12));
    return red_by_2(a, b, c, one);
}


template <bool ALIGNED>
static void reduceby2_hv_rgba(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    const __m128i one = _mm_set1_epi8(1);

    for (size_t y = 0; y < height - 2; y += 2) {
        __m128i ret0 = red_by_2(
            load<ALIGNED>(srcp),
            load<ALIGNED>(srcp + sstride),
            load<ALIGNED>(srcp + 2 * sstride), one);

        for (size_t x = 0; x < width - 2; x += 8) {
            __m128i ret1 = red_by_2(
                load<ALIGNED>(srcp + 4 * x + 16),
                load<ALIGNED>(srcp + 4 * x + 16 + sstride),
                load<ALIGNED>(srcp + 4 * x + 16 + 2 * sstride), one);

            __m128i ret2 = red_by_2(
                load<ALIGNED>(srcp + 4 * x + 32),
                load<ALIGNED>(srcp + 4 * x + 32 + sstride),
                load<ALIGNED>(srcp + 4 * x + 32 + 2 * sstride), one);

            ret1 = red_by_2_h_rgba(ret0, ret1, ret2, one);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + 2 * x), ret1);

            ret0 = ret2;
        }
        if ((width & 1) == 0) {
            auto d = dstp + (width / 2 - 1) * 4;
            auto s0 = srcp + (width - 2) * 4;
            auto s1 = s0 + sstride;
            d[0] = (s0[0] + s0[4] * 3 + s1[0] * 3 + s1[4] * 9 + 8) / 16;
            d[1] = (s0[1] + s0[5] * 3 + s1[1] * 3 + s1[5] * 9 + 8) / 16;
            d[2] = (s0[2] + s0[6] * 3 + s1[2] * 3 + s1[6] * 9 + 8) / 16;
            d[3] = (s0[3] + s0[7] * 3 + s1[3] * 3 + s1[7] * 9 + 8) / 16;
        }
        srcp += 2 * sstride;
        dstp += dstride;
    }

    if ((height & 1) == 0) {
        __m128i s0 = load<ALIGNED>(srcp);
        __m128i s1 = load<ALIGNED>(srcp + sstride);
        __m128i ret0 = red_by_2(s0, s1, s1, one);

        for (size_t x = 0; x < width - 2; x += 8) {
            s0 = load<ALIGNED>(srcp + 4 * x + 16);
            s1 = load<ALIGNED>(srcp + 4 * x + 16 + sstride);
            __m128i ret1 = red_by_2(s0, s1, s1, one);

            s0 = load<ALIGNED>(srcp + 4 * x + 32);
            s1 = load<ALIGNED>(srcp + 4 * x + 32 + sstride);
            __m128i ret2 = red_by_2(s0, s1, s1, one);

            ret1 = red_by_2_h_rgba(ret0, ret1, ret2, one);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + 2 * x), s1);

            ret0 = ret2;
        }
        if ((width & 1) == 0) {
            auto d = dstp + (width / 2 - 1) * 4;
            auto sa = srcp + (width - 2) * 4;
            auto sb = sa + sstride;
            d[0] = (sa[0] + sa[4] * 3 + sb[0] * 3 + sb[4] * 9 + 8) / 16;
            d[1] = (sa[1] + sa[5] * 3 + sb[1] * 3 + sb[5] * 9 + 8) / 16;
            d[2] = (sa[2] + sa[6] * 3 + sb[2] * 3 + sb[6] * 9 + 8) / 16;
            d[3] = (sa[3] + sa[7] * 3 + sb[3] * 3 + sb[7] * 9 + 8) / 16;
        }
    }
}


template <bool ALIGNED>
static void reduceby2_h_rgba(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    const __m128i one = _mm_set1_epi8(1);

    for (size_t y = 0; y < height; ++y) {
        __m128i s0 = load<ALIGNED>(srcp);
        for (size_t x = 0; x < width - 2; x += 8) {
            __m128i s1 = load<ALIGNED>(srcp + 4 * x + 16);
            __m128i s2 = load<ALIGNED>(srcp + 4 * x + 32);
            __m128i ret = red_by_2_h_rgba(s0, s1, s2, one);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + 2 * x), ret);
            s0 = s2;
        }
        if ((width & 1) == 0) {
            auto d = dstp + (width / 2 - 1) * 4;
            auto s = srcp + (width - 2) * 4;
            d[0] = (s[0] + 3 * s[4] + 2) / 4;
            d[1] = (s[1] + 3 * s[5] + 2) / 4;
            d[2] = (s[2] + 3 * s[6] + 2) / 4;
            d[3] = (s[3] + 3 * s[7] + 2) / 4;
        }
        srcp += sstride;
        dstp += dstride;
    }
}


template <bool ALIGNED>
static void reduceby2_v_rgba(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    const __m128i one = _mm_set1_epi8(1);

    for (size_t y = 0; y < height - 2; y += 2) {
        for (size_t x = 0; x < width; x += 4) {
            __m128i ret = red_by_2(
                load<ALIGNED>(srcp + 4 * x),
                load<ALIGNED>(srcp + 4 * x + sstride),
                load<ALIGNED>(srcp + 4 * x + sstride * 2), one);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + 4 * x), ret);
        }
        srcp += 2 * sstride;
        dstp += dstride;
    }

    if ((height & 1) == 0) {
        for (size_t x = 0; x < width; x += 4) {
            __m128i s0 = load<ALIGNED>(srcp + 4 * x);
            __m128i s1 = load<ALIGNED>(srcp + 4 * x + sstride);
            s1 = red_by_2(s0, s1, s1, one);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + 4 * x), s1);
        }
    }
}


// ReduceBy2 for GREY8
static F_INLINE __m128i red_by_2_h_grey(
    const __m128i& l0, const __m128i& l1, const __m128i& l2, const __m128i& mask,
    const __m128i& one)
{
    __m128i l = _mm_packus_epi16(_mm_and_si128(l0, mask), _mm_and_si128(l1, mask));
    __m128i m = _mm_packus_epi16(_mm_srli_epi16(l0, 8), _mm_srli_epi16(l1, 8));
    __m128i r = _mm_or_si128(_mm_srli_si128(l, 1), _mm_slli_si128(l2, 15));
    return red_by_2(l, m , r, one);
}


template <bool ALIGNED>
static void reduceby2_hv_grey(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    const __m128i one = _mm_set1_epi8(1);
    const __m128i mask = _mm_set1_epi16(0x00FF);

    for (size_t y = 0; y < height - 2; y += 2) {
        __m128i ret0 = red_by_2(
            load<ALIGNED>(srcp),
            load<ALIGNED>(srcp + sstride),
            load<ALIGNED>(srcp + 2 * sstride), one);

        for (size_t x = 0; x < width - 2; x += 32) {
            __m128i ret1 = red_by_2(
                load<ALIGNED>(srcp + x + 16),
                load<ALIGNED>(srcp + x + 16 + sstride),
                load<ALIGNED>(srcp + x + 16 + 2 * sstride), one);

            __m128i ret2 = red_by_2(
                load<ALIGNED>(srcp + x + 32),
                load<ALIGNED>(srcp + x + 32 + sstride),
                load<ALIGNED>(srcp + x + 32 + 2 * sstride), one);

            ret1 = red_by_2_h_grey(ret0, ret1, ret2, mask, one);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + x / 2), ret1);

            ret0 = ret2;
        }
        if ((width & 1) == 0) {
            dstp[width / 2 - 1] = (srcp[width - 2] + srcp[width - 1] * 3 + 2) / 4;
        }
        srcp += 2 * sstride;
        dstp += dstride;
    }

    if ((height & 1) == 0) {
        __m128i s0 = load<ALIGNED>(srcp);
        __m128i s1 = load<ALIGNED>(srcp + sstride);
        __m128i ret0 = red_by_2(s0, s1, s1, one);

        for (size_t x = 0; x < width; x += 32) {
            s0 = load<ALIGNED>(srcp + x + 16);
            s1 = load<ALIGNED>(srcp + x + 16 + sstride);
            __m128i ret1 = red_by_2(s0, s1, s1, one);

            s0 = load<ALIGNED>(srcp + x + 32);
            s1 = load<ALIGNED>(srcp + x + 32 + sstride);
            __m128i ret2 = red_by_2(s0, s1, s1, one);

            ret1 = red_by_2_h_grey(ret0, ret1, ret2, mask, one);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + x / 2), ret1);

            ret0 = ret2;
        }
    }
}


template <bool ALIGNED>
static void reduceby2_h_grey(const uint8_t* srcp, uint8_t* dstp, const size_t width,
    const size_t height, const size_t sstride,
    const size_t dstride) noexcept
{
    const __m128i one = _mm_set1_epi8(1);
    const __m128i mask = _mm_set1_epi16(0x00FF);
    for (size_t y = 0; y < height; ++y) {
        __m128i s0 = load<ALIGNED>(srcp);

        for (size_t x = 0; x < width - 2; x += 32) {
            __m128i s1 = load<ALIGNED>(srcp + x + 16);
            __m128i s2 = load<ALIGNED>(srcp + x + 32);
            __m128i ret = red_by_2_h_grey(s0, s1, s2, mask, one);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + x / 2), ret);
            s0 = s2;
        }
        if ((width & 1) == 0) {
            dstp[width / 2 - 1] = (srcp[width - 2] + srcp[width - 1] * 3 + 2) / 4;
        }
        srcp += sstride;
        dstp += dstride;
    }
}


template <bool ALIGNED>
static void reduceby2_v_grey(const uint8_t* srcp, uint8_t* dstp, const size_t width,
    const size_t height, const size_t sstride,
    const size_t dstride) noexcept
{
    const __m128i one = _mm_set1_epi8(1);
    for (size_t y = 0; y < height - 2; y += 2) {
        for (size_t x = 0; x < width; x += 16) {
            __m128i ret = red_by_2(
                load<ALIGNED>(srcp + x),
                load<ALIGNED>(srcp + x + sstride),
                load<ALIGNED>(srcp + x + 2 * sstride), one);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + x), ret);
        }
        srcp += 2 * sstride;
        dstp += dstride;
    }

    if ((height & 1) == 0) {
        for (size_t x = 0; x < width; x += 16) {
            __m128i s0 = load<ALIGNED>(srcp + x);
            __m128i s1 = load<ALIGNED>(srcp + x + sstride);
            __m128i ret = red_by_2(s0, s1, s1, one);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + x), ret);
        }
    }
}


// ReduceBy2 for RGB888
static void reduceby2_hv_rgb(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    // not implemented yet
}


static void reduceby2_h_rgb(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    // not implemented yet
}


static void reduceby2_v_rgb(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    // not implemented yet...
}

#endif  // __SSE2__


// ReduceBy2 for RGBA (no SIMD)
static void reduceby2_hv_rgba_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    // not implemented yet...
}


static void reduceby2_h_rgba_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    // not implemented yet...
}


static void reduceby2_v_rgba_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    // not implemented yet...
}


// ReduceBy2 for GREY8 (no SIMD)
static void reduceby2_hv_grey_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    // not implemented yet...
}


static void reduceby2_h_grey_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    // not implemented yet...
}


static void reduceby2_v_grey_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    // not implemented yet...
}


// ReduceBy2 for RGB888 (no SIMD)
static void reduceby2_hv_rgb_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    // not implemented yet...
}


static void reduceby2_h_rgb_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    // not implemented yet...
}


static void reduceby2_v_rgb_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    // not implemented yet...
}

#endif // REDUCE_BY_2_FUNCTIONS_H

