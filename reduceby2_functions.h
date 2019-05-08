#ifndef REDUCE_BY_2_FUNCTIONS_H
#define REDUCE_BY_2_FUNCTIONS_H

#include "rh_common.h"

#if defined(__SSE2__)

// ReduceBy2 helper
#if 1
static F_INLINE __m128i red_by_2(
    const __m128i& s0, const __m128i& s1, const __m128i& s2, const __m128i& one)
{
    return _mm_avg_epu8(
        _mm_avg_epu8(s0, s1), _mm_subs_epu8(_mm_avg_epu8(s2, s1), one));
}
#else
static F_INLINE __m128i red_by_2(   // accurate version
    const __m128i& s0, const __m128i& s1, const __m128i& s2, const __m128i& one)
{
    __m128i t0 = _mm_avg_epu8(s0, s1);
    __m128i t1 = _mm_avg_epu8(s2, s1);
    __m128i error = _mm_or_si128(_mm_xor_si128(s0, s1), _mm_xor_si128(s2, s1));
    __m128i err2 = _mm_xor_si128(t0, t1);
    error = _mm_and_si128(_mm_and_si128(error, err2), one);
    return _mm_subs_epu8(_mm_avg_epu8(t0, t1), error);
}
#endif

// ReduceBy2 for RGBA
static F_INLINE __m128i red_by_2_h_rgba(
    const __m128i& _l, const __m128i& _c, const __m128i& _r, const __m128i& one)
{
    __m128i t0 = _mm_shuffle_epi32(_l, _MM_SHUFFLE(3, 1, 2, 0));
    __m128i t1 = _mm_shuffle_epi32(_c, _MM_SHUFFLE(3, 1, 2, 0));
    __m128i l = _mm_unpacklo_epi64(t0, t1);
    __m128i c = _mm_unpackhi_epi64(t0, t1);
#if defined(__SSSE3__)
    __m128i r = _mm_alignr_epi8(_r, l, 4);
#else
    __m128i r = _mm_or_si128(_mm_srli_si128(l, 4), _mm_slli_si128(_r, 12));
#endif
    return red_by_2(l, c, r, one);
}


template <bool ALIGNED>
static void reduceby2_hv_rgba(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    const __m128i one = _mm_set1_epi8(1);

    for (size_t y = 0; y < height - 2; y += 2) {
        auto sb = srcp + sstride;
        auto sc = sb + sstride;
        __m128i left = red_by_2(
            load<ALIGNED>(srcp), load<ALIGNED>(sb), load<ALIGNED>(sc), one);

        for (size_t x = 0; x < width - 2; x += 8) {
            __m128i center = red_by_2(
                load<ALIGNED>(srcp + 4 * x + 16), load<ALIGNED>(sb + 4 * x + 16),
                load<ALIGNED>(sc + 4 * x + 16), one);

            __m128i right = red_by_2(
                load<ALIGNED>(srcp + 4 * x + 32), load<ALIGNED>(sb + 4 * x + 32),
                load<ALIGNED>(sc + 4 * x + 32), one);

            center = red_by_2_h_rgba(left, center, right, one);
            stream(dstp + 2 * x, center);

            left = right;
        }
        if ((width & 1) == 0) {
            auto d = reinterpret_cast<RGBA*>(dstp) + width / 2 - 1;
            auto s0 = reinterpret_cast<const RGBA*>(srcp) + width - 2;
            auto s1 = reinterpret_cast<const RGBA*>(sb) + width - 2;
            auto s2 = reinterpret_cast<const RGBA*>(sc) + width - 2;
            *d = (
                RGBAi(s0[0], 1) + RGBAi(s0[1], 3) +
                RGBAi(s1[0], 2) + RGBAi(s1[1], 6) +
                RGBAi(s2[0], 1) + RGBAi(s2[1], 3)).div16<RGBA>();
        }
        srcp += 2 * sstride;
        dstp += dstride;
    }

    if ((height & 1) == 0) {
        auto d = reinterpret_cast<RGBA*>(dstp);
        auto sa = reinterpret_cast<const RGBA*>(srcp);
        auto sb = reinterpret_cast<const RGBA*>(srcp + sstride);
        for (size_t x = 0; x < width - 2; x += 2) {
            d[x / 2] = (
                RGBAi(sa[x], 1) + RGBAi(sa[x + 1], 2) + RGBAi(sa[x + 2], 1) +
                RGBAi(sb[x], 3) + RGBAi(sb[x + 1], 6) + RGBAi(sb[x + 2], 3)
                ).div16<RGBA>();

        }
        if ((width & 1) == 0) {
            d[width / 2 - 1] = (
                RGBAi(sa[width - 2], 1) + RGBAi(sa[width - 1], 3) +
                RGBAi(sb[width - 2], 3) + RGBAi(sb[width - 1], 9)).div16<RGBA>();
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
            stream(dstp + 2 * x, ret);
            s0 = s2;
        }
        if ((width & 1) == 0) {
            auto d = reinterpret_cast<RGBA*>(dstp) + width / 2 - 1;
            auto s = reinterpret_cast<const RGBA*>(srcp) + width - 2;
            *d = (RGBAi(s[0], 1) + RGBAi(s[1], 3)).div4<RGBA>();
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
            stream(dstp + 4 * x, ret);
        }
        srcp += 2 * sstride;
        dstp += dstride;
    }

    if ((height & 1) == 0) {
        for (size_t x = 0; x < width; x += 4) {
            __m128i s0 = load<ALIGNED>(srcp + 4 * x);
            __m128i s1 = load<ALIGNED>(srcp + 4 * x + sstride);
            s1 = red_by_2(s0, s1, s1, one);
            stream(dstp + 4 * x, s1);
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
#if defined(__SSSE3__)
    __m128i r = _mm_alignr_epi8(l2, l, 1);
#else
    __m128i r = _mm_or_si128(_mm_srli_si128(l, 1), _mm_slli_si128(l2, 15));
#endif
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
        auto sb = srcp + sstride;
        auto sc = sb + sstride;
        __m128i left = red_by_2(
            load<ALIGNED>(srcp), load<ALIGNED>(sb), load<ALIGNED>(sc), one);

        for (size_t x = 0; x < width - 2; x += 32) {
            __m128i center = red_by_2(
                load<ALIGNED>(srcp + x + 16), load<ALIGNED>(sb + x + 16),
                load<ALIGNED>(sc + x + 16), one);

            __m128i right = red_by_2(
                load<ALIGNED>(srcp + x + 32), load<ALIGNED>(sb + x + 32),
                load<ALIGNED>(sc + x + 32), one);

            center = red_by_2_h_grey(left, center, right, mask, one);
            stream(dstp + x / 2, center);

            left = right;
        }
        if ((width & 1) == 0) {
            auto w2 = width - 2;
            auto w1 = width - 1;
            dstp[width / 2 - 1] = (
                srcp[w2] + 3 * srcp[w1] +
                2 * sb[w2] + 6 * sb[w1] +
                sc[w2] + 3 * sc[w1] + 8) / 16;
        }
        srcp += 2 * sstride;
        dstp += dstride;
    }

    if ((height & 1) == 0) {
        auto sb = srcp + sstride;
        __m128i s0 = load<ALIGNED>(srcp);
        __m128i s1 = load<ALIGNED>(sb);
        __m128i left = red_by_2(s0, s1, s1, one);

        for (size_t x = 0; x < width; x += 32) {
            s0 = load<ALIGNED>(srcp + x + 16);
            s1 = load<ALIGNED>(sb + x + 16);
            __m128i center = red_by_2(s0, s1, s1, one);

            s0 = load<ALIGNED>(srcp + x + 32);
            s1 = load<ALIGNED>(sb + x + 32);
            __m128i right = red_by_2(s0, s1, s1, one);

            center = red_by_2_h_grey(left, center, right, mask, one);
            stream(dstp + x / 2, center);

            left = right;
        }
        if ((width & 1) == 0) {
            dstp[width / 2 - 1] = (
                srcp[width - 2] + srcp[width - 1] * 3 +
                sb[width - 2] * 3 + sb[width - 1] * 9 + 8) / 16;
        }
    }
}


template <bool ALIGNED>
static void reduceby2_h_grey(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    const __m128i one = _mm_set1_epi8(1);
    const __m128i mask = _mm_set1_epi16(0x00FF);

    for (size_t y = 0; y < height; ++y) {
        __m128i left = load<ALIGNED>(srcp);

        for (size_t x = 0; x < width - 2; x += 32) {
            __m128i center = load<ALIGNED>(srcp + x + 16);
            __m128i right = load<ALIGNED>(srcp + x + 32);
            center = red_by_2_h_grey(left, center, right, mask, one);
            stream(dstp + x / 2, center);
            left = center;
        }
        if ((width & 1) == 0) {
            dstp[width / 2 - 1] = (
                srcp[width - 2] + srcp[width - 1] * 3 + 2) / 4;
        }
        srcp += sstride;
        dstp += dstride;
    }
}


template <bool ALIGNED>
static void reduceby2_v_grey(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    const __m128i one = _mm_set1_epi8(1);

    for (size_t y = 0; y < height - 2; y += 2) {
        for (size_t x = 0; x < width; x += 16) {
            __m128i ret = red_by_2(
                load<ALIGNED>(srcp + x),
                load<ALIGNED>(srcp + x + sstride),
                load<ALIGNED>(srcp + x + 2 * sstride), one);
            stream(dstp + x, ret);
        }
        srcp += 2 * sstride;
        dstp += dstride;
    }

    if ((height & 1) == 0) {
        for (size_t x = 0; x < width; x += 16) {
            __m128i s0 = load<ALIGNED>(srcp + x);
            __m128i s1 = load<ALIGNED>(srcp + x + sstride);
            __m128i ret = red_by_2(s0, s1, s1, one);
            stream(dstp + x, ret);
        }
    }
}


// ReduceBy2 for RGB888
#if defined(__SSSE3__)
static void reduceby2_hv_rgb888(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    const __m128i one = _mm_set1_epi8(1);
    const __m128i smask0 = _mm_setr_epi8(
        0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
    const __m128i smask1 = _mm_setr_epi8(
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);

    for (size_t y = 0; y < height - 2; y += 2) {
        auto sb = srcp + sstride;
        auto sc = sb + sstride;
        __m128i left = _mm_shuffle_epi8(
            red_by_2(load<false>(srcp),
                     load<false>(sb),
                     load<false>(sc), one), smask0);

        for (size_t x = 0; x < width - 2; x += 8) {
            __m128i center = _mm_shuffle_epi8(
                red_by_2(load<false>(srcp + 3 * x + 12),
                         load<false>(sb + 3 * x + 12),
                         load<false>(sc + 3 * x + 12), one), smask0);
            __m128i right = _mm_shuffle_epi8(
                red_by_2(load<false>(srcp + 3 * x + 24),
                         load<false>(sb + 3 * x + 24),
                         load<false>(sc + 3 * x + 24), one), smask0);

            center = _mm_shuffle_epi8(
                red_by_2_h_rgba(left, center, right, one), smask1);
            storeu(dstp + 3 * x / 2, center);
            left = right;
        }
        if ((width & 1) == 0) {
            auto d = reinterpret_cast<RGB24*>(dstp) + width / 2 - 1;
            auto s0 = reinterpret_cast<const RGB24*>(srcp) + width - 2;
            auto s1 = reinterpret_cast<const RGB24*>(sb) + width - 2;
            auto s2 = reinterpret_cast<const RGB24*>(sc) + width - 2;
            *d = (
                RGBAi(s0[0], 1) + RGBAi(s0[1], 3) +
                RGBAi(s1[0], 2) + RGBAi(s1[1], 6) +
                RGBAi(s2[0], 1) + RGBAi(s2[1], 3)).div16<RGB24>();
        }
        srcp += 2 * sstride;
        dstp += dstride;
    }

    if ((height & 1) == 0) {
        auto sb = srcp + sstride;
        __m128i s0 = load<false>(srcp);
        __m128i s1 = load<false>(sb);
        __m128i left = red_by_2(s0, s1, s1, one);

        for (size_t x = 0; x < width - 2; x += 8) {
            s0 = load<false>(srcp + 3 * x + 12);
            s1 = load<false>(sb + 3 * x + 12);
            __m128i center = red_by_2(s0, s1, s1, one);

            s0 = load<false>(srcp + 3 * x + 32);
            s1 = load<false>(sb + 3 * x + 32);
            __m128i right = red_by_2(s0, s1, s1, one);

            center = _mm_shuffle_epi8(
                red_by_2_h_rgba(left, center, right, one), smask1);
            storeu(dstp + 3 * x / 2, center);
            left = right;
        }
        if ((width & 1) == 0) {
            auto d = reinterpret_cast<RGB24*>(dstp) + width / 2 - 1;
            auto sc = reinterpret_cast<const RGB24*>(srcp) + width - 2;
            auto sd = reinterpret_cast<const RGB24*>(srcp + sstride) + width - 2;
            *d = (
                RGBAi(sc[0], 1) + RGBAi(sc[1], 3) +
                RGBAi(sd[0], 3) + RGBAi(sd[1], 9)).div16<RGB24>();
        }
    }
}


static void reduceby2_h_rgb888(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    const __m128i one = _mm_set1_epi8(1);
    const __m128i smask0 = _mm_setr_epi8(
        0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
    const __m128i smask1 = _mm_setr_epi8(
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);

    for (size_t y = 0; y < height; ++y) {
        __m128i s0 = _mm_shuffle_epi8(load<false>(srcp), smask0);
        for (size_t x = 0; x < width - 2; x += 8) {
            __m128i s1 = _mm_shuffle_epi8(load<false>(srcp + 3 * x + 12), smask0);
            __m128i s2 = _mm_shuffle_epi8(load<false>(srcp + 3 * x + 24), smask0);
            __m128i ret = _mm_shuffle_epi8(red_by_2_h_rgba(s0, s1, s2, one), smask1);
            storeu(dstp + 3 * x / 2, ret);
            s0 = s2;
        }
        if ((width & 1) == 0) {
            auto d = reinterpret_cast<RGB24*>(dstp) + width / 2 - 1;
            auto s = reinterpret_cast<const RGB24*>(srcp) + width - 2;
            *d = (RGBAi(s[0]) + RGBAi(s[1], 3)).div4<RGB24>();
        }
        srcp += sstride;
        dstp += dstride;
    }
}
#endif  // __SSSE3__

static void reduceby2_v_rgb888(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    reduceby2_v_grey<false>(srcp, dstp, width * 3, height, sstride, dstride);
}

#endif  // __SSE2__


// ReduceBy2 for RGBA (no SIMD)
static void reduceby2_hv_rgba_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    for (size_t y = 0; y < height - 2; y += 2) {
        auto sa = reinterpret_cast<const RGBA*>(srcp);
        auto sb = reinterpret_cast<const RGBA*>(srcp + sstride);
        auto sc = reinterpret_cast<const RGBA*>(srcp + 2 * sstride);
        auto d = reinterpret_cast<RGBA*>(dstp);
        for (size_t x = 0; x < width - 2; x += 2) {
            d[x / 2] = (
                RGBAi(sa[x], 1) + RGBAi(sa[x + 1], 2) + RGBAi(sa[x + 2], 1) +
                RGBAi(sb[x], 2) + RGBAi(sb[x + 1], 4) + RGBAi(sb[x + 2], 2) +
                RGBAi(sc[x], 1) + RGBAi(sc[x + 1], 2) + RGBAi(sc[x + 2], 1)
                ).div16<RGBA>();
        }
        if ((width & 1) == 0) {
            d[width / 2 - 1] = (
                RGBAi(sa[width - 2], 1) + RGBAi(sa[width - 1], 3) +
                RGBAi(sb[width - 2], 2) + RGBAi(sb[width - 1], 6) +
                RGBAi(sc[width - 2], 1) + RGBAi(sc[width - 1], 3)).div16<RGBA>();
        }
        srcp += 2 * sstride;
        dstp += dstride;
    }

    if ((height & 1) == 0) {
        auto sa = reinterpret_cast<const RGBA*>(srcp);
        auto sb = reinterpret_cast<const RGBA*>(srcp + sstride);
        auto d = reinterpret_cast<RGBA*>(dstp);
        for (size_t x = 0; x < width - 2; x += 2) {
            d[x / 2] = (
                RGBAi(sa[x], 1) + RGBAi(sa[x + 1], 2) + RGBAi(sa[x + 2], 1) +
                RGBAi(sb[x], 3) + RGBAi(sb[x + 1], 6) + RGBAi(sb[x + 2], 3)
                ).div16<RGBA>();

        }
        if ((width & 1) == 0) {
            d[width / 2 - 1] = (
                RGBAi(sa[width - 2], 1) + RGBAi(sa[width - 1], 3) +
                RGBAi(sb[width - 2], 3) + RGBAi(sb[width - 1], 9)).div16<RGBA>();
        }
    }
}


static void reduceby2_h_rgba_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    for (size_t y = 0; y < height; ++y) {
        auto s = reinterpret_cast<const RGBA*>(srcp);
        auto d = reinterpret_cast<RGBA*>(dstp);
        for (size_t x = 0; x < width - 2; x += 2) {
            d[x / 2] = (
                RGBAi(s[x], 1) + RGBAi(s[x + 1], 2) + RGBAi(s[x + 2], 1)
                ).div4<RGBA>();
        }
        if ((width & 1) == 0) {
            d[width / 2 - 1] = (
                RGBAi(s[width - 2], 1) + RGBAi(s[width - 1], 3)).div4<RGBA>();
        }
        srcp += sstride;
        dstp += dstride;
    }
}


static void reduceby2_v_rgba_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    auto s = reinterpret_cast<const RGBA*>(srcp);
    auto d = reinterpret_cast<RGBA*>(dstp);
    auto ss = sstride / sizeof(RGBA);
    auto ds = dstride / sizeof(RGBA);

    for (size_t y = 0; y < height - 2; y += 2) {
        auto sb = s + ss;
        auto sc = sb + ss;
        for (size_t x = 0; x < width; ++x) {
            d[x] = (
                RGBAi(s[x], 1) + RGBAi(sb[x], 2) + RGBAi(sc[x], 1)).div4<RGBA>();
        }
        s += 2 * ss;
        d += ds;
    }

    if ((height & 1) == 0) {
        auto sb = s + ss;
        for (size_t x = 0; x < width; ++x) {
            d[x] = (RGBAi(s[x], 1) + RGBAi(sb[x], 3)).div4<RGBA>();
        }
    }
}


// ReduceBy2 for GREY8 (no SIMD)
static void reduceby2_hv_grey_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    for (size_t y = 0; y < height - 2; y += 2) {
        auto sb = srcp + sstride;
        auto sc = sb + sstride;
        for (size_t x = 0; x < width - 2; x += 2) {
            dstp[x / 2] = (srcp[x] + 2 * srcp[x + 1] + srcp[x + 2]
                         + 2 * sb[x] + 4 * sb[x + 1] + 2 * sb[x + 2]
                         + sc[x] + 2 * sc[x + 1] + sc[x + 2] + 8) / 16;
        }
        if ((width & 1) == 0) {
            dstp[width / 2 - 1] = (srcp[width - 2] + srcp[width - 1] * 3
                + 2 * sb[width - 2] + 6 * sb[width - 1] + sc[width - 2]
                + sc[width - 1] + 8) / 16;
        }
        srcp += 2 * sstride;
        dstp += dstride;
    }
}


static void reduceby2_h_grey_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width - 2; x += 2) {
            dstp[x / 2] = (srcp[x] + 2 * srcp[x + 1] + srcp[x + 2] + 2) / 4;
        }
        if ((width & 1) == 0) {
            dstp[width / 2 - 1] = (srcp[width - 2] + srcp[width - 1] * 3 + 2) / 4;
        }
        srcp += sstride;
        dstp += dstride;
    }
}


static void reduceby2_v_grey_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    for (size_t y = 0; y < height - 2; y += 2) {
        for (size_t x = 0; x < width; ++x) {
            dstp[x] = (
                srcp[x] + 2 * srcp[x + sstride] + srcp[x + 2 * sstride] + 2) / 4;
        }
        srcp += 2 * sstride;
        dstp += dstride;
    }

    if ((height & 1) == 0) {
        for (size_t x = 0; x < width; ++x) {
            dstp[x] = (srcp[x] + 3 * srcp[x + sstride] + 2) / 4;
        }
    }
}


// ReduceBy2 for RGB888 (no SIMD)
static void reduceby2_hv_rgb888_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    for (size_t y = 0; y < height - 2; y += 2) {
        auto sa = reinterpret_cast<const RGB24*>(srcp);
        auto sb = reinterpret_cast<const RGB24*>(srcp + sstride);
        auto sc = reinterpret_cast<const RGB24*>(srcp + 2 * sstride);
        auto d = reinterpret_cast<RGB24*>(dstp);
        for (size_t x = 0; x < width - 2; x += 2) {
            d[x / 2] = (
                RGBAi(sa[x], 1) + RGBAi(sa[x + 1], 2) + RGBAi(sa[x + 2], 1) +
                RGBAi(sb[x], 2) + RGBAi(sb[x + 1], 4) + RGBAi(sb[x + 2], 2) +
                RGBAi(sc[x], 1) + RGBAi(sc[x + 1], 2) + RGBAi(sc[x + 2], 1)
                ).div16<RGB24>();
        }
        if ((width & 1) == 0) {
            auto w2 = width - 2;
            auto w1 = width - 1;
            d[width / 2 - 1] = (
                RGBAi(sa[w2], 1) + RGBAi(sa[w1], 3) +
                RGBAi(sb[w2], 2) + RGBAi(sb[w1], 6) +
                RGBAi(sc[w2], 1) + RGBAi(sc[w1], 3)).div16<RGB24>();
        }
        srcp += 2 * sstride;
        dstp += dstride;
    }

    if ((height & 1) == 0) {
        auto sa = reinterpret_cast<const RGB24*>(srcp);
        auto sb = reinterpret_cast<const RGB24*>(srcp + sstride);
        auto d = reinterpret_cast<RGB24*>(dstp);
        for (size_t x = 0; x < width - 2; x += 2) {
            d[x / 2] = (
                RGBAi(sa[x], 1) + RGBAi(sa[x + 1], 2) + RGBAi(sa[x + 2], 1) +
                RGBAi(sb[x], 3) + RGBAi(sb[x + 1], 6) + RGBAi(sb[x + 2], 3)
                ).div16<RGB24>();
        }
        if ((width & 1) == 0) {
            auto w2 = width - 2;
            auto w1 = width - 1;
            d[width / 2 - 1] = (
                RGBAi(sa[w2], 1) + RGBAi(sa[w1], 3) +
                RGBAi(sb[w2], 3) + RGBAi(sb[w1], 9)).div16<RGB24>();
        }
    }
}


static void reduceby2_h_rgb888_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    for (size_t y = 0; y < height; ++y) {
        auto s = reinterpret_cast<const RGB24*>(srcp);
        auto d = reinterpret_cast<RGB24*>(dstp);
        for (size_t x = 0; x < width - 2; x += 2) {
            d[x / 2] = (
                RGBAi(s[x], 1) + RGBAi(s[x + 1], 2) + RGBAi(s[x + 2], 1)
                ).div4<RGB24>();
        }
        if ((width & 1) == 0) {
            d[width / 2 - 1] = (
                RGBAi(s[width - 2], 1) + RGBAi(s[width - 1], 3)).div4<RGB24>();
        }
        srcp += sstride;
        dstp += dstride;
    }
}


static void reduceby2_v_rgb888_c(
    const uint8_t* srcp, uint8_t* dstp, const size_t width, const size_t height,
    const size_t sstride, const size_t dstride) noexcept
{
    reduceby2_v_grey_c(srcp, dstp, width * 3, height, sstride, dstride);
}

#endif // REDUCE_BY_2_FUNCTIONS_H

