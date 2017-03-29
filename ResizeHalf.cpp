#include <cstring>

#include "rh_common.h"
#include "bilinear_functions.h"
#include "reduceby2_functions.h"

#include "ResizeHalf.h"



enum ProcType : int {
    PROC_HV,
    PROC_H,
    PROC_V,
};

enum : int {
    UNALIGNED_IMAGE = 0,
    ALIGNED_IMAGE = (1 << 16),
};


ResizeHalf::
ResizeHalf(const FMT fmt, const MODE m, const size_t a) :
    format(fmt), mode(m), image(nullptr), buffsize(0), width(0), height(0),
    stride(0)
{
    if ((a & 15) != 0) {
        throw std::runtime_error("invalid align was specified");
    }
    align = a - 1;
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


void ResizeHalf::setFormat(const FMT f) noexcept
{
    format = f;
}


void ResizeHalf::setProcMode(const MODE m) noexcept
{
    mode = m;
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
prepare(const uint8_t* srcp, const size_t sw, const size_t sh, const size_t ss,
        int pt)
{
    if (sw < 16 || sh < 16) {
        throw std::runtime_error("source image is too small.");
    }
    if (!srcp) {
        throw std::runtime_error("null pointer exception.");
    }

    size_t sstride = ss;
    if (sstride == 0) {
        if (format == GREY8) {
            sstride = (sw + 3) & ~3;
        } else if (format == RGB888) {
            sstride = (sw * 3 + 3) & ~3;
        } else {
            sstride = sw * 4;
        }
    }
    if (sstride < sw * format) {
        throw std::runtime_error("inavlid stride was specified.");
    }

    width = pt == PROC_V ? sw : sw / 2;
    height = pt == PROC_H ? sh : sh / 2;
    auto f = format == RGB888 ? 4 : format;
    stride = (width * f + align) & ~align;
    if (height * stride > buffsize) {
        alloc();
    }

    return sstride;
}


void ResizeHalf::
copyToDst(uint8_t* dstp, const size_t ds) noexcept
{
    if (!dstp) {
        return;
    }

    auto dstride = ds;
    if (dstride == 0) {
        if (format == GREY8) {
            dstride = (width + 3) & ~3;
        } else if (format == RGB888) {
            dstride = (width * 3 + 3) & ~3;
        } else {
            dstride = width * 4;
        }
    }

    const uint8_t* s = image;
    auto rowsize = width * format;
    if (rowsize == dstride && rowsize == stride) {
        std::memcpy(dstp, s, rowsize * height);
    } else {
        for (size_t y = 0; y < height; ++y) {
            std::memcpy(dstp, s, rowsize);
            s += stride;
            dstp += dstride;
        }
    }
}


void ResizeHalf::
resizeHV(uint8_t* dstp, const uint8_t* srcp, const size_t sw, const size_t sh,
         const size_t ds, const size_t ss)
{
    auto sstride = prepare(srcp, sw, sh, ss, PROC_HV);
    int proc = (mode | format);

#if defined(__SSE2__)
    if (format != RGB888) {
        if (((reinterpret_cast<size_t>(srcp) | sstride) & align) == 0) {
            proc |= ALIGNED_IMAGE;
        }
    }

    switch (proc) {
    case (ALIGNED_IMAGE | BILINEAR | GREY8):
        bilinear_hv_grey<true>(srcp, image, sw, sh, sstride, stride);
        break;
    case (ALIGNED_IMAGE | BILINEAR | RGBA8888):
        bilinear_hv_rgba<true>(srcp, image, sw, sh, sstride, stride);
        break;
    case (ALIGNED_IMAGE | REDUCE_BY_2 | GREY8):
        reduceby2_hv_grey<true>(srcp, image, sw, sh, sstride, stride);
        break;
    case (ALIGNED_IMAGE | REDUCE_BY_2 | RGBA8888):
        reduceby2_hv_rgba<true>(srcp, image, sw, sh, sstride, stride);
        break;
    case (UNALIGNED_IMAGE | BILINEAR | GREY8):
        bilinear_hv_grey<false>(srcp, image, sw, sh, sstride, stride);
        break;
    case (UNALIGNED_IMAGE | BILINEAR | RGB888):
        bilinear_hv_rgb888(srcp, image, sw, sh, sstride, stride);
        break;
    case (UNALIGNED_IMAGE | BILINEAR | RGBA8888):
        bilinear_hv_rgba<false>(srcp, image, sw, sh, sstride, stride);
        break;
    case (UNALIGNED_IMAGE | REDUCE_BY_2 | GREY8):
        reduceby2_hv_grey<false>(srcp, image, sw, sh, sstride, stride);
        break;
    case (UNALIGNED_IMAGE | REDUCE_BY_2 | RGB888):
        reduceby2_hv_rgb(srcp, image, sw, sh, sstride, stride);
        break;
    case (UNALIGNED_IMAGE | REDUCE_BY_2 | RGBA8888):
        reduceby2_hv_rgba<false>(srcp, image, sw, sh, sstride, stride);
        break;
    default:
        break;
    }
#else
    switch (proc) {
    case (BILINEAR | GREY8):
        bilinear_hv_grey_c(srcp, image, sw, sh, sstride, stride);
        break;
    case (BILINEAR | RGB888):
        bilinear_hv_rgb888_c(srcp, image, sw, sh, sstride, stride);
        break;
    case (BILINEAR | RGBA8888):
        bilinear_hv_rgba_c(srcp, image, sw, sh, sstride, stride);
        break;
    case (REDUCE_BY_2 | GREY8):
        reduceby2_hv_grey_c(srcp, image, sw, sh, sstride, stride);
        break;
    case (REDUCE_BY_2 | RGB888):
        reduceby2_hv_rgb_c(srcp, image, sw, sh, sstride, stride);
        break;
    case (REDUCE_BY_2 | RGBA8888):
        reduceby2_hv_rgba_c(srcp, image, sw, sh, sstride, stride);
        break;
    default:
        break;
    }
#endif

    copyToDst(dstp, ds);
}


void ResizeHalf::
resizeHorizontal(uint8_t* dstp, const uint8_t* srcp, const size_t sw,
                 const size_t sh, const size_t ds, const size_t ss)
{
    auto sstride = prepare(srcp, sw, sh, ss, PROC_H);
    int proc = (mode | format);

#if defined(__SSE2__)
    if (format != RGB888) {
        if (((reinterpret_cast<size_t>(srcp) | sstride) & align) == 0) {
            proc |= ALIGNED_IMAGE;
        }
    }

    switch (proc) {
    case (ALIGNED_IMAGE | BILINEAR | GREY8):
        bilinear_h_grey<true>(srcp, image, sw, sh, sstride, stride);
        break;
    case (ALIGNED_IMAGE | BILINEAR | RGBA8888):
        bilinear_h_rgba<true>(srcp, image, sw, sh, sstride, stride);
        break;
    case (ALIGNED_IMAGE | REDUCE_BY_2 | GREY8):
        reduceby2_h_grey<true>(srcp, image, sw, sh, sstride, stride);
        break;
    case (ALIGNED_IMAGE | REDUCE_BY_2 | RGBA8888):
        reduceby2_h_rgba<true>(srcp, image, sw, sh, sstride, stride);
        break;
    case (UNALIGNED_IMAGE | BILINEAR | GREY8):
        bilinear_h_grey<false>(srcp, image, sw, sh, sstride, stride);
        break;
    case (UNALIGNED_IMAGE | BILINEAR | RGB888):
        bilinear_h_rgb888(srcp, image, sw, sh, sstride, stride);
        break;
    case (UNALIGNED_IMAGE | BILINEAR | RGBA8888):
        bilinear_h_rgba<false>(srcp, image, sw, sh, sstride, stride);
        break;
    case (UNALIGNED_IMAGE | REDUCE_BY_2 | GREY8):
        reduceby2_h_grey<false>(srcp, image, sw, sh, sstride, stride);
        break;
    case (UNALIGNED_IMAGE | REDUCE_BY_2 | RGB888):
        reduceby2_h_rgb(srcp, image, sw, sh, sstride, stride);
        break;
    case (UNALIGNED_IMAGE | REDUCE_BY_2 | RGBA8888):
        reduceby2_h_rgba<false>(srcp, image, sw, sh, sstride, stride);
        break;
    default:
        break;
    }
#else
    switch (proc) {
    case (BILINEAR | GREY8):
        bilinear_h_grey_c(srcp, image, sw, sh, sstride, stride);
        break;
    case (BILINEAR | RGB888):
        bilinear_h_rgb888_c(srcp, image, sw, sh, sstride, stride);
        break;
    case (BILINEAR | RGBA8888):
        bilinear_h_rgba_c(srcp, image, sw, sh, sstride, stride);
        break;
    case (REDUCE_BY_2 | GREY8):
        reduceby2_h_grey_c(srcp, image, sw, sh, sstride, stride);
        break;
    case (REDUCE_BY_2 | RGB888):
        reduceby2_h_rgb_c(srcp, image, sw, sh, sstride, stride);
        break;
    case (REDUCE_BY_2 | RGBA8888):
        reduceby2_h_rgba_c(srcp, image, sw, sh, sstride, stride);
        break;
    default:
        break;
    }
#endif

    copyToDst(dstp, ds);
}


void ResizeHalf::
resizeVertical(uint8_t* dstp, const uint8_t* srcp, const size_t sw,
               const size_t sh, const size_t ds, const size_t ss)
{
    auto sstride = prepare(srcp, sw, sh, ss, PROC_V);
    int proc = (mode | format);

#if defined(__SSE2__)
    if (format != RGB888) {
        if (((reinterpret_cast<size_t>(srcp) | sstride) & align) != 0) {
            proc |= ALIGNED_IMAGE;
        }
    }

    switch (proc) {
    case (ALIGNED_IMAGE | BILINEAR | GREY8):
        bilinear_v_grey<true>(srcp, image, sw, sh, sstride, stride);
        break;
    case (ALIGNED_IMAGE | BILINEAR | RGBA8888):
        bilinear_v_rgba<true>(srcp, image, sw, sh, sstride, stride);
        break;
    case (ALIGNED_IMAGE | REDUCE_BY_2 | GREY8):
        reduceby2_v_grey<true>(srcp, image, sw, sh, sstride, stride);
        break;
    case (ALIGNED_IMAGE | REDUCE_BY_2 | RGBA8888):
        reduceby2_v_rgba<true>(srcp, image, sw, sh, sstride, stride);
        break;
    case (UNALIGNED_IMAGE | BILINEAR | GREY8):
        bilinear_v_grey<false>(srcp, image, sw, sh, sstride, stride);
        break;
    case (UNALIGNED_IMAGE | BILINEAR | RGB888):
        bilinear_v_rgb888(srcp, image, sw, sh, sstride, stride);
        break;
    case (UNALIGNED_IMAGE | BILINEAR | RGBA8888):
        bilinear_v_rgba<false>(srcp, image, sw, sh, sstride, stride);
        break;
    case (UNALIGNED_IMAGE | REDUCE_BY_2 | GREY8):
        reduceby2_v_grey<false>(srcp, image, sw, sh, sstride, stride);
        break;
    case (UNALIGNED_IMAGE | REDUCE_BY_2 | RGB888):
        reduceby2_v_rgb(srcp, image, sw, sh, sstride, stride);
        break;
    case (UNALIGNED_IMAGE | REDUCE_BY_2 | RGBA8888):
        reduceby2_v_rgba<false>(srcp, image, sw, sh, sstride, stride);
        break;
    default:
        break;
    }
#else
    switch (proc) {
    case (BILINEAR | GREY8):
        bilinear_v_grey_c(srcp, image, sw, sh, sstride, stride);
        break;
    case (BILINEAR | RGB888):
        bilinear_v_rgb888_c(srcp, image, sw, sh, sstride, stride);
        break;
    case (BILINEAR | RGBA8888):
        bilinear_v_rgba_c(srcp, image, sw, sh, sstride, stride);
        break;
    case (REDUCE_BY_2 | GREY8):
        reduceby2_v_grey_c(srcp, image, sw, sh, sstride, stride);
        break;
    case (REDUCE_BY_2 | RGB888):
        reduceby2_v_rgb_c(srcp, image, sw, sh, sstride, stride);
        break;
    case (REDUCE_BY_2 | RGBA8888):
        reduceby2_v_rgba_c(srcp, image, sw, sh, sstride, stride);
        break;
    default:
        break;
    }
#endif

    copyToDst(dstp, ds);
}

