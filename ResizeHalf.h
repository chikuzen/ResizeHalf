#ifndef RESIZE_HALF_H
#define RESIZE_HALF_H

#include <cstdint>
#include <stdexcept>

#define RESIZE_HALF_VERSION_MAJOR   0
#define RESIZE_HALF_VERSION_MINOR   0
#define RESIZE_HALF_VERSION_PATCH   2
#define RESIZE_HALF_VERSION_STRING  "0.0.3"

// Add -mssse3 when compiling for x86 with GCC/clang.

// Note that this class throws std::runtime_error if any errors occur during processing.

// width:       Image width. The unit is not byte.
// rowsize:     Number of bytes of valid data in each line of the image.
//                  This value is equal to width for GRAY8.
//                  For RGB888, this value is three times the width.
//                  For RGBA888, this value is four times the width.
// height:      Image height.
// padding:     Data inserted after each line of image to adjust memory alignment.
// stride:      The number of real bytes in each line of the image (rowsize + padding),
//                  In most cases it will be a multiple of 4 in Windows Bitmap.
//                  This value should be a multiple of 16 or more power of 2 when using SSE.


class ResizeHalf {
    const size_t align;
    int format;
    int mode;
    uint8_t* image;
    size_t buffsize;
    size_t width;
    size_t height;
    size_t stride;

    int getFlag(const void* ptr, size_t bytes) const noexcept;
    void alloc();
    const size_t prepare(const uint8_t* s, const size_t sw, const size_t sh,
                         const size_t ss, const size_t ds, int pt);
    void copyToDst(uint8_t* d, const size_t ds) noexcept;

public:
    // Format of image to resize.
    enum FMT : int {
        GREY8       = 1,
        RGB888      = 3,
        RGBA8888    = 4,
    };

    // Resize method.
    enum MODE : int {
        BILINEAR    = (1 << 8),
        REDUCE_BY_2 = (1 << 9), // port from VirtualDub filter (better).
    };

    ResizeHalf(const FMT format, const MODE mode=REDUCE_BY_2);
    ~ResizeHalf();

    // Change the image format to process.
    void setFormat(const FMT format) noexcept;

    // Change the methid to process.
    void setProcMode(const MODE mode) noexcept;

    // Reduce the image to vertical and horizontal halves (round down after the decimal point).
    // dstp      : Start address of buffer to write the image after reduction.
    //             If this value is nullptr, do not copy from the intermediate buffer.
    // srcp      : Start address of original image.
    // src_width : Width of original image.
    // src_height: Height of original image.
    // dst_stride: Stride of processed image.
    // src_stride: Stride of original image.
    // ※ If src_stride and dst_stride are 0, they are treated as Windows Bitmap standard respectively.
    void resizeHV(uint8_t* dstp, const uint8_t* srcp, const size_t src_width,
                  const size_t src_height, const size_t dst_stride=0,
                  const size_t src_stride=0);

    // Reduce the image horizontally by half (round down after the decimal point).
    void resizeHorizontal(uint8_t* dstp, const uint8_t* srcp,
                          const size_t src_width, const size_t src_height,
                          const size_t dst_stride=0, const size_t src_stride=0);

    // Reduce the image vertically by half (round down after the decimal point)
    void resizeVertical(uint8_t* dstp, const uint8_t* srcp,
                        const size_t src_width, const size_t src_height,
                        const size_t dst_stride=0, const size_t src_stride=0);

    // Returns the start address of the intermediate buffer where processed image data is stored.
    const uint8_t* data() const noexcept { return image; }

    // Returns the width of the processed image currently stored in the intermediate buffer.
    const size_t getWidth() const noexcept { return width; }

    // Returns the number of valid bytes per line of the processed image currently stored in the intermediate buffer.
    const size_t getRowsize() const noexcept { return width * format; }

    // Returns the hidth of the processed image currently stored in the intermediate buffer.
    const size_t getHeight() const noexcept { return height; }

    // Returns the stride of the processed image currently stored in the intermediate buffer.
    const size_t getStride() const noexcept { return stride; }

    // Returns the currently set image format to process
    const int getFormat() const noexcept { return format; }

    // Returns the currently set method to process
    const int getProcMode() const noexcept { return mode; }
};


#endif // RESIZE_HALF_H
