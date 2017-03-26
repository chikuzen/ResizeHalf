#ifndef RESIZE_HALF_H
#define RESIZE_HALF_H

#include <cstdint>
#include <stdexcept>


//処理中になんらかのエラーが発生した場合はstd::runtime_errorを投げるので注意

class ResizeHalf {
    uint8_t* image;
    size_t align;
    size_t buffsize;
    size_t width;
    size_t height;
    size_t stride;

    void alloc();

    enum ProcType {
        PROC_HV,
        PROC_H,
        PROC_V,
    };
    const size_t prepare(const size_t sw, const size_t sh,
                           const size_t ss, ProcType pt);
    template <ProcType TYPE>
    void resize(const uint8_t* srcRGBA, const size_t src_width,
                const size_t src_height, const size_t src_stride);

public:
    ResizeHalf(const size_t src_max_width, const size_t src_max_height,
               const size_t align=16);
    ~ResizeHalf();

    // 画像(32bitRGBA限定)をBilinearで縦横2分の一に縮小する
    // srcRGBA   :元画像の先頭アドレス。
    // src_width :　〃　　幅。ラインごとのbyte数ではない。
    // src_height:　〃　　高さ
    // src_stride:　〃　　stride(paddingを含めたライン当たりのbyte数)
    //    src_stride=0の場合は自動的にsrc_width*4(Windows Bitmapの標準)として扱う
    void resizeHV(const uint8_t* srcRGBA, const size_t src_width,
                  const size_t src_height, const size_t src_stride=0);

    // 画像(32bitRGBA限定)をBilinearで水平方向のみ2分の一に縮小する
    void resizeHorizontal(const uint8_t* srcRGBA, const size_t src_width,
                          const size_t src_height, const size_t src_stride=0);

    // 画像(32bitRGBA限定)をBilinearで垂直方向のみ2分の一に縮小する
    void resizeVertical(const uint8_t* srcRGBA, const size_t src_width,
                        const size_t src_height, const size_t src_stride=0);

    // resize後の画像の先頭ポインタ(16bytes aligned)
    // resizeXXX()においてsrc_width/src_heightがコンストラクタで指定した
    // 最大値よりも大きかったい場合、この値は変更される可能性がある
    const uint8_t* data() const noexcept { return image; }

    // resize後の画像の幅
    const size_t getWidth() const noexcept { return width; }

    // resize後の画像の高さ
    const size_t getHeight() const noexcept { return height; }

    // resize後の画像のstride(byte, 16の倍数保証)
    const size_t getStride() const noexcept { return stride; }
};




#endif // !RESIZE_HALF_H
