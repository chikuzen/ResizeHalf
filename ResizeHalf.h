#ifndef RESIZE_HALF_H
#define RESIZE_HALF_H

#include <cstdint>
#include <stdexcept>


//処理中になんらかのエラーが発生した場合はstd::runtime_errorを投げるので注意

class ResizeHalf {
public:
    enum FMT : int {
        GREY8 = 1,
        RGBA8888 = 4,
    };

private:
    FMT format;
    uint8_t* image;
    size_t align;
    size_t buffsize;
    size_t width;
    size_t height;
    size_t stride;

    void alloc();
    const size_t prepare(const uint8_t* srcp, const size_t sw, const size_t sh,
                         const size_t ss, int pt);
    void copyToDst(uint8_t* dstp, const size_t ds) noexcept;

public:
    // format: 処理を行う画像形式
    // align: 中間バッファのメモリアライメント
    ResizeHalf(const FMT format, const size_t align=16);
    ~ResizeHalf();

    // 処理を行う画像形式を変更する
    void setFormat(const FMT format) noexcept;

    // 画像をBilinearで縦横2分の1(小数点以下切り上げ)に縮小する
    // dstp      : 縮小後の画像を書き込むバッファの先頭アドレス
    //             nullptrの場合は中間バッファからのコピーを行わない
    // srcp      : 元画像の先頭アドレス。
    // src_width :　〃　　幅。ラインごとのbyte数ではない。
    // src_height:　〃　　高さ
    // dst_stride: 縮小後の画像を書き込むバッファのstride
    // src_stride: 元画像のstride
    // dst_strideとsrc_strideは0の場合は自動的にWindows Bitmapの標準(下記)として扱う
    //    RGBA8888 -> (width * 4), GREY8 -> ((width + 3) & ~3)
    void resizeHV(uint8_t* dstp, const uint8_t* srcp, const size_t src_width,
                  const size_t src_height, const size_t dst_stride=0,
                  const size_t src_stride=0);

    // 画像をBilinearで水平方向のみ2分の1(小数点以下切り上げ)に縮小する
    void resizeHorizontal(uint8_t* dstp, const uint8_t* srcp,
                          const size_t src_width, const size_t src_height,
                          const size_t dst_stride=0, const size_t src_stride=0);

    // 画像(32bitRGBA限定)をBilinearで垂直方向のみ2分の1(小数点以下切り上げ)に縮小する
    void resizeVertical(uint8_t* dstp, const uint8_t* srcp,
                        const size_t src_width, const size_t src_height,
                        const size_t dst_stride=0, const size_t src_stride=0);

    const uint8_t* data() const noexcept { return image; }
    const size_t getWidth() const noexcept { return width; }
    const size_t getHeight() const noexcept { return height; }
    const size_t getStride() const noexcept { return stride; }
    const FMT getFormat() const noexcept { return format; }
};




#endif // !RESIZE_HALF_H
