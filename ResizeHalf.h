#ifndef RESIZE_HALF_H
#define RESIZE_HALF_H

#include <cstdint>
#include <stdexcept>

#define RESIZE_HALF_VERSION_MAJOR   0
#define RESIZE_HALF_VERSION_MINOR   0
#define RESIZE_HALF_VERSION_PATCH   1
#define RESIZE_HALF_VERSION_STRING  "0.0.1"

// GCC/clangでx86系用にコンパイルする際は、-mssse3を付けること

// 処理中になんらかのエラーが発生した場合はstd::runtime_errorを投げるので注意

// 用語
// width:       画像の幅。byte数ではない。
// rowsize:     画像の各ラインにおける有効データのbyte数
//                  GREY8ならばwidthと等しい
//                  RGB888ならばwidth*3
//                  RGBA8888ならばwidth*4
// height:      画像の高さ
// padding:     メモリアライメントを調節するために画像データの各ラインの後ろに
//              挿入される無効データ領域
// stride:      画像の各ラインの本当のbyte数(rowsize + padding)
//                  Windows Bitmapでは、ほとんどの場合4の倍数になる
//                  SSEを使う場合は16以上の２の累乗の倍数であることが望ましい


class ResizeHalf {
    int format;
    int mode;
    uint8_t* image;
    size_t align;
    size_t buffsize;
    size_t width;
    size_t height;
    size_t stride;

    void alloc();
    const size_t prepare(const uint8_t* s, const size_t sw, const size_t sh,
                         const size_t ss, int pt);
    void copyToDst(uint8_t* d, const size_t ds) noexcept;

public:
    // 縮小する画像の形式
    enum FMT : int {
        GREY8       = 1,
        RGB888      = 3,
        RGBA8888    = 4,
    };

    // 縮小方法
    enum MODE : int {
        BILINEAR    = (1 << 8),
        REDUCE_BY_2 = (1 << 9), // port from VirtualDub filter(ちょっと綺麗かも)
    };

    // format:  処理を行う画像形式
    // mode:    縮小方法
    // align:   中間バッファのメモリアライメント(16以上の2の累乗)
    ResizeHalf(const FMT format, const MODE mode=BILINEAR, const size_t align=16);
    ~ResizeHalf();

    // 処理を行う画像形式を変更する
    void setFormat(const FMT format) noexcept;

    // 縮小方法を変更する
    void setProcMode(const MODE mode) noexcept;

    // 画像を縦横2分の1(小数点以下切り捨て)に縮小する
    // dstp      : 縮小後の画像を書き込むバッファの先頭アドレス
    //             nullptrの場合は中間バッファからのコピーを行わない
    // srcp      : 元画像の先頭アドレス
    // src_width :　〃　　幅
    // src_height:　〃　　高さ
    // dst_stride: 縮小後の画像を書き込むバッファのstride
    // src_stride: 元画像のstride
    // ※ dst_strideとsrc_strideは0の場合はそれぞれWindows Bitmapの標準として扱う
    void resizeHV(uint8_t* dstp, const uint8_t* srcp, const size_t src_width,
                  const size_t src_height, const size_t dst_stride=0,
                  const size_t src_stride=0);

    // 画像を水平方向のみ2分の1(小数点以下切り捨て)に縮小する
    void resizeHorizontal(uint8_t* dstp, const uint8_t* srcp,
                          const size_t src_width, const size_t src_height,
                          const size_t dst_stride=0, const size_t src_stride=0);

    // 画像を垂直方向のみ2分の1(小数点以下切り捨て)に縮小する
    void resizeVertical(uint8_t* dstp, const uint8_t* srcp,
                        const size_t src_width, const size_t src_height,
                        const size_t dst_stride=0, const size_t src_stride=0);

    // 縮小処理済みの画像データが格納されている中間バッファの先頭アドレスを返す
    const uint8_t* data() const noexcept { return image; }

    // 中間バッファに現在格納されている縮小済み画像の幅
    const size_t getWidth() const noexcept { return width; }

    // 中間バッファに現在格納されている縮小済み画像のラインあたりの有効byte数
    const size_t getRowsize() const noexcept { return width * format; }

    // 中間バッファに現在格納されている縮小済み画像の高さ
    const size_t getHeight() const noexcept { return height; }

    // 中間バッファに現在格納されている縮小済み画像のstride
    const size_t getStride() const noexcept { return stride; }

    // 現在設定されている処理する画像の形式
    const int getFormat() const noexcept { return format; }

    // 現在設定されている縮小方法
    const int getProcMode() const noexcept { return mode; }
};


#endif // RESIZE_HALF_H
