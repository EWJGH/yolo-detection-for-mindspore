#include <opencv2/opencv.hpp>
#include "inferHandle.h"
#include "hilog/log.h"
#include <sys/stat.h>
#include <errno.h>
#include <string.h>
#include <cstdio>

#ifdef LOG_DOMAIN
#undef LOG_DOMAIN
#endif
#ifdef LOG_TAG
#undef LOG_TAG
#endif
#define LOG_DOMAIN 0x0001
#define LOG_TAG "test_infer"

// 简单的 PPM 保存（兜底用，不依赖任何编解码）
static bool save_ppm_rgb(const std::string &path, const cv::Mat &bgr) {
    // OpenCV 默认是 BGR，这里转成 RGB 再写
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) {
        OH_LOG_ERROR(LOG_APP, "PPM fopen failed: %s (%d)", strerror(errno), errno);
        return false;
    }
    fprintf(f, "P6\n%d %d\n255\n", rgb.cols, rgb.rows);
    fwrite(rgb.data, 1, rgb.cols * rgb.rows * 3, f);
    fclose(f);
    return true;
}

int main() {
    // 建议统一使用应用可写目录（Stage 模型沙箱）：
    const char *kFilesDir = "/data/storage/el2/base/files";
    // 确保目录存在（通常存在；防御一下）
    mkdir(kFilesDir, 0755);

    std::string inPath = std::string(kFilesDir) + "/test.jpg";   // 把测试图片放这里
    std::string outPng = std::string(kFilesDir) + "/result.png"; // 先用 PNG，成功率高
    std::string outPpm = std::string(kFilesDir) + "/result.ppm"; // 兜底：PPM 总能成

    OH_LOG_INFO(LOG_APP, "准备初始化 infer 模型...");
    InferHandle infer;
    int init_ret = infer.init();
    if (init_ret != 0) {
        OH_LOG_ERROR(LOG_APP, "infer.init() 失败，返回值：%d", init_ret);
        return -1;
    }
    OH_LOG_INFO(LOG_APP, "模型初始化完成");

    OH_LOG_INFO(LOG_APP, "读取图片: %s", inPath.c_str());
    cv::Mat image = cv::imread(inPath, cv::IMREAD_COLOR);
    if (image.empty()) {
        OH_LOG_ERROR(LOG_APP, "读取图片失败。请确认图片已推到: %s", inPath.c_str());
        // 你可以先用 hdc 推一张上去:
        // hdc file send ./test.jpg /data/storage/el2/base/files/test.jpg
        return -1;
    }

    OH_LOG_INFO(LOG_APP, "准备调用 inferenceWithPtr");
    cv::Mat result = infer.inferenceWithPtr(image);
    OH_LOG_INFO(LOG_APP, "推理完成，准备保存结果");

    // 先尝试 PNG（大多数构建都支持）
    bool ok = cv::imwrite(outPng, result);
    if (!ok) {
        OH_LOG_ERROR(LOG_APP, "cv::imwrite PNG 失败，尝试 PPM 兜底: %s", outPng.c_str());
        bool ok2 = save_ppm_rgb(outPpm, result);
        if (!ok2) {
            OH_LOG_ERROR(LOG_APP, "PPM 兜底也失败，请检查目录权限/磁盘/路径。");
            return -1;
        } else {
            OH_LOG_INFO(LOG_APP, "PPM 兜底保存成功: %s", outPpm.c_str());
        }
    } else {
        OH_LOG_INFO(LOG_APP, "结果 PNG 保存成功: %s", outPng.c_str());
    }

    infer.destroy();
    return 0;
}
