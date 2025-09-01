#include "hilog/log.h"
#include <cstdio> // for std::snprintf
#include <bits/alltypes.h>
#include <inferHandle.h>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <cmath> // for std::isfinite

#ifdef LOG_DOMAIN
#undef LOG_DOMAIN
#endif
#ifdef LOG_TAG
#undef LOG_TAG
#endif
#define LOG_DOMAIN 0x0001
#define LOG_TAG "testAIinfer"

// OH_AI_ContextHandle context;
// OH_AI_ModelHandle model;
static const char *class_names[] = {"refrigerator", "air_conditioner", "washing_machine"};
// const char *model_path = "/data/storage/el2/base/haps/entry/files/industry_sandbox_fp32_sim_rk3568.ms";
static const char *model_path = "/data/storage/el2/base/haps/entry/files/sange.ms";

struct Object {
    float x0, y0, x1, y1;
    int label;
    float prob;
};
constexpr int numclass = 3;

// ============ 可视化 ============
// 4. isfinite_box 和 drawRectangle 函数
static inline bool isfinite_box(const Object &o) {
    return std::isfinite(o.x0) && std::isfinite(o.y0) && std::isfinite(o.x1) && std::isfinite(o.y1);
}

static void drawRectangle(cv::Mat &image, std::vector<Object> &objects) {
    for (size_t i = 0; i < objects.size(); i++) {
        Object obj = objects[i];

        if (!isfinite_box(obj))
            continue;
        if (obj.x1 < obj.x0)
            std::swap(obj.x0, obj.x1);
        if (obj.y1 < obj.y0)
            std::swap(obj.y0, obj.y1);

        obj.x0 = std::max(0.f, std::min(obj.x0, (float)image.cols - 1));
        obj.y0 = std::max(0.f, std::min(obj.y0, (float)image.rows - 1));
        obj.x1 = std::max(0.f, std::min(obj.x1, (float)image.cols - 1));
        obj.y1 = std::max(0.f, std::min(obj.y1, (float)image.rows - 1));
        if (obj.x1 - obj.x0 < 1.f || obj.y1 - obj.y0 < 1.f)
            continue;

        int lab = (obj.label >= 0 && obj.label < numclass) ? obj.label : 0;

        cv::rectangle(image, cv::Point((int)obj.x0, (int)obj.y0), cv::Point((int)obj.x1, (int)obj.y1),
                      cv::Scalar(255, 0, 0), 2);

        char text[256];
        std::snprintf(text, sizeof(text), "%s %.1f%%", class_names[lab], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = (int)obj.x0;
        int y = (int)obj.y0 - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);
        cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0), 1);
    }
}

// ============ NMS/排序 ============
static inline float intersection_area(const Object &a, const Object &b) {
    float inter_w = std::max(0.f, std::min(a.x1, b.x1) - std::max(a.x0, b.x0));
    float inter_h = std::max(0.f, std::min(a.y1, b.y1) - std::max(a.y0, b.y0));
    return inter_w * inter_h;
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;
    while (i <= j) {
        while (faceobjects[i].prob > p)
            i++;
        while (faceobjects[j].prob < p)
            j--;
        if (i <= j) {
            std::swap(faceobjects[i], faceobjects[j]);
            i++;
            j--;
        }
    }
    if (left < j)
        qsort_descent_inplace(faceobjects, left, j);
    if (i < right)
        qsort_descent_inplace(faceobjects, i, right);
}
static void qsort_descent_inplace(std::vector<Object> &faceobjects) {
    if (faceobjects.empty())
        return;
    qsort_descent_inplace(faceobjects, 0, (int)faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked, float nms_threshold,
                              bool /*agnostic*/ = false) {
    picked.clear();
    const int n = (int)faceobjects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = (faceobjects[i].x1 - faceobjects[i].x0) * (faceobjects[i].y1 - faceobjects[i].y0);
    }
    for (int i = 0; i < n; i++) {
        const Object &a = faceobjects[i];
        bool keep = true;
        for (int j : picked) {
            const Object &b = faceobjects[j];
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[j] - inter_area;
            if (union_area <= 0.f)
                continue;
            if (inter_area / union_area > nms_threshold) {
                keep = false;
                break;
            }
        }
        if (keep)
            picked.push_back(i);
    }
}

// ============ 前处理：letterbox & HWC->CHW ============
struct LetterboxInfo {
    cv::Mat img640; // BGR, 640x640
    float r;        // 缩放比例
    float dw;       // 左右 padding
    float dh;       // 上下 padding
};

static LetterboxInfo letterbox_to_640(const cv::Mat &src) {
    const int target = 640;
    float r = std::min(target * 1.f / src.cols, target * 1.f / src.rows);
    int new_w = (int)std::round(src.cols * r);
    int new_h = (int)std::round(src.rows * r);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));

    int dw = (target - new_w) / 2;
    int dh = (target - new_h) / 2;

    cv::Mat out(target, target, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(dw, dh, new_w, new_h)));
    return {out, r, (float)dw, (float)dh};
}

// src: 640x640x3, CV_32FC3, RGB, [0,1]
// dst: float[3*640*640]（CHW）
static void hwc_to_chw(const cv::Mat &src, float *dst) {
    const int C = 3, H = src.rows, W = src.cols;
    std::vector<cv::Mat> channels;
    cv::split(src, channels); // channels[0]=R, [1]=G, [2]=B（见调用前的 BGR->RGB）
    for (int c = 0; c < C; ++c) {
        memcpy(dst + c * H * W, channels[c].ptr<float>(), H * W * sizeof(float));
    }
}

static void postprocess(std::vector<float *> outputPtr, std::vector<int64_t> outputSize, std::vector<Object> &proposal,
                        float r, float dw, float dh, // 这三个要用！
                        int orig_w, int orig_h, cv::Mat &debug_draw) {
    proposal.clear();
    if (outputPtr.empty() || outputSize.empty())
        return;

    const int64_t elems = outputSize[0];

    // 推断每行维度 D（常见为 11 或 13）：elems = rows * D，rows 通常 25200
    int D = (elems % 25200 == 0) ? (int)(elems / 25200) : (numclass + 5);
    if (D < numclass + 5)
        D = numclass + 5; // 至少包含 [cx,cy,w,h,obj]
    const int rows = (int)(elems / D);

    float *data = outputPtr[0];

    for (int i = 0; i < rows; ++i) {
        float cx = data[0];
        float cy = data[1];
        float ww = data[2];
        float hh = data[3];

        float obj = data[4]; // 如果你的模型输出没做 sigmoid，这里请自行加 sigmoid
        if (obj >= 0.50f) {
            // 最大类别
            int cls_id = 0;
            float cls_p = 0.f;
            for (int k = 0; k < numclass && (5 + k) < D; ++k) {
                float p = data[5 + k]; // 同理，如需 sigmoid 请加
                if (p > cls_p) {
                    cls_p = p;
                    cls_id = k;
                }
            }
            if (cls_p >= 0.25f) {
                // 先在 640×640（含 padding）坐标系下计算四角
                float x0 = (cx - 0.5f * ww);
                float y0 = (cy - 0.5f * hh);
                float x1 = (cx + 0.5f * ww);
                float y1 = (cy + 0.5f * hh);

                // 去 padding 再按等比缩放 r 还原到原图
                x0 = (x0 - dw) / r;
                y0 = (y0 - dh) / r;
                x1 = (x1 - dw) / r;
                y1 = (y1 - dh) / r;

                // 裁剪到原图边界
                x0 = std::max(0.f, std::min(x0, (float)orig_w - 1));
                y0 = std::max(0.f, std::min(y0, (float)orig_h - 1));
                x1 = std::max(0.f, std::min(x1, (float)orig_w - 1));
                y1 = std::max(0.f, std::min(y1, (float)orig_h - 1));

                if ((x1 - x0) >= 1.f && (y1 - y0) >= 1.f) {
                    // 调试：画中心点
                    if (!debug_draw.empty()) {
                        float dcx = 0.5f * (x0 + x1);
                        float dcy = 0.5f * (y0 + y1);
                        cv::circle(debug_draw, cv::Point((int)dcx, (int)dcy), 2, cv::Scalar(128, 128, 128), -1);
                    }
                    proposal.push_back({x0, y0, x1, y1, cls_id, obj * cls_p});
                }
            }
        }
        data += D; // ★★ 每次前进真实的 D，而不是 numclass+5
    }

    // 置信度降序 + NMS
    qsort_descent_inplace(proposal);
    std::vector<int> picked;
    nms_sorted_bboxes(proposal, picked, 0.25f);

    // 收集 NMS 后结果并做可选中心去重
    std::vector<Object> objects;
    objects.reserve(picked.size());
    for (int idx : picked)
        objects.push_back(proposal[idx]);

    auto center_dist = [](const Object &a, const Object &b) {
        float ax = 0.5f * (a.x0 + a.x1), ay = 0.5f * (a.y0 + a.y1);
        float bx = 0.5f * (b.x0 + b.x1), by = 0.5f * (b.y0 + b.y1);
        float dx = ax - bx, dy = ay - by;
        return std::sqrt(dx * dx + dy * dy);
    };

    const float CENTER_MERGE_THRESH = 12.0f;
    std::vector<Object> dedup;
    std::vector<bool> used(objects.size(), false);
    for (size_t i = 0; i < objects.size(); ++i) {
        if (used[i])
            continue;
        used[i] = true;
        const Object &keep = objects[i];
        for (size_t j = i + 1; j < objects.size(); ++j) {
            if (used[j])
                continue;
            if (center_dist(keep, objects[j]) < CENTER_MERGE_THRESH)
                used[j] = true;
        }
        dedup.push_back(keep);
    }
    proposal = std::move(dedup);
}


// ============ OH_AI 初始化/销毁 ============
int InferHandle::init() {
    OH_LOG_INFO(LOG_APP, "Model path: %{public}s", model_path);

    context = OH_AI_ContextCreate();
    if (context == NULL) {
        OH_LOG_INFO(LOG_APP, "OH_AI_ContextCreate failed.");
        return OH_AI_STATUS_LITE_ERROR;
    }
    OH_LOG_INFO(LOG_APP, "OH_AI_ContextCreate success.");

    OH_AI_DeviceInfoHandle nnrt_device_info = OH_AI_CreateNNRTDeviceInfoByType(OH_AI_NNRTDEVICE_ACCELERATOR);
    if (nnrt_device_info == NULL) {
        OH_LOG_INFO(LOG_APP, "OH_AI_DeviceInfoCreate failed.");
        OH_AI_ContextDestroy(&context);
        return OH_AI_STATUS_LITE_ERROR;
    }
    OH_AI_DeviceInfoSetPerformanceMode(nnrt_device_info, OH_AI_PERFORMANCE_EXTREME);
    OH_AI_ContextAddDeviceInfo(context, nnrt_device_info);

    OH_AI_DeviceInfoHandle cpu_device_info = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
    if (cpu_device_info == NULL) {
        OH_LOG_INFO(LOG_APP, "OH_AI_DeviceInfoCreate failed.");
        OH_AI_ContextDestroy(&context);
        return OH_AI_STATUS_LITE_ERROR;
    }
    OH_AI_ContextAddDeviceInfo(context, cpu_device_info);
    OH_LOG_INFO(LOG_APP, "Device infos added.");

    model = OH_AI_ModelCreate();
    if (model == NULL) {
        OH_LOG_INFO(LOG_APP, "OH_AI_ModelCreate failed.");
        OH_AI_ContextDestroy(&context);
        return OH_AI_STATUS_LITE_ERROR;
    }
    OH_LOG_INFO(LOG_APP, "OH_AI_ModelCreate success.");

    int trytimes = 5;
    int ret;
    while ((ret = OH_AI_ModelBuildFromFile(model, model_path, OH_AI_MODELTYPE_MINDIR, context)) !=
           OH_AI_STATUS_SUCCESS) {
        OH_LOG_INFO(LOG_APP, "OH_AI_ModelBuildFromFile failed, ret=%{public}d, try_left=%{public}d", ret, trytimes - 1);
        trytimes--;
        if (trytimes <= 0)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (ret != OH_AI_STATUS_SUCCESS) {
        OH_LOG_INFO(LOG_APP, "OH_AI_ModelBuildFromFile failed, ret=%{public}d", ret);
        OH_AI_ModelDestroy(&model);
    } else {
        OH_LOG_INFO(LOG_APP, "OH_AI_ModelBuildFromFile success, ret=%{public}d", ret);
    }
    return ret;
}

int InferHandle::destroy() {
    OH_AI_ModelDestroy(&model);
    OH_AI_ContextDestroy(&context);
    return OH_AI_STATUS_SUCCESS;
}

// ============ OH_AI 输入拷贝工具 ============
static int GenerateInputDataWithPtr(OH_AI_TensorHandleArray inputs, const void *inputdata, size_t bytes) {
    if (inputs.handle_num == 0)
        return OH_AI_STATUS_LITE_ERROR;
    void *input_data = OH_AI_TensorGetMutableData(inputs.handle_list[0]);
    if (!input_data) {
        OH_LOG_INFO(LOG_APP, "MSTensorGetMutableData failed.");
        return OH_AI_STATUS_LITE_ERROR;
    }
    memcpy(input_data, inputdata, bytes);
    return OH_AI_STATUS_SUCCESS;
}

// ============ 推理入口（单帧） ============
cv::Mat InferHandle::inferenceWithPtr(cv::Mat src) {
    OH_LOG_INFO(LOG_APP, "Input image: w=%{public}d h=%{public}d c=%{public}d", src.cols, src.rows, src.channels());

    // 1) letterbox 到 640×640（BGR）
    LetterboxInfo lb = letterbox_to_640(src);

    // 2) BGR -> RGB
    cv::Mat rgb;
    cv::cvtColor(lb.img640, rgb, cv::COLOR_BGR2RGB);

    // 3) float32 /255（OpenCV 的 CV_32FC3 默认是 HWC 布局）
    cv::Mat rgb32f;
    rgb.convertTo(rgb32f, CV_32FC3, 1.0 / 255.0);
    if (!rgb32f.isContinuous())
        rgb32f = rgb32f.clone(); // 确保连续内存

    // 4) 绘框底图（BGR 原图）
    cv::Mat drawbox = src.clone();

    // ==== 推理 ====
    int ret;
    OH_AI_TensorHandleArray inputs = OH_AI_ModelGetInputs(model);
    if (inputs.handle_list == NULL) {
        OH_LOG_INFO(LOG_APP, "OH_AI_ModelGetInputs failed.");
        return drawbox;
    }

    // 核对输入 shape 是否为 NHWC: 1x640x640x3
    size_t in_rank = 0;
    const int64_t *in_shape = OH_AI_TensorGetShape(inputs.handle_list[0], &in_rank);
    int64_t in_elems = OH_AI_TensorGetElementNum(inputs.handle_list[0]);

    if (!(in_rank == 4 && in_shape[0] == 1 && in_shape[1] == 640 && in_shape[2] == 640 && in_shape[3] == 3)) {
        OH_LOG_INFO(LOG_APP,
                    "Unexpected input shape (expect NHWC 1x640x640x3). rank=%{public}zu, "
                    "shape=%{public}ld,%{public}ld,%{public}ld,%{public}ld",
                    in_rank, in_rank > 0 ? in_shape[0] : -1, in_rank > 1 ? in_shape[1] : -1,
                    in_rank > 2 ? in_shape[2] : -1, in_rank > 3 ? in_shape[3] : -1);
        // 你也可以直接 return，避免错布局继续跑
    }

    size_t need_bytes = (size_t)in_elems * sizeof(float); // 应为 1*640*640*3*sizeof(float)

    // ★★ 关键：直接拷贝 NHWC(HWC) 的 rgb32f 数据
    ret = GenerateInputDataWithPtr(inputs, (const void *)rgb32f.data, need_bytes);
    if (ret != OH_AI_STATUS_SUCCESS) {
        OH_LOG_INFO(LOG_APP, "GenerateInputDataWithPtr failed, ret=%{public}d.", ret);
        return drawbox;
    }
    OH_LOG_INFO(LOG_APP, "GenerateInputDataWithPtr success.");

    // 前向
    OH_AI_TensorHandleArray outputs;
    ret = OH_AI_ModelPredict(model, inputs, &outputs, NULL, NULL);
    if (ret != OH_AI_STATUS_SUCCESS) {
        OH_LOG_INFO(LOG_APP, "OH_AI_ModelPredict failed, ret=%{public}d.", ret);
        return drawbox;
    }
    OH_LOG_INFO(LOG_APP, "OH_AI_ModelPredict success, outputs=%{public}zu", outputs.handle_num);

    // 收集输出
    std::vector<float *> outTensorPtr;
    std::vector<int64_t> outTensorSize;
    for (size_t i = 0; i < outputs.handle_num; ++i) {
        OH_AI_TensorHandle tensor = outputs.handle_list[i];
        size_t rank = 0;
        const int64_t *shape = OH_AI_TensorGetShape(tensor, &rank);
        int64_t element_num = OH_AI_TensorGetElementNum(tensor);
        OH_LOG_INFO(LOG_APP, "Tensor name=%{public}s size=%{public}zu elems=%{public}ld rank=%{public}zu",
                    OH_AI_TensorGetName(tensor), (size_t)OH_AI_TensorGetDataSize(tensor), (long)element_num, rank);
        for (size_t r = 0; r < rank; ++r)
            OH_LOG_INFO(LOG_APP, "  dim[%{public}zu]=%{public}ld", r, (long)shape[r]);

        // 打印前 16 个元素
        float *p = (float *)OH_AI_TensorGetData(tensor);
        std::string s = "  head=";
        int take = std::min<int64_t>(16, element_num);
        for (int t = 0; t < take; ++t) {
            char buf[48];
            std::snprintf(buf, sizeof(buf), (t ? ",%g" : "%g"), p[t]);
            s += buf;
        }
        OH_LOG_INFO(LOG_APP, "%{public}s", s.c_str());

        outTensorPtr.push_back((float *)OH_AI_TensorGetData(tensor));
        outTensorSize.push_back(element_num);
    }

    // 挑元素最多的输出做后处理（一般是主输出）
    size_t best = 0;
    for (size_t i = 1; i < outTensorSize.size(); ++i)
        if (outTensorSize[i] > outTensorSize[best])
            best = i;

    std::vector<float *> chosenPtr = {outTensorPtr[best]};
    std::vector<int64_t> chosenSize = {outTensorSize[best]};
    OH_LOG_INFO(LOG_APP, "choose output index %{public}zu elems=%{public}ld as main", best, (long)chosenSize[0]);

    // 后处理（用 r/dw/dh 做坐标回映射）
    OH_LOG_INFO(LOG_APP, "post process start");
    std::vector<Object> proposal;
    postprocess(chosenPtr, chosenSize, proposal, lb.r, lb.dw, lb.dh, src.cols, src.rows, drawbox);
    drawRectangle(drawbox, proposal);
    OH_LOG_INFO(LOG_APP, "post process end, dets=%{public}zu", proposal.size());

    return drawbox;
}
