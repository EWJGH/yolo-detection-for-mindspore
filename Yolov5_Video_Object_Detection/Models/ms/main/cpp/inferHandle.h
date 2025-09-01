//
// Created on 2024/3/4.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef TestXComponent_inferHandle_H
#define TestXComponent_inferHandle_H

#include <mindspore/model.h>
#include <mindspore/context.h>
#include <mindspore/status.h>
#include <mindspore/tensor.h>
#include <mindspore/types.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class InferHandle {
    public:
    OH_AI_ContextHandle context;
    OH_AI_ModelHandle model;
    int init();
    cv::Mat inferenceWithPtr(cv::Mat src);
    int destroy();
    
};
#endif //TestXComponent_infer_H
