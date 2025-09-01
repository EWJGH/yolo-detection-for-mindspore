//
// Created on 2024/3/4.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".
#include <arv.h>
#include <bits/alltypes.h>
#include <chrono>
#include <thread>
#include <cameraLinkThreadWorker.h>
#include "hilog/log.h"
#ifdef LOG_DOMAIN
#undef LOG_DOMAIN
#endif

#ifdef LOG_TAG
#undef LOG_TAG
#endif

#define LOG_DOMAIN 0x0001
#define LOG_TAG "testAIrender"

ArvCamera *arvCamera = NULL;
ArvStream *arvStream = NULL;

std::queue<cv::Mat> DisplayQueue;
std::mutex DisplayMutex;

void releaseCamera() {
    if (arvCamera) {
        arv_camera_stop_acquisition(arvCamera, NULL);
    }
    if (arvStream) {
        arv_stream_set_emit_signals(arvStream, FALSE);
    }
    if (arvStream) {
        /* Destroy the stream object */
        g_clear_object(&arvStream);
        arvStream = NULL;
    }
    if (arvCamera) {
        /* Destroy the camera instance */
        g_clear_object(&arvCamera);
        arvCamera = NULL;
    }
}

bool CameraLinkThreadWorker::isConnected = false;
bool CameraLinkThreadWorker::isTimeout = false;
// cv::Mat* CameraLinkThreadWorker::_matBuffer[IMAGE_BUFFER_SIZE]={};
uint64_t CameraLinkThreadWorker::lastTimeStamp = 0;

static uint64_t frameID = 0;
// 相机数据回调函数
void CameraLinkThreadWorker::new_buffer_cb(void *stream, void *user_data) {

    OH_LOG_INFO(LOG_APP, "enter new buffer callback");
    ArvStream *arvStream = static_cast<ArvStream *>(stream);
    /* This code is called from the stream receiving thread, which means all the time spent there is less time
     * available for the reception of incoming packets */
    ArvBuffer *frameBuffer = arv_stream_pop_buffer(arvStream);
    ArvBufferStatus status = arv_buffer_get_status(frameBuffer);
    if (status == ARV_BUFFER_STATUS_SUCCESS) {
        isTimeout = false;
        lastTimeStamp = arv_buffer_get_system_timestamp(frameBuffer) / 1000; // 帧收到时的系统时间戳
        frameID = arv_buffer_get_frame_id(frameBuffer);
        OH_LOG_INFO(LOG_APP, "frameBuffer system timestamp is %llu , frameID is %llu", lastTimeStamp, frameID);
        if (frameID % 10 == 1) {
            size_t size;
            const void *data = arv_buffer_get_data(frameBuffer, &size);
            CameraLinkThreadWorker *worker = static_cast<CameraLinkThreadWorker *>(user_data);
            worker->_gotNewFrame(data, lastTimeStamp);
        }
    }
    if (status == ARV_BUFFER_STATUS_TIMEOUT) {
        isTimeout = true;
        OH_LOG_INFO(LOG_APP, "frameBuffer timeout", lastTimeStamp, frameID);
    }
    /* Don't destroy the buffer, but put it back into the buffer pool */
    arv_stream_push_buffer(arvStream, frameBuffer);
}

int CameraLinkThreadWorker::_configCamera() {
    GError *error = NULL;
    arv_camera_set_acquisition_mode(arvCamera, ARV_ACQUISITION_MODE_CONTINUOUS, &error);
    if (error != NULL) {
        return 1;
    }
    arvStream = arv_camera_create_stream(arvCamera, NULL, NULL, &error);
    if (!ARV_IS_STREAM(arvStream)) {
        return 2;
    }
    size_t payload;
    /* Retrieve the payload size for buffer creation */
    payload = arv_camera_get_payload(arvCamera, &error);
    if (error != NULL) {
        g_clear_object(&arvStream);
        arvStream = NULL;
        return 3;
    }
    /* Insert some buffers in the stream buffer pool */
    for (int i = 0; i < 30; i++) {
        arv_stream_push_buffer(arvStream, arv_buffer_new(payload, NULL));
    }
    /*缓存*/
    int x, y;
    arv_camera_get_region(arvCamera, &x, &y, &m_imageWidth, &m_imageHeight, &error);
    if (error != NULL) {
        g_clear_object(&arvStream);
        arvStream = NULL;
        return 4;
    }
    OH_LOG_INFO(LOG_APP, "payload:%zu x:%d y:%d width:%d height:%d", payload, x, y, m_imageWidth, m_imageHeight);
    _initImageDataBuffers(payload, IMAGE_WIDTH /*m_imageWidth*/, IMAGE_HEIGHT /*m_imageHeight*/);

    g_signal_connect(arvStream, "new-buffer", G_CALLBACK(&new_buffer_cb), this);
    arv_stream_set_emit_signals(arvStream, TRUE);
    /* Start the acquisition */
    arv_camera_start_acquisition(arvCamera, &error);
    return 0;
}

void CameraLinkThreadWorker::_gotNewFrame(const void *data, uint64_t timestamp) {

    OH_LOG_INFO(LOG_APP, "to produce Display Queue size is： %d", DisplayQueue.size());
    if (DisplayQueue.size() < IMAGE_BUFFER_SIZE) {
        // TODO 图像处理
        cv::Mat cachedMat = _convertAndCacheFrame((unsigned char *)data);
        std::unique_lock<std::mutex> lock(DisplayMutex);
        DisplayQueue.push(cachedMat);
        // OH_LOG_INFO(LOG_APP, "push new frame to queue");
        lock.unlock();
    }
    return;
}

void CameraLinkThreadWorker::startConnect() {
    // 枚举设备并打开第一台设备
    do {
        GError *error = NULL;
        /* Connect to the first available camera */
        arvCamera = arv_camera_new(NULL, &error);
        if (!ARV_IS_CAMERA(arvCamera)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        } else {
            OH_LOG_INFO(LOG_APP, "Found camera:", arv_camera_get_model_name(arvCamera, NULL));
            int rtn = _configCamera();
            if (rtn == 0) {
                // TODO offlinecheck
                OH_LOG_INFO(LOG_APP, "camera collecting finished");
                isConnected = true;
                isTimeout = false;

                break;
            } else {
                OH_LOG_INFO(LOG_APP, "config camera error %d ", rtn);
                if (arvCamera) {
                    /* Destroy the camera instance */
                    g_clear_object(&arvCamera);
                    arvCamera = NULL;
                }
            }
        }
    } while (1);
}
void CameraLinkThreadWorker::disConnect() {
    OH_LOG_INFO(LOG_APP, "CameraLinkThreadWorker slt_disConnect in");
    isConnected = false;
    releaseCamera();
    OH_LOG_INFO(LOG_APP, "CameraLinkThreadWorker slt_disConnect out");
}
void CameraLinkThreadWorker::_initImageDataBuffers(int bufferSize, int imageWidth, int imageHeight) {
    static bool inited = false;
    DisplayQueue.empty();
    OH_LOG_INFO(LOG_APP, "CameraLinkThreadWorker _initImageDataBuffers %d %d ", imageWidth, imageHeight);
    if (!inited) {
        inited = true;
        return;
    }
}

cv::Mat CameraLinkThreadWorker::_convertAndCacheFrame(unsigned char *data) {
    cv::Mat rgbMat;
    cv::Mat bayerrgMat = cv::Mat(m_imageHeight, m_imageWidth, CV_8UC1, (void *)data);
    // cv::cvtColor(bayerrgMat, rgbMat, cv::COLOR_BayerRGGB2BGR);
    cv::cvtColor(bayerrgMat, rgbMat, cv::COLOR_BayerRGGB2RGB);
    return rgbMat;
}
