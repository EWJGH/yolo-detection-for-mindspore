//
// Created on 2024/3/4.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef TestXComponent_cameraLinkThreadWorker_H
#define TestXComponent_cameraLinkThreadWorker_H



#include <bits/alltypes.h>
#define IMAGE_BUFFER_SIZE 10
#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 640
#include <opencv2/opencv.hpp>
constexpr int cacheQueue = 10;
constexpr int img_size = 640;
extern std::queue<cv::Mat> DisplayQueue;
extern std::mutex DisplayMutex;

class CameraLinkThreadWorker 
{
public:
    static void new_buffer_cb(void *stream, void *user_data);
    //~CameraLinkThreadWorker();
    static bool isConnected;
    static bool isTimeout;
    static uint64_t lastTimeStamp;
    // 配置相机参数
    int _configCamera();
    void startConnect();
    // 得到新的帧画面数据
    void _gotNewFrame(const void *data, uint64_t timestamp);
    
    int m_imageWidth;
    int m_imageHeight;
    
    static void _initImageDataBuffers(int bufferSize, int imageWidth, int imageHeight);
    void disConnect();
        // 将image中的数据缓存起来，返回缓存的地址
    cv::Mat _convertAndCacheFrame(unsigned char *data);
    cv::Mat m_capturedImg;                   // 相机采集的图片
};

#endif //TestXComponent_cameraLinkThreadWorker_H
