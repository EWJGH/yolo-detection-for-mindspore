// ui_main.cpp - VS2019 Win32 GUI（左侧按钮 + 右侧视频面板）+ Galaxy 相机 + YOLO(OpenCV DNN)
//#define UNICODE
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#include <codecvt>
#define NOMINMAX
#include <windows.h>
#include <commctrl.h>
#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>     // ✅ 新增
#include <algorithm>   // ✅ 新增
#include <opencv2/opencv.hpp>

// ====== Galaxy 头文件 ======
#include "GalaxyIncludes.h"
using namespace GxIAPICPP;

// ====== 控件ID / 布局 ======
#define ID_BTN_START  1001
#define ID_BTN_STOP   1002
const int PANEL_W = 220; // 左侧控制栏宽度

// ====== GUI 全局 ======
HWND g_hMain = nullptr;
HWND g_hVideo = nullptr;

// ====== 共享帧缓存 ======
std::mutex g_mtx;
cv::Mat g_frameBGRA;                  // BGRA（top-down），VideoViewProc 里绘制
std::atomic<bool> g_detecting{ false };

// ====== YOLO 全局 ======
cv::dnn::Net g_net;
std::vector<std::string> g_classes;

// ====== Galaxy 全局 ======
CGXDevicePointer          g_dev;
CGXStreamPointer          g_strm;
CGXFeatureControlPointer  g_feat;
IDeviceOfflineEventHandler* g_pDevOfflineHandler = nullptr;
ICaptureEventHandler* g_pCaptureHandler = nullptr;
GX_DEVICE_OFFLINE_CALLBACK_HANDLE g_hDevOffline = nullptr;

// -------------------- YOLO 工具函数 --------------------
std::vector<std::string> load_class_list() {
    std::vector<std::string> class_list;
    std::ifstream ifs("Models/classes.txt");
    std::string line;
    while (getline(ifs, line)) class_list.push_back(line);
    return class_list;
}

void load_net(cv::dnn::Net& net, bool is_cuda) {
    auto result = cv::dnn::readNet("Models/sange.onnx");
    if (is_cuda) {
        std::cout << "Using CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else {
        std::cout << "CPU Mode\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

const std::vector<cv::Scalar> colors = {
    cv::Scalar(255,255,0), cv::Scalar(0,255,0),
    cv::Scalar(0,255,255), cv::Scalar(255,0,0)
};

const float INPUT_WIDTH = 640.0f;
const float INPUT_HEIGHT = 640.0f;
const float SCORE_THRESHOLD = 0.1f;
const float NMS_THRESHOLD = 0.3f;
const float CONFIDENCE_THRESHOLD = 0.3f;

struct Detection { int class_id; float confidence; cv::Rect box; };

cv::Mat format_yolov5(const cv::Mat& source) {
    int col = source.cols, row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void detect(cv::Mat& image, cv::dnn::Net& net,
    std::vector<Detection>& output,
    const std::vector<std::string>& className) {
    cv::Mat blob;
    auto input_image = format_yolov5(image);

    cv::dnn::blobFromImage(input_image, blob, 1.f / 255.f,
        cv::Size(INPUT_WIDTH, INPUT_HEIGHT),
        cv::Scalar(), false, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    static bool once = false;
    if (!once) {
        std::cout << "out dims: ";
        for (int i = 0; i < outputs[0].dims; ++i) std::cout << outputs[0].size[i] << " ";
        std::cout << "  total=" << outputs[0].total() << "\n";
        // 粗查 obj 最大值
        float* p = (float*)outputs[0].data; float max_obj = 0.f;
        for (int i = 0; i < 25200; ++i) { if (p[4] > max_obj) max_obj = p[4]; p += 8; }
        std::cout << "max obj = " << max_obj << "\n";
        once = true;
    }

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float* data = (float*)outputs[0].data;
    const int dimensions = 8;   // 
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float obj = data[4];
        if (obj >= CONFIDENCE_THRESHOLD) {
            float* classes_scores = data + 5;
            cv::Mat scores(1, (int)className.size(), CV_32FC1, classes_scores);
            cv::Point class_id; double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            if (max_class_score > SCORE_THRESHOLD) {
                float conf = obj * (float)max_class_score;
                confidences.push_back(conf);
                class_ids.push_back(class_id.x);

                float x = data[0], y = data[1], w = data[2], h = data[3];
                int left = int((x - 0.5f * w) * x_factor);
                int top = int((y - 0.5f * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.emplace_back(left, top, width, height);
            }
        }
        data += dimensions;
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, keep);
    for (int idx : keep) output.push_back({ class_ids[idx], confidences[idx], boxes[idx] });
}

// -------------------- 绘制窗口：把 g_frameBGRA 画出来 --------------------
LRESULT CALLBACK VideoViewProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_PAINT: {
        PAINTSTRUCT ps; HDC hdc = BeginPaint(hWnd, &ps);
        RECT rc; GetClientRect(hWnd, &rc);
        HBRUSH bg = CreateSolidBrush(RGB(20, 20, 20));
        FillRect(hdc, &rc, bg); DeleteObject(bg);

        cv::Mat local;
        { std::lock_guard<std::mutex> lk(g_mtx); if (!g_frameBGRA.empty()) local = g_frameBGRA.clone(); }
        if (!local.empty()) {
            BITMAPINFO bmi{};
            bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
            bmi.bmiHeader.biWidth = local.cols;
            bmi.bmiHeader.biHeight = -local.rows; // top-down
            bmi.bmiHeader.biPlanes = 1;
            bmi.bmiHeader.biBitCount = 32;
            bmi.bmiHeader.biCompression = BI_RGB;

            RECT rcw; GetClientRect(hWnd, &rcw);
            int cw = rcw.right - rcw.left, ch = rcw.bottom - rcw.top;
            double s = std::min(double(cw) / local.cols, double(ch) / local.rows);
            int dw = int(local.cols * s), dh = int(local.rows * s);
            int dx = (cw - dw) / 2, dy = (ch - dh) / 2;

            SetStretchBltMode(hdc, HALFTONE);
            StretchDIBits(hdc, dx, dy, dw, dh, 0, 0, local.cols, local.rows,
                local.data, &bmi, DIB_RGB_COLORS, SRCCOPY);
        }
        EndPaint(hWnd, &ps);
        return 0;
    }
    case WM_ERASEBKGND: return 1;
    }
    return DefWindowProc(hWnd, msg, wParam, lParam);
}

// -------------------- Galaxy 回调：取帧→YOLO→写缓存→重绘 --------------------
class CSampleDeviceOfflineEventHandler : public IDeviceOfflineEventHandler {
public:
    void DoOnDeviceOfflineEvent(void*) override {
        MessageBox(g_hMain, L"相机掉线！", L"提示", MB_ICONWARNING);
    }
};

class CSampleCaptureEventHandler : public ICaptureEventHandler {
public:
    void DoOnImageCaptured(CImageDataPointer& img, void*) override {
        if (!g_detecting.load()) return;
        if (img.IsNull() || img->GetStatus() != GX_FRAME_STATUS_SUCCESS) return;

        GX_PIXEL_FORMAT_ENTRY pf = img->GetPixelFormat();
        GX_VALID_BIT_LIST bitsel = GX_BIT_0_7;
        switch (pf) {
        case GX_PIXEL_FORMAT_MONO10:
        case GX_PIXEL_FORMAT_MONO12:
        case GX_PIXEL_FORMAT_MONO16:
        case GX_PIXEL_FORMAT_BAYER_RG10:
        case GX_PIXEL_FORMAT_BAYER_BG10:
        case GX_PIXEL_FORMAT_BAYER_GR10:
        case GX_PIXEL_FORMAT_BAYER_GB10:
        case GX_PIXEL_FORMAT_BAYER_RG12:
        case GX_PIXEL_FORMAT_BAYER_BG12:
        case GX_PIXEL_FORMAT_BAYER_GR12:
        case GX_PIXEL_FORMAT_BAYER_GB12:
            bitsel = GX_BIT_8_15; break;
        default:
            bitsel = GX_BIT_0_7; break;
        }

        void* pRGB24 = img->ConvertToRGB24(bitsel, GX_RAW2RGB_NEIGHBOUR, true);
        if (!pRGB24) return;

        cv::Mat rgb(img->GetHeight(), img->GetWidth(), CV_8UC3, pRGB24);
        cv::Mat bgr; cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);

        std::vector<Detection> dets;
        detect(bgr, g_net, dets, g_classes);

        for (const auto& d : dets) {
            cv::Scalar color = colors[d.class_id % colors.size()];
            cv::rectangle(bgr, d.box, color, 2);
            if (d.class_id >= 0 && d.class_id < (int)g_classes.size()) {
                cv::putText(bgr, g_classes[d.class_id], { d.box.x, std::max(0, d.box.y - 4) },
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
            }
        }

        cv::Mat shown; cv::cvtColor(bgr, shown, cv::COLOR_BGR2BGRA);
        { std::lock_guard<std::mutex> lk(g_mtx); g_frameBGRA = shown.clone(); }
        InvalidateRect(g_hVideo, nullptr, FALSE);
    }
};

std::wstring utf8_to_wstring(const std::string& str) {
    if (str.empty()) return std::wstring();
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
    return wstrTo;
}

// -------------------- Start / Stop --------------------
bool StartDetect() {
    if (g_detecting.load()) {
        OutputDebugStringW(L"[StartDetect] 已经在检测中\n");
        return true;
    }

    // 首次加载模型与类别
    static bool inited = false;
    if (!inited) {
        OutputDebugStringW(L"[StartDetect] 正在加载模型与类别...\n");
        g_classes = load_class_list();
        load_net(g_net, /*is_cuda=*/false);
        inited = true;
    }

    try {
        OutputDebugStringW(L"[StartDetect] 初始化 Galaxy SDK...\n");
        IGXFactory::GetInstance().Init();

        OutputDebugStringW(L"[StartDetect] 枚举设备...\n");
        gxdeviceinfo_vector devs;
        IGXFactory::GetInstance().UpdateDeviceList(1000, devs);
        if (devs.empty()) {
            MessageBox(g_hMain, L"未发现相机", L"错误", MB_ICONERROR);
            OutputDebugStringW(L"[StartDetect] 未发现相机\n");
            return false;
        }

        OutputDebugStringA(("[StartDetect] 打开设备: " + std::string(devs[0].GetSN()) + "\n").c_str());
        g_dev = IGXFactory::GetInstance().OpenDeviceBySN(devs[0].GetSN(), GX_ACCESS_EXCLUSIVE);

        OutputDebugStringW(L"[StartDetect] 打开数据流...\n");
        g_strm = g_dev->OpenStream(0);

        OutputDebugStringW(L"[StartDetect] 获取特性控制器...\n");
        g_feat = g_dev->GetRemoteFeatureControl();

        // 去掉 AcquisitionMode 设置，直接启动采集
        OutputDebugStringW(L"[StartDetect] 注册采集回调...\n");
        g_pCaptureHandler = new CSampleCaptureEventHandler();
        g_strm->RegisterCaptureCallback(g_pCaptureHandler, nullptr);

        OutputDebugStringW(L"[StartDetect] 开始抓取图像...\n");
        g_strm->StartGrab();

        OutputDebugStringW(L"[StartDetect] 执行 AcquisitionStart...\n");
        g_feat->GetCommandFeature("AcquisitionStart")->Execute();

        OutputDebugStringW(L"[StartDetect] 注册掉线回调...\n");
        g_pDevOfflineHandler = new CSampleDeviceOfflineEventHandler();
        g_hDevOffline = g_dev->RegisterDeviceOfflineCallback(g_pDevOfflineHandler, nullptr);

        g_detecting = true;
        OutputDebugStringW(L"[StartDetect] 检测启动成功\n");
        return true;

    }
    catch (CGalaxyException& e) {
        std::wstring msg = L"Galaxy 错误: " + std::to_wstring(e.GetErrorCode());

        // 如果 SDK 有 GetErrorString()
        // msg += L"\n消息: " + utf8_to_wstring(e.GetErrorString());

        // 如果没有，就用 what()
        msg += L"\n消息: " + utf8_to_wstring(e.what());

        MessageBox(g_hMain, msg.c_str(), L"错误", MB_ICONERROR);
        OutputDebugStringW((L"[StartDetect] Galaxy 异常: " + msg + L"\n").c_str());
    }

    catch (std::exception& e) {
        std::wstring msg = L"标准异常: " + std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(e.what());
        MessageBox(g_hMain, msg.c_str(), L"错误", MB_ICONERROR);
        OutputDebugStringW((L"[StartDetect] std::exception: " + msg + L"\n").c_str());
    }
    catch (...) {
        MessageBox(g_hMain, L"未知异常", L"错误", MB_ICONERROR);
        OutputDebugStringW(L"[StartDetect] 未知异常\n");
    }
    return false;
}


void StopDetect() {
    if (!g_detecting.load() && g_dev.IsNull()) return;
    g_detecting = false;

    try {
        if (!g_feat.IsNull())  g_feat->GetCommandFeature("AcquisitionStop")->Execute();
        if (!g_strm.IsNull()) { g_strm->StopGrab(); g_strm->UnregisterCaptureCallback(); g_strm->Close(); }
        if (!g_dev.IsNull() && g_hDevOffline) { g_dev->UnregisterDeviceOfflineCallback(g_hDevOffline); g_hDevOffline = nullptr; g_dev->Close(); }
    }
    catch (...) {}

    // 置空智能指针
    g_feat = CGXFeatureControlPointer();
    g_strm = CGXStreamPointer();
    g_dev = CGXDevicePointer();

    // 清空画面
    { std::lock_guard<std::mutex> lk(g_mtx); g_frameBGRA.release(); }
    if (g_hVideo) InvalidateRect(g_hVideo, nullptr, TRUE);

    try { IGXFactory::GetInstance().Uninit(); }
    catch (...) {}
}

// -------------------- 主窗口 --------------------
LRESULT CALLBACK MainProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_CREATE: {
        // 注册右侧视频面板窗口类
        WNDCLASSW vw{}; vw.lpfnWndProc = VideoViewProc;
        vw.hInstance = (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE);
        vw.lpszClassName = L"VideoView";
        vw.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
        RegisterClassW(&vw);

        // 左侧按钮
        CreateWindowExW(0, L"BUTTON", L"开始检测",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            20, 20, PANEL_W - 40, 40, hWnd, (HMENU)ID_BTN_START, vw.hInstance, nullptr);
        CreateWindowExW(0, L"BUTTON", L"关闭检测",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            20, 70, PANEL_W - 40, 40, hWnd, (HMENU)ID_BTN_STOP, vw.hInstance, nullptr);

        // 右侧视频面板
        RECT rc; GetClientRect(hWnd, &rc);
        g_hVideo = CreateWindowExW(0, L"VideoView", L"",
            WS_CHILD | WS_VISIBLE,
            PANEL_W, 0, rc.right - PANEL_W, rc.bottom,
            hWnd, nullptr, vw.hInstance, nullptr);
        return 0;
    }
    case WM_SIZE: {
        int w = LOWORD(lParam), h = HIWORD(lParam);
        if (g_hVideo) MoveWindow(g_hVideo, PANEL_W, 0, std::max(0, w - PANEL_W), h, TRUE);
        return 0;
    }
    case WM_COMMAND:
        switch (LOWORD(wParam)) {
        case ID_BTN_START: StartDetect(); break;
        case ID_BTN_STOP:  StopDetect();  break;
        }
        return 0;
    case WM_DESTROY:
        StopDetect();
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hWnd, msg, wParam, lParam);
}

int APIENTRY wWinMain(HINSTANCE hInst, HINSTANCE, LPWSTR, int nCmdShow) {
    WNDCLASSW wc{}; wc.lpfnWndProc = MainProc; wc.hInstance = hInst;
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1); wc.lpszClassName = L"MainWnd";
    RegisterClassW(&wc);

    HWND hWnd = CreateWindowExW(0, L"MainWnd", L"YOLO 检测（Start/Stop + 右侧视频）",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 1200, 720,
        nullptr, nullptr, hInst, nullptr);
    g_hMain = hWnd;
    ShowWindow(hWnd, nCmdShow);
    UpdateWindow(hWnd);

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
    return (int)msg.wParam;
}
