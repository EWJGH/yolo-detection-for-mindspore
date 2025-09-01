#include "cameraLinkThreadWorker.h"
#include "inferHandle.h"
#include "napi/native_api.h"
#include "ace/xcomponent/native_interface_xcomponent.h"
#include "native_window/external_window.h"
#include "hilog/log.h"

#include <bits/alltypes.h>
#include <sys/mman.h>
#include <mutex>
#include <condition_variable>
#include <thread>


#include <cameraLinkThreadWorker.h>

#ifdef LOG_DOMAIN
#undef LOG_DOMAIN
#endif

#ifdef LOG_TAG
#undef LOG_TAG
#endif

#define LOG_DOMAIN 0x0001
#define LOG_TAG "testAIrender"


static std::condition_variable _condVar;
std::thread _thread;
static bool _exit = false;
static bool _nativeWindowAvailable = false;
static OHNativeWindow* _nativeWindow = nullptr;
static OH_NativeXComponent_Callback _callback;

std::thread inferthread;
static std::queue<cv::Mat> inferQueue;
static std::mutex inferMutex;

CameraLinkThreadWorker camworker;


void RenderProc();
void InferProc();

static napi_value StartRender(napi_env env, napi_callback_info info)
{
    if (_thread.joinable()) return nullptr;
    
    _exit = false;
    _thread = std::thread(RenderProc);
    inferthread = std::thread(InferProc);
    
    napi_value result;
    napi_create_int32(env, 0, &result);
    return result;
}

static napi_value StopRender(napi_env env, napi_callback_info info)
{
    OH_LOG_INFO(LOG_APP, "stop render");
    {
        std::lock_guard<std::mutex> lock(DisplayMutex);
        _exit = true;
        _condVar.notify_one();
    }
    
    if (_thread.joinable()) _thread.join();
    if (inferthread.joinable()) inferthread.join();
    camworker.disConnect();
    
    OH_LOG_INFO(LOG_APP, "disconnect camera");
    //MyInferEngine.destroy();
    napi_value result;
    napi_create_int32(env, 0, &result);
    return result;
}
void InferProc() {
    InferHandle MyInferEngine;
    OH_LOG_INFO(LOG_APP, "infer handle init start");
    MyInferEngine.init();
    OH_LOG_INFO(LOG_APP, "infer handle init finish");
    while (true) {
        std::unique_lock<std::mutex> lock(DisplayMutex);
        
//        _condVar.wait(lock, [] { return _exit; });
        if (_exit) {
            OH_LOG_INFO(LOG_APP, "infer loop break");
            break;
        }
        
        //OH_LOG_INFO(LOG_APP, "to comsume Display Queue size is： %lu", DisplayQueue.size());
        if (DisplayQueue.size() == 0) {
            lock.unlock();
            OH_LOG_INFO(LOG_APP, "DisplayQueue is empty");
            std::this_thread::sleep_for(std::chrono::milliseconds(24));
            continue;
        }
        auto element = DisplayQueue.front();
        DisplayQueue.pop();
        lock.unlock();
        
        std::unique_lock<std::mutex> inferlock(inferMutex);
        //skip infer when buffer is full            
        if (inferQueue.size()>=IMAGE_BUFFER_SIZE){
            continue;
        }
        OH_LOG_INFO(LOG_APP, "start to infer frame");
        auto detectionResult = MyInferEngine.inferenceWithPtr(element);
        
        inferQueue.push(detectionResult);
        OH_LOG_INFO(LOG_APP, "push frame to InferQueue");
        _condVar.notify_one();
        inferlock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(24));
        
        
    }
    MyInferEngine.destroy();
}

void cameraWatcherThread(){
    while (!_exit) {
        std::this_thread::sleep_for(std::chrono::seconds(3)); // 每3秒检查一次
        OH_LOG_INFO(LOG_APP, "offline check");
        if (camworker.isConnected && camworker.isTimeout) {
            camworker.disConnect();
            camworker.startConnect();
            camworker.isTimeout = false;
            OH_LOG_INFO(LOG_APP, "offline camera reconnect");
        }

    }
}

void RenderProc() {
    OH_LOG_INFO(LOG_APP, "thread start");
    camworker.startConnect();
    std::thread watcherThread(cameraWatcherThread);
        
    while (true) {
        std::unique_lock<std::mutex> lock(inferMutex);
        _condVar.wait(lock, [] { return _nativeWindowAvailable  || _exit; });

        if (!_nativeWindowAvailable || _exit) {
            OH_LOG_INFO(LOG_APP, "loop break");
            watcherThread.join();
            break;
        }
        //OH_LOG_INFO(LOG_APP, "to comsume infer Queue size is： %llu",inferQueue.size());
        if (inferQueue.size()==0) {
            lock.unlock();
            OH_LOG_INFO(LOG_APP, "inferQueue is empty");
            std::this_thread::sleep_for(std::chrono::milliseconds(24));
            continue;
        }
        // 相机内容写入buffer
        auto element = inferQueue.front();
        inferQueue.pop();
        
        OHNativeWindowBuffer* buffer = nullptr;
        int fenceFd;
        // 通过 OH_NativeWindow_NativeWindowRequestBuffer 获取 OHNativeWindowBuffer 实例
        OH_NativeWindow_NativeWindowRequestBuffer(_nativeWindow, &buffer, &fenceFd);
        // 通过 OH_NativeWindow_GetBufferHandleFromNative 获取 buffer 的 handle
        BufferHandle* bufferHandle = OH_NativeWindow_GetBufferHandleFromNative(buffer);

        // 使用系统mmap接口拿到bufferHandle的内存虚拟地址
        void* mappedAddr = mmap(bufferHandle->virAddr, bufferHandle->size, PROT_READ | PROT_WRITE, MAP_SHARED, bufferHandle->fd, 0);
        if (mappedAddr == MAP_FAILED) {
            // mmap failed
            OH_LOG_INFO(LOG_APP, "mmap failed");
        }

        // 将生产的内容写入Buffer
        static uint8_t value = 0x00;
        value++;
        uint8_t *pixel = static_cast<uint8_t *>(mappedAddr); // 使用mmap获取到的地址来访问内存
        OH_LOG_INFO(LOG_APP, "bufferHandle width,height,stride is %d %d %d",bufferHandle->width,bufferHandle->height,bufferHandle->stride);
        
        uint8_t *cvdata = element.data;
        for (uint32_t x=0;x<bufferHandle->width*bufferHandle->height;x++){
            //rgba -- bgr
            pixel[2] = cvdata[0];
            pixel[1] = cvdata[1];
            pixel[0] = cvdata[2];
            pixel[3] = 0xFF;
            pixel += 4;
            cvdata +=3;
        }

        // 设置刷新区域，如果Region中的Rect为nullptr,或者rectNumber为0，则认为OHNativeWindowBuffer全部有内容更改。
        Region region{nullptr, 0};
        // 通过OH_NativeWindow_NativeWindowFlushBuffer 提交给消费者使用，例如：显示在屏幕上。
        OH_NativeWindow_NativeWindowFlushBuffer(_nativeWindow, buffer, fenceFd, region);

        // 内存使用完记得去掉内存映射
        int result = munmap(mappedAddr, bufferHandle->size);
        if (result == -1) {
            // munmap failed
            OH_LOG_INFO(LOG_APP, "munmap failed");
        }
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(24));
    }
    
    OH_LOG_INFO(LOG_APP, "thread exit");
}

// 定义回调函数
void OnSurfaceCreatedCB(OH_NativeXComponent* component, void* window)
{
    // 可获取 OHNativeWindow 实例
    OHNativeWindow* nativeWindow = static_cast<OHNativeWindow*>(window);
    OH_LOG_INFO(LOG_APP, "surface created: %{public}p", window);
    
    // 设置 OHNativeWindowBuffer 的宽高
    int32_t width = 640;
    int32_t height = 640;
    // 这里的nativeWindow是从上一步骤中的回调函数中获得的
    int32_t ret = OH_NativeWindow_NativeWindowHandleOpt(nativeWindow, SET_BUFFER_GEOMETRY, width, height);

    std::lock_guard<std::mutex> lock(DisplayMutex);
    _nativeWindow = nativeWindow;
    _nativeWindowAvailable = true;
    _condVar.notify_one();
}

void OnSurfaceChangedCB(OH_NativeXComponent* component, void* window)
{
    // 可获取 OHNativeWindow 实例
    OHNativeWindow* nativeWindow = static_cast<OHNativeWindow*>(window);
    // ...
}

void OnSurfaceDestroyedCB(OH_NativeXComponent* component, void* window)
{
    OH_LOG_INFO(LOG_APP, "surface destroyed: %{public}p", window);
    
    // 可获取 OHNativeWindow 实例
    OHNativeWindow* nativeWindow = static_cast<OHNativeWindow*>(window);
    
    std::lock_guard<std::mutex> lock(DisplayMutex);
    _nativeWindow = nullptr;
    _nativeWindowAvailable = false;
    _condVar.notify_one();
}

void DispatchTouchEventCB(OH_NativeXComponent* component, void* window)
{
    // 可获取 OHNativeWindow 实例
    OHNativeWindow* nativeWindow = static_cast<OHNativeWindow*>(window);
    // ...
}

EXTERN_C_START
static napi_value Init(napi_env env, napi_value exports)
{
    OH_LOG_INFO(LOG_APP, "module init");
    
    napi_property_descriptor desc[] = {
        { "startRender", nullptr, StartRender, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "stopRender", nullptr, StopRender, nullptr, nullptr, nullptr, napi_default, nullptr }
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);

    napi_value exportInstance = nullptr;
    // 用来解析出被wrap了NativeXComponent指针的属性
    napi_get_named_property(env, exports, OH_NATIVE_XCOMPONENT_OBJ, &exportInstance);
    OH_NativeXComponent *nativeXComponent = nullptr;
    // 通过napi_unwrap接口，解析出NativeXComponent的实例指针
    napi_unwrap(env, exportInstance, reinterpret_cast<void**>(&nativeXComponent));
        
    // 获取XComponentId
    char idStr[OH_XCOMPONENT_ID_LEN_MAX + 1] = {};
    uint64_t idSize = OH_XCOMPONENT_ID_LEN_MAX + 1;
    OH_NativeXComponent_GetXComponentId(nativeXComponent, idStr, &idSize);
    
    OH_LOG_INFO(LOG_APP, "component: %{public}s, %{public}p", idStr, nativeXComponent);
    
    if (nativeXComponent && strcmp(idStr, "xcomponent1") == 0) {
        // 初始化 OH_NativeXComponent_Callback
        _callback.OnSurfaceCreated = OnSurfaceCreatedCB;
        _callback.OnSurfaceChanged = OnSurfaceChangedCB;
        _callback.OnSurfaceDestroyed = OnSurfaceDestroyedCB;
        _callback.DispatchTouchEvent = DispatchTouchEventCB;

        // 注册回调函数
        OH_NativeXComponent_RegisterCallback(nativeXComponent, &_callback);
    }
    
    return exports;
}
EXTERN_C_END

static napi_module demoModule = {
    .nm_version =1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "entry",
    .nm_priv = ((void*)0),
    .reserved = { 0 },
};

extern "C" __attribute__((constructor)) void RegisterEntryModule(void)
{
    napi_module_register(&demoModule);
}
