#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>

using namespace nvinfer1;

// Config 
static const int INPUT_W = 640;
static const int INPUT_H = 640;
static const float CONF_THRESH = 0.25f;
static const float NMS_THRESH  = 0.45f;
static const int NUM_CLASSES   = 80;

// Error-checking macro for CUDA functions
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

// Logger
class TRTLogger : public ILogger 
{
public:
    void log(Severity severity, const char* msg) noexcept override 
    {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

struct Detection 
{
    cv::Rect2f box;
    int class_id;
    float score;
};

// Scale and keeping aspect
static cv::Mat letterbox(const cv::Mat& src, int newW, int newH, cv::Scalar color, float& scale, int& padw, int& padh) {
    int w = src.cols, h = src.rows;
    scale = std::min((float)newW / w, (float)newH / h);
    int nw = (int)(w * scale);
    int nh = (int)(h * scale);
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(nw, nh));

    cv::Mat out(newH, newW, src.type(), color);
    padw = (newW - nw) / 2;
    padh = (newH - nh) / 2;
    resized.copyTo(out(cv::Rect(padw, padh, nw, nh)));
    return out;
}

// Rearranges the pixel data
static void hwc_to_chw(const cv::Mat& img, float* data) 
{
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    std::vector<cv::Mat> bgr;
    cv::split(img, bgr);
    std::vector<cv::Mat> rgb = {bgr[2], bgr[1], bgr[0]};
    int channelSize = img_h * img_w;
    for (int i = 0; i < channels; ++i) 
    {
        for (int y = 0; y < img_h; ++y) 
        {
            const uchar* row = rgb[i].ptr<uchar>(y);
            for (int x = 0; x < img_w; ++x) 
            {
                data[i * channelSize + y * img_w + x] = row[x] / 255.0f;
            }
        }
    }
}

// Intersection over Union
static float iou(const cv::Rect2f& a, const cv::Rect2f& b) 
{
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return inter / (uni + 1e-6f);
}

// Non-Maximum Suppression
static std::vector<Detection> nms(const std::vector<Detection>& dets, float iou_thresh) 
{
    std::vector<Detection> res;
    std::vector<int> idx(dets.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int i, int j){ return dets[i].score > dets[j].score; });

    std::vector<bool> removed(dets.size(), false);
    for (size_t _i = 0; _i < idx.size(); ++_i) {
        int i = idx[_i];
        if (removed[i]) continue;
        res.push_back(dets[i]);
        for (size_t _j = _i + 1; _j < idx.size(); ++_j) {
            int j = idx[_j];
            if (removed[j]) continue;
            if (dets[i].class_id != dets[j].class_id) continue;
            if (iou(dets[i].box, dets[j].box) > iou_thresh) removed[j] = true;
        }
    }
    return res;
}

// Decode YOLO output for class probabilities
static std::vector<Detection> decodeYoloV5(const float* out, int64_t elems,
                                           float conf_th, float nms_th,
                                           float gain, int padw, int padh,
                                           int imw, int imh) 
{
    std::vector<Detection> dets;
    int stride = 5 + NUM_CLASSES;
    int rows = elems / stride;

    for (int i = 0; i < rows; ++i) {
        const float* p = out + i * stride;
        float obj = p[4];
        if (obj < 1e-5f) continue;

        int cls = -1;
        float cls_score = 0.0f;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float sc = p[5 + c];
            if (sc > cls_score) { cls_score = sc; cls = c; }
        }
        float score = obj * cls_score;
        if (score < conf_th) continue;

        float cx = p[0], cy = p[1], w = p[2], h = p[3];
        float x0 = cx - w / 2.0f;
        float y0 = cy - h / 2.0f;
        float x1 = cx + w / 2.0f;
        float y1 = cy + h / 2.0f;

        x0 = (x0 - padw) / gain;
        y0 = (y0 - padh) / gain;
        x1 = (x1 - padw) / gain;
        y1 = (y1 - padh) / gain;

        x0 = std::max(0.f, std::min((float)imw, x0));
        y0 = std::max(0.f, std::min((float)imh, y0));
        x1 = std::max(0.f, std::min((float)imw, x1));
        y1 = std::max(0.f, std::min((float)imh, y1));

        Detection d;
        d.box = cv::Rect2f(cv::Point2f(x0, y0), cv::Point2f(x1, y1));
        d.class_id = cls;
        d.score = score;
        dets.push_back(d);
    }
    return nms(dets, nms_th);
}

// Load TensorRT engine 
static std::vector<char> loadFile(const std::string& path) 
{
    std::ifstream f(path, std::ios::binary);
    if (!f) 
    {
        std::cerr << "Failed to open " << path << std::endl;
        std::exit(1);
    }
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}


int main()
{

    // Load class names
    std::vector<std::string> classes;
    std::ifstream ifs("coco.names");
    std::string line;
    while (getline(ifs, line)) classes.push_back(line);

    // Open Camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) { std::cerr << "Cannot open webcam\n"; return 1; }

    // Logger
    TRTLogger logger;

    // Load engine
    std::string enginePath = "yolov5s.trt";
    std::vector<char> engineData = loadFile(enginePath);
    if (engineData.empty()) 
    {
        std::cerr << "Engine file is empty! Check path: " << enginePath << std::endl;
        return -1;
    }

    // TensorRT runtime
    auto runtime = std::unique_ptr<nvinfer1::IRuntime, void(*)(nvinfer1::IRuntime*)>(
        nvinfer1::createInferRuntime(logger),
        [](nvinfer1::IRuntime* p){ delete p; }
    );

    // Deserialized engine
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)>(
        runtime->deserializeCudaEngine(engineData.data(), engineData.size()),
        [](nvinfer1::ICudaEngine* p){ delete p; }
    );

    // Execution context
    auto context = std::unique_ptr<nvinfer1::IExecutionContext, void(*)(nvinfer1::IExecutionContext*)>(
        engine->createExecutionContext(),
        [](nvinfer1::IExecutionContext* p){ delete p; }
    );

    // Check if engine is dynamic
    bool isDynamic = false;
    for (int i = 0; i < engine->getNbBindings(); ++i) 
    {
        if (engine->isShapeBinding(i))
        {
            isDynamic = true;
            break;
        }
    }

    // Allocate buffers based on engine sizes
    const int inputIndex = 0;
    nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
    size_t inputSize = 1;
    for (int i = 0; i < inputDims.nbDims; ++i) inputSize *= inputDims.d[i];
    inputSize *= sizeof(float);

    const int outputIndex = 1;
    nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);
    size_t outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; ++i) outputSize *= outputDims.d[i];
    outputSize *= sizeof(float);

    void* dInput{nullptr};
    void* dOutput{nullptr};
    CHECK_CUDA(cudaMalloc(&dInput, inputSize));
    CHECK_CUDA(cudaMalloc(&dOutput, outputSize));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    std::vector<float> hostInput(inputSize / sizeof(float));
    std::vector<float> hostOutput(outputSize / sizeof(float));

    // OpenCV window
    cv::Mat frame;
    cv::namedWindow("YOLOv5s TensorRT", cv::WINDOW_NORMAL);

    while (true) 
    {

        // Read frame
        if (!cap.read(frame) || frame.empty()) break;

        // Scale
        float gain; 
        int padw, padh;
        cv::Mat lb = letterbox(frame, INPUT_W, INPUT_H, cv::Scalar(114,114,114), gain, padw, padh);
        hwc_to_chw(lb, hostInput.data());

        // Copy data 
        CHECK_CUDA(cudaMemcpyAsync(dInput, hostInput.data(), inputSize, cudaMemcpyHostToDevice, stream));

        // Set dynamic shape if needed
        if (isDynamic) 
        {
            nvinfer1::Dims4 dims{1, 3, INPUT_H, INPUT_W};
            context->setBindingDimensions(inputIndex, dims);
        }

        void* buffers[] = { dInput, dOutput };
        if (!context->enqueueV2(buffers, stream, nullptr)) 
        {
            std::cerr << "Inference failed\n";
            break;
        }

        // Copy data 
        CHECK_CUDA(cudaMemcpyAsync(hostOutput.data(), dOutput, outputSize, cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));


        // Decode YOLO output for class probabilities
        auto dets = decodeYoloV5(hostOutput.data(), (int64_t)hostOutput.size(),
                                 CONF_THRESH, NMS_THRESH,
                                 gain, padw, padh, frame.cols, frame.rows);

        // Draw rectangle and class label
        for (const auto& d : dets) 
        {
            cv::rectangle(frame, d.box, cv::Scalar(0,255,0), 2);
            char label[128];
            std::snprintf(label, sizeof(label), "%s %.2f", classes[d.class_id].c_str(), d.score);
            int base=0; cv::Size t = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &base);
            cv::rectangle(frame, { (int)d.box.x, (int)(d.box.y - t.height - 6) },
                          { (int)(d.box.x + t.width + 6), (int)d.box.y }, cv::Scalar(0,255,0), cv::FILLED);
            cv::putText(frame, label, { (int)d.box.x + 3, (int)d.box.y - 4 }, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 2);
        }

        // Measuring FPS of video frame
        static auto last = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        double fps = 1e9 / std::chrono::duration_cast<std::chrono::nanoseconds>(now - last).count();
        last = now;
        char fpsText[64]; std::snprintf(fpsText, sizeof(fpsText), "FPS: %.1f", fps);
        cv::putText(frame, fpsText, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255,0,0), 2);

        // Display
        cv::imshow("YOLOv5s TensorRT", frame);

        // Press ESC to exit
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q') break;
    }

    // Release memory
    cudaStreamDestroy(stream);
    cudaFree(dInput);
    cudaFree(dOutput);
    return 0;
}
