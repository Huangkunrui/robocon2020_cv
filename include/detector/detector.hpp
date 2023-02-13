
#ifndef _DETECTOR_HPP_
#define _DETECTOR_HPP_

#include <NvInfer.h>

#include <opencv2/core.hpp>

#include "data_types.h"

class TRTModule {
  static constexpr int TOPK_NUM = 128;
  static constexpr float KEEP_THRES = 0.1f;

 public:
  explicit TRTModule(const std::string &onnx_file);

  ~TRTModule();

  TRTModule(const TRTModule &) = delete;

  TRTModule operator=(const TRTModule &) = delete;

  std::vector<bbox_t> operator()(const cv::Mat &src) const;

 private:
  void buildEngineFromOnnx(const std::string &onnx_file);

  void buildEngineFromCache(const std::string &cache_file);

  void cacheEngine(const std::string &cache_file);

  nvinfer1::ICudaEngine *engine_;
  nvinfer1::IExecutionContext *context_;
  mutable void *device_buffer_[2];
  float *output_buffer_;
  cudaStream_t stream_;
  int input_idx, output_idx_;
  size_t input_size_, output_size_;
};

#endif /* _ONNXTRTMODULE_HPP_ */
