#include "detector.hpp"

int main() {
  //Receive camera sensor data//
  // cv::Mat img;
  TRTModule model(onnx_file);
  auto detections = model(img);
  /* publish detection results */
  return 0;
}
