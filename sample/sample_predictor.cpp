#include <iostream>
#include <opencv2/core.hpp>
#include "EKF_predictor.h"
#include "data_types.h"
int main() {
  std::cout << "hello world!!!" << std::endl;
  EKFPredictor predictor;

  // receive detections
  DetectionPack detections;
  RobotCmd robot_cmd;
  cv::Mat im2show;
  bool is_show = true;
  bool ok = predictor.predict(detections, robot_cmd, im2show,
                             is_show);
  // publish robot cmd//
  return 0;
}