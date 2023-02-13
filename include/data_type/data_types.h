
#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <array>
#include <cstdint>
#include <opencv2/core.hpp>
//检测框输出
struct alignas(4) BBox {
  cv::Point2f points_[4]; //检测框四个顶点
  float confidence_;//置信度
  int category_id_; //类别id
};
//检测线程发出的数据包
struct DetectionPack{
    std::vector<BBox> detections_;
    cv::Mat img_;
    std::array<double, 4> q_;
    double timestamp_;
};


// 预测线程发出的数据包
struct RobotCmd {
  uint8_t start_ = (unsigned)'s';
  float pitch_angle_ = 0;    // 单位：度
  float yaw_angle_ = 0;      // 单位：度
  float pitch_speed_ = 0;    // 单位：弧度/s
  float yaw_speed_ = 0;      // 单位：弧度/s
  uint8_t distance_ = 0;     // 计算公式 (int)(distance * 10)
  uint8_t end_ = (unsigned)'e';
} __attribute__((packed));

#endif  // DATA_TYPES_HPP