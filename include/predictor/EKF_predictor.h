
#ifndef EKFPredictor_H
#define EKFPredictor_H

#include <ceres/ceres.h>

#include <Eigen/Dense>
#include <cmath>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "EKF.hpp"
#include "data_types.h"

struct Predict {
  /*
   * 匀速直线运动模型
   */
  template <class T>
  void operator()(const T x0[5], T x1[5]) {
    x1[0] = x0[0] + delta_t_ * x0[1];  // 0.1
    x1[1] = x0[1];                     // 100
    x1[2] = x0[2] + delta_t_ * x0[3];  // 0.1
    x1[3] = x0[3];                     // 100
    x1[4] = x0[4];                     // 0.01
  }

  double delta_t_;
};

template <class T>
void Xyz2PitchYawDistance(T xyz[3], T pyd[3]);

struct Measure {
  template <class T>
  void operator()(const T x[5], T y[3]) {
    T x_[3] = {x[0], x[2], x[4]};
    Xyz2PitchYawDistance(x_, y);
  }
};

struct Record {
  BBox bbox_;
  Eigen::Vector3d last_position_w_;
  EKF<5, 3> ekf_;
  bool updated = false, init = false;
  double last_yaw_ = 0, last_pitch_ = 0;
  float yaw_angle_ = 0., pitch_angle_ = 0., yaw_speed_ = 0, pitch_speed_ = 0;
  int distance_ = 0.;
  bool is_distance_valid_ = false;

  Record(BBox &armor)
      : bbox_(armor),
        updated(false),
        init(true),
        last_yaw_(0.),
        last_pitch_(0.),
        yaw_angle_(0.),
        pitch_angle_(0.),
        yaw_speed_(0.),
        pitch_speed_(0.),
        is_distance_valid_(false) {}
};

class EKFPredictor {
 public:
  void loadParameter(bool update_all = true);

  explicit EKFPredictor();

  bool predict(DetectionPack &, RobotCmd &, cv::Mat &, bool);

  void clear();

  ~EKFPredictor() = default;

 private:
  EKF<5, 3> ekf_;
  double last_time_;
  Eigen::Matrix3d rotation_ci_;
  Eigen::Matrix3d cam_intrinsic_matrix_;
  Eigen::Matrix<double, 1, 5> distortion_vec_;
  cv::Mat rotation_ci_cv_mat_;
  cv::Mat distortion_vector_cv_mat_;
  cv::Mat cam_intrinsic_cv_mat_;
  std::vector<BBox> last_boxes_;
  BBox last_sbbox_;
  Eigen::Matrix3d last_rotation_iw_;
  Eigen::Vector3d last_position_w_;
  bool last_shoot_ = false;
  double last_yaw_ = 0, last_pitch_ = 0;
  bool is_right_ = true;

  float robot_speed_mps_ = 28.;
  const double height_thres_ = 0.;
  const double shoot_delay_ = 0.110;
  const float high_thres_ = 0.6f;
  const float low_thres_ = 0.2f;
  std::vector<Record> candidates_;

 private:
  inline Eigen::Vector3d pc2Pw(const Eigen::Vector3d &p_c,
                               const Eigen::Matrix3d &rotation_iw) {
    auto rotation_wc = (rotation_ci_ * rotation_iw).transpose();
    return rotation_wc * p_c;
  }

  inline Eigen::Vector3d pWorld2pCam(const Eigen::Vector3d &pw,
                                     const Eigen::Matrix3d &rotation_iw) {
    auto R_CW = rotation_ci_ * rotation_iw;
    return R_CW * pw;
  }

  inline Eigen::Vector3d pCam2Ppixel(const Eigen::Vector3d &pc) {
    return cam_intrinsic_matrix_ * pc / pc(2, 0);
  }
  void reprojectPoint(cv::Mat &image, const Eigen::Vector3d &pw,
                      const Eigen::Matrix3d &rotation_iw,
                      const cv::Scalar &color);
  cv::Point2f getPointCenter(cv::Point2f points_[4]);
  float bbOverlap(const cv::Rect2f &box1, const cv::Rect2f &box2);
  cv::Rect2f getROI(BBox &armor, float coefficient);
  double getQuadrangleArea(const BBox &bx);
  bool isSameBBox(const Eigen::Vector3d old_m_pw, const BBox &new_armor,
                  const Eigen::Matrix3d &rotation_iw,
                  const double distance_threshold);
  void matchArmors(BBox &armor, bool &selected,
                   const std::vector<BBox> &detections,
                   const Eigen::Matrix3d &rotation_iw, const bool is_right_);
  Eigen::Vector3d getCoordInCamera(const cv::Point2f p[4], int armor_number);
  double getVerticalCompensation(Eigen::Vector3d &p_world,
                                 double robot_speed_mps);
  void getAngularSpeed(Eigen::Matrix<double, 5, 1> &Xp, Eigen::Vector3d &s_pw,
                       RobotCmd &send, Eigen::Matrix3d &rotation_iw);
};

#endif  // EKFPredictor_H
