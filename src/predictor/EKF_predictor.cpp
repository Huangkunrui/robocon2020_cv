#include "EKF_predictor.h"

struct TimeManager {
  TimeManager(double current_time, double &ref_time)
      : c_time(current_time), r_time(ref_time){};

  ~TimeManager() { r_time = c_time; }

  double c_time;
  double &r_time;
};

double EKFPredictor::getVerticalCompensation(Eigen::Vector3d &p_world,
                                             double robot_speed_mps) {
  double p_pitch = atan2(p_world(2, 0), p_world.topRows<2>().norm());
  double distance = p_world.norm();
  double a = 9.8 * 9.8 * 0.25;
  double b = -robot_speed_mps * robot_speed_mps -
             distance * 9.8 * cos(M_PI_2 + p_pitch);
  double c = distance * distance;
  double t_2 = (-sqrt(b * b - 4 * a * c) - b) / (2 * a);
  double fly_time = sqrt(t_2);
  double height = 0.5 * 9.8 * t_2;
  return height;
}

void EKFPredictor::getAngularSpeed(Eigen::Matrix<double, 5, 1> &Xp,
                                   Eigen::Vector3d &s_pw, RobotCmd &send,
                                   Eigen::Matrix3d &R_IW) {
  Eigen::Vector3d s_pc = pWorld2pCam(s_pw, R_IW);
  double distance = s_pc.norm();
  double s_yaw = atan(s_pc(0, 0) / s_pc(2, 0)) / M_PI * 180.;
  double s_pitch = atan(s_pc(1, 0) / s_pc(2, 0)) / M_PI * 180.;
  Predict predictfunc;
  predictfunc.delta_t_ = 0.001;
  Eigen::Matrix<double, 5, 1> Xd;
  predictfunc(Xp.data(), Xd.data());
  Eigen::Vector3d d_pw{Xd(0, 0), Xd(2, 0), Xd(4, 0)};
  double d_pitch = atan2(d_pw(2, 0), d_pw.topRows<2>().norm());
  double d_distance = d_pw.norm();
  double d_a = 9.8 * 9.8 * 0.25;
  double d_b = -robot_speed_mps_ * robot_speed_mps_ -
               d_distance * 9.8 * cos(M_PI_2 + d_pitch);
  double d_c = d_distance * d_distance;
  double d_t_2 = (-sqrt(d_b * d_b - 4 * d_a * d_c) - d_b) / (2 * d_a);
  double d_fly_time = sqrt(d_t_2);
  double d_height = 0.5 * 9.8 * d_t_2;
  Eigen::Vector3d ds_pw{d_pw(0, 0), d_pw(1, 0), d_pw(2, 0) + d_height};
  Eigen::Vector3d ds_pc = pWorld2pCam(ds_pw, R_IW);
  double ds_yaw = atan(ds_pc(0, 0) / ds_pc(2, 0)) / M_PI * 180.;
  double ds_pitch = atan(ds_pc(1, 0) / ds_pc(2, 0)) / M_PI * 180.;
  send.distance_ = (int)(distance * 10);
  send.yaw_angle_ = (float)s_yaw;
  send.pitch_angle_ = (float)s_pitch;
  send.yaw_speed_ = (float)(ds_yaw - s_yaw) / 0.001 / 180. * M_PI;
  send.pitch_speed_ = (float)(ds_pitch - s_pitch) / 0.001 / 180. * M_PI;
  return;
}

bool EKFPredictor::predict(DetectionPack &data, RobotCmd &send,
                           cv::Mat &im2show, bool is_show) {
  std::cout << "===============predict=================" << std::endl;
  auto &[detections, img, q_, t] = data;
  TimeManager helper(t, last_time_);
  im2show = img.clone();
  Eigen::Quaternionf q_raw(q_[0], q_[1], q_[2], q_[3]);
  Eigen::Quaternionf q(q_raw.matrix().transpose());
  Eigen::Matrix3d R_IW = q.matrix().cast<double>();
  last_rotation_iw_ = R_IW;
  double delta_t = t - last_time_;
  bool selected = false;
  BBox goal;
  if (detections.empty()) {
    clear();
    return false;
  }
  bool same_goal = false, need_reset = false;
  std::vector<BBox> new_detections;
  for (auto &d : detections) {
    Eigen::Vector3d m_pc = getCoordInCamera(d.points_, d.category_id_);
    Eigen::Vector3d m_pw = pc2Pw(m_pc, R_IW);
    if (m_pw[2] > height_thres_) continue;  
    if (d.confidence_ >= high_thres_) 
      new_detections.push_back(d);
    else if (d.confidence_ >= low_thres_) {
      auto center = getPointCenter(d.points_);
      for (auto &tmp : last_boxes_) {
        Eigen::Vector3d tmp_last_m_pc =
            getCoordInCamera(tmp.points_, tmp.category_id_);
        Eigen::Vector3d tmp_last_m_pw = pc2Pw(tmp_last_m_pc, last_rotation_iw_);
        if (center.inside(getROI(tmp, 1.0)) ||
            isSameBBox(tmp_last_m_pw, d, R_IW, 0.15)) {
          new_detections.push_back(d);
          break;
        }
      }
    }
  }
  if (new_detections.empty()) {
    clear();
    return false;
  }
  for (auto &d : new_detections) {
    auto center = getPointCenter(d.points_);
    if (last_shoot_ && (center.inside(getROI(last_sbbox_, 1.0)) ||
                        isSameBBox(last_position_w_, d, R_IW, 0.15))) {
      goal = d;
      selected = true;
      same_goal = true;
      need_reset = false;
      break;
    }
  }

  if (same_goal) {
    Eigen::Vector3d m_pc = getCoordInCamera(goal.points_, goal.category_id_);
    Eigen::Vector3d m_pw = pc2Pw(m_pc, R_IW);
    double mc_yaw = std::atan2(m_pc(1, 0), m_pc(0, 0));
    double m_yaw = std::atan2(m_pw(1, 0), m_pw(0, 0));
    double mc_pitch = std::atan2(
        m_pc(2, 0), sqrt(m_pc(0, 0) * m_pc(0, 0) + m_pc(1, 0) * m_pc(1, 0)));
    double m_pitch = std::atan2(
        m_pw(2, 0), sqrt(m_pw(0, 0) * m_pw(0, 0) + m_pw(1, 0) * m_pw(1, 0)));

    last_position_w_ = m_pw;
    last_yaw_ = m_yaw;
    last_pitch_ = m_pitch;

    Predict predictfunc;
    Measure measure;

    Eigen::Matrix<double, 5, 1> Xr;
    Xr << m_pw(0, 0), 0, m_pw(1, 0), 0, m_pw(2, 0);
    Eigen::Matrix<double, 3, 1> Yr;
    measure(Xr.data(), Yr.data());
    predictfunc.delta_t_ = delta_t;
    ekf_.predict(predictfunc);
    Eigen::Matrix<double, 5, 1> Xe = ekf_.update(measure, Yr);
    double predict_time = m_pw.norm() + shoot_delay_;
    predictfunc.delta_t_ = predict_time;
    Eigen::Matrix<double, 5, 1> Xp;
    predictfunc(Xe.data(), Xp.data());
    Eigen::Vector3d c_pw{Xe(0, 0), Xe(2, 0), Xe(4, 0)};
    Eigen::Vector3d p_pw{Xp(0, 0), Xp(2, 0), Xp(4, 0)};
    double height = getVerticalCompensation(p_pw, robot_speed_mps_);
    Eigen::Vector3d s_pw{p_pw(0, 0), p_pw(1, 0), p_pw(2, 0) + height};
    getAngularSpeed(Xp, s_pw, send, R_IW);
    if (is_show) {
      reprojectPoint(im2show, c_pw, R_IW, {0, 255, 0});
      reprojectPoint(im2show, p_pw, R_IW, {255, 0, 0});
      reprojectPoint(im2show, s_pw, R_IW, {0, 0, 255});
      for (int i = 0; i < 4; ++i)
        cv::circle(im2show, goal.points_[i], 3, {255, 0, 0});
      cv::circle(im2show, {im2show.cols / 2, im2show.rows / 2}, 3, {0, 255, 0});
    }
  }
  if (need_reset) {
    Eigen::Vector3d m_pc = getCoordInCamera(goal.points_, goal.category_id_);
    Eigen::Vector3d m_pw = pc2Pw(m_pc, R_IW);
    double mc_yaw = std::atan2(m_pc(1, 0), m_pc(0, 0));
    double m_yaw = std::atan2(m_pw(1, 0), m_pw(0, 0));
    double mc_pitch = std::atan2(
        m_pc(2, 0), sqrt(m_pc(0, 0) * m_pc(0, 0) + m_pc(1, 0) * m_pc(1, 0)));
    double m_pitch = std::atan2(
        m_pw(2, 0), sqrt(m_pw(0, 0) * m_pw(0, 0) + m_pw(1, 0) * m_pw(1, 0)));
    Eigen::Matrix<double, 5, 1> Xr;
    Xr << m_pw(0, 0), 0, m_pw(1, 0), 0, m_pw(2, 0);
    ekf_.init(Xr);
    last_position_w_ = m_pw;
    last_yaw_ = m_yaw;
    last_pitch_ = m_pitch;

    double height = getVerticalCompensation(m_pw, robot_speed_mps_);
    Eigen::Vector3d s_pw{m_pw(0, 0), m_pw(1, 0), m_pw(2, 0) + height};
    Eigen::Vector3d s_pc = pWorld2pCam(s_pw, R_IW);
    double s_yaw = atan(s_pc(0, 0) / s_pc(2, 0)) / M_PI * 180.;
    double s_pitch = atan(s_pc(1, 0) / s_pc(2, 0)) / M_PI * 180.;
    send.distance_ = (int)(s_pc.norm() * 10);
    send.yaw_angle_ = (float)s_yaw;
    send.pitch_angle_ = (float)s_pitch;
    send.yaw_speed_ = 0.;
    send.pitch_speed_ = 0.;
  }

  int len = candidates_.size();
  for (long unsigned int i = 0; i < candidates_.size(); ++i) {
    if (!candidates_[i].updated) {
      candidates_.erase(candidates_.begin() + i);
      --i;
      --len;
    }
  }
  loadParameter(true);
  return true;
}

float EKFPredictor::bbOverlap(const cv::Rect2f &box1, const cv::Rect2f &box2) {
  if (box1.x > box2.x + box2.width) {
    return 0.0;
  }
  if (box1.y > box2.y + box2.height) {
    return 0.0;
  }
  if (box1.x + box1.width < box2.x) {
    return 0.0;
  }
  if (box1.y + box1.height < box2.y) {
    return 0.0;
  }
  float colInt = std::min(box1.x + box1.width, box2.x + box2.width) -
                 std::max(box1.x, box2.x);
  float rowInt = std::min(box1.y + box1.height, box2.y + box2.height) -
                 std::max(box1.y, box2.y);
  float intersection = colInt * rowInt;
  float area1 = box1.width * box1.height;
  float area2 = box2.width * box2.height;
  return intersection / (area1 + area2 - intersection);
}

double EKFPredictor::getQuadrangleArea(const BBox &bx) {
  auto bx_a = sqrt(pow(bx.points_[0].x - bx.points_[1].x, 2) +
                   pow(bx.points_[0].y - bx.points_[1].y, 2));
  auto bx_b = sqrt(pow(bx.points_[1].x - bx.points_[2].x, 2) +
                   pow(bx.points_[1].y - bx.points_[2].y, 2));
  auto bx_c = sqrt(pow(bx.points_[2].x - bx.points_[3].x, 2) +
                   pow(bx.points_[2].y - bx.points_[3].y, 2));
  auto bx_d = sqrt(pow(bx.points_[3].x - bx.points_[0].x, 2) +
                   pow(bx.points_[3].y - bx.points_[0].y, 2));
  auto bx_z = (bx_a + bx_b + bx_c + bx_d) / 2;
  auto bx_size =
      2 * sqrt((bx_z - bx_a) * (bx_z - bx_b) * (bx_z - bx_c) * (bx_z - bx_d));
  return bx_size;
}

void EKFPredictor::matchArmors(BBox &goal, bool &selected,
                               const std::vector<BBox> &detections,
                               const Eigen::Matrix3d &R_IW,
                               const bool is_right_) {
  if (!last_shoot_) {
    selected = false;
    return;
  }

  std::vector<BBox> temp_armors;
  for (auto &d : detections) {
    BBox tmp = d;
    if (d.category_id_ == last_sbbox_.category_id_) temp_armors.push_back(tmp);
  }
  if (temp_armors.size() == 2) {
    selected = false;
    Eigen::Vector3d m_pw1, m_pw2;
    bool flag = false;
    for (auto &t : temp_armors) {
      if (t.category_id_ == last_sbbox_.category_id_) {
        Eigen::Vector3d tmp_m_pc = getCoordInCamera(t.points_, t.category_id_);
        Eigen::Vector3d tmp_m_pw = pc2Pw(tmp_m_pc, R_IW);
        if (!flag) {
          m_pw1 = tmp_m_pw;
          flag = true;
        } else {
          m_pw2 = tmp_m_pw;
          flag = false;
        }
      }
    }
    if (flag) std::cout << "[Error] in matchArmors." << std::endl;
    if (is_right_) {
      if (m_pw1(0, 0) > m_pw2(0, 0))
        goal = temp_armors.front();
      else
        goal = temp_armors.back();
    } else {
      if (m_pw1(0, 0) < m_pw2(0, 0))
        goal = temp_armors.front();
      else
        goal = temp_armors.back();
    }
    selected = true;
  } else {
    selected = false;
    return;
  }
}

Eigen::Vector3d EKFPredictor::getCoordInCamera(const cv::Point2f p[4],
                                               int armor_number) {
  static const std::vector<cv::Point3d> pw_small = {// 单位：m
                                                    {-0.066, 0.027, 0.},
                                                    {-0.066, -0.027, 0.},
                                                    {0.066, -0.027, 0.},
                                                    {0.066, 0.027, 0.}};
  static const std::vector<cv::Point3d> pw_big = {// 单位：m
                                                  {-0.115, 0.029, 0.},
                                                  {-0.115, -0.029, 0.},
                                                  {0.115, -0.029, 0.},
                                                  {0.115, 0.029, 0.}};
  std::vector<cv::Point2d> pu(p, p + 4);
  cv::Mat rvec, tvec;

  if (armor_number == 0 || armor_number == 1 || armor_number == 8)
    cv::solvePnP(pw_big, pu, cam_intrinsic_cv_mat_, distortion_vector_cv_mat_,
                 rvec, tvec);
  else
    cv::solvePnP(pw_small, pu, cam_intrinsic_cv_mat_, distortion_vector_cv_mat_,
                 rvec, tvec);

  Eigen::Vector3d pc;
  cv::cv2eigen(tvec, pc);
  pc[0] += 0.0475;
  pc[1] += 0.0165;
  pc[2] += 0.02385;
  return pc;
}

void EKFPredictor::loadParameter(bool update_all) {
  cv::FileStorage fin(PROJECT_DIR "/asset/autoaim-param.yml",
                      cv::FileStorage::READ);

  fin["Q00"] >> ekf_.predict_cov_(0, 0);
  fin["Q11"] >> ekf_.predict_cov_(1, 1);
  fin["Q22"] >> ekf_.predict_cov_(2, 2);
  fin["Q33"] >> ekf_.predict_cov_(3, 3);
  fin["Q44"] >> ekf_.predict_cov_(4, 4);
  fin["R00"] >> ekf_.observe_cov_(0, 0);
  fin["R11"] >> ekf_.observe_cov_(1, 1);
  fin["R22"] >> ekf_.observe_cov_(2, 2);

  if (update_all) {
    for (auto &a : candidates_) {
      fin["Q00"] >> a.ekf_.predict_cov_(0, 0);
      fin["Q11"] >> a.ekf_.predict_cov_(1, 1);
      fin["Q22"] >> a.ekf_.predict_cov_(2, 2);
      fin["Q33"] >> a.ekf_.predict_cov_(3, 3);
      fin["Q44"] >> a.ekf_.predict_cov_(4, 4);
      fin["R00"] >> a.ekf_.observe_cov_(0, 0);
      fin["R11"] >> a.ekf_.observe_cov_(1, 1);
      fin["R22"] >> a.ekf_.observe_cov_(2, 2);
    }
  }
  return;
}


cv::Rect2f EKFPredictor::getROI(BBox &goal, float coefficient) {
  auto center = getPointCenter(goal.points_);
  auto w = std::max({goal.points_[0].x, goal.points_[1].x, goal.points_[2].x,
                     goal.points_[3].x}) -
           std::min({goal.points_[0].x, goal.points_[1].x, goal.points_[2].x,
                     goal.points_[3].x});
  auto h = std::max({goal.points_[0].y, goal.points_[1].y, goal.points_[2].y,
                     goal.points_[3].y}) -
           std::min({goal.points_[0].y, goal.points_[1].y, goal.points_[2].y,
                     goal.points_[3].y});
  return cv::Rect2f(center.x - w / 2, center.y - h / 2, w * coefficient,
                    h * coefficient);
}

bool EKFPredictor::isSameBBox(const Eigen::Vector3d old_m_pw,
                              const BBox &new_armor,
                              const Eigen::Matrix3d &R_IW,
                              const double distance_threshold) {
  Eigen::Vector3d new_m_pc =
      getCoordInCamera(new_armor.points_, new_armor.category_id_);
  Eigen::Vector3d new_m_pw = pc2Pw(new_m_pc, R_IW);
  Eigen::Vector3d m_pw_delta = new_m_pw - old_m_pw;
  double distance = m_pw_delta.norm();
  if (distance < distance_threshold)
    return true;
  else
    return false;
}
cv::Point2f EKFPredictor::getPointCenter(cv::Point2f points_[4]) {
  cv::Point2f center;
  center.x = (points_[0].x + points_[1].x + points_[2].x + points_[3].x) / 4;
  center.y = (points_[0].y + points_[1].y + points_[2].y + points_[3].y) / 4;
  return center;
}
void EKFPredictor::reprojectPoint(cv::Mat &image, const Eigen::Vector3d &pw,
                                  const Eigen::Matrix3d &R_IW,
                                  const cv::Scalar &color) {
  Eigen::Vector3d pc = pWorld2pCam(pw, R_IW);
  Eigen::Vector3d pu = pCam2Ppixel(pc);
  cv::circle(image, {int(pu(0, 0)), int(pu(1, 0))}, 3, color, 2);
}

void EKFPredictor::clear() {
  last_boxes_.clear();
  last_shoot_ = false;
  candidates_.clear();
}

template <class T>
void Xyz2PitchYawDistance(T xyz[3], T pyd[3]) {
  pyd[0] = ceres::atan2(
      xyz[2], ceres::sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1]));  // pitch
  pyd[1] = ceres::atan2(xyz[1], xyz[0]);                        // yaw
  pyd[2] = ceres::sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1] +
                       xyz[2] * xyz[2]);  // distance
}

EKFPredictor::EKFPredictor() {
  cv::FileStorage fin(PROJECT_DIR "/asset/camera-param.yml",
                      cv::FileStorage::READ);
  fin["Tcb"] >> rotation_ci_cv_mat_;
  fin["K"] >> cam_intrinsic_cv_mat_;
  fin["D"] >> distortion_vector_cv_mat_;
  cv::cv2eigen(rotation_ci_cv_mat_, rotation_ci_);
  cv::cv2eigen(cam_intrinsic_cv_mat_, cam_intrinsic_matrix_);
  cv::cv2eigen(distortion_vector_cv_mat_, distortion_vec_);

  loadParameter();
}