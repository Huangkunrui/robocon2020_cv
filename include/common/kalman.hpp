#ifndef _KALMAN_H_
#define _KALMAN_H_

#include <Eigen/Dense>

template <int V_OBSERVE = 1, int V_STATE = 3>
class Kalman {
 public:
  using Matrix_zzd = Eigen::Matrix<double, V_OBSERVE, V_OBSERVE>;
  using Matrix_xxd = Eigen::Matrix<double, V_STATE, V_STATE>;
  using Matrix_zxd = Eigen::Matrix<double, V_OBSERVE, V_STATE>;
  using Matrix_xzd = Eigen::Matrix<double, V_STATE, V_OBSERVE>;
  using Matrix_x1d = Eigen::Matrix<double, V_STATE, 1>;
  using Matrix_z1d = Eigen::Matrix<double, V_OBSERVE, 1>;

 private:
  Matrix_x1d x_last_;
  Matrix_xzd k_;
  Matrix_xxd predict_matrix_;
  Matrix_zxd observe_matrix_;
  Matrix_xxd noise_cov_;
  Matrix_zzd observe_cov_;
  Matrix_xxd state_cov_;
  double last_t{0};

 public:
  Kalman() = default;

  Kalman(Matrix_xxd predict_matrix, Matrix_zxd observe_matrix, Matrix_xxd R,
         Matrix_zzd observe_cov, Matrix_x1d init, double t) {
    reset(predict_matrix, observe_matrix, R, observe_cov, init, t);
  }

  void reset(Matrix_xxd predict_matrix, Matrix_zxd observe_matrix, Matrix_xxd R,
             Matrix_zzd observe_cov, Matrix_x1d init, double t) {
    this->predict_matrix_ = predict_matrix;
    this->observe_matrix_ = observe_matrix;
    this->state_cov_ = Matrix_xxd::Zero();
    this->noise_cov_ = R;
    this->observe_cov_ = observe_cov;
    x_last_ = init;
    last_t = t;
  }

  void reset(Matrix_x1d init, double t) {
    x_last_ = init;
    last_t = t;
  }

  void reset(double x, double t) {
    x_last_(0, 0) = x;
    last_t = t;
  }

  Matrix_x1d update(Matrix_z1d z_k, double t) {
    for (int i = 1; i < V_STATE; i++) {
      predict_matrix_(i - 1, i) = t - last_t;
    }
    last_t = t;
    Matrix_x1d p_x_k = predict_matrix_ * x_last_;
    state_cov_ =
        predict_matrix_ * state_cov_ * predict_matrix_.transpose() + noise_cov_;
    k_ = state_cov_ * observe_matrix_.transpose() *
         (observe_matrix_ * state_cov_ * observe_matrix_.transpose() +
          observe_cov_)
             .inverse();
    x_last_ = p_x_k + k_ * (z_k - observe_matrix_ * p_x_k);
    state_cov_ = (Matrix_xxd::Identity() - k_ * observe_matrix_) * state_cov_;
    return x_last_;
  }
};

#endif /* _KALMAN_H_ */
