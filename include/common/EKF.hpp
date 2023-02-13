
#ifndef EKF_HPP_
#define EKF_HPP_

#include <ceres/jet.h>
#include <Eigen/Dense>

template<int N_STATE, int N_OBSERVE>
class EKF {
    using MatrixXX = Eigen::Matrix<double, N_STATE, N_STATE>;
    using MatrixYX = Eigen::Matrix<double, N_OBSERVE, N_STATE>;
    using MatrixXY = Eigen::Matrix<double, N_STATE, N_OBSERVE>;
    using MatrixYY = Eigen::Matrix<double, N_OBSERVE, N_OBSERVE>;
    using VectorX = Eigen::Matrix<double, N_STATE, 1>;
    using VectorY = Eigen::Matrix<double, N_OBSERVE, 1>;

public:
    explicit EKF(const VectorX &X0 = VectorX::Zero())
            : x_estimate_(X0), state_cov_(MatrixXX::Identity()), predict_cov_(MatrixXX::Identity()), observe_cov_(MatrixYY::Identity()) {}

    void init(const VectorX &X0 = VectorX::Zero()) {
        x_estimate_ = X0;
    }

    template<class Func>
    VectorX predict(Func &&func) {
        ceres::Jet<double, N_STATE> Xe_auto_jet[N_STATE];
        for (int i = 0; i < N_STATE; i++) {
            Xe_auto_jet[i].a = x_estimate_[i];
            Xe_auto_jet[i].v[i] = 1;
        }
        ceres::Jet<double, N_STATE> Xp_auto_jet[N_STATE];
        func(Xe_auto_jet, Xp_auto_jet);
        for (int i = 0; i < N_STATE; i++) {
            x_predict_[i] = Xp_auto_jet[i].a;
            predict_jacobi_.block(i, 0, 1, N_STATE) = Xp_auto_jet[i].v.transpose();
        }
        state_cov_ = predict_jacobi_ * state_cov_ * predict_jacobi_.transpose() + predict_cov_;
        return x_predict_;
    }

    template<class Func>
    VectorX update(Func &&func, const VectorY &Y) {
        ceres::Jet<double, N_STATE> Xp_auto_jet[N_STATE];
        for (int i = 0; i < N_STATE; i++) {
            Xp_auto_jet[i].a = x_predict_[i];
            Xp_auto_jet[i].v[i] = 1;
        }
        ceres::Jet<double, N_STATE> Yp_auto_jet[N_OBSERVE];
        func(Xp_auto_jet, Yp_auto_jet);
        for (int i = 0; i < N_OBSERVE; i++) {
            predict_observe_[i] = Yp_auto_jet[i].a;
            observe_jacobi_.block(i, 0, 1, N_STATE) = Yp_auto_jet[i].v.transpose();
        }
        kalman_gain_ = state_cov_ * observe_jacobi_.transpose() * (observe_jacobi_ * state_cov_ * observe_jacobi_.transpose() + observe_cov_).inverse();
        x_estimate_ = x_predict_ + kalman_gain_ * (Y - predict_observe_);
        state_cov_ = (MatrixXX::Identity() - kalman_gain_ * observe_jacobi_) * state_cov_;
        return x_estimate_;
    }
    VectorX x_estimate_;     
    VectorX x_predict_;    
    MatrixXX predict_jacobi_;    
    MatrixYX observe_jacobi_;     
    MatrixXX state_cov_;    
    MatrixXX predict_cov_;    
    MatrixYY observe_cov_;     
    MatrixXY kalman_gain_;     
    VectorY predict_observe_;     
};


#endif /* EKF_HPP_ */
