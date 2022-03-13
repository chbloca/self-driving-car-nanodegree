#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/*
 * Please note that the Eigen library does not initialize
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
   * TODO: predict the state
   */
    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * TODO: update the state by using Kalman Filter equations
   */
    const VectorXd y = z - H_ * x_;
    UpdateNexus(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * TODO: update the state by using Extended Kalman Filter equations
   */
    double px = x_(0);
    double py = x_(1);
    double vx = x_(2);
    double vy = x_(3);

    double rho = sqrt(px * px + py * py);
    double theta = atan2(py, px);
    double rho_dot = (px * vx + py * vy) / rho;

    VectorXd h = VectorXd(3);
    h << rho, theta, rho_dot;

    VectorXd y = z - h;

    // Make sure theta is comprehended between the indicated ranges
    while(y(1) < -M_PI || y(1) > M_PI){
        if(y(1) < M_PI)
            y(1) += M_PI;
        else
            y(1) -= M_PI;
    }
    UpdateNexus(y);
}

void KalmanFilter::UpdateNexus(const VectorXd &y){
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Sinv = S.inverse();
    MatrixXd K = P_ * Ht * Sinv;
    MatrixXd I = MatrixXd::Identity(P_.rows(), P_.cols());

    x_ += K * y;
    P_ = (I - K * H_) * P_;
}
