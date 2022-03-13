#include "PID.h"

/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
  /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   */
    Kp = Kp_;
    Ki = Ki_;
    Kd = Kd_;

    p_error = 0;
    i_error = 0;
    d_error = 0;

    step_0 = true;
}

void PID::UpdateError(double cte) {
  /**
   * TODO: Update PID errors based on cte.
   */
    double dt = 1;

    p_error = cte;
    i_error += cte * dt;

    if(step_0)
        step_0 = false;
    else
        d_error = (cte - prev_cte) / dt;
    prev_cte = cte;
}

double PID::TotalError() {
  /**
   * TODO: Calculate and return the total error
   */
  return - Kp * p_error - Ki * i_error - Kd * d_error;  // TODO: Add your total error calc here!

}
