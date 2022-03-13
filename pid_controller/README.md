# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

---

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1(mac, linux), 3.81(Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `./install-mac.sh` or `./install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Simulator. You can download these from the [project intro page](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.

Fellow students have put together a guide to Windows set-up for the project [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/files/Kidnapped_Vehicle_Windows_Setup.pdf) if the environment you have set up for the Sensor Fusion projects does not work for this project. There's also an experimental patch for windows in this [PR](https://github.com/udacity/CarND-PID-Control-Project/pull/3).

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`. Optionally, you can set Kp, Ki and Kd parameters as additional arguments (e.g. ./pid 1.0 1.0 1.0)

Tips for setting up your environment can be found [here](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/23d376c7-0195-4276-bdf0-e02f1f3c665d)

## PID components

1. Proportional gain: in charge of producing a proportional effect in the output with regards the cross-track error (CTE). This component causes considerable correction values that overshoots the desired reference producing oscillations.

2. Derivative gain: it minimizes the abovementioned overshooting effect with a temporal derivative of CTE. So that when CTE decreases, this gain mitigates the proportional gain effect. The opposite happens when CTE increases.

3. Integral gain: this factor is proportional to sum of the CTE over time. When there is a systematic bias that affects the output in the long-term, then this gain is needed in order to compensate this undesired effect. 

## How to choose satisfactory parameters

There are two ways to tune this parameters: by either setting them manually or using iterative algorithms.

In this case, the implementation has been done by manually iterating.

The process to do that is:

0. Consider Kp, Ki and Kd 0 as initial values.

1. Tune Kp: by increasing it until the point that the car moves too aggressively towards the lane center. Once this is done, oscillations have been observed since there is no Derivative gain at this stage.

2. Tune Kd: by increasing it so that oscillations are mitigated and there is no considerable overshooting effect.

3. Tune Ki: this parameter requires a much smaller magnitude order than the previous ones. It is increased carefully so that the uncalibrated steering is mitigated (the car slightly drifts to the right since this simulates an uncalibrated direction of the vehicle).

The tunned parameters have been set to default.
