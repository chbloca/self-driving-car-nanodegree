#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"
#include "spline.h"

// for convenience
using nlohmann::json;
using std::string;
using std::vector;

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  int lane = 1; // start in middle lane
  double ref_vel = 0.0; // mph

  h.onMessage([&lane, &ref_vel, &map_waypoints_x,&map_waypoints_y,&map_waypoints_s,
               &map_waypoints_dx,&map_waypoints_dy]
              (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
               uWS::OpCode opCode) {

    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
          // Main car's localization Data
          double car_x = j[1]["x"];
          double car_y = j[1]["y"];
          double car_s = j[1]["s"];
          double car_d = j[1]["d"];
          double car_yaw = j[1]["yaw"];
          double car_speed = j[1]["speed"];

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values 
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];

          // Sensor Fusion Data, a list of all other cars on the same side 
          //   of the road.
          auto sensor_fusion = j[1]["sensor_fusion"];

          json msgJson;

          vector<double> next_x_vals;
          vector<double> next_y_vals;

          /**
           * TODO: define a path made up of (x,y) points that the car will visit
           *   sequentially every .02 seconds
           */
          /* Case 1: Linear path on cartesian coordinates
          double dist_inc = .5;
          for (int i = 0; i < 50; ++i) {
            next_x_vals.push_back(car_x + (dist_inc * i) * cos(deg2rad(car_yaw)));
            next_y_vals.push_back(car_y + (dist_inc * i) * sin(deg2rad(car_yaw)));
          }
          */

          /* Case 2: Linear path on Frenet coordinates
          double dist_inc = .5;
          for (int i = 0; i < 50; i++) {
            double next_s = car_s + dist_inc * (i + 1);
            double next_d = 6;
            vector<double> xy = getXY(next_s, next_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);

            next_x_vals.push_back(xy[0]);
            next_y_vals.push_back(xy[1]);
          }
          */

          /* Case 3: Circular path
          double pos_x;
          double pos_y;
          double angle;
          int path_size = previous_path_x.size();

          for (int i = 0; i < path_size; ++i) {
            next_x_vals.push_back(previous_path_x[i]);
            next_y_vals.push_back(previous_path_y[i]);
          }

          if (path_size == 0) {
            pos_x = car_x;
            pos_y = car_y;
            angle = deg2rad(car_yaw);
          } else {
            pos_x = previous_path_x[path_size-1];
            pos_y = previous_path_y[path_size-1];

            double pos_x2 = previous_path_x[path_size-2];
            double pos_y2 = previous_path_y[path_size-2];
            angle = atan2(pos_y-pos_y2,pos_x-pos_x2);
          }

          double dist_inc = .5;
          for (int i = 0; i < 50-path_size; ++i) {
            next_x_vals.push_back(pos_x+(dist_inc)*cos(angle+(i+1)*(pi()/100)));
            next_y_vals.push_back(pos_y+(dist_inc)*sin(angle+(i+1)*(pi()/100)));
            pos_x += (dist_inc)*cos(angle+(i+1)*(pi()/100));
            pos_y += (dist_inc)*sin(angle+(i+1)*(pi()/100));
          }
          */

          int prev_size = previous_path_x.size();
          if(prev_size > 0)
              car_s = end_path_s;

          // Perception module

          bool too_close_ahead = false;
          bool free_left = true;
          bool free_right = true;

          for(int i = 0; i < sensor_fusion.size(); i++){
              // Other's car is in my lane
              float detected_d = sensor_fusion[i][6]; // transversal displacement from the center of the highway
              float detected_s = sensor_fusion[i][5]; // longitudinal displacement from starting point
              float vx = sensor_fusion[i][3];
              float vy = sensor_fusion[i][4];
              float detected_s_dot = sqrt(vx * vx + vy * vy); // speed
              // predicted s a step ahead in the future
              detected_s += (double)prev_size * .02 * detected_s_dot;

              int car_on_lane;
              if(detected_d > 0 && detected_d < 4)
                  car_on_lane = 0;
              else if(detected_d > 4 && detected_d < 8)
                  car_on_lane = 1;
              else if(detected_d > 8 && detected_d < 12)
                  car_on_lane = 2;
              else
                  continue;

              // if ahead of us there is a car
              if(car_on_lane == lane){
                  if(detected_s - car_s > 0 && detected_s - car_s < 20){
                      too_close_ahead = true;
                  }
              }
              // if free left side
              else if(car_on_lane + 1 == lane){
                      if(abs(car_s + 20) > detected_s && abs(car_s - 15) < detected_s)
                          free_left = false;
              }
              // if free right side
              else if(car_on_lane - 1 == lane){
                          if(abs(car_s + 20) > detected_s && abs(car_s - 15) < detected_s)
                              free_right = false;
              }
          }

          // Behavioral planner

          const double Max_S_dot = 49.5; // max speed mph
          const double Max_S_dot_dot = .224; // ~5 m/s² (< 10 m/s² max. comfortable)

          if(too_close_ahead){
              ref_vel -= Max_S_dot_dot;

              if(free_left && lane > 0)
                  lane--;
              else if(free_right && lane < 2)
                  lane++;

          }else if(ref_vel < Max_S_dot)
              ref_vel += Max_S_dot_dot;


          std::cout << "lane: " << lane << "\ttoo close ahead: " << too_close_ahead
                    << "\tfree on left: " << free_left << "\tfree on right: " << free_right
                    << std::endl;

          // Path planner

          // Create a list of widely spaced waypoints (x, y) which later will be interpolated by a spline
          vector<double> ptsx;
          vector<double> ptsy;

          // Reference x, y, yaw states
          // Either referenciated the starting point as where the car is or ath the previous path's end point
          double ref_x = car_x;
          double ref_y = car_y;
          double ref_yaw = deg2rad(car_yaw);

          if(prev_size < 2){ // if previous size is almost empty
              // Use two points that make the path tangent to the car
              double prev_car_x = car_x - cos(ref_yaw);
              double prev_car_y = car_y - sin(ref_yaw);

              ptsx.push_back(prev_car_x);
              ptsx.push_back(car_x);

              ptsy.push_back(prev_car_y);
              ptsy.push_back(car_y);
          }else{ // use the previous path's end point as starting reference
              // Redefine reference state as previous path end point
              ref_x = previous_path_x[prev_size - 1];
              ref_y = previous_path_y[prev_size - 1];

              double ref_x_prev = previous_path_x[prev_size - 2];
              double ref_y_prev = previous_path_y[prev_size - 2];
              ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

              // Use two points that make the path tangent to the previous path's end point
              ptsx.push_back(ref_x_prev);
              ptsx.push_back(ref_x);

              ptsy.push_back(ref_y_prev);
              ptsy.push_back(ref_y);
          }
          // In Frenet add evenly 30m spaced points ahead of the starting reference
          vector<double> next_wp0 = getXY(car_s + 30, 2 + 4 * lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
          vector<double> next_wp1 = getXY(car_s + 60, 2 + 4 * lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
          vector<double> next_wp2 = getXY(car_s + 90, 2 + 4 * lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);

          ptsx.push_back(next_wp0[0]);
          ptsx.push_back(next_wp1[0]);
          ptsx.push_back(next_wp2[0]);

          ptsy.push_back(next_wp0[1]);
          ptsy.push_back(next_wp1[1]);
          ptsy.push_back(next_wp2[1]);

          for(int i = 0; i < ptsx.size(); i++){
              // Shift the car reference angle to 0 degrees
              double shift_x = ptsx[i] - ref_x;
              double shift_y = ptsy[i] - ref_y;

              ptsx[i] = shift_x * cos(0 - ref_yaw) - shift_y * sin(0 - ref_yaw);
              ptsy[i] = shift_x * sin(0 - ref_yaw) + shift_y * cos(0 - ref_yaw);
          }

          // Create a spline
          tk::spline s;

          // Set (x, y) points to the spline
          s.set_points(ptsx, ptsy);

          // Start with all of the previous path points from last time
          for(int i = 0; i < previous_path_x.size(); i++){
              next_x_vals.push_back(previous_path_x[i]);
              next_y_vals.push_back(previous_path_y[i]);
          }

          // Calculate how to break up spline points so that we travel at our desired reference speed
          double target_x = 30.0;
          double target_y = s(target_x);
          double target_dist = sqrt(target_x * target_x + target_y * target_y);

          double x_initial = 0;

          // Fill up the rest of our path planner after filling it with previous points
          // Output 50 points

          for(int i = 0; i <= 50 - previous_path_x.size(); i++){              
              double N = target_dist / (.02 * ref_vel / 2.24);
              double x_point = x_initial + target_x / N;
              double y_point = s(x_point);

              x_initial = x_point;

              double x_ref = x_point;
              double y_ref = y_point;

              // Rotate back to normal after rotating it earlier
              x_point = x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw);
              y_point = x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw);

              x_point += ref_x;
              y_point += ref_y;

              next_x_vals.push_back(x_point);
              next_y_vals.push_back(y_point);
          }

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}
