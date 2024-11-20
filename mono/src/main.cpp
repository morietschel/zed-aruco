///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2020, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

/***********************************************************************************************
 ** This sample demonstrates how to reloc a ZED camera using an ArUco marker.                  **
 ** Images are captured with the ZED SDK and cameras poses is then computed from ArUco pattern **
 ** to reset ZED tracking with this known position.                                            **
 ***********************************************************************************************/

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "aruco.hpp"
#include "GLViewer.hpp"

// OCV includes
#include <opencv2/opencv.hpp>

#include <fstream>

using namespace sl;
using namespace std;

const int MAX_CHAR = 128;

inline void setTxt(sl::float3 value, char *ptr_txt)
{
  snprintf(ptr_txt, MAX_CHAR, "%3.2f; %3.2f; %3.2f", value.x, value.y, value.z);
}

std::map<int, sl::Transform> readArucoTransforms(const std::string& filename) {
  std::map<int, sl::Transform> transforms;

  std::ifstream file(filename);
  if (!file.is_open()) {
    // File could not be opened
    std::cerr << "Error: could not open file " << filename << std::endl;
    return transforms;
  }

  std::string line;
  std::getline(file, line); // Ignore the header line

  while (std::getline(file, line)) {
    // Skip lines starting with "#"
    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::istringstream iss(line);
    std::string idStr, txStr, tyStr, tzStr, qxStr, qyStr, qzStr, qwStr;

    if (!(iss >> idStr >> txStr >> tyStr >> tzStr >> qxStr >> qyStr >> qzStr >> qwStr)) {
      // Failed to extract values from the line
      continue;
    }

    int id;
    float tx, ty, tz, qx, qy, qz, qw;
    try {
      id = std::stoi(idStr);
      tx = std::stof(txStr);
      ty = std::stof(tyStr);
      tz = std::stof(tzStr);
      qx = std::stof(qxStr);
      qy = std::stof(qyStr);
      qz = std::stof(qzStr);
      qw = std::stof(qwStr);
    } catch (...) {
      // Failed to convert values to the correct types
      continue;
    }

    sl::Transform transform;
    transform.setTranslation(sl::float3(tx, ty, tz));
    transform.setOrientation(sl::float4(qx, qy, qz, qw));

    transforms[id] = transform;
  }

  file.close();
  return transforms;
}

bool isTagValidForReset(const vector<cv::Point2f> &corners, const cv::Size &image_size, float ratio = 0.05) {
  float image_area = image_size.width * image_size.height;

  double side1 = cv::norm(corners[1] - corners[0]);
  double side2 = cv::norm(corners[2] -corners[1]);

  float tag_area = side1 * side2;

  auto size_ratio = tag_area / image_area;

  return size_ratio > ratio;
}

struct MarkerObservation {
    int marker_id;
    sl::Transform camera_to_marker;
    double confidence;
    double timestamp;
};

struct MarkerPosition {
    sl::float3 position;
    sl::float3 orientation;
    double confidence;
    int observation_count;
    std::vector<MarkerObservation> observations;

    MarkerPosition() : confidence(0), observation_count(0) {}
};

std::map<int, MarkerPosition> marker_positions;

double calculateConfidence(const vector<cv::Point2f>& corners, const cv::Size& image_size) {
    // Calculate marker area ratio to image area
    double side1 = cv::norm(corners[1] - corners[0]);
    double side2 = cv::norm(corners[2] - corners[1]);
    double marker_area = side1 * side2;
    double image_area = image_size.width * image_size.height;
    return marker_area / image_area;
}

sl::Transform estimateMarkerPose(const cv::Vec3d& rvec, const cv::Vec3d& tvec) {
    sl::Transform pose;
    pose.setTranslation(sl::float3(tvec(0), tvec(1), tvec(2)));
    pose.setRotationVector(sl::float3(rvec(0), rvec(1), rvec(2)));
    return pose;
}

void parse_args(int argc, char **argv, InitParameters &param, std::map<int, sl::Transform> &aruco_transforms)
{
  if (argc > 1 && string(argv[1]).find(".svo") != string::npos)
  {
    // SVO input mode
    param.input.setFromSVOFile(argv[1]);
    param.svo_real_time_mode = true;

    cout << "[Sample] Using SVO File input: " << argv[1] << endl;

    if (argc > 2 && string(argv[2]).find(".txt") != string::npos) {
      aruco_transforms = readArucoTransforms(std::string(argv[2]));
    }
  }
  else if (argc > 1 && string(argv[1]).find(".txt") != string::npos)
  {
    aruco_transforms = readArucoTransforms(std::string(argv[1]));

  }
  else if (argc > 1 && string(argv[1]).find(".svo") == string::npos)
  {
    string arg = string(argv[1]);
    unsigned int a, b, c, d, port;
    if (sscanf(arg.c_str(), "%u.%u.%u.%u:%d", &a, &b, &c, &d, &port) == 5)
    {
      // Stream input mode - IP + port
      string ip_adress = to_string(a) + "." + to_string(b) + "." + to_string(c) + "." + to_string(d);
      param.input.setFromStream(String(ip_adress.c_str()), port);
      cout << "[Sample] Using Stream input, IP : " << ip_adress << ", port : " << port << endl;
    }
    else if (sscanf(arg.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4)
    {
      // Stream input mode - IP only
      param.input.setFromStream(String(argv[1]));
      cout << "[Sample] Using Stream input, IP : " << argv[1] << endl;
    }
    else if (arg.find("HD2K") != string::npos)
    {
      param.camera_resolution = RESOLUTION::HD2K;
      cout << "[Sample] Using Camera in resolution HD2K" << endl;
    }
    else if (arg.find("HD1200") != string::npos)
    {
      param.camera_resolution = RESOLUTION::HD1200;
      cout << "[Sample] Using Camera in resolution HD1200" << endl;
    }
    else if (arg.find("HD1080") != string::npos)
    {
      param.camera_resolution = RESOLUTION::HD1080;
      cout << "[Sample] Using Camera in resolution HD1080" << endl;
    }
    else if (arg.find("HD720") != string::npos)
    {
      param.camera_resolution = RESOLUTION::HD720;
      cout << "[Sample] Using Camera in resolution HD720" << endl;
    }
    else if (arg.find("SVGA") != string::npos)
    {
      param.camera_resolution = RESOLUTION::SVGA;
      cout << "[Sample] Using Camera in resolution SVGA" << endl;
    }
    else if (arg.find("VGA") != string::npos)
    {
      param.camera_resolution = RESOLUTION::VGA;
      cout << "[Sample] Using Camera in resolution VGA" << endl;
    }
  }
  else
  {
    // Default
  }
}

void saveMarkerPositions(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    file << "marker_id,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,confidence,observations\n";
    for (const auto& marker : marker_positions) {
        file << marker.first << ","
             << marker.second.position.x << ","
             << marker.second.position.y << ","
             << marker.second.position.z << ","
             << marker.second.orientation.x << ","
             << marker.second.orientation.y << ","
             << marker.second.orientation.z << ","
             << marker.second.confidence << ","
             << marker.second.observation_count << "\n";
    }
    file.close();
    std::cout << "Marker positions saved to: " << filename << std::endl;
}

int main(int argc, char **argv)
{

  // Create a ZED camera object
  Camera zed;

  // Create optional Aruco transforms container
  std::map<int, sl::Transform> aruco_transforms;

  // Set configuration parameters
  InitParameters init_params;
  init_params.camera_resolution = RESOLUTION::HD720;
  init_params.coordinate_units = UNIT::METER;
  init_params.sensors_required = false;
  init_params.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
  parse_args(argc, argv, init_params, aruco_transforms);
  init_params.svo_real_time_mode = false;
  init_params.camera_image_flip = sl::FLIP_MODE::AUTO;

  //  Open the camera
  ERROR_CODE err = zed.open(init_params);
  if (err != ERROR_CODE::SUCCESS)
  {
    cerr << "Error, unable to open ZED camera: " << sl::toString(err).c_str() << "\n";
    zed.close();
    return EXIT_FAILURE; // Quit if an error occurred
  }

  auto cameraInfo = zed.getCameraInformation();
  Resolution image_size = cameraInfo.camera_configuration.resolution;
  Mat image_zed(image_size, MAT_TYPE::U8_C4);
  cv::Mat image_ocv = cv::Mat(image_zed.getHeight(), image_zed.getWidth(), CV_8UC4, image_zed.getPtr<sl::uchar1>(MEM::CPU));
  cv::Mat image_ocv_rgb;

  GLViewer viewer;
  char text_rotation[MAX_CHAR];
  char text_translation[MAX_CHAR];
  viewer.init(argc, argv, cameraInfo.camera_model, init_params.coordinate_system);
  viewer.setArucoTags(aruco_transforms);

  auto calibInfo = cameraInfo.camera_configuration.calibration_parameters.left_cam;
  cv::Matx33d camera_matrix = cv::Matx33d::eye();
  camera_matrix(0, 0) = calibInfo.fx;
  camera_matrix(1, 1) = calibInfo.fy;
  camera_matrix(0, 2) = calibInfo.cx;
  camera_matrix(1, 2) = calibInfo.cy;

  cv::Matx<float, 4, 1> dist_coeffs = cv::Vec4f::zeros();

  float actual_marker_size_meters = 0.17; // real marker size in meters -> IT'S IMPORTANT THAT THIS VARIABLE CONTAINS THE CORRECT SIZE OF THE TARGET

  auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_100);

  cout << "Make sure the ArUco marker is a 6x6 (100), measuring " << actual_marker_size_meters * 1000 << " mm" << endl;

  Transform pose;
  Pose zed_pose;
  vector<cv::Vec3d> rvecs, tvecs;
  vector<int> ids;
  vector<vector<cv::Point2f>> corners;
  string position_txt;

  float auto_reset_aruco_screen_ratio = 0.01;
  int last_aruco_reset = -1;

  bool can_reset = false;

  PositionalTrackingParameters tracking_params;
  tracking_params.enable_imu_fusion =
      false; // for this sample, IMU (of ZED-M) is disable, we use the gravity
             // given by the marker.
  tracking_params.set_gravity_as_origin = false;
  tracking_params.enable_area_memory = false;
  zed.enablePositionalTracking(tracking_params);

  // Loop until 'q' or 'Q' is pressed
  char key = '.';
  int wait_time = 10;
  bool has_reset = false;
  bool zed_tracking = false;
  bool showArucoPath = false;

  sl::Transform ARUCO_TO_IMAGE_basis_change;
  ARUCO_TO_IMAGE_basis_change.r00 = -1;
  ARUCO_TO_IMAGE_basis_change.r11 = -1;
  sl::Transform IMAGE_TO_ARUCO_basis_change;
  IMAGE_TO_ARUCO_basis_change = sl::Transform::inverse(ARUCO_TO_IMAGE_basis_change);

  POSITIONAL_TRACKING_STATE tracking_state;
  bool pause_camera = false;

  // find the nearest aruco to the camera
  int nearest_aruco_index = 0;


  while (viewer.isAvailable() && key != 'q' && key != 'Q')
  {
    if (!pause_camera && zed.grab() == ERROR_CODE::SUCCESS) {
      // Retrieve the left image
      zed.retrieveImage(image_zed, VIEW::LEFT, MEM::CPU, image_size);

      // convert to RGB
      cv::cvtColor(image_ocv, image_ocv_rgb, cv::COLOR_BGRA2BGR);

      cv::Mat grayImage;
      cv::cvtColor(image_ocv_rgb, grayImage, cv::COLOR_BGR2GRAY);

      // detect marker
      cv::aruco::detectMarkers(image_ocv_rgb, dictionary, corners, ids);


      for (size_t i = 0; i < corners.size(); ++i) {
        cv::cornerSubPix(grayImage, corners[i], cv::Size(5, 5), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
      }


      // get actual ZED position : basis change from camera to world expressed in init_params.coordinate_system
      tracking_state = zed.getPosition(zed_pose);

      // display ZED position
      cv::rectangle(image_ocv_rgb, cv::Point(0, 0), cv::Point(490, 75),
                    cv::Scalar(0, 0, 0), -1);
      cv::putText(image_ocv_rgb,
                  "Loaded dictionary : 6x6.     Press 'SPACE' to reset "
                  "the camera position",
                  cv::Point(10, 15), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                  cv::Scalar(220, 220, 220));
      position_txt = "ZED  x: " + to_string(zed_pose.pose_data.tx) +
                     "; y: " + to_string(zed_pose.pose_data.ty) +
                     "; z: " + to_string(zed_pose.pose_data.tz);
      position_txt =
          position_txt + " rx " +
          to_string(zed_pose.pose_data.getEulerAngles(false).x) +
          " ry : " + to_string(zed_pose.pose_data.getEulerAngles(false).y) +
          " rz : " + to_string(zed_pose.pose_data.getEulerAngles(false).z);
      if (zed_tracking)
        position_txt += " *";
      cv::putText(image_ocv_rgb, position_txt, cv::Point(10, 35),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(236, 188, 26), 2);

      // if at least one marker detected
      if (ids.size() > 0) {
        cv::aruco::estimatePoseSingleMarkers(corners, actual_marker_size_meters,
                                             camera_matrix, dist_coeffs, rvecs, tvecs);

        // Process each detected marker
        for (size_t i = 0; i < ids.size(); i++) {
            int marker_id = ids[i];
            
            // Calculate observation confidence based on marker size in image
            double confidence = calculateConfidence(corners[i], cv::Size(image_zed.getWidth(), image_zed.getHeight()));
            
            // Skip if marker is too small/far
            if (confidence < 0.001) continue;

            // Create marker observation in camera coordinate system
            MarkerObservation observation;
            observation.marker_id = marker_id;
            observation.camera_to_marker = estimateMarkerPose(rvecs[i], tvecs[i]);
            observation.confidence = confidence;
            observation.timestamp = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE).getMilliseconds();

            // Update marker position
            auto& marker = marker_positions[marker_id];
            marker.observations.push_back(observation);
            marker.observation_count++;

            // Update running average position and orientation
            sl::float3 new_position = observation.camera_to_marker.getTranslation();
            sl::float3 new_orientation = observation.camera_to_marker.getEulerAngles();
            
            if (marker.observation_count == 1) {
                marker.position = new_position;
                marker.orientation = new_orientation;
                marker.confidence = confidence;
            } else {
                // Weighted average based on confidence
                float alpha = confidence / (marker.confidence + confidence);
                marker.position = marker.position * (1 - alpha) + new_position * alpha;
                marker.orientation = marker.orientation * (1 - alpha) + new_orientation * alpha;
                marker.confidence = (marker.confidence + confidence) / 2;
            }

            // Output current marker position (you can modify this based on your needs)
            std::cout << "Marker " << marker_id << " Position (in camera frame):" 
                      << " x: " << new_position.x
                      << " y: " << new_position.y
                      << " z: " << new_position.z 
                      << " confidence: " << confidence << std::endl;
        }

        // Optional: Visualize markers
        cv::aruco::drawDetectedMarkers(image_ocv_rgb, corners, ids);
        for (size_t i = 0; i < ids.size(); i++) {
            cv::aruco::drawAxis(image_ocv_rgb, camera_matrix, dist_coeffs, 
                               rvecs[i], tvecs[i], actual_marker_size_meters * 0.5f);
        }
      }
      else
        can_reset = false;

      if (ids.size() == 0) {
        rvecs.clear();
        tvecs.clear();
        rvecs.resize(1);
        tvecs.resize(1);
      }

      // display zed tracking world into image
      // this world should not move if the zed tracking is not drifting
      // we can see the drift of the zed tracking by visualizing the motion of the world
      if (zed_tracking) {
        if (has_reset) {
          // zed pose is camera to world basis change, expressed in init_params.coordinate_system coordinate system
          auto transform = zed_pose.pose_data;

          // drawAxis uses world to camera (IMAGE coordinate system) basis change, so inverse the zed pose to match
          transform.inverse();

          // get user_coordinate_to_image from SDK
          auto user_coordinate_to_image = sl::getCoordinateTransformConversion4f(
              init_params.coordinate_system, sl::COORDINATE_SYSTEM::IMAGE);

          // to project the right axis in the image, we apply the
          // user_coordinate_to_image transform to the pose
          transform = user_coordinate_to_image * transform;

          // set the zed pose, world (expressed in init_params.coordinate_system)
          // to camera (expressed in IMAGE) to rvecs and tvecs
          // for display
          sl::float3 rotation = transform.getRotationVector();
          rvecs[nearest_aruco_index](0) = rotation.x;
          rvecs[nearest_aruco_index](1) = rotation.y;
          rvecs[nearest_aruco_index](2) = rotation.z;
          tvecs[nearest_aruco_index](0) = transform.tx;
          tvecs[nearest_aruco_index](1) = transform.ty;
          tvecs[nearest_aruco_index](2) = transform.tz;

        }
      } else {
        // recompute rvecs, tvecs from aruco pose for display
        auto transform = pose;

        // drawAxis uses world to camera basis change, so inverse the aruco pose to match. Note that aruco pose was changed to follow the zed SDK convention (camera to world expressed in user coordinate system)
        transform.inverse();

        auto user_coordinate_to_image = sl::getCoordinateTransformConversion4f(
            init_params.coordinate_system, sl::COORDINATE_SYSTEM::IMAGE);

        // to project the right axis in the image, we apply the
        // user_coordinate_to_image transform to the pose
        transform = user_coordinate_to_image * transform;

        // set the aruco pose, world (expressed in
        // init_params.coordinate_system) to camera (expressed in IMAGE) to
        // rvecs and tvecs for display
        sl::float3 rotation = transform.getRotationVector();
        rvecs[nearest_aruco_index](0) = rotation.x;
        rvecs[nearest_aruco_index](1) = rotation.y;
        rvecs[nearest_aruco_index](2) = rotation.z;
        tvecs[nearest_aruco_index](0) = transform.tx;
        tvecs[nearest_aruco_index](1) = transform.ty;
        tvecs[nearest_aruco_index](2) = transform.tz;
      }

      // Apply auto aruco reset when last reset was made with no or different aruco
      if (!ids.empty() && can_reset && last_aruco_reset != ids[nearest_aruco_index]) {
        bool resetPose = isTagValidForReset(corners[nearest_aruco_index], cv::Size(image_zed.getWidth(), image_zed.getHeight()), auto_reset_aruco_screen_ratio);
        if (resetPose) {
          std::cout << "Auto Pose Reset!" << std::endl;
          zed.resetPositionalTracking(pose);
          has_reset = true;
          last_aruco_reset = ids[nearest_aruco_index];
        }
      }


      cv::aruco::drawAxis(image_ocv_rgb, camera_matrix, dist_coeffs, rvecs[nearest_aruco_index],
                          tvecs[nearest_aruco_index], actual_marker_size_meters * 0.5f);
      position_txt = "Aruco x: " + to_string(pose.tx) +
                     "; y: " + to_string(pose.ty) +
                     "; z: " + to_string(pose.tz);
      position_txt = position_txt + " rx " +
                     to_string(pose.getEulerAngles(false).x) +
                     " ry : " + to_string(pose.getEulerAngles(false).y) +
                     " rz : " + to_string(pose.getEulerAngles(false).z);
      if (!zed_tracking)
        position_txt += " *";
      cv::putText(image_ocv_rgb, position_txt, cv::Point(10, 60),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(124, 252, 124), 2);


      sl::Transform viewer_pose = zed_pose.pose_data;

      auto z_t = viewer_pose.getTranslation();
      viewer_pose.setTranslation(sl::Translation(z_t.x, z_t.y, 0.0f));
      auto aruco_t = pose.getTranslation();
      pose.setTranslation(sl::Translation(aruco_t.x, aruco_t.y, 0.0f));

      setTxt(viewer_pose.getEulerAngles(false), text_rotation);
      setTxt(viewer_pose.getTranslation(), text_translation);
      // Update rotation, translation and tracking state values in the OpenGL window
      viewer.updateData(viewer_pose, pose, string(text_translation),
                        string(text_rotation), tracking_state);

      // Display image
      // Calculate the new size
      cv::imshow("Image", image_ocv_rgb);
      }

    // Handle key event
    key = cv::waitKey(wait_time);

    // if KEY_SPACE is pressed and aruco marker is visible, then reset ZED position
    if ((key == ' ') && can_reset)
    {
      //  the pose is already in init_params.coordinate_system and represent a
      //  camera to world basis change set it to resetPositionalTracking
      //  function
      std::cout << "Pose Reset to pose: " << pose.getTranslation() << ", " << pose.getEulerAngles() << std::endl;
      zed.resetPositionalTracking(pose);
      has_reset = true;
    }
    // pause the video
    if ((key == 'p' || key == 'P'))
    {
      pause_camera = !pause_camera;
    }
    // switch the view to zed tracking visualization
    if ((key == 'z' || key == 'Z'))
    {
      zed_tracking = true;
    }

    if ((key == 'd' || key == 'D'))
    {
      showArucoPath = !showArucoPath;
      viewer.setShowArucoPath(showArucoPath);
    }

    // switch the view to aruco tracking visualization
    if ((key == 'a' || key == 'A'))
    {
      zed_tracking = false;
    }

    if (key == 's' || key == 'S') {
        saveMarkerPositions("marker_positions.csv");
    }
  }
  zed.close();
  return 0;
}
