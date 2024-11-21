
// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "aruco.hpp"
#include "GLViewer.hpp"

// OCV includes
#include <opencv2/opencv.hpp>

#include <fstream>
#include <chrono>

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

// First, add all structure definitions
struct MarkerObservation {
    int marker_id;
    sl::Transform camera_to_marker;
    double confidence;
    double timestamp;
};

struct MarkerPosition {
    int marker_id;
    sl::Transform relative_transform;  // Transform relative to reference marker
    double confidence;
    int observation_count;

    MarkerPosition() : marker_id(-1), confidence(0), observation_count(0) {}
};

// Add this struct near other struct definitions
struct TimestampedObservations {
    double timestamp;
    std::vector<MarkerObservation> observations;
};

// Modify RecordingSession struct
struct RecordingSession {
    bool is_recording;
    double start_time;
    double end_time;
    std::vector<TimestampedObservations> timestamped_observations; // Replace observations map
    std::set<int> seen_markers;
    
    RecordingSession() : is_recording(false), start_time(0), end_time(0) {}
};

struct MarkerRelation {
    int marker1_id;
    int marker2_id;
    sl::Transform relative_transform;
    double confidence;
    int observation_count;
};

// Then, add global variables
std::vector<RecordingSession> recording_sessions;
std::map<std::pair<int, int>, MarkerRelation> marker_relations;

// Then, add function declarations (prototypes)
void processRecordingSession(const RecordingSession& session);
void startRecording();
void stopRecording();
//void convertZedToRhino(const sl::Transform& zed_pose, sl::Transform& rhino_pose);
void saveMarkerRelations(const std::string& filename);
void calculateAbsolutePositions(const std::vector<RecordingSession>& sessions, 
                               std::map<int, MarkerPosition>& absolute_positions);
void saveAbsolutePositions(const std::string& filename);
bool validateTransform(const sl::Transform& transform);
// Add these function declarations
sl::Transform convertZedToRhino(const sl::Transform& zed_pose);
sl::float3 convertZedToRhino(const sl::float3& zed_coords);
// Add this helper function near the top with other utility functions
sl::Transform interpolateTransforms(const sl::Transform& t1, const sl::Transform& t2, float alpha) {
    sl::Transform result;
    
    // Interpolate translation linearly
    auto trans1 = t1.getTranslation();
    auto trans2 = t2.getTranslation();
    result.setTranslation(sl::float3(
        trans1.x * (1 - alpha) + trans2.x * alpha,
        trans1.y * (1 - alpha) + trans2.y * alpha,
        trans1.z * (1 - alpha) + trans2.z * alpha
    ));
    
    // Retrieve and normalize quaternions
    sl::float4 quat1 = t1.getOrientation();
    sl::float4 quat2 = t2.getOrientation();
    
    float norm1 = sqrt(quat1.x * quat1.x + quat1.y * quat1.y + quat1.z * quat1.z + quat1.w * quat1.w);
    float norm2 = sqrt(quat2.x * quat2.x + quat2.y * quat2.y + quat2.z * quat2.z + quat2.w * quat2.w);
    
    quat1 = sl::float4(quat1.x / norm1, quat1.y / norm1, quat1.z / norm1, quat1.w / norm1);
    quat2 = sl::float4(quat2.x / norm2, quat2.y / norm2, quat2.z / norm2, quat2.w / norm2);
    
    // Compute the dot product to determine the shortest path
    float dot_product = quat1.x * quat2.x + quat1.y * quat2.y + quat1.z * quat2.z + quat1.w * quat2.w;
    
    // If the dot product is negative, invert one quaternion to ensure the shortest path
    if (dot_product < 0.0f) {
        quat2 = sl::float4(-quat2.x, -quat2.y, -quat2.z, -quat2.w);
        dot_product = -dot_product;
    }
    
    const float DOT_THRESHOLD = 0.9995f;
    sl::float4 interpolated_quat;
    
    if (dot_product > DOT_THRESHOLD) {
        // Quaternions are very close, use linear interpolation
        interpolated_quat.x = quat1.x + alpha * (quat2.x - quat1.x);
        interpolated_quat.y = quat1.y + alpha * (quat2.y - quat1.y);
        interpolated_quat.z = quat1.z + alpha * (quat2.z - quat1.z);
        interpolated_quat.w = quat1.w + alpha * (quat2.w - quat1.w);
    } else {
        // Use SLERP for interpolation
        float theta_0 = acos(dot_product);            // Angle between quaternions
        float theta = theta_0 * alpha;                // Interpolated angle
        float sin_theta = sin(theta);
        float sin_theta_0 = sin(theta_0);
        
        float s1 = cos(theta) - dot_product * sin_theta / sin_theta_0;
        float s2 = sin_theta / sin_theta_0;
        
        interpolated_quat.x = (s1 * quat1.x) + (s2 * quat2.x);
        interpolated_quat.y = (s1 * quat1.y) + (s2 * quat2.y);
        interpolated_quat.z = (s1 * quat1.z) + (s2 * quat2.z);
        interpolated_quat.w = (s1 * quat1.w) + (s2 * quat2.w);
    }
    
    // Normalize the interpolated quaternion
    float norm_result = sqrt(interpolated_quat.x * interpolated_quat.x + 
                             interpolated_quat.y * interpolated_quat.y +
                             interpolated_quat.z * interpolated_quat.z +
                             interpolated_quat.w * interpolated_quat.w);
    interpolated_quat.x /= norm_result;
    interpolated_quat.y /= norm_result;
    interpolated_quat.z /= norm_result;
    interpolated_quat.w /= norm_result;
    
    result.setOrientation(interpolated_quat);
    
    return result;
}

// Then, implement the functions
void processRecordingSession(const RecordingSession& session) {
    std::cout << "Processing recording session..." << std::endl;
    
    marker_relations.clear();
    
    if (session.timestamped_observations.empty()) {
        std::cerr << "Warning: Recording session contains no observations." << std::endl;
        return;
    }

    for (const auto& frame : session.timestamped_observations) {
        const auto& observations = frame.observations;
        
        // Process all pairs of markers visible at this timestamp
        for (size_t i = 0; i < observations.size(); i++) {
            for (size_t j = i + 1; j < observations.size(); j++) {
                const auto& obs1 = observations[i];
                const auto& obs2 = observations[j];
                
                // Ensure marker IDs are ordered
                int marker1_id = std::min(obs1.marker_id, obs2.marker_id);
                int marker2_id = std::max(obs1.marker_id, obs2.marker_id);
                
                // Calculate relative transform
                sl::Transform relative_transform;
                try {
                    if (obs1.marker_id == marker1_id) {
                        relative_transform = sl::Transform::inverse(obs1.camera_to_marker) * obs2.camera_to_marker;
                    } else {
                        relative_transform = sl::Transform::inverse(obs2.camera_to_marker) * obs1.camera_to_marker;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Error calculating relative transform between markers " 
                              << marker1_id << " and " << marker2_id 
                              << ": " << e.what() << std::endl;
                    continue;
                }
                
                if (!validateTransform(relative_transform)) {
                    std::cerr << "Invalid relative transform between markers " 
                              << marker1_id << " and " << marker2_id << std::endl;
                    continue;
                }
                
                // Update marker relation
                auto relation_key = std::make_pair(marker1_id, marker2_id);
                auto& relation = marker_relations[relation_key];
                
                if (relation.observation_count == 0) {
                    relation.marker1_id = marker1_id;
                    relation.marker2_id = marker2_id;
                    relation.relative_transform = relative_transform;
                    relation.confidence = (obs1.confidence + obs2.confidence) / 2.0;
                    relation.observation_count = 1;
                }
                else {
                    // Update running average with SLERP interpolated rotation
                    float total_confidence = relation.confidence * relation.observation_count;
                    float new_confidence = (obs1.confidence + obs2.confidence) / 2.0;
                    float alpha = new_confidence / (total_confidence + new_confidence);
                    
                    relation.relative_transform = interpolateTransforms(
                        relation.relative_transform, relative_transform, alpha);
                    relation.confidence = 
                        (relation.confidence * relation.observation_count + new_confidence) / 
                        (relation.observation_count + 1);
                    relation.observation_count++;
                }
            }
        }
    }
    
    // Print summary
    std::cout << "\nProcessed marker relations:" << std::endl;
    for (const auto& relation : marker_relations) {
        std::cout << "Markers " << relation.first.first << " -> " << relation.first.second 
                  << " (" << relation.second.observation_count << " observations, confidence: "
                  << relation.second.confidence << ")" << std::endl;
    }
}
// Add this function to save raw recording data
void saveRawRecording(const RecordingSession& session, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    file << "# Recording Session\n";
    file << "# start_time: " << session.start_time << "\n";
    file << "# end_time: " << session.end_time << "\n";
    file << "# Format: timestamp,marker_id,tx,ty,tz,qx,qy,qz,qw,confidence\n";

    for (const auto& frame : session.timestamped_observations) {
        for (const auto& obs : frame.observations) {
            auto trans = obs.camera_to_marker.getTranslation();
            auto orient = obs.camera_to_marker.getOrientation();
            
            file << frame.timestamp << ","
                 << obs.marker_id << ","
                 << trans.x << ","
                 << trans.y << ","
                 << trans.z << ","
                 << orient.x << ","
                 << orient.y << ","
                 << orient.z << ","
                 << orient.w << ","
                 << obs.confidence << "\n";
        }
    }

    file.close();
    std::cout << "Raw recording data saved to: " << filename << std::endl;
}
void startRecording() {
    RecordingSession session;
    session.is_recording = true;
    session.start_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    recording_sessions.push_back(session);
}

void stopRecording() {
    if (!recording_sessions.empty()) {
        auto& session = recording_sessions.back();
        session.is_recording = false;
        session.end_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        std::string timestamp = std::to_string(session.end_time);
        
        // Save raw data
        saveRawRecording(session, "recording_" + timestamp + ".csv");
        
        // Process the session
        processRecordingSession(session);
        
        // Save the processed relations
        saveMarkerRelations("relations_" + timestamp + ".csv");
        
        // Calculate and save absolute positions
        saveAbsolutePositions("positions_" + timestamp + ".csv");
        
        std::cout << "Recording processed and saved:\n"
                  << "  - Raw data: recording_" << timestamp << ".csv\n"
                  << "  - Marker relations: relations_" << timestamp << ".csv\n"
                  << "  - Absolute positions: positions_" << timestamp << ".csv" << std::endl;
    }
}

std::map<int, MarkerPosition> marker_positions;

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

    file << "marker_id,x,y,z,qx,qy,qz,qw,confidence,observations\n";
    for (const auto& marker : marker_positions) {
        sl::Transform rhino_pose = convertZedToRhino(marker.second.relative_transform);
        auto trans = rhino_pose.getTranslation();
        auto orient = rhino_pose.getOrientation();
        
        file << marker.first << ","
             << trans.x << ","
             << trans.y << ","
             << trans.z << ","
             << orient.x << ","
             << orient.y << ","
             << orient.z << ","
             << orient.w << ","
             << marker.second.confidence << ","
             << marker.second.observation_count << "\n";
    }
    file.close();
    std::cout << "Marker positions saved to: " << filename << std::endl;
}

void saveMarkerRelations(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    file << "# Marker Relations File\n";
    file << "# Format: marker1_id,marker2_id,tx,ty,tz,qx,qy,qz,qw,confidence,observations\n";
    
    for (const auto& relation : marker_relations) {
        const auto& r = relation.second;
        auto trans = r.relative_transform.getTranslation();
        auto orient = r.relative_transform.getOrientation();
        
        file << r.marker1_id << ","
             << r.marker2_id << ","
             << trans.x << ","
             << trans.y << ","
             << trans.z << ","
             << orient.x << ","
             << orient.y << ","
             << orient.z << ","
             << orient.w << ","
             << r.confidence << ","
             << r.observation_count << "\n";
    }
    
    file.close();
    std::cout << "Marker relations saved to: " << filename << std::endl;
}

std::map<std::pair<int, int>, MarkerRelation> loadMarkerRelations(const std::string& filename) {
    std::map<std::pair<int, int>, MarkerRelation> relations;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return relations;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string value;
        std::vector<std::string> values;
        
        while (std::getline(iss, value, ',')) {
            values.push_back(value);
        }

        if (values.size() != 11) continue; // Skip invalid lines

        try {
            MarkerRelation relation;
            relation.marker1_id = std::stoi(values[0]);
            relation.marker2_id = std::stoi(values[1]);
            
            sl::Transform transform;
            transform.setTranslation(sl::float3(
                std::stof(values[2]),
                std::stof(values[3]),
                std::stof(values[4])
            ));
            transform.setOrientation(sl::float4(
                std::stof(values[5]),
                std::stof(values[6]),
                std::stof(values[7]),
                std::stof(values[8])
            ));
            
            relation.relative_transform = transform;
            relation.confidence = std::stod(values[9]);
            relation.observation_count = std::stoi(values[10]);

            auto key = std::make_pair(relation.marker1_id, relation.marker2_id);
            relations[key] = relation;
            
        } catch (const std::exception& e) {
            std::cerr << "Error parsing line: " << line << std::endl;
            continue;
        }
    }

    file.close();
    std::cout << "Loaded " << relations.size() << " marker relations from: " << filename << std::endl;
    return relations;
}

// Add this function to convert relative positions into absolute positions
void calculateAbsolutePositions(const std::vector<RecordingSession>& sessions, 
                               std::map<int, MarkerPosition>& absolute_positions) {
    if (sessions.empty()) {
        std::cout << "No recording sessions available." << std::endl;
        return;
    }

    // First, find all unique marker IDs and count their observations
    std::map<int, int> marker_observations;
    for (const auto& session : sessions) {
        for (const auto& frame : session.timestamped_observations) {
            for (const auto& obs : frame.observations) {
                marker_observations[obs.marker_id]++;
            }
        }
    }

    if (marker_observations.empty()) {
        std::cout << "No marker observations found." << std::endl;
        return;
    }

    // Find the most observed marker to use as reference
    int reference_marker = -1;
    int max_observations = 0;
    for (const auto& marker_pair : marker_observations) {
        if (marker_pair.second > max_observations) {
            max_observations = marker_pair.second;
            reference_marker = marker_pair.first;
        }
    }

    std::cout << "Using marker " << reference_marker << " as reference (observed " 
              << marker_observations[reference_marker] << " times)" << std::endl;

    // Set reference marker at identity transform
    absolute_positions.clear();
    absolute_positions[reference_marker] = MarkerPosition();
    absolute_positions[reference_marker].marker_id = reference_marker;
    absolute_positions[reference_marker].relative_transform = sl::Transform(); // Identity transform
    absolute_positions[reference_marker].confidence = 1.0;
    absolute_positions[reference_marker].observation_count = marker_observations[reference_marker];

    // Create a graph of marker connections
    std::map<int, std::set<int>> marker_graph;
    for (const auto& relation : marker_relations) {
        marker_graph[relation.first.first].insert(relation.first.second);
        marker_graph[relation.first.second].insert(relation.first.first);
    }

    // Use breadth-first search to process markers in order of distance from reference
    std::queue<int> to_process;
    std::set<int> processed;
    to_process.push(reference_marker);
    processed.insert(reference_marker);

    while (!to_process.empty()) {
        int current_marker = to_process.front();
        to_process.pop();

        // Process all neighbors
        for (int neighbor : marker_graph[current_marker]) {
            if (processed.find(neighbor) != processed.end()) continue;

            // Find the relation between current_marker and neighbor
            auto relation_key = std::make_pair(std::min(current_marker, neighbor), 
                                             std::max(current_marker, neighbor));
            auto relation_it = marker_relations.find(relation_key);
            if (relation_it == marker_relations.end()) continue;

            MarkerPosition pos;
            pos.marker_id = neighbor;
            
            // Calculate transform relative to reference through current_marker
            if (relation_key.first == current_marker) {
                pos.relative_transform = absolute_positions[current_marker].relative_transform * 
                                       relation_it->second.relative_transform;
            } else {
                pos.relative_transform = absolute_positions[current_marker].relative_transform * 
                                       sl::Transform::inverse(relation_it->second.relative_transform);
            }

            pos.confidence = relation_it->second.confidence;
            pos.observation_count = relation_it->second.observation_count;
            absolute_positions[neighbor] = pos;

            // Add neighbor to processing queue
            to_process.push(neighbor);
            processed.insert(neighbor);
        }
    }

    // Print summary
    std::cout << "\nCalculated positions for " << absolute_positions.size() 
              << " markers relative to reference marker " << reference_marker << ":" << std::endl;
    for (const auto& pos : absolute_positions) {
        std::cout << "Marker " << pos.first 
                  << " (confidence: " << pos.second.confidence 
                  << ", observations: " << pos.second.observation_count << "):" << std::endl;
        auto trans = pos.second.relative_transform.getTranslation();
        auto rot = pos.second.relative_transform.getEulerAngles();
        std::cout << "  Translation: " << trans.x << ", " << trans.y << ", " << trans.z << std::endl;
        std::cout << "  Rotation: " << rot.x << ", " << rot.y << ", " << rot.z << std::endl;
    }

    // Print warning for unreachable markers
    std::set<int> unreachable_markers;
    for (const auto& marker : marker_observations) {
        if (processed.find(marker.first) == processed.end()) {
            unreachable_markers.insert(marker.first);
        }
    }
    if (!unreachable_markers.empty()) {
        std::cout << "\nWarning: Could not calculate positions for markers: ";
        for (int marker : unreachable_markers) {
            std::cout << marker << " ";
        }
        std::cout << "\nThese markers are not connected to the reference marker through any chain of observations." << std::endl;
    }
}

void saveAbsolutePositions(const std::string& filename) {
    std::map<int, MarkerPosition> absolute_positions;
    calculateAbsolutePositions(recording_sessions, absolute_positions);

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    // Find reference marker (the one with identity transform)
    int reference_marker = -1;
    for (const auto& marker : absolute_positions) {
        auto trans = marker.second.relative_transform.getTranslation();
        if (trans.x == 0 && trans.y == 0 && trans.z == 0) {
            reference_marker = marker.first;
            break;
        }
    }

    // Write metadata as comments
    file << "# Reference marker ID: " << reference_marker << "\n";
    file << "# Coordinate system: Rhino3D (Z-up, Y-forward)\n";
    file << "# Format: marker_id,x,y,z,qx,qy,qz,qw,confidence,observations\n";

    // Write data
    for (const auto& marker : absolute_positions) {
        sl::Transform rhino_transform = convertZedToRhino(marker.second.relative_transform);
        auto rhino_trans = rhino_transform.getTranslation();
        auto orient = rhino_transform.getOrientation();
        
        file << marker.first << ","
             << rhino_trans.x << ","    // X stays the same
             << rhino_trans.y << ","    // Converted Y (forward in Rhino)
             << rhino_trans.z << ","    // Converted Z (up in Rhino)
             << orient.x << ","
             << orient.y << ","
             << orient.z << ","
             << orient.w << ","
             << marker.second.confidence << ","
             << marker.second.observation_count << "\n";
    }
    
    file.close();
    std::cout << "Absolute marker positions saved to: " << filename << std::endl;
    std::cout << "Reference marker used: " << reference_marker << std::endl;
    std::cout << "Note: Coordinates are in Rhino3D convention (Z-up, Y-forward)" << std::endl;
}

// Add this global variable near the top with other globals
const int PRINT_INTERVAL_MS = 2000; // Print every 2 seconds
const int RECORDING_INTERVAL_MS = 100; // Record every 100 milliseconds
double last_print_time = 0;
double last_record_time = 0;

bool validateTransform(const sl::Transform& transform) {
    // Check for NaN values
    auto trans = transform.getTranslation();
    if (std::isnan(trans.x) || std::isnan(trans.y) || std::isnan(trans.z)) {
        return false;
    }

    // Check quaternion normalization
    auto quat = transform.getOrientation();
    float norm = std::sqrt(quat.x*quat.x + quat.y*quat.y + quat.z*quat.z + quat.w*quat.w);
    if (std::abs(norm - 1.0f) > 1e-3) {
        return false;
    }

    // Check for reasonable translation values (adjust thresholds as needed)
    const float MAX_TRANSLATION = 10.0f; // meters
    if (std::abs(trans.x) > MAX_TRANSLATION || 
        std::abs(trans.y) > MAX_TRANSLATION || 
        std::abs(trans.z) > MAX_TRANSLATION) {
        return false;
    }

    return true;
}

// Add this constant near the top with other constants
const double MIN_CONFIDENCE_THRESHOLD = 0.0005; // Lower threshold for testing

double calculateConfidence(const vector<cv::Point2f>& corners, const cv::Size& image_size) {
    // Calculate marker area ratio to image area
    double side1 = cv::norm(corners[1] - corners[0]);
    double side2 = cv::norm(corners[2] - corners[1]);
    double marker_area = side1 * side2;
    double image_area = image_size.width * image_size.height;
    
    // Normalize confidence between 0 and 1
    double confidence = marker_area / image_area;
    
    // Optional: Apply additional confidence modifiers based on marker visibility
    // For example, reduce confidence if marker is near image edges
    double center_x = (corners[0].x + corners[2].x) / 2;
    double center_y = (corners[0].y + corners[2].y) / 2;
    double distance_from_center = std::sqrt(
        std::pow(center_x - image_size.width/2, 2) + 
        std::pow(center_y - image_size.height/2, 2)
    );
    double max_distance = std::sqrt(
        std::pow(image_size.width/2, 2) + 
        std::pow(image_size.height/2, 2)
    );
    
    // Reduce confidence based on distance from center
    confidence *= (1 - distance_from_center/max_distance);
    
    return confidence;
}

sl::Transform estimateMarkerPose(const cv::Vec3d& rvec, const cv::Vec3d& tvec) {
    sl::Transform pose;

    // Convert rotation vector to rotation matrix using OpenCV
    cv::Mat rot_matrix;
    cv::Rodrigues(rvec, rot_matrix);
    
    // Apply coordinate system correction
    // This matrix transforms:
    // ArUco's Z (forward) -> Rhino's Z (up)
    // ArUco's X (right) -> Rhino's X (right)
    // ArUco's Y (down) -> Rhino's -Y (forward)
    cv::Mat R_correction = (cv::Mat_<double>(3,3) <<
        1,  0,  0,
        0, -1,  0,
        0,  0,  1
    );
    
    cv::Mat R_final = R_correction * rot_matrix;

    // Convert rotation matrix to quaternion
    cv::Matx33d R(R_final);
    double trace = R(0,0) + R(1,1) + R(2,2);
    double qw, qx, qy, qz;

    if (trace > 0) {
        double s = 0.5 / sqrt(trace + 1.0);
        qw = 0.25 / s;
        qx = (R(2,1) - R(1,2)) * s;
        qy = (R(0,2) - R(2,0)) * s;
        qz = (R(1,0) - R(0,1)) * s;
    }
    else {
        if (R(0,0) > R(1,1) && R(0,0) > R(2,2)) {
            double s = 2.0 * sqrt(1.0 + R(0,0) - R(1,1) - R(2,2));
            qw = (R(2,1) - R(1,2)) / s;
            qx = 0.25 * s;
            qy = (R(0,1) + R(1,0)) / s;
            qz = (R(0,2) + R(2,0)) / s;
        }
        else if (R(1,1) > R(2,2)) {
            double s = 2.0 * sqrt(1.0 + R(1,1) - R(0,0) - R(2,2));
            qw = (R(0,2) - R(2,0)) / s;
            qx = (R(0,1) + R(1,0)) / s;
            qy = 0.25 * s;
            qz = (R(1,2) + R(2,1)) / s;
        }
        else {
            double s = 2.0 * sqrt(1.0 + R(2,2) - R(0,0) - R(1,1));
            qw = (R(1,0) - R(0,1)) / s;
            qx = (R(0,2) + R(2,0)) / s;
            qy = (R(1,2) + R(2,1)) / s;
            qz = 0.25 * s;
        }
    }

    // Set pose rotation
    pose.setOrientation(sl::float4(static_cast<float>(qx), 
                                   static_cast<float>(qy), 
                                   static_cast<float>(qz), 
                                   static_cast<float>(qw)));

    // Apply coordinate system correction to translation
    // Transform the translation vector to match Rhino's coordinate system
    pose.setTranslation(sl::float3(
        static_cast<float>(tvec[0]),     // X remains the same
        static_cast<float>(-tvec[1]),    // Y becomes -Y
        static_cast<float>(tvec[2])      // Z remains Z
    ));

    return pose;
}



// Update the implementation
sl::Transform convertZedToRhino(const sl::Transform& zed_pose) {
    sl::Transform rhino_pose;
    
    // Convert translation
    auto trans = zed_pose.getTranslation();
    auto rhino_trans = convertZedToRhino(trans);
    rhino_pose.setTranslation(rhino_trans);
    
    // Keep the orientation as is (assuming it's already in the correct format)
    rhino_pose.setOrientation(zed_pose.getOrientation());
    
    return rhino_pose;
}

sl::float3 convertZedToRhino(const sl::float3& zed_coords) {
    return sl::float3(
        zed_coords.x,      // X remains the same
        -zed_coords.z,     // -Z becomes Y in Rhino3D
        zed_coords.y       // Y becomes Z in Rhino3D
    );
}

int main(int argc, char **argv)
{
    // Add near the top of main(), after the variable declarations
    double current_timestamp;

    // Create a ZED camera object
    Camera zed;

    // Create optional Aruco transforms container
    std::map<int, sl::Transform> aruco_transforms;

    // Set configuration parameters
    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION::HD720;
    init_params.coordinate_units = UNIT::METER;
    init_params.sensors_required = false;
    init_params.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
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

    float actual_marker_size_meters = 0.036; // real marker size in meters -> IT'S IMPORTANT THAT THIS VARIABLE CONTAINS THE CORRECT SIZE OF THE TARGET

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

    POSITIONAL_TRACKING_STATE tracking_state;
    bool pause_camera = false;

    // find the nearest aruco to the camera
    int nearest_aruco_index = 0;

    // Add near the top with other global variables
    std::set<int> seen_marker_ids;

    // Initialize last_record_time
    last_record_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    while (viewer.isAvailable() && key != 'q' && key != 'Q') {
        // Update timestamp at the start of each loop iteration
        current_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();

        if (!pause_camera && zed.grab() == ERROR_CODE::SUCCESS) {
            // Retrieve the left image
            zed.retrieveImage(image_zed, VIEW::LEFT, MEM::CPU, image_size);

            // convert to RGB
            cv::cvtColor(image_ocv, image_ocv_rgb, cv::COLOR_BGRA2BGR);

            cv::Mat grayImage;
            cv::cvtColor(image_ocv_rgb, grayImage, cv::COLOR_BGR2GRAY);

            // Detect markers
            cv::aruco::detectMarkers(image_ocv_rgb, dictionary, corners, ids);

            if (ids.size() > 0) {
                // Estimate pose for each marker
                cv::aruco::estimatePoseSingleMarkers(corners, actual_marker_size_meters, 
                    camera_matrix, dist_coeffs, rvecs, tvecs);

                // Check if it's time to record
                if (current_timestamp - last_record_time >= RECORDING_INTERVAL_MS) {
                    last_record_time = current_timestamp;

                    if (!recording_sessions.empty() && recording_sessions.back().is_recording) {
                        auto& current_session = recording_sessions.back();

                        // Create a single timestamped observation group for this capture
                        TimestampedObservations frame_observations;
                        frame_observations.timestamp = current_timestamp - current_session.start_time;

                        // Process all detected markers
                        for (size_t i = 0; i < ids.size(); i++) {
                            int marker_id = ids[i];

                            // Calculate confidence
                            double confidence = calculateConfidence(corners[i], 
                                cv::Size(image_zed.getWidth(), image_zed.getHeight()));
                            
                            // Add debug output for confidence threshold
                            if (current_timestamp - last_print_time > PRINT_INTERVAL_MS) {
                                std::cout << "Marker " << marker_id << " confidence: " << confidence 
                                          << " (threshold: " << MIN_CONFIDENCE_THRESHOLD << ")" << std::endl;
                            }

                            if (confidence < MIN_CONFIDENCE_THRESHOLD) {
                                if (current_timestamp - last_print_time > PRINT_INTERVAL_MS) {
                                    std::cout << "Marker " << marker_id << " rejected due to low confidence" << std::endl;
                                }
                                continue;
                            }

                            MarkerObservation observation;
                            observation.marker_id = marker_id;
                            observation.camera_to_marker = estimateMarkerPose(rvecs[i], tvecs[i]);
                            observation.confidence = confidence;
                            observation.timestamp = frame_observations.timestamp;

                            frame_observations.observations.push_back(observation);
                        }

                        // Only add frame if we have observations
                        if (!frame_observations.observations.empty()) {
                            current_session.timestamped_observations.push_back(frame_observations);
                            std::cout << "Recorded " << frame_observations.observations.size() 
                                      << " markers at timestamp: " << frame_observations.timestamp << " ms" << std::endl;
                        }
                    }
                }

                // Update last print time
                if (current_timestamp - last_print_time > PRINT_INTERVAL_MS) {
                    last_print_time = current_timestamp;
                }

                // Visualize markers
                cv::aruco::drawDetectedMarkers(image_ocv_rgb, corners, ids);
                for (size_t i = 0; i < ids.size(); i++) {
                    cv::aruco::drawAxis(image_ocv_rgb, camera_matrix, dist_coeffs, 
                                          rvecs[i], tvecs[i], actual_marker_size_meters * 0.5f);
                }
            }

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
            saveMarkerRelations("marker_relations.csv");
        }

        if (key == 'l' || key == 'L') {
            marker_relations = loadMarkerRelations("marker_relations.csv");
        }

        if (key == 'r' || key == 'R') {
            if (recording_sessions.empty() || !recording_sessions.back().is_recording) {
                startRecording();
                std::cout << "Recording started" << std::endl;
            } else {
                stopRecording();
                std::cout << "Recording stopped" << std::endl;
                // Automatically save both raw data and processed relations
                std::string timestamp = std::to_string(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()
                    ).count()
                );
                saveRawRecording(recording_sessions.back(), "recording_" + timestamp + ".csv");
                saveMarkerRelations("relations_" + timestamp + ".csv");
            }
        }

        if (!recording_sessions.empty() && recording_sessions.back().is_recording) {
            // Add a red circle indicator when recording
            cv::circle(image_ocv_rgb, cv::Point(image_ocv_rgb.cols - 30, 30), 10, cv::Scalar(0, 0, 255), -1);
            cv::putText(image_ocv_rgb, "REC", cv::Point(image_ocv_rgb.cols - 80, 35),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        }
    }
    zed.close();
    return 0;
}
