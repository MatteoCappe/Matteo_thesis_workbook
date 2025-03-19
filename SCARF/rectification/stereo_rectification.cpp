#include <yarp/os/all.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <event-driven/core.h>
#include <event-driven/algs.h>
#include <event-driven/vis.h>

using namespace cv;
using namespace yarp::os;
using std::string;

// check camera-calibration file to correct the parameters depending on the case, check also when is_rectifed is set to True

int main(int argc, char* argv[])
{
    cv::Size res = {640, 480}; //{640, 480}; depending on the disparity maps to use
    int fps = 240;
    double period = 1.0/fps;

    std::string file_path_left = "/home/cappe/Desktop/uni5/Tesi/IIT/EventPointDatasets/DSEC/zurich_city_10_b/leftdvs/leftdvs_15/leftdvs_15/data.log";
    std::string file_path_right = "/home/cappe/Desktop/uni5/Tesi/IIT/EventPointDatasets/DSEC/zurich_city_10_b/rightdvs/rightdvs_13/rightdvs_13/data.log";
    std::string timestamp_path = "/home/cappe/Desktop/uni5/Tesi/IIT/EventPointDatasets/DSEC/zurich_city_10_b/zurich_city_10_b_disparity_timestamps_normalized.txt";

    ev::offlineLoader<ev::AE> eloader_left;
    yInfo() << "Loading left log file ... ";    
    if(!eloader_left.load(file_path_left)) {
        yError() << "Could not open log file";
        return false;
    } else {
        yInfo() << eloader_left.getinfo();
    }

    ev::offlineLoader<ev::AE> eloader_right;
    yInfo() << "Loading right log file ... ";    
    if(!eloader_right.load(file_path_right)) {
        yError() << "Could not open log file";
        return false;
    } else {
        yInfo() << eloader_right.getinfo();
    }

    std::string timestamp_SCARF_left = std::string(std::getenv("HOME")) + "/Downloads/rectified_timestamp_SCARF_left_15/";
    if (!std::filesystem::exists(timestamp_SCARF_left)) {
        std::filesystem::create_directory(timestamp_SCARF_left);
    }

    std::string timestamp_SCARF_right = std::string(std::getenv("HOME")) + "/Downloads/rectified_timestamp_SCARF_right_13/";
    if (!std::filesystem::exists(timestamp_SCARF_right)) {
        std::filesystem::create_directory(timestamp_SCARF_right);
    }

    std::string path_left = std::string(std::getenv("HOME")) + "/Downloads/events_rectified_left.mp4";
    std::string path_right = std::string(std::getenv("HOME")) + "/Downloads/events_rectified_right.mp4";

    cv::VideoWriter dw_left;
    dw_left.open(path_left, cv::VideoWriter::fourcc('a','v','c','1'), fps, res, true);

    cv::VideoWriter dw_right;
    dw_right.open(path_right, cv::VideoWriter::fourcc('a','v','c','1'), fps, res, true);

    const string calibration_file = "/home/cappe/Desktop/uni5/Tesi/IIT/EventPointDatasets/DSEC/zurich_city_10_b/zurich_city_10_b_calibration/camera_calibration.txt";

    ev::vIPT ipt;
    ipt.configure(calibration_file, 1);

    double virtual_timer = period;

    // SCARF init
    ev::SCARF scarf_left, scarf_right;
    scarf_left.initialise(res, 14, 1.5, 0.3); // res, block_size, alpha, C # !!!
    scarf_right.initialise(res, 14, 1.5, 0.3); // res, block_size, alpha, C # !!!

    std::vector<double> normalized_timestamps;
    std::ifstream timestamp_file("/home/cappe/Desktop/uni5/Tesi/IIT/EventPointDatasets/DSEC/zurich_city_10_b/zurich_city_10_b_disparity_timestamps_normalized.txt");
    if (!timestamp_file.is_open()) {
        yError() << "Failed to open normalized timestamps file.";
    }
    double value;
    while (timestamp_file >> value) {
        normalized_timestamps.push_back(value);
    }
    timestamp_file.close();

    yInfo() << "Loaded normalized timestamps: " << normalized_timestamps;

    int i = 1170;
    const double save_interval = 0.1; // Save frames every 0.1 seconds
    double next_save_time = 117.0;
    bool first_frame_saved = true;
    const double epsilon = 1e-4; // needed as timestamps are not exactly equal

    while(eloader_left.incrementReadTill(virtual_timer)) {

        //yInfo() << "Left";
        
        cv::Mat img_left, img8U_left;

        //yInfo () << "save time: " << next_save_time;

        for(auto &v : eloader_left) {

            int x = v.x;
            int y = v.y;

            ipt.sparseForwardTransform(0, y, x); //d

            scarf_left.update(x, y, v.p); //d
        }

        yInfo() << "virtual timer: " << virtual_timer;

        scarf_left.getSurface().convertTo(img8U_left, CV_8U, 255);
        cv::cvtColor(img8U_left, img_left, cv::COLOR_GRAY2BGR);

        dw_left << img_left;

        if (!first_frame_saved) {
            std::string first_frame_name = timestamp_SCARF_left + "0000.png";
            if (!cv::imwrite(first_frame_name, img_left)) {
                yError() << "Failed to save first frame.";
            } else {
                yInfo() << "First frame saved as " << first_frame_name;
            }
            first_frame_saved = true;
            next_save_time += save_interval;
        }

        if (std::abs(next_save_time - virtual_timer) < epsilon && virtual_timer) {
            std::ostringstream oss;
            oss << timestamp_SCARF_left << std::setw(4) << std::setfill('0') << i << ".png";
            std::string file_name = oss.str();
            if (!cv::imwrite(file_name, img_left)) {
                yError() << "Failed to save frame " << i;
            } else {
                yInfo() << "Frame saved as " << file_name;
                i++;
            }
            next_save_time += save_interval;
        }

        virtual_timer += period;
        std::cout.flush();     
    }

    std::cout << std::endl;

    i = 1177;
    next_save_time = 117.7;
    first_frame_saved = true;
    virtual_timer = period;

    while(eloader_right.incrementReadTill(virtual_timer)) {
        
        cv::Mat img_right, img8U_right;

        for(auto &v : eloader_right) {
            int x = v.x;
            int y = v.y;
            ipt.sparseForwardTransform(0, y, x);

            scarf_right.update(x, y, v.p);
        }

        yInfo() << "virtual timer: " << virtual_timer;

        scarf_right.getSurface().convertTo(img8U_right, CV_8U, 255);
        cv::cvtColor(img8U_right, img_right, cv::COLOR_GRAY2BGR);

        dw_right << img_right;

        if (!first_frame_saved) {
            std::string first_frame_name = timestamp_SCARF_right + "0000.png";
            if (!cv::imwrite(first_frame_name, img_right)) {
                yError() << "Failed to save first frame.";
            } else {
                yInfo() << "First frame saved as " << first_frame_name;
            }
            first_frame_saved = true;
            next_save_time += save_interval;
        }

        if (std::abs(next_save_time - virtual_timer) < epsilon && virtual_timer) {
            std::ostringstream oss;
            oss << timestamp_SCARF_right << std::setw(4) << std::setfill('0') << i << ".png";
            std::string file_name = oss.str();
            if (!cv::imwrite(file_name, img_right)) {
                yError() << "Failed to save frame " << i;
            } else {
                yInfo() << "Frame saved as " << file_name;
                i++;
            }
            next_save_time += save_interval;
        }

        virtual_timer += period;
        std::cout.flush();
            
    }

    std::cout << std::endl;
    dw_left.release();
    dw_right.release();

    return 0;
}
