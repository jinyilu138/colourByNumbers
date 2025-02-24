#pragma once
#include <opencv4/opencv2/opencv.hpp>
#include <vector>

struct regionInfo {
    std::vector<cv::Point> contour;
    int clusterLabel;
    cv::Point centroid;
    double area;
};