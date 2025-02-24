#pragma once
#include <SFML/Graphics.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <mapbox/polylabel.hpp>
#include "regionInfo.hpp"

class imageProcess 
{
    public:
        imageProcess(const std::string &filename);
        bool processImage (int clusters);
        const cv::Mat& getProcessedImage() const;

    private:
        sf::Texture texture;
        sf::Image sfImage;
        sf::Vector2u size; 
        cv::Mat imgBGR, imgWithBorders, edges, centre, labels;
        std::vector<cv::Point> labelPositions; // Store label positions to check for overlaps
        std::vector<regionInfo> regions;

        bool groupColours(int clusters);
        bool reformQuantize();
        bool detectEdges();
        bool getContours();
        bool validContour(const std::vector<std::pair<cv::Point, double>>& storedRegions, cv::Point centroid, double area, double minDist, double areaThreshold);
        bool drawBorders();
        bool labelRegions();
        bool highlightContours();
        void toOpenCV();
};