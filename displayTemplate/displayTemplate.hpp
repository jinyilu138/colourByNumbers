#pragma once
#include <SFML/Graphics.hpp>
#include <opencv4/opencv2/opencv.hpp>

class displayTemplate 
{
    public:
        displayTemplate(const cv::Mat &img);
        void run();

    private:
        sf::Texture processedTexture;
        sf::Sprite imageSprite;

        void toSFML(const cv::Mat& img);
};