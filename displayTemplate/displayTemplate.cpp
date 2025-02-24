#include "displayTemplate.hpp"

displayTemplate::displayTemplate(const cv::Mat& img) 
{
    toSFML(img);
}

void displayTemplate::run() 
{
    sf::RenderWindow window(sf::VideoMode(1280, 960), "Colour with numbers template");
    sf::Vector2u windowSize = window.getSize();
    sf::Vector2u textureSize = processedTexture.getSize();

    float scale = std::min(static_cast<float>(windowSize.x) / textureSize.x,
                            static_cast<float>(windowSize.y) / textureSize.y);

    imageSprite.setTexture(processedTexture);
    imageSprite.setScale(scale, scale);
    imageSprite.setPosition((windowSize.x - textureSize.x * scale) / 2, (windowSize.y - textureSize.y * scale) / 2);

    while (window.isOpen()) 
    {
        sf::Event event;
        while (window.pollEvent(event)) 
        {
            if (event.type == sf::Event::Closed) 
            {
                window.close();
            }
        }
        window.clear();
        window.draw(imageSprite);
        window.display();
    }
}

void displayTemplate::toSFML(const cv::Mat& img) 
{
    cv::Mat imgRGBA;
    cv::cvtColor(img, imgRGBA, cv::COLOR_BGR2RGBA);
    sf::Image processedImage;
    processedImage.create(imgRGBA.cols, imgRGBA.rows, imgRGBA.ptr());
    processedTexture.loadFromImage(processedImage);
}
