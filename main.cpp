#include <SFML/Graphics.hpp>
#include <iostream>
#include <filesystem>
#include <opencv4/opencv2/opencv.hpp>

int main()
{
    sf::Texture texture;

    if (!texture.loadFromFile("../images/penguin-cartoon.png")) 
    {
        std::cout << "Could not load texture" << std::endl;
        std::cout << "Current working directory: " << std::__1::__fs::filesystem::current_path() << std::endl;
        return 0;
    }

    // sfml to opencv
    sf::Image sfImage = texture.copyToImage();
    sf::Vector2u size = sfImage.getSize();
    const sf::Uint8* pixels = sfImage.getPixelsPtr();
    cv::Mat img (cv::Size(size.x, size.y), CV_8UC4, (void*)pixels);
    cv::Mat imgBGR;
    cv::cvtColor(img, imgBGR, cv::COLOR_RGBA2BGR);

    // group colours
    cv::Mat imgReshaped = imgBGR.reshape(1, imgBGR.rows * imgBGR.cols);
    imgReshaped.convertTo(imgReshaped, CV_32F);  // Convert to float for k-means
    cv::Mat labels, centre;
    int k = 4; //for now
    cv::kmeans (imgReshaped, k, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 3.0), 3, cv::KMEANS_RANDOM_CENTERS, centre); 
    
    // Print the contents of centers
    std::cout << "Cluster centers:" << std::endl;
    for (int i = 0; i < centre.rows; ++i) {
        cv::Vec3b color = centre.at<cv::Vec3b>(i, 0);  // Access each row as Vec3b
        std::cout << "Cluster " << i << ": B=" << static_cast<int>(color[0])
                  << " G=" << static_cast<int>(color[1])
                  << " R=" << static_cast<int>(color[2]) << std::endl;
    }

    // reform a photo to check quantization results
    centre.convertTo(centre, CV_8U); // cap 255
    cv::Mat imgQuantized(imgBGR.size(), imgBGR.type());
    for (int i = 0; i < imgBGR.rows; i++)
    {
        for (int j = 0; j < imgBGR.cols; j++)
        {
        int cluster_idx = labels.at<int>(i * img.cols + j);
        imgQuantized.at<cv::Vec3b>(i, j) = centre.row(cluster_idx);  
        }
    }

    // Convert back to SFML format for display
    cv::Mat imgForDisplay;
    sf::Image processedImage;
    cv::cvtColor(imgQuantized, imgForDisplay, cv::COLOR_BGR2RGBA);
    processedImage.create(imgForDisplay.cols, imgForDisplay.rows, imgForDisplay.ptr());

    sf::Texture processedTexture;
    processedTexture.loadFromImage(processedImage);
    sf::Sprite imageSprite;
    imageSprite.setTexture(processedTexture);
    imageSprite.setPosition(sf::Vector2f(100,100));
    imageSprite.scale(sf::Vector2f(1,1.5));

    sf::RenderWindow window(sf::VideoMode(640, 480), "Jinyi Rocks!", sf::Style::Default);
    while (window.isOpen())
    {
        sf::Event event;

        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        window.draw(imageSprite);
        window.display();
    }
}