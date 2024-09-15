#include <SFML/Graphics.hpp>
#include <iostream>
#include <filesystem>
#include <opencv4/opencv2/opencv.hpp>



bool isCloseToBlack(const cv::Vec3b& color, int threshold = 50) {
    return (color[0] < threshold && color[1] < threshold && color[2] < threshold);
}

int main()
{
    sf::Texture texture;
    //TODO support for transparent bg pictures
    if (!texture.loadFromFile("../images/who-is-arguably-the-most-famous-cartoon-character-of-all-v0-1y8q3zvs2rzb1.jpg")) 
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
    int k = 16; //for now
    cv::kmeans (imgReshaped, k, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 3.0), 3, cv::KMEANS_RANDOM_CENTERS, centre); 
    
    // print the contents of centers
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

    // cv::imshow("quantized", imgQuantized);
    
    
    // Convert the image from BGR to LAB color space (perceptual color differences)
    cv::Mat imgLAB;
    cv::cvtColor(imgQuantized, imgLAB, cv::COLOR_BGR2Lab);

    // Calculate the color difference between neighboring pixels
    cv::Mat diffX, diffY;
    cv::Sobel(imgLAB, diffX, CV_32F, 1, 0, 3);  // Horizontal color differences
    cv::Sobel(imgLAB, diffY, CV_32F, 0, 1, 3);  // Vertical color differences

    // Calculate the magnitude of the color gradient (difference)
    cv::Mat magnitude;
    cv::magnitude(diffX, diffY, magnitude);

    // Normalize the gradient to display
    cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Threshold to create a binary image of color transitions (edges)
    cv::Mat edges;
    cv::threshold(magnitude, edges, 50, 255, cv::THRESH_BINARY); // Adjust the threshold value as needed
    cv::Mat imgBW(img.size(), CV_8UC1, cv::Scalar(255));  // Initialize with white

    // Ensure edges is a single-channel image
    if (edges.type() != CV_8UC1) {
        std::cout << "Image is not single-channel, converting..." << std::endl;
        cv::cvtColor(edges, edges, cv::COLOR_BGR2GRAY); // Ensure it's a single channel
    }

    // Draw black borders around color regions
    cv::Mat imgWithBorders = cv::Mat::zeros(edges.size(), CV_8UC1);  // Initialize with black
    imgWithBorders.setTo(cv::Scalar(255), ~edges);  // Set non-edges to white

    // Find contours (all regions with color transitions)
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);  // edges is CV_8UC1

    // Draw black borders around color regions
    for (const auto& contour : contours) {
        cv::drawContours(imgWithBorders, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(0, 0, 0), 2);  
        
        cv::Moments moments = cv::moments(contour, false);
        if (moments.m00 != 0) {
            int centerX = static_cast<int>(moments.m10 / moments.m00);  // x-coordinate of the centroid
            int centerY = static_cast<int>(moments.m01 / moments.m00);  // y-coordinate of the centroid

            // Find the corresponding label for the pixel at (centerX, centerY)
            int labelIndex = centerY * img.cols + centerX;
            int clusterLabel = labels.at<int>(labelIndex);  // Retrieve the cluster label
            cv::Vec3b clusterColor = centre.row(clusterLabel).at<cv::Vec3b>(0);

            if (isCloseToBlack(clusterColor)) 
            {
                // std::cout << clusterLabel << std::endl;
                // Fill the region with black
                cv::drawContours(imgWithBorders, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(0, 0, 0), cv::FILLED);
                // // Draw border around regions
                // cv::drawContours(imgWithBorders, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(0, 0, 0), 2);  // Black border
                // // Draw the number inside the closed area
                // cv::putText(imgWithBorders, std::to_string(clusterLabel), 
                //             cv::Point(centerX, centerY), cv::FONT_HERSHEY_COMPLEX_SMALL, 
                //             1, cv::Scalar(0, 0, 0), 1);  // Smaller text size and thinner text
            } 
            else 
            {
                // Draw the number inside the closed area
                cv::putText(imgWithBorders, std::to_string(clusterLabel), 
                            cv::Point(centerX, centerY), cv::FONT_HERSHEY_COMPLEX_SMALL, 
                            1, cv::Scalar(0, 0, 0), 1);  // Smaller text size and thinner text
            }
        }
    }


    // Convert back to SFML format for display
    cv::Mat imgForDisplay;
    sf::Image processedImage;
    cv::cvtColor(imgWithBorders, imgForDisplay, cv::COLOR_BGR2RGBA);
    processedImage.create(imgForDisplay.cols, imgForDisplay.rows, imgForDisplay.ptr());

    sf::Texture processedTexture;
    sf::Sprite imageSprite;
    sf::RenderWindow window(sf::VideoMode(1280, 960), "Jinyi Rocks!", sf::Style::Default);
    sf::Vector2u windowSize = window.getSize(); 
    processedTexture.loadFromImage(processedImage);
    imageSprite.setTexture(processedTexture);
    sf::Vector2u textureSize = processedTexture.getSize();

    float scaleX = static_cast<float>(windowSize.x) / textureSize.x;
    float scaleY = static_cast<float>(windowSize.y) / textureSize.y;
    float scale = std::min(scaleX, scaleY);  // Use the smaller scale to keep the aspect ratio


    sf::Vector2f spritePos;
    spritePos.x = (windowSize.x - (textureSize.x * scale)) / 2;
    spritePos.y = (windowSize.y - (textureSize.y * scale)) / 2;

    imageSprite.scale(sf::Vector2f(scale, scale));
    imageSprite.setPosition(sf::Vector2f(spritePos.x,spritePos.y));
    while (window.isOpen())
    {
        sf::Event event;

        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        
        window.clear();
        // window.clear(sf::Color(avgColor[2], avgColor[1], avgColor[0]));
        window.draw(imageSprite);
        window.display();
    }
}