#include <SFML/Graphics.hpp>
#include <iostream>
#include <filesystem>
#include <opencv4/opencv2/opencv.hpp>

struct RegionInfo {
    std::vector<cv::Point> contour;
    int clusterLabel;
    cv::Point centroid;
    double area;
};

struct Point {
    double x, y;
};

class PoleOfInaccessibility {
public:
    PoleOfInaccessibility(double precision = 1.0) : precision(precision) {}

    Point findOptimalLabelPosition(const std::vector<cv::Point>& contour) {
        std::vector<std::vector<cv::Point>> polygon = { contour };
        cv::Rect bbox = cv::boundingRect(contour);
        double minX = bbox.x, minY = bbox.y;
        double maxX = bbox.x + bbox.width, maxY = bbox.y + bbox.height;

        double width = maxX - minX;
        double height = maxY - minY;
        double cellSize = std::min(width, height);
        double h = cellSize / 2;

        std::priority_queue<Cell> cellQueue;

        for (double x = minX; x < maxX; x += cellSize) {
            for (double y = minY; y < maxY; y += cellSize) {
                cellQueue.push(Cell(x + h, y + h, h, polygon));
            }
        }

        Cell bestCell = getCentroidCell(polygon);

        while (!cellQueue.empty()) {
            Cell cell = cellQueue.top();
            cellQueue.pop();

            if (cell.d > bestCell.d) {
                bestCell = cell;
            }

            if (cell.max - bestCell.d <= precision) continue;

            h = cell.h / 2;
            cellQueue.push(Cell(cell.x - h, cell.y - h, h, polygon));
            cellQueue.push(Cell(cell.x + h, cell.y - h, h, polygon));
            cellQueue.push(Cell(cell.x - h, cell.y + h, h, polygon));
            cellQueue.push(Cell(cell.x + h, cell.y + h, h, polygon));
        }

        return { bestCell.x, bestCell.y };
    }

private:
    double precision;

    struct Cell {
        double x, y, h, d, max;
        Cell(double x, double y, double h, const std::vector<std::vector<cv::Point>>& polygon)
            : x(x), y(y), h(h) {
            d = PoleOfInaccessibility::pointToPolygonDist(x, y, polygon); // Call static function
            max = d + h * std::sqrt(2);
        }

        bool operator<(const Cell& other) const {
            return max < other.max;
        }
    };

    static double pointToPolygonDist(double x, double y, const std::vector<std::vector<cv::Point>>& polygon) {
        bool inside = false;
        double minDistSq = std::numeric_limits<double>::max();

        for (const auto& ring : polygon) {
            for (size_t i = 0, j = ring.size() - 1; i < ring.size(); j = i++) {
                const cv::Point& a = ring[i];
                const cv::Point& b = ring[j];

                if ((a.y > y) != (b.y > y) &&
                    (x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x)) {
                    inside = !inside;
                }

                minDistSq = std::min(minDistSq, getSegDistSq(x, y, a, b));
            }
        }

        return (inside ? 1 : -1) * std::sqrt(minDistSq);
    }

    static double getSegDistSq(double px, double py, const cv::Point& a, const cv::Point& b) {
        double x = a.x, y = a.y;
        double dx = b.x - x, dy = b.y - y;

        if (dx != 0 || dy != 0) {
            double t = ((px - x) * dx + (py - y) * dy) / (dx * dx + dy * dy);
            if (t > 1) {
                x = b.x;
                y = b.y;
            } else if (t > 0) {
                x += dx * t;
                y += dy * t;
            }
        }

        dx = px - x;
        dy = py - y;
        return dx * dx + dy * dy;
    }

    Cell getCentroidCell(const std::vector<std::vector<cv::Point>>& polygon) {
        double area = 0, x = 0, y = 0;
        const auto& points = polygon[0];

        for (size_t i = 0, len = points.size(), j = len - 1; i < len; j = i++) {
            const cv::Point& a = points[i];
            const cv::Point& b = points[j];
            double f = a.x * b.y - b.x * a.y;
            x += (a.x + b.x) * f;
            y += (a.y + b.y) * f;
            area += f * 3;
        }

        if (area == 0) {
            return Cell(points[0].x, points[0].y, 0, polygon);
        }

        return Cell(x / area, y / area, 0, polygon);
    }
};

class imageProcess {
    public:
        imageProcess(const std::string &filename)
        {
            // TODO support for transparent bg pictures
            if (!texture.loadFromFile(filename)) 
            {
                std::cerr << "Error: Could not load texture from: " << filename << std::endl;
            }
            toOpenCV();
        }

        bool processImage (int clusters)
        {
            if (!groupColours(clusters)) return false;
            if (!detectEdges()) return false;
            if (!highlightContours()) return false;
            return true;
        }

        const cv::Mat& getProcessedImage() const { return imgWithBorders; }

    private:
        sf::Texture texture;
        sf::Image sfImage;
        sf::Vector2u size; 
        cv::Mat imgBGR, imgWithBorders, edges, centre, labels;
        std::vector<cv::Point> labelPositions; // Store label positions to check for overlaps
        std::vector<RegionInfo> regions;

        bool groupColours(int clusters)
        {   
            // TODO add error cases
            cv::Mat imgReshaped = imgBGR.reshape(1, imgBGR.rows * imgBGR.cols);
            imgReshaped.convertTo(imgReshaped, CV_32F);  // Convert to float for k-means
            cv::kmeans (imgReshaped, clusters, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 3.0), 3, cv::KMEANS_RANDOM_CENTERS, centre); 
            return reformQuantize();
        }

        bool reformQuantize()
        {
            // reform a photo to check quantization results
            centre.convertTo(centre, CV_8U); // cap 255
            cv::Mat imgQuantized(imgBGR.size(), imgBGR.type());
            for (int i = 0; i < imgBGR.rows; i++)
            {
                for (int j = 0; j < imgBGR.cols; j++)
                {
                    int cluster_idx = labels.at<int>(i * imgBGR.cols + j);
                    imgQuantized.at<cv::Vec3b>(i, j) = centre.row(cluster_idx);  
                }
            }

            // cv::imshow("quantized", imgQuantized);
            // cv::waitKey(0);
            return true;
        }

        bool detectEdges() 
        {
            cv::Mat imgLAB;
            cv::cvtColor(imgBGR, imgLAB, cv::COLOR_BGR2Lab);

            std::vector<cv::Mat> labChannels;
            cv::split(imgLAB, labChannels);

            // process separate chanels
            std::vector<cv::Mat> edgeChannels;
            for (const auto& channel : labChannels) {
                cv::Mat channelEdges;
                cv::Mat filtered;
                cv::bilateralFilter(channel, filtered, 9, 75, 75);
                
                cv::Canny(filtered, channelEdges, 30, 90);
                edgeChannels.push_back(channelEdges);
            }

            edges = cv::Mat::zeros(imgLAB.size(), CV_8UC1);
            for (const auto& channelEdges : edgeChannels) {
                edges |= channelEdges;
            }

            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, kernel);

            return true;
        }

        bool getContours()
        {
            // Find contours (all regions with color transitions)
            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

            std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) { return cv::contourArea(a) > cv::contourArea(b); });
            
            std::vector<std::pair<cv::Point, double>> storedRegions; // Store (centroid, area)
            double minDist = 3.0;  
            double areaThreshold = 0.1; 

            
            // Draw black borders around color regions
            for (const auto& contour : contours) {
                cv::drawContours(imgWithBorders, std::vector<std::vector<cv::Point>>{contour}, 0, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
                
                double area = cv::contourArea(contour);
                // Only process large enough regions
                // if (area < 200) continue;

                cv::Moments moments = cv::moments(contour, false);
                if (moments.m00 != 0) {
                    int centerX = static_cast<int>(moments.m10 / moments.m00);  // x-coordinate of the centroid
                    int centerY = static_cast<int>(moments.m01 / moments.m00);  // y-coordinate of the centroid
                    
                    // Check if this centroid is too close to an existing one
                    cv::Point centroid(centerX, centerY);

                    if (validContour(storedRegions, centroid, area, minDist, areaThreshold)) {

                        storedRegions.emplace_back(centroid, area); // Store (centroid, area)
                        
                        // Ensure centroid is within image bounds
                        if (centerX >= 0 && centerX < imgBGR.cols && centerY >= 0 && centerY < imgBGR.rows) {
                            int labelIndex = centerY * imgBGR.cols + centerX;

                            if (labelIndex >= 0 && labelIndex < labels.total()) {
                                int clusterLabel = labels.at<int>(labelIndex);

                                if (clusterLabel >= 0 && clusterLabel < centre.rows) {
                                    RegionInfo region;
                                    region.contour = contour;
                                    region.clusterLabel = clusterLabel;
                                    region.centroid = cv::Point(centerX, centerY);
                                    region.area = area;
                                    regions.push_back(region);
                                    
                                }
                            }
                        }
                    }
                }
            }
            return true;
        }

        bool validContour(const std::vector<std::pair<cv::Point, double>>& storedRegions, cv::Point centroid, double area, double minDist, double areaThreshold) 
        {
            for (const auto& stored : storedRegions) {
                double distance = cv::norm(centroid - stored.first);
                double areaDiff = std::abs(area - stored.second) / stored.second; // Relative area difference
                if (distance < minDist && areaDiff < areaThreshold) {
                    return false; // reject it, too similar
                }
            }
            return true; 
        }

        bool drawBorders() {
            imgWithBorders = cv::Mat(edges.size(), CV_8UC3, cv::Scalar(255, 255, 255));
            
            // Draw all borders
            for (const auto& region : regions) {
                cv::drawContours(imgWithBorders, std::vector<std::vector<cv::Point>>{region.contour}, 0, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
            }
            
            return true;
        }

        bool labelRegions() {

            PoleOfInaccessibility poleFinder;

            for (const auto& region : regions) 
            {
                Point labelPos = poleFinder.findOptimalLabelPosition(region.contour);
                std::string labelText = std::to_string(region.clusterLabel);
        
                int fontFace = cv::FONT_HERSHEY_PLAIN;
                double fontScale = 1;
                int thickness = 1;
                int baseline = 0;
                cv::Size textSize = cv::getTextSize(labelText, fontFace, fontScale, thickness, &baseline);

                // Ensure label is within image bounds
                int textX = std::max(0, std::min(imgWithBorders.cols - textSize.width, static_cast<int>(labelPos.x - textSize.width / 2)));
                int textY = std::max(textSize.height, std::min(imgWithBorders.rows, static_cast<int>(labelPos.y + textSize.height / 2)));

                // Draw the label
                cv::putText(imgWithBorders, labelText, cv::Point(textX, textY), fontFace, fontScale, cv::Scalar(0, 0, 0), thickness, cv::LINE_AA);
            }
            return true;
        }

        bool highlightContours() {
            // Ensure edges is a single-channel image
            if (edges.type() != CV_8UC1) {
                cv::cvtColor(edges, edges, cv::COLOR_BGR2GRAY);
            }

            // Extract all regions
            if (!getContours()) return false;

            // Draw borders
            if (!drawBorders()) return false;

            // Place labels
            if (!labelRegions()) return false;

            return true;
        }
        void toOpenCV()
        {
            sfImage = texture.copyToImage();
            size = sfImage.getSize();
            const sf::Uint8* pixels = sfImage.getPixelsPtr();
            cv::Mat img(cv::Size(size.x, size.y), CV_8UC4, (void*)pixels);
            cv::cvtColor(img, imgBGR, cv::COLOR_RGBA2BGR);
        }
};

class displayTemplate {
    public:
        displayTemplate(const cv::Mat& img) {
        toSFML(img);
    }

    void run() {
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

    private:
        sf::Texture processedTexture;
        sf::Sprite imageSprite;

        void toSFML(const cv::Mat& img) {
            cv::Mat imgRGBA;
            cv::cvtColor(img, imgRGBA, cv::COLOR_BGR2RGBA);
            sf::Image processedImage;
            processedImage.create(imgRGBA.cols, imgRGBA.rows, imgRGBA.ptr());
            processedTexture.loadFromImage(processedImage);
        }
};

int main()
{   
    imageProcess spongebob("../images/who-is-arguably-the-most-famous-cartoon-character-of-all-v0-1y8q3zvs2rzb1.jpg");
    spongebob.processImage(10);
    displayTemplate display(spongebob.getProcessedImage());
    display.run();
    return 0;
}