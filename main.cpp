#include "imageProcess/imageProcess.hpp"
#include "displayTemplate/displayTemplate.hpp"
#include <iostream>
#include <filesystem>

void printUsage(const char* programName) {
    std::cout << "\nUsage: " << programName << " <image_path> [number_of_colors]\n"
              << "  image_path: path to the image file\n"
              << "  number_of_colors: (optional) number of colors to use (default: 10)\n" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        printUsage(argv[0]);
        return 0;
    }
    
    // Check for minimum required arguments
    if (argc < 2) {
        std::cerr << "Error: No image path provided\n\n";
        printUsage(argv[0]);
        return 1;
    }
    
    std::string imagePath = argv[1];
    
    int numColors = 10; // default value
    if (argc > 2) {
        try {
            numColors = std::stoi(argv[2]);
            if (numColors < 2 || numColors > 20) {
                std::cerr << "Error: Number of colors must be between 2 and 20\n";
                return 1;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid number of colors specified\n";
            return 1;
        }
    }
    
    try {
        imageProcess image(imagePath);
        if (!image.processImage(numColors)) {
            std::cerr << "Error: Failed to process image\n";
            return 1;
        }
        
        displayTemplate display(image.getProcessedImage());
        display.run();
    } catch (const std::exception& err) {
        std::cerr << "Error processing image: " << err.what() << std::endl;
        return 1;
    }
    return 0;
}