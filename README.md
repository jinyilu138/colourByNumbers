# colour by numbers project!

**why:** I started working on this because I was interested in image processing, and I also had a phase of playing colour by numbers games on my ipad. 

**what:** Given a picture, create a colour by numbers template from it! Can specify number of colour quantiles for the template to have.  

**future plans:** adding some type of graphics processing/effect to this! I'm super interested in graphics and low level driver code to create cool graphics, especially in games!

## build dependencies:
- SFML 2.6 or higher
- OpenCV 4.x 
- Mapbox polylabel 
- C++17 compatible compiler
- CMake 3.20 or higher

## to run:
1. clone repo in local env
2. install dependencies (I used vcpkg)
3. run from command line
   ```console
   ./colour <image_path> [number_of_colors]
   ```
