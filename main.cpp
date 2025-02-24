#include "imageProcess.hpp"
#include "displayTemplate.hpp"

int main()
{   
    imageProcess spongebob("../images/who-is-arguably-the-most-famous-cartoon-character-of-all-v0-1y8q3zvs2rzb1.jpg");
    spongebob.processImage(10);
    displayTemplate display(spongebob.getProcessedImage());
    display.run();
    return 0;
}