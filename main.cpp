#include <SFML/Graphics.hpp>
#include <iostream>
#include <filesystem>

int main()
{
    sf::RenderWindow window(sf::VideoMode(640, 480), "Jinyi Rocks!", sf::Style::Default);
    sf::Texture texture;

    if (!texture.loadFromFile("images/sfml-logo-small.png")) {
        std::cout << "Could not load texture" << std::endl;
        std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

        return 0;
    }

    while (window.isOpen())
    {
        sf::Event event;

        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        sf::Sprite image;
        image.setTexture(texture);
        image.setPosition(sf::Vector2f(100,100));
        image.scale(sf::Vector2f(1,1.5));
        window.draw(image);
        window.display();
    }
}