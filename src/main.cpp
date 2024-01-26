#include "vulkan_playground.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main() {
	VulkanPlayground vapp;
    try {
        vapp.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
	return 0;
}
