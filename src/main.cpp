#include "vulkan_playground.hpp"

#include <iostream>
#include <stdexcept>
#include <cstdlib>

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
