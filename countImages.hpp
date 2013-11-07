#ifndef COUNT_IMAGES_HPP
#define COUNT_IMAGES_HPP

#include <string>

// This function returns true if the file exists and false otherwise
bool fileExists(const std::string& filename);

// Returns the number of images in the folder "face_samples/sample_{sample}/"
int countImages(char sample);

#endif
