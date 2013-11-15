/* CSCI435 Computer Vision
 * Project: Facial Recognition
 * Paul Foster - 3648370
 */
#ifndef DETECT_HPP
#define DETECT_HPP

#include <string>

// Counts the number of faces in the given image and displays the image with bounding boxes around
// the faces. Alternatively, if there is only one face, it will try to identify it using various
// facial recognition algorithms.
void detect(const std::string& imageFile);

#endif
