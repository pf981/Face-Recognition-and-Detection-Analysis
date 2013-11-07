/* CSCI435 Computer Vision
 * Project: Facial Recognition
 * Paul Foster - 3648370
 */
#ifndef IMAGES_HPP
#define IMAGES_HPP

#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

// This function assumes that all images are the same size
// It takes a vector of identically sized images and combines them horizontally into a single
// image that then gets displayed in a window
cv::Mat mergeImages(
    const cv::Mat& quadrant1,
    const cv::Mat& quadrant2,
    const cv::Mat& quadrant3,
    const cv::Mat& quadrant4);


// Reads in all images specified by the command line arguments. It then scales them
// such that their area is approximately 480*800.
void readAndRescaleImages(std::vector<cv::Mat>& imgs, int argc, char** argv);

// This function resizes the image such that its new area is approximately 480/600
void scaleImage(cv::Mat& mat);
void scaleGroupImage(cv::Mat& mat);

#endif
