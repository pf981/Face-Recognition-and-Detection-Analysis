/* CSCI435 Computer Vision
 * Project: Facial Recognition
 * Paul Foster - 3648370
 */
#ifndef IMAGES_HPP
#define IMAGES_HPP

#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

// This function resizes the image such that its new area is approximately 480/600
void scaleImage(cv::Mat& mat);
void scaleGroupImage(cv::Mat& mat);

#endif
