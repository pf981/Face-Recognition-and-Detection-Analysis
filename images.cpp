/* CSCI435 Computer Vision
 * Project: Facial Recognition
 * Paul Foster - 3648370
 */
#include <assert.h>
#include <iostream>
#include <cstdlib>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "images.hpp"


// This function resizes the image to exactly 128*128. It does this by first  uniformly scaling the
// image and then cropping it. The cropping is done by taking the 128*128 square centred at the
// centre of the image.
//
// The reason we must crop it is because we have a choice between scaling isotropically or cropping.
// We chose cropping over isotropic scaling because some of the face detectors are not invariant
// to isotropic scaling. This has a disadvantage that images with a significantly different ratio
// to 480*600 will experience significant cropping.
//
// The reason we need the mat to be EXACTLY 128*128 (and not approximately) is because the face
// detectors require training/test images to be of equal size.
void scaleImage(cv::Mat& mat)
{
    const int ROWS = 128;
    const int COLS = 128;

    cv::Mat dest = mat.clone();
    float scale = std::max(float(ROWS)/mat.rows, float(COLS)/mat.cols);

    // Scale the image uniformly
    cv::resize(mat, dest, cv::Size(), scale, scale);

    // Crop the image
    // To crop the image, we take the region of interest of the appropriate size
    // centred about the centre of the original image
    // mat = dest(cv::Rect(0, 0, COLS, ROWS));
    mat = dest(cv::Rect(
        (dest.cols - COLS)/2,
        (dest.rows - ROWS)/2,
        COLS,
        ROWS));

    // Note that even though mat is a shallow copy of dest, the data won't be destroyed when
    // dest goes out of scope because OpenCV does reference counting for Mats
}


// Scales the image to approximately 480*600.
// Note that, for group images, we do not need the dimensions to be exact because the haar cascade
// implementation does not require it.
void scaleGroupImage(cv::Mat& mat)
{
    cv::Mat dest = mat.clone();
    float scale = sqrt((480.0f * 600.0f) / (mat.rows * mat.cols));

    // Don't make the image bigger
    if (scale > 1)
        return;
    cv::resize(mat, dest, cv::Size(), scale, scale);

    // Note that even though mat is a shallow copy of dest, the data won't be destroyed when
    // dest goes out of scope because OpenCV does reference counting for Mats
    mat = dest;
}
