/* CSCI435 Computer Vision
 * Project: Facial Recognition
 * Paul Foster - 3648370
 */
#include <assert.h>
#include <iostream>
#include <stdlib.h> // Needed for exit

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp" // Needed for resize
#include "opencv2/highgui/highgui.hpp" // Needed for imshow

#include "images.hpp"

// This function assumes that all images are the same size
// It takes a 4 identically sized images and combines them into a single
// image with 4 quadrants.
cv::Mat mergeImages( // FIXME: Unused
    const cv::Mat& quadrant1,
    const cv::Mat& quadrant2,
    const cv::Mat& quadrant3,
    const cv::Mat& quadrant4)
{
    assert(quadrant2.size() == quadrant1.size());
    assert(quadrant3.size() == quadrant1.size());
    assert(quadrant4.size() == quadrant1.size());

    assert(quadrant1.type() == CV_8UC3);
    assert(quadrant2.type() == CV_8UC3);
    assert(quadrant3.type() == CV_8UC3);
    assert(quadrant4.type() == CV_8UC3);

    const unsigned int h = quadrant1.size().height;
    const unsigned int w = quadrant1.size().width;

    // Create the image which holds all the images
    cv::Mat composite(h * 2, w * 2, CV_8UC3);

    quadrant1.copyTo(composite(cv::Rect(w * 1, h * 0, w, h)));
    quadrant2.copyTo(composite(cv::Rect(w * 0, h * 0, w, h)));
    quadrant3.copyTo(composite(cv::Rect(w * 0, h * 1, w, h)));
    quadrant4.copyTo(composite(cv::Rect(w * 1, h * 1, w, h)));

    return composite;
}


// Reads in all images specified by the command line arguments. It then scales them
// such that their area is approximately 480*800.
void readAndRescaleImages(std::vector<cv::Mat>& imgs, int argc, char** argv)
{
    for (int i = 1; i < argc; ++i)
    {
        // Attempt to load the image
        cv::Mat curImage(cv::imread(argv[i], CV_LOAD_IMAGE_GRAYSCALE));
        if (!curImage.data)
        {
            std::cerr << "Could not open image " << argv[i] << "\nExiting.\n";
            return exit(1);
        }

        // Rescale and put into vector
        scaleImage(curImage);
        imgs.push_back(curImage);
    }
}


// This function resizes the image to exactly COLUMNS*ROWS. It does this by first
// uniformly scaling the image and then cropping it.
//
// The reason we must crop it is because we have a choice between scaling isotropically or cropping.
// We chose cropping over isotropic scaling because some of the face detectors are not invariant
// to isotropic scaling. This has a disadvantage that images with a significantly different ratio
// to 480*600 will experience significant cropping.
//
// The reason we need the mat to be EXACTLY 480*600 (and not approximately) is because the face
// detectors require training/test images to be of equal size.
void scaleImage(cv::Mat& mat)
{
//    const int ROWS = 600;
//    const int COLS = 480;
    const int ROWS = 128; // FIXME: We need to CENTRE the image - not just chop off the ends
  const int COLS = 128;// FIXME: Test
   // const int ROWS = 60;
   // const int COLS = 48;// FIXME: Test

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

// FIXME:
// This has different dimensions to the other scaling as the group image has a significantly
// different row-to-column ratio
void scaleGroupImage(cv::Mat& mat) // FIXME: Does inlining in cpp even work?
{
    const int ROWS = 480;
    const int COLS = 600;

    cv::Mat dest = mat.clone();
    float scale = std::max(float(ROWS)/mat.rows, float(COLS)/mat.cols);

    // Scale the image uniformly
    cv::resize(mat, dest, cv::Size(), scale, scale);

    // Crop the image
    mat = dest(cv::Rect(0, 0, COLS, ROWS));

    // Note that even though mat is a shallow copy of dest, the data won't be destroyed when
    // dest goes out of scope because OpenCV does reference counting for Mats
}


// inline void scaleImage(cv::Mat& mat)
// {
//     cv::Mat dest = mat.clone();
//     float scale = sqrt((480.0f * 600.0f) / (mat.rows * mat.cols));

//     // Don't make the image bigger
//     if (scale > 1)
//         return;
//     cv::resize(mat, dest, cv::Size(), scale, scale);

//     // Note that even though mat is a shallow copy of dest, the data won't be destroyed when
//     // dest goes out of scope because OpenCV does reference counting for Mats
//     mat = dest;
// }
