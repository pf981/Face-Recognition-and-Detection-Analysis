#include <assert.h>
#include <iostream> // FIXME: Remove
#include <vector>

// FIXME: Only include what you need
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/objdetect/objdetect.hpp"
// #include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
// #include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
// #include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
// #include <opencv2/features2d/features2d.hpp>
// #include <opencv2/contrib/detection_based_tracker.hpp>



#include "lid.hpp"
#include "params.hpp"


void normalizeHistograms(std::vector<cv::Mat>& hists)
{
    cv::Mat normalizedMat;
    for (std::vector<cv::Mat>::const_iterator it = hists.begin(); it !=  hists.end(); ++it)
    {
        cv::normalize(*it, normalizedMat);
        normalizedMat.copyTo(*it);
    }
}


namespace lid
{
// Populates allKeyPoints and descriptors
void Lidfaces::detectKeypointsAndDescriptors(
    cv::InputArrayOfArrays src,
    std::vector<std::vector<cv::KeyPoint> >& allKeyPoints,// FIXME: Put into a single array
    cv::Mat& descriptors)
{
    // FIXME: TODO: Check that it is 8 bit grayscale
    cv::SIFT detector(
        params::sift::nfeatures,
        params::sift::nOctaveLayers,
        params::sift::contrastThreshold,
        params::sift::edgeThreshold,
        params::sift::sigma);

    std::vector<cv::Mat> images;
    src.getMatVector(images);// FIXME: This might crash

    for (unsigned int i = 0; i < images.size(); ++i)
    {
        // Determine the SIFT keypoints (but discard SIFT descriptors)
        std::vector<cv::KeyPoint> keyPointsForCurrentImage;
        detector(
            images[i], // Get keypoints in the current image
            cv::noArray(), // No mask
            keyPointsForCurrentImage,
            cv::noArray()); // We don't care about the SIFT descriptors (We will use LID descriptors)
        allKeyPoints.push_back(keyPointsForCurrentImage);

        // For each of the keypoints, calculate the LID descriptor
        cv::Mat singleImgDescriptors;
        for (size_t j = 0; j < keyPointsForCurrentImage.size(); ++j)
            descriptors.push_back(lid(images[i], keyPointsForCurrentImage[j].pt, mInradius));
    }

    // We only want to check asserts when we are debugging
    // If we aren't debugging then the loop is a waste of time
#ifndef NDEBUG // FIXME: Uncomment
    int totalNumberOfKeyPoints = 0;
    for (unsigned int i = 0; i < allKeyPoints.size(); ++i)
        totalNumberOfKeyPoints += allKeyPoints[i].size();
    assert(descriptors.rows == totalNumberOfKeyPoints);
#endif
    assert(descriptors.cols = 8*mInradius); // Ensure that each descriptors size is the number of neighbors
}


// void computeDescriptors(
//     const std::vector<std::vector<cv::KeyPoint> >& allKeyPoints, // FIXME: After this, we don't specifically care what the keypoints are. We just need to know how many keypoints are associated with each image. Ie allKeypoints[i].size()
//     cv::Mat& descriptors
//     )

cv::Ptr<cv::FaceRecognizer> createLidFaceRecognizer(int inradius, double threshold)
{
//  return makePtr<Lidfaces>(inradius, threshold); // FIXME: makePtr gives error for some reason
    return cv::Ptr<Lidfaces>(new Lidfaces(inradius, threshold)); // This is equivalent to makePtr
}

// Returns the LID descriptor of mat about p
// For an image I : Z^2 -> R
// lid(I, p) = [d(p1, p), ..., d(pn, p)]
// where d(pi, p) = I(pi) - I(p)
cv::Mat lid(const cv::Mat& src, cv::Point p, int inradius)
{
    assert(src.type() == CV_8UC1);
    assert(inradius >= 1);

    // For illustration, if p is the point and X are the neighbors of inradius=2 (N=16) and o are ignored points
    // XXXXX
    // XoooX
    // XopoX
    // XoooX
    // XXXXX
//    int totalNeighbors = (2*inradius + 1)*(2*inradius + 1) - 1; // FIXME: NO THIS IS WRONG. FIX EVERYTHING
    int totalNeighbors = 8*inradius; // This is the formula for the perimeter of a square given the inradius

    cv::Mat lidDescriptor(1, totalNeighbors, CV_8UC1);

    // Calculate the real bounds (making sure not to go off the end of the image)
    // These are the bounds of the square with appropriate inradius centred about p
    const int MIN_X = std::max(p.x - inradius, 0);
    const int MAX_X = std::min(p.x + inradius, src.cols);
    const int MIN_Y = std::max(p.y - inradius, 0);
    const int MAX_Y = std::min(p.y + inradius, src.cols);
    const unsigned char centerIntensity = src.at<unsigned char>(p.y, p.x);

    // neighborIndex is i where p_i is the ith neighbor
    // It goes from 0 to totalNeighbors-1
    int neighborIndex = 0;

    // For each pixel in the square perimeter (going clockwise from the top right)
    // Set the nth descriptor element
    // Top (left to rigth)
    for (int x = MIN_X; x < MAX_X; ++x)
    {
        lidDescriptor.at<unsigned char>(neighborIndex++) = std::max(
            src.at<unsigned char>(MIN_Y, x) - centerIntensity,
            0);
    }
    // Right (top to bottom)
    for (int y = MIN_Y; y < MAX_Y; ++y)
    {
        lidDescriptor.at<unsigned char>(neighborIndex++) = std::max(
            src.at<unsigned char>(y, MAX_X) - centerIntensity,
            0);
    }
    // Bottom (right to left)
    for (int x = MAX_X; x > MIN_X; --x)
    {
        // Set the nth descriptor element
        lidDescriptor.at<unsigned char>(neighborIndex++) = std::max(
            src.at<unsigned char>(MIN_Y, x) - src.at<unsigned char>(p.y, p.x),
            0);
    }
    // Left (bottom to top)
    for (int y = MAX_Y; y > MIN_Y; --y)
    {
        // Set the nth descriptor element
        lidDescriptor.at<unsigned char>(neighborIndex++) = std::max(
            src.at<unsigned char>(y, MIN_X) - src.at<unsigned char>(p.y, p.x),
            0);
    }



    // for (int x = MIN_X; x <= MAX_X; ++x) // FIXME:
    // {
    //     for (int y = MIN_Y; y <= MAX_Y; ++y)
    //     {
    //         if (x == p.x && y == p.y) // Don't calculate d(pi, p) when pi==p
    //             continue;

    //         // Set the nth descriptor element
    //         lidDescriptor.at<unsigned char>(neighborIndex++) = std::max(src.at<unsigned char>(y, x) - src.at<unsigned char>(p.y, p.x), 0);
    //     }
    // }

    // FIXME: Are you getting INTENSITIES???
    // FIXME: Are we meant to equalise intensities? (Note that I think I DONT equalised them elsewhere)
    // FIXME: Note that reading in as CV_LOAD_IMAGE_GRAYSCALE(0) loads image as an intensity image (What we want)
    return lidDescriptor;
}


// Computes an Lidfaces model with images in src and corresponding labels
// in labels.
 // FIXME:
void Lidfaces::train(cv::InputArrayOfArrays src, cv::InputArray labels)
{
    std::vector<std::vector<cv::KeyPoint> > allKeyPoints;
    cv::Mat descriptors;

    // Get SIFT keypoints and LID descriptors
    detectKeypointsAndDescriptors(src, allKeyPoints, descriptors);
    // FIXME: TODO
}

// Predicts the label of a query image in src.
// FIXME:
int Lidfaces::predict(cv::InputArray src) const
{
    return 0;
}

// Predicts the label and confidence for a given sample.
// FIXME
void Lidfaces::predict(cv::InputArray _src, int &label, double &dist) const
{
}

// see FaceRecognizer::load.
// FIXME:
void Lidfaces::load(const cv::FileStorage& fs)
{
}

// See FaceRecognizer::save.
// FIXME:
void Lidfaces::save(cv::FileStorage& fs) const
{
}

// FIXME:
cv::AlgorithmInfo* Lidfaces::info() const
{
    return NULL;
}

} // namespace lid
