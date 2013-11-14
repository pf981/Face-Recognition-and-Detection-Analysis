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


// Generates the histograms for each label and puts them in the hists vector
void generateHistograms(std::vector<cv::Mat>& hists, const std::vector<cv::Mat>& separatedLabels, int clusterCount)
{
    // Histogram paramaters
    const int nimages = 1; // Only 1 image (The labels)
    const int channels[] = {0}; // Use the 0 index channel (none)
    const int dims = 1; // Only 1 channel
    const int histSize[] = {clusterCount}; // The number of bins is the number of clusters
    const float hranges[] = {0,clusterCount}; // Cluster group varies from 0 to the number of clusters
    const float* ranges[] = {hranges};

    // For each image, calculate its histogram
    for (unsigned int i = 0; i < separatedLabels.size(); ++i)
    {
        cv::calcHist(
            &separatedLabels[i],
            nimages,
            channels,
            cv::Mat(), // Do not use a mask
            hists[i],
            dims,
            histSize,
            ranges,
            true, // The histogram is uniform
            false); // Do not accumulate
    }
}

void normalizeHistograms(std::vector<cv::Mat>& hists)
{
    cv::Mat normalizedMat;
    for (std::vector<cv::Mat>::const_iterator it = hists.begin(); it !=  hists.end(); ++it)
    {
        cv::normalize(*it, normalizedMat);
        normalizedMat.copyTo(*it);
    }
}

size_t getSize(cv::InputArrayOfArrays src)
{
    std::vector<cv::Mat> images;
    src.getMatVector(images);
    return images.size();
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
    assert(descriptors.cols = 8*mInradius); // Ensure that each descriptors size is the number of neighbors // FIXME: 8*mInradius is NOT correct - you need to specify P (the number of columns) as a paramater of LID
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

    // FIXME: Don't know if you normalize descriptors...

    // kmeans function requires points to be CV_32F
    descriptors.convertTo(descriptors, CV_32FC1);

    // Do k-means clustering
    const int CLUSTER_COUNT = descriptors.rows; // FIXME: Don't know if we take a fraction of the descriptors
    cv::Mat histogramLabels;

    // This function populates histogram bin labels
    // The nth element of histogramLabels is an integer which represents the cluster that the
    // nth element of allKeyPoints is a member of.
    kmeans(
        descriptors, // The points we are clustering are the descriptors
        CLUSTER_COUNT, // The number of clusters (K)
        histogramLabels, // The label of the corresponding keypoint
        params::kmeans::termCriteria,
        params::kmeans::attempts,
        params::kmeans::flags);
//        mCenters); // FIXME: This is not using the right distance equation. This is using Euclidean distance when it should be using D defined in the paper. // FIXME: NOTE THAT YOU NEED TO CREATE mCENTERS FIRST!!! GIVE IT THE RIGHT DIMENSIONS!!!
// FIXME:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// FIXME: This is very important to change, however, it is still a distance measure, so it will probably still work okay

    // FIXME: I don't think I've done the following right. How big is the codebook meant to be? What happens to multiple images of the same person? How do they end up getting merged into one - or are they not meant to be merged at all?


    // Convert to single channel 32 bit float as the matrix needs to be in a form supported
    // by calcHist
    histogramLabels.convertTo(histogramLabels, CV_32FC1);

    // We end up with a histogram for each image
//    std::vector<cv::Mat> hists(imgs.size()); // FIXME: How do we get the size of src???
    const size_t NUM_IMAGES = getSize(src);
    std::vector<cv::Mat> hists(NUM_IMAGES); // FIXME: How do we get the size of src???
    // mCodebook.resize(NUM_IMAGES);

    // The histogramLabels vector contains ALL the points from EVERY image. We need to split
    // it up into groups of points for each image.
    // Because there are the same number of points in each image, and the points were put
    // into the labels vector in order, we can simply divide the labels vector evenly to get
    // the individual image's points.
    std::vector<cv::Mat> separatedLabels;
    for (unsigned int i = 0, startRow = 0; i < NUM_IMAGES; ++i)
    {
        separatedLabels.push_back(
            histogramLabels.rowRange(
                startRow,
                startRow + allKeyPoints[i].size()));
        startRow += allKeyPoints[i].size();
    }

    // Populate the hists vector
    generateHistograms(hists, separatedLabels, CLUSTER_COUNT); // FIXME: Uncomment

    // Make the magnitude of each histogram equal to 1
    normalizeHistograms(hists); // FIXME: Are we meant to normalise the histograms (Yes I think so - especially if you have a different number of keypoints for each image

    // FIXME: What exactly is the codebook? hists? I think so...
}

// Predicts the label of a query image in src by creating a histogram by clustering the sift
// descriptors using the centres we generated in training. The distances between the histogram and
// the histogram of every training image is calculated. The smallest average distance to a class of
// images is used to classify the image.
int Lidfaces::predict(cv::InputArray src) const
{
    std::vector<std::vector<cv::KeyPoint> > keyPoints;
    cv::Mat descriptors;
//    cv::Mat image;
    std::vector<cv::Mat> imageVector; // A vector containing just one image (this is so we can use the same detectKeypointsAndDescriptors function
//    src.getMat(image); // FIXME: This is convoluted
    imageVector.push_back(src.getMat());

    // Get SIFT keypoints and LID descriptors
//    detectKeypointsAndDescriptors(imageVector, keyPoints, descriptors);
    // FIXME: TODO BIGTIME!!!
    // Cluster the image using the training centres
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
