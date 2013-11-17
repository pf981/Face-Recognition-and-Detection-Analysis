/* CSCI435 Computer Vision
 * Project: Facial Recognition
 * Paul Foster - 3648370
 */
#ifndef PARAMS_HPP
#define PARAMS_HPP


// This is where all the paramaters that I can tweak are located.
// They are put in one place so it is easier to play with them.
//
// Note that variables declared as const and not explicitly delared extern have internal linkage
// meaning we won't get duplicate symbols when linking translation units.
namespace params
{
    namespace training
    {
        // The faces must be broken up into training images and images used to measure performance
        const double trainingToValidationRatio = 0.7f;
    }
    namespace eigenFace
    {
        const int numComponents = 80;
        const double threshold = 4710.0f;
    }
    namespace fisherFace
    {
        const int numComponents = 0; // OpenCV will set numComponents to {classes}-1
        const double threshold = DBL_MAX;
    }
    namespace lbphFace
    {
        const int radius = 1;
        const int neighbors = 8;
        const int gridX = 7;
        const int gridY = 7;
        const double threshold = 52.0f;
    }
    namespace lidFace
    {
        const int inradius = 1;
        const double threshold = DBL_MAX;
        const double clustersAsPercentageOfKeypoints = 0.9f;
    }
    namespace cascadeClassifier
    {
        const double scaleFactor = 1.01;
        const int minNeighbors = 9;
        const int flags = 0; // This is a legacy parameter and it will not do anything with our application

        // Chose not to have a min/max size so that we could handle close/far faces.
        const cv::Size minSize;
        const cv::Size maxSize; // Note: In OpenCV 2.4.2, this parameter does not work for haar
    }
    namespace sift
    {
        // There is a bug in 2.4.2 where this parameter does not work, so I have to leave it at 0
        const int nfeatures = 0; // Default 0
        const int nOctaveLayers = 5; // Default 3
        const double contrastThreshold = 0.06; // Default 0.04
        const double edgeThreshold = 10; // Default 10
        const double sigma = 3; // Default 1.6
    }
    namespace kmeans
    {
        const cv::TermCriteria termCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 1000, 0.01);
        const int attempts = 5;
        const int flags = cv::KMEANS_PP_CENTERS;
        // K (the number of clusters) is not specified here as it is determined by the number of
        // keypoints (and not a compile-time constant)
    }
}

#endif
