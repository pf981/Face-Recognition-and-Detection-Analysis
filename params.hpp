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
        const double trainingToValidationRatio = 2/3.0f; // 66.66% of the images will be used for training
    }
    namespace eigenFace
    {
//        const int numComponents = 0;
        const int numComponents = 3;
        // const double threshold = 10.0;
        // FIXME: Below are the good params
//        const int numComponents = 0; // Setting this to 0 means the algorithm will automatically pick c-1 // FIXME: WHY IS THIS STILL CRASHING!!?? // Still crash with 8 and 5 and 4 and 3 and 2. Only 1 works. WHY??? I get "killed" message. HUH??!?? Even TRAINING data is failing!!
//        const double threshold = 50.0;
        const double threshold = DBL_MAX; // FIXME: There is a BIG BUG in your code. Your distances for eigen and fisher are WAY too big.
//        const double threshold = 100.0; // FIXME: This should NOT be DBL_MAX. I think that will mean it will ALWAYS classify a face (never return -1). Furthermore, reducing this WILL NOT reduce the training time / file size. This value needs to be tuned so that it correctly classifies faces but doesn't have false hits. Also must be able to say -1 for faces not in the training set.
    }
    namespace fisherFace
    {
//        const int numComponents = 0;
        const int numComponents = 0;
        const double threshold = DBL_MAX;
//        const double threshold = 10.0;
    }
    namespace lbphFace
    {
        const int radius = 1;
        const int neighbors = 8;
        const int gridX = 8;
        const int gridY = 8;
        const double threshold = DBL_MAX;
    }
    namespace lidFace
    {
        const int inradius = 1;
        const double threshold = DBL_MAX;
    }
    namespace cascadeClassifier
    {
        const double scaleFactor = 1.017;
        const int minNeighbors = 9;
        const int flags = 0; // This is a legacy parameter and it will not do anything with our application

        // Chose not to have a min/max size so that we could handle close/far faces.
        const cv::Size minSize;
        const cv::Size maxSize;
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
}

#endif
