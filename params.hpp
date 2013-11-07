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
        // const int numComponents = 10;
        // const double threshold = 10.0;
        // FIXME: Below are the good params
        const int numComponents = 1; // Setting this to 0 means the algorithm will automatically pick c-1 // FIXME: WHY IS THIS STILL CRASHING!!?? // Still crash with 8 and 5 and 4 and 3 and 2. Only 1 works. WHY??? I get "killed" message. HUH??!?? Even TRAINING data is failing!!
//        const double threshold = DBL_MAX;
        const double threshold = 100.0; // FIXME: This should NOT be DBL_MAX. I think that will mean it will ALWAYS classify a face (never return -1). Furthermore, reducing this WILL NOT reduce the training time / file size. This value needs to be tuned so that it correctly classifies faces but doesn't have false hits. Also must be able to say -1 for faces not in the training set.
    }
    namespace fisherFace
    {
        const int numComponents = 3;
//        const double threshold = DBL_MAX;
        const double threshold = 10.0;
    }
    namespace lbphFace
    {
        const int radius = 1;
        const int neighbors = 8;
        const int gridX = 8;
        const int gridY = 8;
        const double threshold = DBL_MAX;
    }
    namespace cascadeClassifier
    {
//        const double scaleFactor = 1.1;
//        const double scaleFactor = 1.01;
        const double scaleFactor = 1.08;
//        const int minNeighbors = 3;
        const int minNeighbors = 7; // 7 is the smallest that removes the jacket mis-hit
//        const int minNeighbors = 1; // 7 is the smallest that removes the jacket mis-hit
        const int flags = 0;
        const cv::Size minSize(10,10); // FIXME: Don't need min size
//        const cv::Size maxSize(1, 1);
        const cv::Size maxSize(11, 11); // FIXME: This doesn't do anything :-(
    }
}

#endif
