

// FIXME: Only include what you need
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
// #include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
// #include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
// #include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
// #include <opencv2/features2d/features2d.hpp>
// #include <opencv2/contrib/detection_based_tracker.hpp>



#include "lid.hpp"


namespace lid
{

cv::Ptr<cv::FaceRecognizer> createLidFaceRecognizer(int inradius, double threshold)
{
//  return makePtr<Lidfaces>(inradius, threshold); // FIXME: makePtr gives error for some reason
    return cv::Ptr<Lidfaces>(new Lidfaces(inradius, threshold)); // This is equivalent to makePtr
}

// FIXME: TODO
cv::Mat lid(cv::Point p)
{
    cv::Mat toReturn;
    return toReturn;
}


// Computes an Lidfaces model with images in src and corresponding labels
// in labels.
 // FIXME:
void Lidfaces::train(cv::InputArrayOfArrays src, cv::InputArray labels)
{
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
