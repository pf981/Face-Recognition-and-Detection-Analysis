

// FIXME: Only include what you need
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "lid.hpp"



cv::Ptr<cv::FaceRecognizer> createLidFaceRecognizer(int inradius, double threshold)
{
    return cv::makePtr<Lidfaces>(inradius, threshold);
}

cv::Mat lid(cv::Point p)
{
    cv::Mat toReturn;
    return toReturn;
}
