#ifndef LID_HPP
#define LID_HPP

#include "opencv2/core/core.hpp"

// inradius is the perpendicular distance from the centre of a square to the edge.
// It is used to determine the size of the region the LID descriptor describes
cv::Ptr<cv::FaceRecognizer> createLidFaceRecognizer(int inradius = 1, double threshold = DBL_MAX);

#endif
