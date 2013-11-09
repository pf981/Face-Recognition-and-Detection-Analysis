#ifndef LID_HPP
#define LID_HPP

#include <cstdlib> // FIXME: Just for null

#include "opencv2/core/core.hpp"

namespace lid
{

cv::Mat lid(const cv::Mat& src, cv::Point p, int inradius);

// inradius is the perpendicular distance from the centre of a square to the edge.
// It is used to determine the size of the region the LID descriptor describes
cv::Ptr<cv::FaceRecognizer> createLidFaceRecognizer(int inradius = 1, double threshold = DBL_MAX);

class Lidfaces : public cv::FaceRecognizer
{
private:
    int mInradius;
    double mThreshold;
    // std::vector<Mat> _projections;
    // Mat _labels;
    // Mat _eigenvectors;
    // Mat _eigenvalues;
    // Mat _mean;

public:
    using cv::FaceRecognizer::save;
    using cv::FaceRecognizer::load;

    // Initializes an empty Lidfaces model.
    Lidfaces(int inradius = 1, double threshold = DBL_MAX) :
        mInradius(inradius),
        mThreshold(threshold)
    {
    }

    // Initializes and computes an Lidfaces model with images in src and
    // corresponding labels in labels. num_components will be kept for
    // classification.
    Lidfaces(cv::InputArrayOfArrays src, cv::InputArray labels,
               int inradius = 1, double threshold = DBL_MAX) :
        mInradius(inradius),
        mThreshold(threshold)
    {
        train(src, labels);
    }

    // Computes an Lidfaces model with images in src and corresponding labels
    // in labels.
    void train(cv::InputArrayOfArrays src, cv::InputArray labels);

    // Predicts the label of a query image in src.
    int predict(cv::InputArray src) const;

    // Predicts the label and confidence for a given sample.
    void predict(cv::InputArray _src, int &label, double &dist) const;

    // see FaceRecognizer::load.
    void load(const cv::FileStorage& fs);

    // See FaceRecognizer::save.
    void save(cv::FileStorage& fs) const;

    cv::AlgorithmInfo* info() const;
};

} // namespace lid

#endif
