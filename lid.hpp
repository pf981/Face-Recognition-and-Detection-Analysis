#ifndef LID_HPP
#define LID_HPP

#include "opencv2/core/core.hpp"

// inradius is the perpendicular distance from the centre of a square to the edge.
// It is used to determine the size of the region the LID descriptor describes
cv::Ptr<cv::FaceRecognizer> createLidFaceRecognizer(int inradius = 1, double threshold = DBL_MAX);

class Eigenfaces : public FaceRecognizer
{
private:
    int _num_components;
    double _threshold;
    std::vector<Mat> _projections;
    Mat _labels;
    Mat _eigenvectors;
    Mat _eigenvalues;
    Mat _mean;

public:
    using FaceRecognizer::save;
    using FaceRecognizer::load;

    // Initializes an empty Eigenfaces model.
    Eigenfaces(int num_components = 0, double threshold = DBL_MAX) :
        _num_components(num_components),
        _threshold(threshold) {}

    // Initializes and computes an Eigenfaces model with images in src and
    // corresponding labels in labels. num_components will be kept for
    // classification.
    Eigenfaces(InputArrayOfArrays src, InputArray labels,
               int num_components = 0, double threshold = DBL_MAX) :
        _num_components(num_components),
        _threshold(threshold) {
        train(src, labels);
    }

    // Computes an Eigenfaces model with images in src and corresponding labels
    // in labels.
    void train(InputArrayOfArrays src, InputArray labels);

    // Predicts the label of a query image in src.
    int predict(InputArray src) const;

    // Predicts the label and confidence for a given sample.
    void predict(InputArray _src, int &label, double &dist) const;

    // See FaceRecognizer::load.
    void load(const FileStorage& fs);

    // See FaceRecognizer::save.
    void save(FileStorage& fs) const;

    AlgorithmInfo* info() const;
};

#endif
