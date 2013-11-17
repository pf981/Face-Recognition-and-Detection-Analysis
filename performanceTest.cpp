/* CSCI435 Computer Vision
 * Project: Facial Recognition
 * Paul Foster - 3648370
 */
#include <assert.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "concatenate.hpp"
#include "countImages.hpp"
#include "images.hpp"
#include "lid.hpp"
#include "performanceTest.hpp"
#include "params.hpp"
#include "train.hpp"


void loadPerformanceImagesAndLabels(std::vector<cv::Mat>& images, std::vector<int>& labels)
{
    for (char sample = 'A'; sample <= 'W'; ++sample)
    {
        std::cout << sample << ": " << countImages(sample) << std::endl;
        for (int photoNum = (int)(params::training::trainingToValidationRatio*countImages(sample)) + 1;
             ;
             ++photoNum)
        {
            // Filenames are in the form "face_samples/sample_A/A1.JPG"
            std::string filename =
                Concatenate("face_samples/sample_")(sample)("/")(sample)(photoNum)(".JPG").str;

            if (!fileExists(filename))
                break;

            std::cout << "Reading " << filename << std::endl;

            cv::Mat image(cv::imread(filename, 0));
            scaleImage(image);
            cv::Mat grayscale(image.clone());

            // Add the image and its label to the vectors
            images.push_back(grayscale);
            labels.push_back(sample - 'A');
        }
    }
}


// This function loads the trained XML files and uses them to classify the validation images. It
// will output which tests passed (classified correctly) and which did not
void performanceTest()
{
    // Load the models
    cv::Ptr<cv::FaceRecognizer> modelEigen = cv::createEigenFaceRecognizer(
        params::eigenFace::numComponents,
        params::eigenFace::threshold);
    std::cout << "Loading trained_eigen.xml..." << std::endl;
    modelEigen->load("trained_eigen.xml");

    cv::Ptr<cv::FaceRecognizer> modelFisher = cv::createFisherFaceRecognizer(
        params::fisherFace::numComponents,
        params::fisherFace::threshold);
    std::cout << "Loading trained_fisher.xml..." << std::endl;
    modelFisher->load("trained_fisher.xml");

    cv::Ptr<cv::FaceRecognizer> modelLbp = cv::createLBPHFaceRecognizer(
        params::lbphFace::radius,
        params::lbphFace::neighbors,
        params::lbphFace::gridX,
        params::lbphFace::gridY,
        params::lbphFace::threshold);
    std::cout << "Loading trained_lbp.xml..." << std::endl;
    modelLbp->load("trained_lbp.xml");

    cv::Ptr<cv::FaceRecognizer> modelLid = lid::createLidFaceRecognizer(
        params::lidFace::inradius,
        params::lidFace::threshold);
    std::cout << "Loading trained_lid.xml..." << std::endl;
    modelLid->load("trained_lid.xml");


    std::vector<cv::Mat> images;
    std::vector<int> labels;

    loadPerformanceImagesAndLabels(images, labels);

    int failsEigen = 0, failsFisher = 0, failsLbp = 0, failsLid = 0;

    int result = 0;
    double dist = 0.0f;

    assert(images.size() == labels.size());
    for (size_t i = 0; i < images.size(); ++i)
    {
        modelEigen->predict(images[i], result, dist);
        if (result != labels[i])
            ++failsEigen;

        // cerr chosen because it is not buffered
        std::cerr << labels[i] << "\t" << result << "(" << dist << ")";

        modelFisher->predict(images[i], result, dist);
        if (result != labels[i])
            ++failsFisher;

        std::cerr << "\t" << result << "(" << dist << ")";

        modelLbp->predict(images[i], result, dist);
        if (result != labels[i])
            ++failsLbp;

        std::cerr << "\t" << result << "(" << dist << ")";

        modelLid->predict(images[i], result, dist);
        if (result != labels[i])
            ++failsLid;

        std::cerr << "\t" << result << "(" << dist << ")" << std::endl;
    }

    // Add small number to denominator to prevent division by 0 and prevent integer division
    std::cout << "\nEigen failures: " << failsEigen*100/(images.size() + 0.000001) << "%" << std::endl;
    std::cout << "Fisher failures: " << failsFisher*100/(images.size() +  + 0.000001) << "%" << std::endl;
    std::cout << "LBP failures: " << failsLbp*100/(images.size()  + 0.000001) << "%" << std::endl;
    std::cout << "LID failures: " << failsLid*100/(images.size()  + 0.000001) << "%" << std::endl;
}
