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
//    for (char sample = 'A'; sample <= 'W'; ++sample) // FIXME: USE THIS
//    for (char sample = 'A'; sample <= 'F'; ++sample) // FIXME: Use other
    for (char sample = 'A'; sample <= 'B'; ++sample) // FIXME: Use other
    {
        // FIXME: Of the images, we need to RANDOMLY select 60% of them for training
        std::cerr << sample << ": " << countImages(sample) << std::endl; // FIXME: Remove
        // for (int photoNum = 1;
        //      photoNum <= (int)(params::training::trainingToValidationRatio*countImages(sample));
        //      ++photoNum) // FIXME: This is using the training data
        for (int photoNum = 1;
             ;
             ++photoNum) // FIXME: This is using the ALL data
        // for (int photoNum = (int)(params::training::trainingToValidationRatio*countImages(sample)) + 1;
        //      ;
        //      ++photoNum) // FIXME: Use this one
        {
            // Filenames are in the form "face_samples/sample_A/A1.JPG"
            std::string filename =
                Concatenate("face_samples/sample_")(sample)("/")(sample)(photoNum)(".JPG").str;

            if (!fileExists(filename))
                break;

            std::cout << "Reading " << filename << std::endl;

            // Read the image in. Fisher only works on greyscale
//            cv::Mat image(cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE));




            // cv::Mat image(cv::imread(filename));
            // scaleImage(image);
            // cv::Mat grayscale;
            // cv::cvtColor(image, grayscale, CV_BGR2GRAY);
            // equalizeHist(grayscale, grayscale); // FIXME: Why do this?

//            cv::Mat image(cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE));







            // FIXME: THIS WAS THE CONFIGURATION THAT DIDN'T WORK
            cv::Mat image(cv::imread(filename, 0));// FIXME: Attempt to fix
            scaleImage(image);
            cv::Mat grayscale(image.clone());
            // FIXME: This is an attempt to fix eigen and fisher. I think you train with color, test with grayscale
            // cv::Mat image(cv::imread(filename));
            // scaleImage(image);
            // cv::Mat grayscale(image.clone());
            // cv::cvtColor(image, grayscale, CV_BGR2GRAY);
            // equalizeHist(grayscale, grayscale);





            // cv::cvtColor(image, grayscale, CV_BGR2GRAY);
            // equalizeHist(grayscale, grayscale); // FIXME: Why do this?






            // Scale the images as they so they are faster to process. All images need to have the same
            // dimensions as it is required for the recognisers
//            scaleImage(image);
//            equalizeHist(image, image); // FIXME: added this

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
    cv::Ptr<cv::FaceRecognizer> modelEigen = cv::createEigenFaceRecognizer( // FIXME: This should be its own function
        params::eigenFace::numComponents,
        params::eigenFace::threshold);
    std::cout << "Loading trained_eigen.xml..." << std::endl;
    modelEigen->load("trained_eigen.xml"); // FIXME: How the heck do I error-check loading??

    cv::Ptr<cv::FaceRecognizer> modelFisher = cv::createFisherFaceRecognizer(
        params::fisherFace::numComponents,
        params::fisherFace::threshold);
    std::cout << "Loading trained_fisher.xml..." << std::endl;
    modelFisher->load("trained_fisher.xml"); // FIXME: How the heck do I error-check loading??

    cv::Ptr<cv::FaceRecognizer> modelLbp = cv::createLBPHFaceRecognizer( // FIXME: This should be its own function
        params::lbphFace::radius,
        params::lbphFace::neighbors,
        params::lbphFace::gridX,
        params::lbphFace::gridY,
        params::lbphFace::threshold);
    std::cout << "Loading trained_lbp.xml..." << std::endl;
    modelLbp->load("trained_lbp.xml"); // FIXME: How the heck do I error-check loading??


    std::vector<cv::Mat> images;
    std::vector<int> labels;

    loadPerformanceImagesAndLabels(images, labels);

    int failsEigen = 0, failsFisher = 0, failsLbp = 0;

    int result = 0;
    double dist = 0.0f;

    assert(images.size() == labels.size());
    for (size_t i = 0; i < images.size(); ++i)
    {
//        result = modelEigen->predict(images[i]);
        modelEigen->predict(images[i], result, dist);
        if (result != labels[i])
            ++failsEigen;

        // FIXME: cerr chosen so not buffered. You need to use cout but flush the buffer
        std::cerr << labels[i] << "\t" << result << "(" << dist << ")";

//        result = modelFisher->predict(images[i]);
        modelFisher->predict(images[i], result, dist);
        if (result != labels[i])
            ++failsFisher;

        std::cerr << "\t" << result << "(" << dist << ")";

//        result = modelLbp->predict(images[i]);
        modelLbp->predict(images[i], result, dist);
        if (result != labels[i])
            ++failsLbp;

        std::cerr << "\t" << result << "(" << dist << ")" << std::endl;
    }

    // FIXME: DEBUGGING
    switch (images[0].type())
    {
    case CV_8SC1:
        std::cerr << "Type: " << "8sc1" << std::endl;
        break;
    case CV_8UC1: // FIXME: IT IS THIS!!!
        std::cerr << "Type: " << "8uc1" << std::endl;
        break;
    case CV_16SC1:
        std::cerr << "Type: " << "16sc1" << std::endl;
        break;
    case CV_16UC1:
        std::cerr << "Type: " << "16uc1" << std::endl;
        break;
    default:
        std::cerr << "Type: " << "other" << std::endl;
    }
    std::cerr << lid::lid(images[0], cv::Point(50, 50), 1) << std::endl; // FIXME:

    // Add small number to denominator to prevent division by 0 and prevent integer division
    std::cout << "\nEigen failures: " << failsEigen*100/(images.size() + 0.000001) << "%" << std::endl;
    std::cout << "Fisher failures: " << failsFisher*100/(images.size() +  + 0.000001) << "%" << std::endl;
    std::cout << "LBP failures: " << failsLbp*100/(images.size()  + 0.000001) << "%" << std::endl;
}
