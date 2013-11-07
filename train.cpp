/* CSCI435 Computer Vision
 * Project: Facial Recognition
 * Paul Foster - 3648370
 */
// This file defines the functions needed to processes the training images. They generate xml files
// that are needed for facial recognition.
#include <fstream>
#include <iostream>
#include <sstream> // FIXME: Certainly don't need all these includes
#include <string>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "concatenate.hpp"
#include "countImages.hpp"
#include "images.hpp"
#include "params.hpp"
#include "train.hpp"


// This function iterates through all training images and store the image and its label in vectors.
// It assumes the following hierarchy:
//      .
//      |-- sample_A
//      |   |-- A1.JPG
//      |   |-- ...
//      |   |-- An_1.JPG
//      |-- sample_B
//      |   |-- B1.JPG
//      |   |-- ...
//      |   |-- Bn_2.JPG
//      ...
//      |-- sample_W
//      |   |-- W1.JPG
//      |   |-- ...
//      |   |-- Wn_13.JPG
//
// That is, it expects samples to be in their own folder with differing numbers of images.
void loadTrainingImagesAndLabels(std::vector<cv::Mat>& images, std::vector<int>& labels)
{
    // For each training image
    for (char sample = 'A'; sample <= 'W'; ++sample) // FIXME: Use this
//    for (char sample = 'A'; sample <= 'B'; ++sample)
    {
        // FIXME: Of the images, we need to RANDOMLY select 60% of them for training
        std::cerr << sample << ": " << countImages(sample) << std::endl; // FIXME: Remove
        for (int photoNum = 1;
             photoNum <= (int)(params::training::trainingToValidationRatio*countImages(sample));
             ++photoNum)
        {
            // Filenames are in the form "face_samples/sample_A/A1.JPG"
            std::string filename =
                Concatenate("face_samples/sample_")(sample)("/")(sample)(photoNum)(".JPG").str;

            if (!fileExists(filename))
                break;

            std::cout << "Reading " << filename << std::endl;

            // Read the image in. Fisher only works on greyscale
//            cv::Mat image(cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE)); // FIXME: Don't think this worked

            // Scale the images as they so they are faster to process. All images need to have the same
            // dimensions as it is required for the recognisers
//            scaleImage(image); // FIXME:


            // cv::Mat image(cv::imread(filename));
            // scaleImage(image);
            // cv::Mat grayscale;
            // cv::cvtColor(image, grayscale, CV_BGR2GRAY);
            // equalizeHist(grayscale, grayscale); // FIXME: Why do this?

            // FIXME: Added this to try and fix eigen and fisher not working
            cv::Mat image(cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE));
            scaleImage(image);
            cv::Mat grayscale(image.clone());
            // cv::cvtColor(image, grayscale, CV_BGR2GRAY);
            //equalizeHist(grayscale, grayscale); // FIXME: Why do this?


            // Add the image and its label to the vectors
            images.push_back(grayscale);
            labels.push_back(sample - 'A');
        }
    }
}


// This function reads in all the training data a uses them to generate xml files.
// Note that this function assumes the heirarchy of the image files. This is not intended to
// be run more than once - once the xml files are generated, there is no need to regenerate
// them.
void train()
{
    std::vector<cv::Mat> images;
    std::vector<int> labels;

    loadTrainingImagesAndLabels(images, labels);


    // Train Eigen, Fisher, LBP and LID face recognisers with the images and produce xml files
//    cv::Ptr<cv::FaceRecognizer> model; // FIXME: REMOVE
    // FIXME: UNCOMMENT
    std::cout << "Training Eigen Face Recogniser..." << std::endl;
    cv::Ptr<cv::FaceRecognizer> model = cv::createEigenFaceRecognizer(
        params::eigenFace::numComponents,
        params::eigenFace::threshold);
    model->train(images, labels);
    model->save("trained_eigen.xml");

    // std::cout << "Training Fisher Face Recogniser..." << std::endl;
    // // Note that Ptr handles reference counting when we use the assignment operator,
    // // so there won't be a memory leak
    // model = cv::createFisherFaceRecognizer(
    //     params::fisherFace::numComponents,
    //     params::fisherFace::threshold);
    // model->train(images, labels);
    // model->save("trained_fisher.xml");

    std::cout << "Training LBP Face Recogniser..." << std::endl;
    model = cv::createLBPHFaceRecognizer(
        params::lbphFace::radius,
        params::lbphFace::neighbors,
        params::lbphFace::gridX,
        params::lbphFace::gridY,
        params::lbphFace::threshold);
    model->train(images, labels);
    model->save("trained_lbp.xml");


    // FIXME: Train LID Face Recogniser
}
