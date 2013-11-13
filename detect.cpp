/* CSCI435 Computer Vision
 * Project: Facial Recognition
 * Paul Foster - 3648370
 */
//#include <cmath> // FIXME: REMOVE
#include <iostream>
#include <string>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "detect.hpp"
#include "images.hpp"
#include "params.hpp"
#include "concatenate.hpp"

// Runs the cascade classifier and popultes the faces vector
void detectFaces(const cv::Mat& grayscale, std::vector<cv::Rect>& faces)
{
    cv::CascadeClassifier faceCascade;
    if(!faceCascade.load("haarcascade_frontalface_alt.xml"))
    {
        std::cerr << "Error: Unable to load haarcascade_frontalface_alt.xml.\nExiting.\n";
        exit(1);
    };

    // Detect the faces
    faceCascade.detectMultiScale(
        grayscale,
        faces,
        params::cascadeClassifier::scaleFactor,
        params::cascadeClassifier::minNeighbors,
        params::cascadeClassifier::flags,
        // cv::Size(),
        // cv::Size());
        cv::Size(params::cascadeClassifier::minSize),
//        cv::Size(params::cascadeClassifier::maxSize));
        params::cascadeClassifier::maxSize);
        // params::cascadeClassifier::minSize,
        // params::cascadeClassifier::maxSize);
        // 1.1,
        // 2,
        // 0,
        // cv::Size(40, 40));
}


// Returns the letter denoting the face ID or "Unknown" if prediction == -1.
inline std::string getPrediction(int prediction)
{
    return ((prediction == -1) ? "Unknown" : Concatenate((char)(prediction + 'A')).str);
}

void detect(const std::string& imageFile)
{
    cv::Mat original(cv::imread(imageFile));
    if (!original.data)
    {
        std::cerr << "Could not open image " << imageFile << "\nExiting.\n";
        exit(1);
    }

    // Scale/crop the group image
    cv::Mat groupImage(original.clone());
    scaleGroupImage(groupImage);

    // Convert to grayscale (required by FaceRecognizer)
    cv::Mat grayscale(groupImage.clone());
    cv::cvtColor(groupImage, grayscale, CV_BGR2GRAY);
    equalizeHist(grayscale, grayscale); // FIXME: THIS IS ABSOLUTELY NEEDED - MISSES A FACE OTHERWISE


    std::vector<cv::Rect> faces;
    detectFaces(grayscale, faces);

    // If it is a group photo
    if (faces.size() > 1)
    {
        // Draw the bounding boxes
        for(size_t i = 0; i < faces.size(); i++)
        {
            cv::Mat faceROI = grayscale(faces[i]);
            int x = faces[i].x;
            int y = faces[i].y;
            int h = y + faces[i].height;
            int w = x + faces[i].width;
            rectangle(
                groupImage,
                cv::Point(x, y),
                cv::Point(w, h),
                cv::Scalar(255, 0, 255),
                2,
                8,
                0);
        }

        std::cout << faces.size() << " faces detected." << std::endl;
        imshow("INFO435 Project", groupImage);
    }
    else // If it is an individual photo
    {
        // Scale it (different dimensions to a group image) and convert it to grayscale
        cv::Mat individualImage(original.clone());
        scaleImage(individualImage);
        cv::cvtColor(individualImage, grayscale, CV_BGR2GRAY);
        equalizeHist(grayscale, grayscale); // FIXME: Why do this?

        // Eigen
        cv::Ptr<cv::FaceRecognizer> model = cv::createEigenFaceRecognizer( // FIXME: This should be its own function
            params::eigenFace::numComponents,
            params::eigenFace::threshold);
        model->load("trained_eigen.xml"); // FIXME: How the heck do I error-check loading??

        int prediction = model->predict(grayscale);
        std::cout << "Eigen Face: Person is " << getPrediction(prediction) << std::endl;

        // Fisher
        model = cv::createFisherFaceRecognizer( // FIXME: This should be its own function
            params::fisherFace::numComponents,
            params::fisherFace::threshold);
        model->load("trained_fisher.xml"); // FIXME: How the heck do I error-check loading??

        prediction = model->predict(grayscale);
        std::cout << "Fisher Face: Person is "<< getPrediction(prediction) << std::endl;

        // LBP
        model = cv::createLBPHFaceRecognizer( // FIXME: This should be its own function
        params::lbphFace::radius,
        params::lbphFace::neighbors,
        params::lbphFace::gridX,
        params::lbphFace::gridY,
        params::lbphFace::threshold);
        model->load("trained_lbp.xml"); // FIXME: How the heck do I error-check loading??

        prediction = model->predict(grayscale);
        std::cout << "LBP: Person is " << getPrediction(prediction) << std::endl;


        imshow("INFO435 Project", individualImage);
    }



    cv::waitKey(0);
}
