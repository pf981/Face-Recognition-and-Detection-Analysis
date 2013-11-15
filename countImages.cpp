/* CSCI435 Computer Vision
 * Project: Facial Recognition
 * Paul Foster - 3648370
 */
#include <iostream>
#include <fstream>

#include "concatenate.hpp"
#include "countImages.hpp"

// This function returns true if the file exists and false otherwise
bool fileExists(const std::string& filename)
{
    std::ifstream fin(filename.c_str());

    // Note that RAII will ensure that the file gets closed, so we shouldn't close it explicitly.
    return fin.good();
}


// Returns the number of images in the folder "face_samples/sample_{sample}/"
// Note that, because this function assumes the structure of the filesystem, it is not intended to
// be used by the markers. I used this function when training the images, however you will not need
// to use it as you will already have the trained XML files.
int countImages(char sample)
{
    for (int photoNum = 1;; ++photoNum)
    {
        // Filenames are in the form "face_samples/sample_A/A1.JPG"
        std::string filename =
            Concatenate("face_samples/sample_")(sample)("/")(sample)(photoNum)(".JPG").str;

        if (!fileExists(filename))
            return photoNum - 1;
    }
}
