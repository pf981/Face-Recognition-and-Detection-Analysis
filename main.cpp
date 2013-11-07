/* CSCI435 Computer Vision
 * Project: Facial Recognition
 * Paul Foster - 3648370
 */
#include <iostream>

#include "detect.hpp"
#include "performanceTest.hpp"
#include "train.hpp"


int main(int argc, char** argv)
{
// We need to have different executables for training, performance-testing and detection. I could
// have multiple files with main() and use different compile directives, however this might confuse the
// markers - which main should you compile?
// So instead, I chose to have #ifdef's that the Makefile will handle. This way, compiling this on any platform
// (even without the Makefile) will result in the correct detection executable being created.
//
// Note to markers: you do not need to worry about the training or performance test executables as the results for
// them are in the XML files and the report respectively.
#ifdef TRAINING
    // Produce the XML training results if we are in training mode
    // Do not #define TRAINING - Makefile handles this
    train();
    return 0;
#endif


#ifdef PERFORMANCE_TEST
    // Given the trained XML files, it will use test data to measure its performance
    // Do not #define PERFORMANCE_TEST - Makefile handles this
    performanceTest();
    return 0;
#endif


    // Check arguments
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " [image]\n";
        return 1;
    }

    detect(argv[1]);
    return 0;
}
