/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

bool printfNPPinfo(int argc, char *argv[])
{
    const NppLibraryVersion *libVer = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
           libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
           (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
           (runtimeVersion % 100) / 10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}

int main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);

    try
    {
        std::string sFilename;
        char *filePath;

        findCudaDevice(argc, (const char **)argv);

        if (printfNPPinfo(argc, argv) == false)
        {
            exit(EXIT_SUCCESS);
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "input"))
        {
            getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
        }
        else
        {
            filePath = sdkFindFilePath("../data/Lena.png", argv[0]);
        }

        if (filePath)
        {
            sFilename = filePath;
        }
        else
        {
            sFilename = "../data/Lena.png";
        }

        // if we specify the filename at the command line, then we only test
        // sFilename[0].
        int file_errors = 0;
        std::ifstream infile(sFilename.data(), std::ifstream::in);

        if (infile.good())
        {
            std::cout << "nppiRotate opened: <" << sFilename.data()
                      << "> successfully!" << std::endl;
            file_errors = 0;
            infile.close();
        }
        else
        {
            std::cout << "nppiRotate unable to open: <" << sFilename.data() << ">"
                      << std::endl;
            file_errors++;
            infile.close();
        }

        if (file_errors > 0)
        {
            exit(EXIT_FAILURE);
        }

        std::string sResultFilename = sFilename;

        std::string::size_type dot = sResultFilename.rfind('.');

        if (dot != std::string::npos)
        {
            sResultFilename = sResultFilename.substr(0, dot);
        }

        sResultFilename += "_median_filter.png";

        if (checkCmdLineFlag(argc, (const char **)argv, "output"))
        {
            char *outputFilePath;
            getCmdLineArgumentString(argc, (const char **)argv, "output",
                                     &outputFilePath);
            sResultFilename = outputFilePath;
        }

        // 1. Load dimensions and pixel data using STB
        int width, height, channels;
        unsigned char *pData = stbi_load(sFilename.c_str(), &width, &height, &channels, 1);

        if (pData == nullptr)
        {
            std::cerr << "Failed to load image: " << sFilename << std::endl;
            exit(EXIT_FAILURE);
        }

        // 2. Setup NPP Host Image and metadata
        // Initialize host image directly with dimensions
        // 1. Create a clean NPP Host Image object (this allocates its own memory)
        npp::ImageCPU_8u_C1 oHostSrc(width, height);

        // 2. Perform a row-by-row copy to account for NPP memory alignment (pitch)
        for (int y = 0; y < height; ++y)
        {
            unsigned char *pLineData = oHostSrc.data() + (y * oHostSrc.pitch());
            unsigned char *pLineSrc = pData + (y * width);
            memcpy(pLineData, pLineSrc, width);
        }

        // 3. Immediately free the STB buffer now that data is safely in oHostSrc
        stbi_image_free(pData);
        
        NppiSize oSrcSize = {width, height};
        NppiRect oSrcRect = {0, 0, width, height};

        // 3. Upload to Device
        npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

       // create struct with the ROI size
        NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};

        // Calculate the bounding box of the rotated image
        NppiRect oBoundingBox;
        double angle = 45.0; // Rotation angle in degrees
        double aBoundingBox[2][2]; // Array to store the result coordinates
        NPP_CHECK_NPP(nppiGetRotateBound(oSrcRect, aBoundingBox, angle, 0.0, 0.0));

        oBoundingBox.x = (int)aBoundingBox[0][0];
        oBoundingBox.y = (int)aBoundingBox[0][1];
        oBoundingBox.width = (int)(aBoundingBox[1][0] - aBoundingBox[0][0]);
        oBoundingBox.height = (int)(aBoundingBox[1][1] - aBoundingBox[0][1]);

        // allocate device image for the rotated image
        npp::ImageNPP_8u_C1 oDeviceDst(oBoundingBox.width, oBoundingBox.height);

        // Set the rotation point (center of the image)
        NppiPoint oRotationCenter = {(int)(oSrcSize.width / 2), (int)(oSrcSize.height / 2)};

        // AAA.002: Median Filter Config and Scratch Buffer
        int nRadius = 2;
        NppiSize oMaskSize = {2 * nRadius + 1, 2 * nRadius + 1};
        NppiPoint oAnchor = {nRadius, nRadius};

        int nScratchSize;
        NPP_CHECK_NPP(nppiFilterMedianGetBufferSize_8u_C1R(oSizeROI, oMaskSize, &nScratchSize));

        Npp8u *pScratch;
        cudaMalloc((void **)&pScratch, nScratchSize); // Update the shifts to -oBoundingBox.x and -oBoundingBox.y

        // AAA.003: The actual Median Filter call
        NPP_CHECK_NPP(nppiFilterMedian_8u_C1R(
            oDeviceSrc.data(), oDeviceSrc.pitch(),
            oDeviceDst.data(), oDeviceDst.pitch(),
            oSizeROI, oMaskSize, oAnchor, pScratch));

        // declare a host image for the result
        npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
        // and copy the device result data into it
        oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

        stbi_write_png(sResultFilename.c_str(), oHostDst.width(), oHostDst.height(), 1, oHostDst.data(), oHostDst.pitch());
        std::cout << "Saved Median Filtered image: " << sResultFilename << std::endl;

        // Clean up scratch buffer
        cudaFree(pScratch);
        std::cout << "Saved Median Filtered image: " << sResultFilename << std::endl;

        // AAA.001: Free scratch buffer before exiting try block
        if (pScratch != nullptr) cudaFree(pScratch);}
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknown type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
        return -1;
    }

    if (pScratch != nullptr)
    {
        cudaFree(pScratch);
    }

    return 0;
}
