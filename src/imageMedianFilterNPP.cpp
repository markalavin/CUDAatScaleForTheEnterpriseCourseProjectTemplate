/* * Standard Includes and NPP Utility Setup
 */
#include <iostream>
#include <fstream>
#include <string>
#include <string.h>

// 1. Define required NPP utility macros BEFORE including NPP headers
#define NPP_CPP_INCLUDES
#ifndef NPP_CHECK_NPP
#define NPP_CHECK_NPP(token)                                   \
    {                                                          \
        NppStatus status = token;                              \
        if (status != NPP_SUCCESS)                             \
        {                                                      \
            std::cerr << "NPP Error: " << status << std::endl; \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    }
#endif
#ifndef NPP_CHECK_CUDA
#define NPP_CHECK_CUDA(token) (token)
#endif
#ifndef NPP_ASSERT_NOT_NULL
#define NPP_ASSERT_NOT_NULL(token) (token)
#endif

// 2. Include local headers
#include "Exceptions.h"
#include "ImagesCPU.h"
#include "ImagesNPP.h"

// 3. Include CUDA/NPP system headers
#include <cuda_runtime.h>
#include <npp.h>
#include <helper_cuda.h>
#include <helper_string.h>

// 4. Include STB for image loading
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

using namespace npp;

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

    // Move pScratch to function scope so the final cleanup can see it
    Npp8u *pScratch = nullptr;

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

        sFilename = (filePath) ? filePath : "../data/Lena.png";

        std::ifstream infile(sFilename.data(), std::ifstream::in);
        if (!infile.good())
        {
            std::cout << "Unable to open: <" << sFilename.data() << ">" << std::endl;
            exit(EXIT_FAILURE);
        }
        infile.close();

        std::string sResultFilename = sFilename;
        std::string::size_type dot = sResultFilename.rfind('.');
        if (dot != std::string::npos)
            sResultFilename = sResultFilename.substr(0, dot);
        sResultFilename += "_median_filter.png";

        // Get the "radius" argument or default:
        int radius = 3;   // default:  7x7 median filter
        if (checkCmdLineFlag(argc, (const char **)argv, "radius"))
        {
            char *radius_str;
            getCmdLineArgumentString(argc, (const char **)argv, "radius", &radius_str);
            radius = atoi(radius_str);
        }
        if ( radius > 7 )
            fprintf(stderr, "Large radius (%d) detected: Using dynamic downsampling pipeline.\n", radius);
        else
            fprintf(stderr, "radius is %d\n", radius);

        // 1. Load dimensions and pixel data using STB
        int width, height, channels;
        unsigned char *pData = stbi_load(sFilename.c_str(), &width, &height, &channels, 1);

        if (pData == nullptr)
        {
            std::cerr << "Failed to load image: " << sFilename << std::endl;
            exit(EXIT_FAILURE);
        }

        // 2. Create Device Image and Upload directly
        npp::ImageNPP_8u_C1 oDeviceSrc(width, height);
        cudaMemcpy2D(oDeviceSrc.data(), oDeviceSrc.pitch(), pData, width, width, height, cudaMemcpyHostToDevice);
        stbi_image_free(pData);

        NppiSize oSizeROI = {width, height};

        // 3. Allocate device image for the result
        npp::ImageNPP_8u_C1 oDeviceDst(width, height);

        // 4. Median Filter Configuration
        int nRadius = radius;
        NppiSize oMaskSize = {2 * nRadius + 1, 2 * nRadius + 1};
        NppiPoint oAnchor = {nRadius, nRadius};
#ifdef NEVER /* ### */
        // 5. Setup Scratch Buffer
        Npp32s nBufferSize;
        NPP_CHECK_NPP(
            nppiFilterMedianBorderGetBufferSize_8u_C1R(
            oSizeROI,    // The Region of Interest size
            oMaskSize,   // The mask size (e.g., {17, 17} for radius 8)
            &nBufferSize // Pointer to the variable that stores the required size
            )
        );
        NPP_CHECK_CUDA( cudaMalloc((void **)&pScratch, nBufferSize) );

        // 6. Execute Median Filter
        NPP_CHECK_NPP(nppiFilterMedian_8u_C1R(
            oDeviceSrc.data(), oDeviceSrc.pitch(),
            oDeviceDst.data(), oDeviceDst.pitch(),
            oSizeROI, oMaskSize, oAnchor, pScratch));
#endif /* ### */
        // 5. Calculate dynamic decimation factor
        float fFactor = (radius > 7) ? (float)radius / 7.0f : 1.0f;

        NppiSize oSmallSize = {
            (int)(width / fFactor),
            (int)(height / fFactor)};
        oSmallSize.width = std::max(oSmallSize.width, 1);
        oSmallSize.height = std::max(oSmallSize.height, 1);

        // Create the small version workspaces
        npp::ImageNPP_8u_C1 oDeviceSmallSrc(oSmallSize.width, oSmallSize.height);
        npp::ImageNPP_8u_C1 oDeviceSmallDst(oSmallSize.width, oSmallSize.height);

        // 6. Resize Down
        NppiSize oSrcSize = {width, height};
        NppiRect oSrcRect = {0, 0, width, height};
        NppiRect oSmallRect = {0, 0, oSmallSize.width, oSmallSize.height};

        NPP_CHECK_NPP(nppiResize_8u_C1R(
            oDeviceSrc.data(), oDeviceSrc.pitch(), oSrcSize, oSrcRect,
            oDeviceSmallSrc.data(), oDeviceSmallSrc.pitch(), oSmallSize, oSmallRect,
            NPPI_INTER_SUPER));

        // 6.2 Filter the small image
        int nSmallRadius = (radius > 7) ? 7 : radius;
        NppiSize oSmallMask = {2 * nSmallRadius + 1, 2 * nSmallRadius + 1};
        NppiPoint oSmallAnchor = {nSmallRadius, nSmallRadius};

        Npp32u nSmallBufferSize;
        NPP_CHECK_NPP(nppiFilterMedianGetBufferSize_8u_C1R(oSmallSize, oSmallMask, &nSmallBufferSize));

        Npp8u *pSmallScratch = nullptr;
        NPP_CHECK_CUDA(cudaMalloc((void **)&pSmallScratch, nSmallBufferSize));

        NPP_CHECK_NPP(nppiFilterMedian_8u_C1R(
            oDeviceSmallSrc.data(), oDeviceSmallSrc.pitch(),
            oDeviceSmallDst.data(), oDeviceSmallDst.pitch(),
            oSmallSize, oSmallMask, oSmallAnchor, pSmallScratch));

        cudaFree(pSmallScratch);

        // C1. Copy source to destination for background
        cudaMemcpy2D(oDeviceDst.data(), oDeviceDst.pitch(),
                     oDeviceSrc.data(), oDeviceSrc.pitch(),
                     width, height, cudaMemcpyDeviceToDevice);

        // 6.5 Resize Small Filtered Image back to Original Size
        NPP_CHECK_NPP(nppiResize_8u_C1R(
            oDeviceSmallDst.data(), oDeviceSmallDst.pitch(), oSmallSize, oSmallRect,
            oDeviceDst.data(), oDeviceDst.pitch(), oSrcSize, oSrcRect,
            NPPI_INTER_LANCZOS));
        // 7. Download Result to Host
        unsigned char *pHostDstData = new unsigned char[width * height];
        cudaMemcpy2D(pHostDstData, width, oDeviceDst.data(), oDeviceDst.pitch(), width, height, cudaMemcpyDeviceToHost);

        // 8. Save the image using STB
        stbi_write_png(sResultFilename.c_str(), width, height, 1, pHostDstData, width);
        std::cout << "Saved Median Filtered image: " << sResultFilename << std::endl;

        // Cleanup local host memory
        delete[] pHostDstData;
    }
    catch (std::exception &e)
    {
        std::cerr << "Program error: " << e.what() << std::endl;
        if (pScratch)
            cudaFree(pScratch);
        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Unknown exception occurred." << std::endl;
        if (pScratch)
            cudaFree(pScratch);
        exit(EXIT_FAILURE);
    }

    if (pScratch != nullptr)
        cudaFree(pScratch);

    return 0;
}
