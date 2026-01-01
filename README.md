# Image Rotation using NVIDIA NPP with CUDA

## Overview

This project demonstrates the use of NVIDIA Performance Primitives (NPP) library with CUDA to perform image median filtering. The goal is to utilize GPU acceleration to efficiently rotate a given image by a specified angle, leveraging the computational power of modern GPUs. The project is a part of the CUDA at Scale for the Enterprise course and serves as an example for understanding how to implement basic image processing operations using CUDA and NPP.

## Code Organization

This code is maintained in GitHub in the project https://github.com/markalavin/CUDAatScaleForTheEnterpriseCourseProjectTemplate/tree/main; for development, it was downloaded into a Coursera-provided userid with Visual Studio on Linux.  It should be buildable and runnable on a Windows system as well, although the Makefile may have to be adjusted.

```bin/```
This folder should hold all binary/executable code that is built automatically or manually. Executable code should have use the .exe extension or programming language-specific extension.

```data/```
This folder should hold all example data in any format. If the original data is rather large or can be brought in via scripts, this can be left blank in the respository, so that it doesn't require major downloads when all that is desired is the code/structure.

```lib/```
Any libraries that are not installed via the Operating System-specific package manager should be placed here, so that it is easier for inclusion/linking.

```src/```
The source code should be placed here in a hierarchical fashion, as appropriate.

```README.md```
This file should hold the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

```Makefile```
For building your project's code in an automatic fashion.

```3rdParty```
Code from STB for reading and writing image files.

```Common```
Code for NPP to perform CUDA-assisted image processing.


## Key Concepts

Performance Strategies, Image Processing, NPP Library

## Supported SM Architectures

[SM 3.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 3.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, ppc64le, armv7l

## CUDA APIs involved

## Dependencies needed to build/run
[FreeImage](../../README.md#freeimage), [NPP](../../README.md#npp)

## Prerequisites

Download and install the [CUDA Toolkit 11.4](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## Build and Run

### Windows
The Windows samples are built using the Visual Studio IDE. Solution files (.sln) are provided for each supported version of Visual Studio, using the format:
```
*_vs<version>.sln - for Visual Studio <version>
```
Each individual sample has its own set of solution files in its directory:

To build/examine all the samples at once, the complete solution files should be used. To build/examine a single sample, the individual sample solution files should be used.
> **Note:** Some samples require that the Microsoft DirectX SDK (June 2010 or newer) be installed and that the VC++ directory paths are properly set up (**Tools > Options...**). Check DirectX Dependencies section for details."

### Linux
The Linux samples are built using makefiles. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:
```
$ cd <sample_dir>
$ make
```
The samples makefiles can take advantage of certain options:
*  **TARGET_ARCH=<arch>** - cross-compile targeting a specific architecture. Allowed architectures are x86_64, ppc64le, armv7l.
    By default, TARGET_ARCH is set to HOST_ARCH. On a x86_64 machine, not setting TARGET_ARCH is the equivalent of setting TARGET_ARCH=x86_64.<br/>
`$ make TARGET_ARCH=x86_64` <br/> `$ make TARGET_ARCH=ppc64le` <br/> `$ make TARGET_ARCH=armv7l` <br/>
    See [here](http://docs.nvidia.com/cuda/cuda-samples/index.html#cross-samples) for more details.
*   **dbg=1** - build with debug symbols
    ```
    $ make dbg=1
    ```
*   **SMS="A B ..."** - override the SM architectures for which the sample will be built, where `"A B ..."` is a space-delimited list of SM architectures. For example, to generate SASS for SM 50 and SM 60, use `SMS="50 60"`.
    ```
    $ make SMS="50 60"
    ```

*  **HOST_COMPILER=<host_compiler>** - override the default g++ host compiler. See the [Linux Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements) for a list of supported host compilers.
```
    $ make HOST_COMPILER=g++
```

## What the Program Does
The program performs median filtering on an RGB image like "Lena.png", from which a grayscale intensity image is derived.  Median filtering
calculates the value of every output pixel based on the median of the values of a square "mask" surrounding the pixel.  Median filtering is a technique for reducing noise and detail in an image; unlike linear filtering (e.g., Gaussian convolution), median filtering does not "blur out" edges.

### Downsampling to handle large-radius median filtering

As the program was developed, I added an argument ```-radius``` to specify the radius of the filter mask.  However, I discovered that the NPP median filtering function does not behave correctly if the radius is > 7.  With the help of Google Gemini, I developed a technique
for getting around this.  Suppose the specified radius is 14.  In that case, we can downsample the input by a factor of 2x, then run
the NPP median filtering with a radius of 7, then upsample the output by a factor of 2x.  In general, the downsample/upsample by a factor of ```radius / 7``` allows the handling of virtually any radius.

## Running the Program
After building the project, you can run the program using the following commands:

```
cd CUDAatScaleForTheEnterpriseProjectTemplate   # root directory of this median filter project
./bin/imageMedianFilterNPP [ -input filename ] [ -output filename ] [ -radius number ]
```

This command will execute the compiled binary, rotating the input image (Lena.png) by 45 degrees, and save the result image in the ```data/<inputfilename>_median_filter``` file.

