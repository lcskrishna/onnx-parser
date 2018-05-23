[![Build Status](https://travis-ci.org/lcskrishna/onnx-parser.svg?branch=master)](https://travis-ci.org/lcskrishna/onnx-parser)

# onnx-parser

ONNX-Parser is a C++ Inference Code Generator tool that takes an [onnx](https://github.com/onnx/onnx) binary model and generates OpenVX [GDF](https://github.com/GPUOpen-ProfessionalCompute-Libraries/amdovx-core/tree/master/runvx) code for quick prototyping and kernel debugging.

The details of various OpenVX Kernels generated are from [vx_nn](https://github.com/GPUOpen-ProfessionalCompute-Libraries/amdovx-modules/tree/master/vx_nn)

Note: Currently only the float-32 tensor datatypes are supported. Special layers such as ROI Pooling Layer, Deconvolution are not yet supported.

## Supported Models

Network Name | Supported
-------------| -----------
AlexNet      | Yes
VGG-16       | Yes
VGG-19       | 
ResNet-50    |    

## Command-line interface

```
 % onnx_gdf_generator <net.pb> [n c H W]
```
Here net.pb is the onnx binary models which is a mandatory file.
n,c,H,W refers to batch size, number of channels, height and width of an image which are optional parameters.

ONNX Binary models can be found here : [models](https://github.com/onnx/models)

## Pre-requisites
1. Ubuntu 16.04
2. CMAKE 2.8 or newer [download](https://cmake.org/download/)
3. Install the [protobuf](https://github.com/google/protobuf) from C++ install instructions.

## Build Instructions

1. After cloning the repository, create a build folder where the executables has to be present.
2. cmake -DCMAKE_BUILD_TYPE=Release ../onnx-parser
3. make 

Now, the executables are built and present in the build folder.


