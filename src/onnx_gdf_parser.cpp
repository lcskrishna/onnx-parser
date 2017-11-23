#include <iostream>
#include <sstream>
#include <iomanip>
#include <fcntl.h>
#include <fstream>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "onnx.pb.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>

void formatFileName(std::string& str, const std::string& from, const std::string& to)
{
    //Written to avoid conflicts with file creation with filenames that contain "/"
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
}

int dumpOnnxModel(const onnx::GraphProto& graph_proto)
{
	for(int i=0; i < graph_proto.initializer_size(); i++) 
	{
		const onnx::TensorProto& tensor_proto = graph_proto.initializer(i);
		const onnx::TensorProto_DataType& datatype = tensor_proto.data_type();
		int tensor_size  = 1;
	
		for(int j = 0; j < tensor_proto.dims_size(); j++) {
			tensor_size *= tensor_proto.dims(j);
		}

		std::string weight_file_name = tensor_proto.name();
		formatFileName(weight_file_name, "/", "_");
		std::string fileName_weights = "weights/" + weight_file_name + ".f32";

		if(datatype == onnx::TensorProto_DataType_FLOAT) {
			
			FILE * fs;
			fs = fopen(fileName_weights.c_str(), "wb");
			if(!fs) {
				std::cout << "ERROR: Unable to create a file, make sure weights folder is writable." << std::endl;
				return -1;
			}
			std::string raw_data_val = tensor_proto.raw_data();
			const char * val = raw_data_val.c_str();
			
			int count = 0;
			for(int k = 0; k < tensor_size*4 - 4; k+=4) {
				float weight;
				char b[] = {val[k], val[k+1], val[k+2], val[k+3]};
				memcpy(&weight, &b, sizeof(float));
				fwrite(&weight, sizeof(float), 1, fs);
				count++;
			}
		
			fclose(fs);
			std::cout << "INFO: Weights dumped for: " << tensor_proto.name() <<  std::endl;
		}
		else {
			std::cout <<"ERROR: Unsupported data types will be supported in future." << std::endl;
			return -1;
		}
	}

	return 0;
}

int getLayerParams(const onnx::NodeProto& node_proto, std::string& params)
{
	std::string layer_type = node_proto.op_type();
	
	if(layer_type == "Conv") {
		
		int pad_h, pad_w;
		int stride_h, stride_w;
		int kernel_h, kernel_w;
		int dilation_h = 1, dilation_w = 1;
				
		for(int i =0; i < node_proto.attribute_size() ; i++) {
			const onnx::AttributeProto& attribute_proto = node_proto.attribute(i);
			std::string attribute_name = attribute_proto.name();
			
			if(attribute_name == "strides") {
				stride_h = attribute_proto.ints(0);
				stride_w = attribute_proto.ints(1);
			}
			else if(attribute_name == "pads") {
				pad_h = attribute_proto.ints(0);
				pad_w = attribute_proto.ints(1);
			}
			else if(attribute_name == "kernel_shape") {
				kernel_h = attribute_proto.ints(0);
				kernel_w = attribute_proto.ints(1);
			}
		}

		params = std::to_string(kernel_w)
			+ " " + std::to_string(kernel_h)
			+ " " + std::to_string(stride_w)
			+ " " + std::to_string(stride_h)
			+ " " + std::to_string(pad_w)
			+ " " + std::to_string(pad_h)
			+ " " + std::to_string(dilation_w)
			+ " " + std::to_string(dilation_h);

		std::cout << "INFO: The parameters are : " << pad_h << " " << pad_w << " " << stride_w << " " << stride_h << " " << kernel_w << " " << kernel_h << std::endl;		
	}
	else if(layer_type == "MaxPool") {
		
		int pad_h, pad_w;
		int stride_h, stride_w;
		int kernel_h, kernel_w;

		for(int i=0; i < node_proto.attribute_size(); i++) {
			const onnx::AttributeProto& attribute_proto = node_proto.attribute(i);
			std::string attribute_name = attribute_proto.name();
		
			if(attribute_name == "strides") {
				stride_h = attribute_proto.ints(0);
				stride_w = attribute_proto.ints(1);
			}
			else if(attribute_name == "pads") {
				pad_h = attribute_proto.ints(0);
				pad_w = attribute_proto.ints(1);
			}
			else if(attribute_name == "kernel_shape") {
				kernel_h = attribute_proto.ints(0);
				kernel_w = attribute_proto.ints(1);
			}

		}

		params = std::to_string(kernel_w)
			+ " " + std::to_string(kernel_h)
			+ " " + std::to_string(stride_w)
			+ " " + std::to_string(stride_h)
			+ " " + std::to_string(pad_w)
			+ " " + std::to_string(pad_h);
		
		std::cout << "INFO: The parameters are: " << pad_h << " " << pad_w << " " << stride_w << " " << stride_h << " " << kernel_w << " " << kernel_h << std::endl;
	

	}
	

	return 0;
}

int parseOnnxGraph(const onnx::GraphProto& graph_proto)
{
	if(graph_proto.has_name()) {
		std::cout << "INFO: Extracting the weights for : " << graph_proto.name() << std::endl;
	}

	if(dumpOnnxModel(graph_proto) < 0) {
		std::cout << "ERROR: Unable to dump weights from onnx model. " << std::endl;
		return -1;
	}
	else {
		std::cout << "RESULT: Weights and bias extraction successful" << std::endl;
	} 

	//TODO: Extract the network structure and finalize  the GDF.
	
	std::cout << "INFO: Extracting the network structure for : " << graph_proto.name() << std::endl;

	for(int i=0; i < graph_proto.node_size(); i++) {
		const onnx::NodeProto node_proto = graph_proto.node(i);
		std::cout << "INFO: Layer is : " << node_proto.op_type() << std::endl;
		std::string params;
		getLayerParams(node_proto, params);	
		/*for(int j=0; j < node_proto.input_size(); j++) {
			std::cout << "Input is : " << node_proto.input(j) << std::endl;
		} */



		/*for(int j=0; j < node_proto.output_size(); j++) {
			std::cout << "Output is : " << node_proto.output(j) << std::endl;
		}

		std::cout << "INFO: attribute proto size is : " << node_proto.attribute_size() << std::endl;
		if(node_proto.attribute_size() > 0) {
			
			for(int j=0; j < node_proto.attribute_size() ; j++) {
				const onnx::AttributeProto& attribute_proto = node_proto.attribute(j);
				std::cout << "Attribute name is: " << attribute_proto.name() << std::endl;
				std::cout << "Tensors size is : " << attribute_proto.tensors_size() << std::endl;
			}
			
		} */
	}

	
	return 0;
}

int loadOnnxModelFile(const char * fileName)
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;
	
        onnx::ModelProto model_proto;
        std::cout << "INFO: Reading the binary onnx model file." << std::endl;
        std::fstream input(fileName, std::ios::in | std::ios::binary);
        bool isSuccess = model_proto.ParseFromIstream(&input);

        if(isSuccess) {
                std::cout << "INFO: Sucessfully read onnx model file. " << std::endl;
		if(model_proto.has_graph()) {
			std::cout << "DEBUG: Parsing the onnx model." << std::endl;
			const onnx::GraphProto graph_proto = model_proto.graph();
			if(parseOnnxGraph(graph_proto) < 0) {
				std::cout << "ERROR: Unable to parse ONNX model." << std::endl;
				return -1;
			}
		}
		else {
			std::cerr << "ERROR: No network structure found." << std::endl;
			return -1;
		}
        }
        else {
                std::cerr << "ERROR: Failed to read the onnx model file. " << std::endl;
		return -1;
        }

	return 0;

}

int main(int argc, char * argv[])
{
	const char * usage = 
			"Usage: \n"
			" ./onnx_gdf_generator <net.pb> [n c H W]";
	
	if(argc < 2) {
		printf("ERROR: %s\n", usage);
		return -1;
	}
	
	int inputDim[4] = {0, 0, 0, 0};
	const char * fileName = argv[1];
	if(argc > 2) inputDim[0] = atoi(argv[2]);
	if(argc > 3) inputDim[1] = atoi(argv[3]);
	if(argc > 4) inputDim[2] = atoi(argv[4]);
	if(argc > 5) inputDim[3] = atoi(argv[5]);

	//load onnx model.
	if(loadOnnxModelFile(fileName) < 0) {
		return -1;
	}
	

	return 0;
}
