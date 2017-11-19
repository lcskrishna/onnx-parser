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

int parseOnnxGraph(const onnx::GraphProto& graph_proto)
{
	if(graph_proto.has_name()) {
		std::cout << "INFO: Creating a network structure for : " << graph_proto.name() << std::endl;
	}

	for(int i=0; i < graph_proto.node_size(); i++) {
		const onnx::NodeProto node_proto = graph_proto.node(i);
		std::cout << "INFO: Layer is : " << node_proto.op_type() << std::endl;
		
		for(int j=0; j < node_proto.input_size(); j++) {
			std::cout << "Input is : " << node_proto.input(j) << std::endl;
		}

		for(int j=0; j < node_proto.output_size(); j++) {
			std::cout << "Output is : " << node_proto.output(j) << std::endl;
		}
	}

	return 1;
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
