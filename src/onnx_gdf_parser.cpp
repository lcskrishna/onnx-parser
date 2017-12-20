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

int calculateTensorDims
(
	const onnx::GraphProto& graph_proto,
	std::map<int, std::map<std::string, std::string>>& net,
	std::map<int, std::map<std::string, std::vector<int>>>& tensorDims
)
{
	std::map<std::string, std::vector<int>> input_tensor_dim_map;	

	//Inputs to the graph.
	for(int i=0; i < graph_proto.input_size(); i++) {
		const onnx::ValueInfoProto& value_info_proto = graph_proto.input(i);
		std::string layer_input = value_info_proto.name();
		std::vector<int> dims;
		
		const onnx::TypeProto& type_proto = value_info_proto.type();
		const onnx::TypeProto::Tensor& tensor = type_proto.tensor_type();
		const onnx::TensorShapeProto& tensor_shape = tensor.shape();
		
		for(int j=0; j < tensor_shape.dim_size(); j++) {
			const onnx::TensorShapeProto::Dimension& dimension = tensor_shape.dim(j);
			dims.push_back(dimension.dim_value());
		}

		input_tensor_dim_map[layer_input] = dims;
	}

	for(int i=0; i < net.size(); i++) {
		std::map<std::string, std::string> layer_details = net.find(i)->second; 
		std::string layer_type = layer_details.find("type")->second;
		std::string layer_input = layer_details.find("input")->second;
		std::string layer_output = layer_details.find("output")->second;
		std::cout << "Input output type is " << layer_input << " " << layer_output << " " << layer_type << std::endl;
		int in_w, in_h, in_c, in_n;
		int out_w, out_h, out_c, out_n;
		std::vector<int> output_dims;
		
		std::vector<int> input_dims = input_tensor_dim_map.find(layer_input)->second;
		in_n = input_dims[0]; in_c = input_dims[1]; in_h = input_dims[2]; in_w = input_dims[3];
		std::map<std::string, std::vector<int>> in_out_map;
		in_out_map[layer_input] = input_dims;			

		if(layer_type == "Conv") {
			std::string layer_weights = " "; 
			std::vector<int> weight_dims, bias_dims;
			if(layer_details.size() > 4) {
				layer_weights = layer_details.find("weights")->second;
				weight_dims = input_tensor_dim_map.find(layer_weights)->second;
			}
			std::string layer_bias = " ";
			if(layer_details.size() > 5) { 
				std::string layer_bias = layer_details.find("bias")->second;
				bias_dims = input_tensor_dim_map.find(layer_bias)->second;
			}
			std::string params = layer_details.find("params")->second;
			
			int kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h, dilation_w, dilation_h;
			std::stringstream ss(params);
			ss >> kernel_w >> kernel_h >> stride_w >> stride_h >> pad_w >> pad_h >> dilation_w >> dilation_h;

			out_w = ((in_w + 2 * (pad_w) - kernel_w - (kernel_w - 1) * (dilation_w - 1))/stride_w) + 1;
			out_h = ((in_h + 2 * (pad_h) - kernel_h - (kernel_h - 1) * (dilation_h - 1))/stride_h) + 1;
			out_c = weight_dims[0];
			out_n = in_n;
			
			if(layer_details.size() > 4) {
				in_out_map[layer_weights] = weight_dims;
			}
			
			if(layer_details.size() > 5) {
				in_out_map[layer_bias] = bias_dims;
			}
			std::cout << "Conv" << std::endl;
		}
		else if(layer_type == "Relu") {
			out_w = in_w;
			out_h = in_h;
			out_c = in_c;
			out_n = in_n;
			std::cout << "Relu" << std::endl;
		}
		else if(layer_type == "LRN") {
			out_w = in_w;
			out_h = in_h;
			out_c = in_c;
			out_n = in_n;
			std::cout << "LRN" << std::endl;
		}
		else if(layer_type == "Dropout") {
			out_w = in_w;
			out_h = in_h;
			out_c = in_c;
			out_n = in_n;
			std::cout << "Dropout" << std::endl;
		}
		else if(layer_type == "MaxPool") {
			std::string params = layer_details.find("params")->second;
			std::stringstream ss(params);
			int kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h;
			ss >> kernel_w >> kernel_h >> stride_w >> stride_h >> pad_w >> pad_h;

			out_w = static_cast<int>(ceil( static_cast<float> (in_w + 2 * pad_w + stride_w - kernel_w)/stride_w));
			out_h = static_cast<int>(ceil( static_cast<float> (in_h + 2 * pad_h + stride_h - kernel_h)/stride_h));
			if(pad_h > 0) if((out_h-1) * stride_h >= (in_h + pad_h)) out_h = out_h - 1;
			if(pad_w > 0) if((out_w-1) * stride_w >= (in_w + pad_w)) out_w = out_w - 1;
			
			out_c = in_c;
			out_n = in_n;	
			std::cout << "MaxPool" << std::endl;
		}
		else if(layer_type == "Gemm") {
			
			std::string layer_weights = " "; 
			std::vector<int> weight_dims, bias_dims;
			std::vector<int> weight_dims_gemm;
			if(layer_details.size() > 4) {
				layer_weights = layer_details.find("weights")->second;
				weight_dims = input_tensor_dim_map.find(layer_weights)->second;
				weight_dims_gemm.push_back(in_w);
				weight_dims_gemm.push_back(in_h);
				weight_dims_gemm.push_back(in_c);
				weight_dims_gemm.push_back(weight_dims[0]);

			}
			std::string layer_bias = " ";
			if(layer_details.size() > 5) { 
				std::string layer_bias = layer_details.find("bias")->second;
				bias_dims = input_tensor_dim_map.find(layer_bias)->second;
				//TODO: Need to handle bias dims.
			}

			out_n = 1;
			out_c = weight_dims[0];
			out_h = 1;
			out_w = 1;
			
			if(layer_details.size() > 4) {
				in_out_map[layer_weights] = weight_dims_gemm;
			}
			
			if(layer_details.size() > 5) {
				in_out_map[layer_bias] = bias_dims;
			}
			std::cout << "Gemm" << std::endl;

		}
		
		output_dims.push_back(out_n);
		output_dims.push_back(out_c);
		output_dims.push_back(out_h);
		output_dims.push_back(out_w);
		input_tensor_dim_map[layer_output] = output_dims;
		std::cout << "Added output : " << layer_output << std::endl;
		in_out_map[layer_output] = output_dims;

		std::cout << "Out dims: " << out_w << " " << out_h << " " << out_c << " " << out_n << std::endl;	

		tensorDims[i] = in_out_map;
	}

	return 0;	
}

int writeGDF
(
	std::ofstream& ofsGDF,
	std::map<int, std::map<std::string, std::string>> net,
	std::map<int, std::map<std::string, std::vector<int>>> tensorDims
)
{
	ofsGDF << "import vx_nn" << std::endl;
	ofsGDF << std::endl;	
	
	for(int i=0; i < net.size(); i++) {
		std::map<std::string, std::string> layer_details = net.find(i)->second;
		std::map<std::string, std::vector<int>> in_out_map = tensorDims.find(i)->second;

		auto&& layer_input = layer_details.find("input")->second;
		auto&& layer_type = layer_details.find("type")->second;
		auto&& layer_output = layer_details.find("output")->second;

		//input dims.
		auto& input_dims = in_out_map.find(layer_input)->second;
		ofsGDF << "data " << layer_input << " =tensor:4{" << input_dims[0] << "," << input_dims[1] << "," << input_dims[2] << "," << input_dims[3] << "},"
			<< "VX_TYPE_FLOAT32, 0" << std::endl; 		
		
		if(i==0) {
			ofsGDF << "read " << layer_input << " input.f32" << std::endl;
		}

		
		//output dims.
		auto& output_dims = in_out_map.find(layer_output)->second;
		//TODO: Generate output dims.

		//TODO: Generate dims of layers and create nodes.
		//if(type == "Conv") {
			
		//}	

		ofsGDF << std::endl;	
	}		
	
	return 0;
}

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
	else if(layer_type == "LRN") {
		
		int lrn_local_size;
		float alpha, beta, bias;

		for(int i=0; i < node_proto.attribute_size(); i++) {
			const onnx::AttributeProto& attribute_proto = node_proto.attribute(i);
			std::string attribute_name = attribute_proto.name();
		
			if(attribute_name == "size") {
				lrn_local_size = attribute_proto.i();
			}
			else if(attribute_name == "alpha") {
				alpha = attribute_proto.f();
			}
			else if(attribute_name == "beta") {
				beta = attribute_proto.f();
			}
			else if(attribute_name == "bias") {
				bias = attribute_proto.f();
			}
		}

		params = std::to_string(lrn_local_size)
			+ " " + std::to_string(alpha)
			+ " " + std::to_string(beta)
			+ " " + std::to_string(bias);

		std::cout << "INFO: The parameters: " << lrn_local_size << " " << alpha << " " << beta << " " << bias << std::endl;
	}
	
	return 0;
}

int parseOnnxGraph(
	const onnx::GraphProto& graph_proto,
	std::map<int, std::map<std::string, std::string>>& net
)
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

	std::cout << "INFO: Extracting the network structure for : " << graph_proto.name() << std::endl;

	for(int i=0; i < graph_proto.node_size(); i++) {
		const onnx::NodeProto node_proto = graph_proto.node(i);
		std::cout << "INFO: Layer is : " << node_proto.op_type() << std::endl;
		std::string params;
		getLayerParams(node_proto, params);	

		std::map<std::string, std::string> layer_details;
		std::string layer_type = node_proto.op_type();
		std::string layer_input = node_proto.input(0);
		std::string layer_output = node_proto.output(0);		
		
		layer_details["type"] = layer_type;
		layer_details["input"] = layer_input;
		layer_details["output"] = layer_output;
		layer_details["params"] = params;
		
		if(node_proto.input_size() > 2) {
			std::string layer_weights = node_proto.input(1);
			layer_details["weights"] = layer_weights;
			std::cout << "Weights added " << std::endl;
		}
		
		if(node_proto.input_size() > 3) {
			std::string layer_bias = node_proto.input(2);
			layer_details["bias"] = layer_bias;
			std::cout << "Bias added " << std::endl;
		}

		net[i] = layer_details;
	}

	return 0;
}

int loadOnnxModelFile(
	onnx::GraphProto& graph_proto,
	const char * fileName,
	std::map<int, std::map<std::string, std::string>>& net
)
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
			graph_proto = model_proto.graph();
			if(parseOnnxGraph(graph_proto, net) < 0) {
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
			" ./onnx_gdf_generator <net.pb> <n> <c> <H> <W>";
	
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
	onnx::GraphProto graph_proto;
	std::map<int, std::map<std::string, std::string>> net;
	if(loadOnnxModelFile(graph_proto, fileName, net) < 0) {
		return -1;
	}
	else {
		std::cout << "INFO: Network structure is extracted successfully." << std::endl;
	}

	//calculate tensor dimensions.
	std::map<int, std::map<std::string, std::vector<int>>> tensorDims;
	if(calculateTensorDims(graph_proto, net, tensorDims) < 0) {
		std::cout << "ERROR: Unable to calculate tensor dims" << std::endl;
	}
	else {
		std::cout << "INFO: Tensor Dim calculation successful" << std::endl;
	}

	//write gdf
	std::ofstream ofsGDF("net.gdf", std::ios::binary);
	writeGDF(ofsGDF, net, tensorDims);
		
	return 0;
}
