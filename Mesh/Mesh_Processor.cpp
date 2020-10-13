#include "pch.h"
#include "Mesh_Processor.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <time.h>

Mesh_Processor::Mesh_Processor(string graph_path)

{
	Status load_graph_status = LoadGraph(graph_path, &sess);
	if (!load_graph_status.ok()) {
		LOG(ERROR) << "\n!!!!!load_graph_status!!!!!\n";
		LOG(ERROR) << load_graph_status;
		LOG(ERROR) << "!!!!!!!!!!!!!!!\n";

	}

}
Mesh_Processor::~Mesh_Processor()
{

}
wchar_t* Mesh_Processor::GetWC(string str)
{
	const char* c = str.c_str();
	const size_t cSize = strlen(c) + 1;
	wchar_t* wc = new wchar_t[cSize];
	size_t outSize;
	mbstowcs_s(&outSize, wc, cSize, c, cSize - 1);

	return wc;
}

Status Mesh_Processor::LoadGraph(const string& graph_file_name,
	unique_ptr<Session>* session) {
	clock_t start = clock();
	GraphDef graph_def;
	Status load_graph_status =
		ReadBinaryProto(Env::Default(), graph_file_name, &graph_def);
	if (!load_graph_status.ok()) {
		return errors::NotFound("Failed to load compute graph at '",
			graph_file_name, "'");
	}

	///////////
	//if (!use_GPU) {

	SessionOptions options;
	ConfigProto* config = &options.config;
	// disabled GPU entirely
	(*config->mutable_device_count())["GPU"] = 0;
	// place nodes somewhere
	config->set_allow_soft_placement(true);
	session->reset(NewSession(options));

	//}
	///////////
	//else {

	//	SessionOptions session_options;
	//	_putenv("CUDA_VISIBLE_DEVICES=""");
	//	session_options.config.mutable_gpu_options()->set_allow_growth(true);
	//	session->reset(NewSession(session_options));
	//}


	Status session_create_status = (*session)->Create(graph_def);
	if (!session_create_status.ok()) {
		cout << "graph not created\n";

		return session_create_status;
	}
	clock_t end = clock();
	cout << "LoadGraph time: " << end - start << endl;

	return Status::OK();
}

