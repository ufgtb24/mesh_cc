#include "pch.h"
#include "Mesh_Processor.h"
#include <Python.h>
#include <numpy/arrayobject.h>

Mesh_Processor::Mesh_Processor(string graph_path, string python_path, 
	string script_name, int coarsen_times, int coarsen_level,bool use_GPU):
	c_times(coarsen_times), c_level(coarsen_level)

{

	init_python(python_path, script_name);

	Status load_graph_status = LoadGraph(graph_path, &sess, use_GPU);
	if (!load_graph_status.ok()) {
		LOG(ERROR) << "\n!!!!!load_graph_status!!!!!\n";
		LOG(ERROR) << load_graph_status;
		LOG(ERROR) << "!!!!!!!!!!!!!!!\n";

	}

}
Mesh_Processor::~Mesh_Processor()
{
	Py_DECREF(pFunction);

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
int Mesh_Processor::init_numpy() {//初始化 numpy 执行环境，主要是导入包，python2.7用void返回类型，python3.0以上用int返回类型

	import_array();
}

void Mesh_Processor::init_python(string python_path, string script_name)
{
	Py_SetPythonHome(GetWC(python_path));
	Py_Initialize();
	init_numpy();
	PyRun_SimpleString("import sys,os");
	//PyRun_SimpleString("sys.path.append('E:\\VS_Projects\\Mesh_Process\\Test')");
	PyRun_SimpleString("sys.path.append('./')");
	PyRun_SimpleString("print(os.getcwd())");
	//pModule = PyImport_ImportModule("math_test");
	cout << script_name << endl;
	PyObject* pModule = PyImport_ImportModule(script_name.c_str());//"coarsen"
	if (pModule == nullptr)
		cout << "no script is load";
	//pDict = PyModule_GetDict(pModule);
	pFunction = PyObject_GetAttrString(pModule, "multi_coarsen");//multi_coarsen
	if (pFunction == nullptr)
		cout << "no function is load";
}

Status Mesh_Processor::LoadGraph(const string& graph_file_name,
	unique_ptr<Session>* session,bool use_GPU) {

	GraphDef graph_def;
	Status load_graph_status =
		ReadBinaryProto(Env::Default(), graph_file_name, &graph_def);
	if (!load_graph_status.ok()) {
		return errors::NotFound("Failed to load compute graph at '",
			graph_file_name, "'");
	}

	///////////
	if (!use_GPU) {

		SessionOptions options;
		ConfigProto* config = &options.config;
		// disabled GPU entirely
		(*config->mutable_device_count())["GPU"] = 0;
		// place nodes somewhere
		config->set_allow_soft_placement(true);
		session->reset(NewSession(options));

	}
	/////////
	else {

		SessionOptions session_options;
		_putenv("CUDA_VISIBLE_DEVICES=""");
		session_options.config.mutable_gpu_options()->set_allow_growth(true);
		session->reset(NewSession(session_options));
	}


	Status session_create_status = (*session)->Create(graph_def);
	if (!session_create_status.ok()) {
		cout << "graph not created\n";

		return session_create_status;
	}

	return Status::OK();
}

PyObject* Mesh_Processor::coarsen(int* adj, int pt_num,int init_K)
{

	npy_intp Dims[2] = { pt_num, init_K };

	//PyObject* PyArray = PyArray_SimpleNewFromData(2, Dims, NPY_DOUBLE, CArrays);
	PyObject* Adj = PyArray_SimpleNewFromData(2, Dims, NPY_INT, adj);
	PyObject* ArgArray = PyTuple_New(3);
	PyTuple_SetItem(ArgArray, 0, Adj);
	PyTuple_SetItem(ArgArray, 1, Py_BuildValue("i", c_times));
	PyTuple_SetItem(ArgArray, 2, Py_BuildValue("i", c_level));

	//PyObject* pFunc = PyDict_GetItemString(pDict, "multi_coarsen");

	PyObject* FuncOneBack = PyObject_CallObject(pFunction, ArgArray);

	//Py_DECREF(Adj);
	//Py_DECREF(ArgArray);
	return FuncOneBack;
}

static void DeallocateTensor(void* data, std::size_t, void*) {
	//std::free(data);
}


void Mesh_Processor::predict(float* vertice, int* adj, int pt_num, int init_K, float** output_interface)
{
	PyObject* perms_adjs = coarsen(adj, pt_num, init_K);
	//int* imNumPt = new int(1);
	vector<pair<string, Tensor>> inputs;
	vector<string> input_names;
	int* imNumPt = new int(1);

	const int64_t tensorDims[3] = { 1,pt_num ,3 };
	TF_Tensor* tftensor = TF_NewTensor(TF_FLOAT, tensorDims, 3,
		(float*)vertice, ((size_t)pt_num * 3) * sizeof(float),
		DeallocateTensor, NULL);


	Tensor vertice_tensor=TensorCApi::MakeTensor(tftensor->dtype, tftensor->shape, tftensor->buffer);
	inputs.push_back({ "vertice",vertice_tensor });
	input_names.push_back("vertice");

	int perm_idx = 0;

	for (int i = 0; i < c_times; i++) {
		PyArrayObject* perm = (PyArrayObject*)PyList_GetItem(perms_adjs, i);//TODO delete perm

		const int64_t perm_dims[1] = { perm->dimensions[0] };

		TF_Tensor* tftensor = TF_NewTensor(TF_INT32, perm_dims, 1,
			(int*)(perm->data), (perm->dimensions[0]) * sizeof(int),
			DeallocateTensor, NULL);

		Tensor perm_tensor = TensorCApi::MakeTensor(tftensor->dtype, tftensor->shape, tftensor->buffer);

		ostringstream ostr;
		ostr << "perm_" << perm_idx++;
		string node_name = ostr.str();

		inputs.push_back({ node_name,perm_tensor });
		input_names.push_back(node_name);

	}
	int adj_idx = 0;

	for (int i = c_times; i < 2 * c_times + 1; i++) {
		PyArrayObject* adj = (PyArrayObject*)PyList_GetItem(perms_adjs, i);//TODO delete adj

		const int64_t adj_dims[2] = { adj->dimensions[0],adj->dimensions[1] };

		TF_Tensor* tftensor = TF_NewTensor(TF_INT32, adj_dims, 2,
			(int*)(adj->data), (adj->dimensions[0] * adj->dimensions[1]) * sizeof(int),
			DeallocateTensor, NULL);

		Tensor adj_tensor = TensorCApi::MakeTensor(tftensor->dtype, tftensor->shape, tftensor->buffer);

		ostringstream ostr;
		ostr << "adj_" << adj_idx++;
		string node_name = ostr.str();

		inputs.push_back({ node_name,adj_tensor });
		input_names.push_back(node_name);

	}
	vector<Tensor> outputs;

	Status status = sess->Run(inputs, { "output_node" }, {}, &outputs);

	auto output_c = outputs[0].flat<float>();

	output_interface[0][3] = output_interface[1][3] = output_interface[2][3] = \
		output_interface[3][0] = output_interface[3][1] = output_interface[3][2] = 0;
	output_interface[3][3] = 1;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			output_interface[i][j] = output_c(3*i+j);
		}
	}
	//Py_DECREF(perms_adjs);

}
