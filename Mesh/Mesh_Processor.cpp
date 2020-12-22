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


PyObject* Mesh_Processor::init_python(string python_path, string script_name)
{
	clock_t start = clock();

	Py_SetPythonHome(GetWC(python_path));
	Py_Initialize();
	init_numpy();
	PyRun_SimpleString("import sys,os");
	//PyRun_SimpleString("sys.path.append('E:\\VS_Projects\\Mesh_Process\\Test')");
	PyRun_SimpleString("sys.path.append('./')");
	PyRun_SimpleString("print(os.getcwd())");
	//pModule = PyImport_ImportModule("math_test");
	PyObject* pModule = PyImport_ImportModule(script_name.c_str());//"coarsen"
	if (pModule == nullptr)
		cout << "no script is load";
	return pModule;

//天坑！！！！！！！
//脚本无论有多少返回值，一定要放到一个列表里，即使只有一个！！！！！！
//编辑 脚本 一定要用 Pycharm 打开，不然会有看不出来的格式错误！！！！！
//Python中的返回数据一定要显式指定类型为32bit，因为默认是64bit，而
//C++ 中，即使是64位机器 float 和 int 也是32 位的！！！！！！
//同样的，C++向 python 传入的值，在python 中需要先转化为 64位，再计算

}

int Mesh_Processor::init_numpy()
{
	import_array();

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

PyObject* Mesh_Processor::normalize(float* x, int pt_num, PartID part_id)
{
	//init_numpy();

	npy_intp Dims[2] = { pt_num, 3 };
	PyObject* X = PyArray_SimpleNewFromData(2, Dims, NPY_FLOAT, x);
	PyObject* ArgArray = PyTuple_New(2);
	PyTuple_SetItem(ArgArray, 0, X);
	PyTuple_SetItem(ArgArray, 1, Py_BuildValue("i", part_id));

	PyObject* FuncOneBack = PyObject_CallObject(pFunc_Normal, ArgArray);

	return FuncOneBack;
}

PyObject* Mesh_Processor::ivs_normalize(float* feature, float* center, int feature_num, PartID part_id)
{

	npy_intp Dims_f[2] = { feature_num, 3 };
	PyObject* F = PyArray_SimpleNewFromData(2, Dims_f, NPY_FLOAT, feature);

	npy_intp Dims_c[1] = { 3 };
	PyObject* C = PyArray_SimpleNewFromData(1, Dims_c, NPY_FLOAT, center);



	PyObject* ArgArray = PyTuple_New(3);
	PyTuple_SetItem(ArgArray, 0, F);
	PyTuple_SetItem(ArgArray, 1, C);
	PyTuple_SetItem(ArgArray, 2, Py_BuildValue("i", part_id));

	PyObject* FuncOneBack = PyObject_CallObject(pFunc_iNormal, ArgArray);

	return FuncOneBack;
}

static void DeallocateTensor(void* data, std::size_t, void*) {
	//std::free(data);
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

