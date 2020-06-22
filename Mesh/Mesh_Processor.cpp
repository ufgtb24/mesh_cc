#include "pch.h"
#include "Mesh_Processor.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <time.h>
Mesh_Processor::Mesh_Processor(string graph_path, Usage usage, bool use_GPU, string python_path,
	string script_name)

{
	
	init_python(python_path, script_name, usage);

	Status load_graph_status = LoadGraph(graph_path, &sess, use_GPU);
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
int Mesh_Processor::init_numpy() {//初始化 numpy 执行环境，主要是导入包，python2.7用void返回类型，python3.0以上用int返回类型

	import_array();
}

void Mesh_Processor::init_python(string python_path, string script_name,Usage usage)
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

	//天坑！！！！！！！
	//脚本无论有多少返回值，一定要放到一个列表里，即使只有一个！！！！！！
	//编辑 脚本 一定要用 Pycharm 打开，不然会有看不出来的格式错误！！！！！
	//Python中的返回数据一定要显式指定类型为32bit，因为默认是64bit，而
	//C++ 中，即使是64位机器 float 和 int 也是32 位的！！！！！！
	pFunc_Normal = PyObject_GetAttrString(pModule, "normalize");
	if (pFunc_Normal == nullptr)
		cout << "no pFunc_Normal is load" << endl;

	if (usage == Usage::FEAT) {
	   pFunc_iNormal = PyObject_GetAttrString(pModule, "ivs_normalize");
	   if (pFunc_iNormal == nullptr)
		   cout << "no pFunc_iNormal is load" << endl;

	}
	clock_t end = clock();
	cout << "init_python time: " << end - start << endl;

}

Status Mesh_Processor::LoadGraph(const string& graph_file_name,
	unique_ptr<Session>* session,bool use_GPU) {
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



PyObject* Mesh_Processor::normalize(float* x, int pt_num, int part_id)
{
	clock_t start = clock();

	npy_intp Dims[2] = { pt_num, 3 };

	//PyObject* PyArray = PyArray_SimpleNewFromData(2, Dims, NPY_DOUBLE, CArrays);
	PyObject* X = PyArray_SimpleNewFromData(2, Dims, NPY_FLOAT, x);
	PyObject* ArgArray = PyTuple_New(2);
	PyTuple_SetItem(ArgArray, 0, X);
	PyTuple_SetItem(ArgArray, 1, Py_BuildValue("i", part_id));

	PyObject* FuncOneBack = PyObject_CallObject(pFunc_Normal, ArgArray);

	clock_t end = clock();
	cout << "normalize time: " << end - start << endl;

	return FuncOneBack;
}

PyObject* Mesh_Processor::ivs_normalize(float* feature, float* center,int feature_num,int part_id)
{
	clock_t start = clock();

	npy_intp Dims_f[2] = { feature_num, 3 };
	PyObject* F = PyArray_SimpleNewFromData(2, Dims_f, NPY_FLOAT, feature);

	npy_intp Dims_c[1] = { 3 };
	PyObject* C = PyArray_SimpleNewFromData(1, Dims_c, NPY_FLOAT, center);



	PyObject* ArgArray = PyTuple_New(3);
	PyTuple_SetItem(ArgArray, 0, F);
	PyTuple_SetItem(ArgArray, 1, C);
	PyTuple_SetItem(ArgArray, 2, Py_BuildValue("i", part_id));

	PyObject* FuncOneBack = PyObject_CallObject(pFunc_iNormal, ArgArray);
	clock_t end = clock();
	cout << "ivs_normalize time: " << end - start << endl;

	return FuncOneBack;
}

static void DeallocateTensor(void* data, std::size_t, void*) {
	//std::free(data);
}




void Mesh_Processor::predict_orientation(float* vertice_ori, int* adj, int pt_num, int init_K, float** output)
{
	cout << "inside predict_orientation\n";
	PyObject* vertice_center = normalize(vertice_ori, pt_num, 0);
	PyArrayObject* vertice_np = (PyArrayObject*)PyList_GetItem(vertice_center, 0);//TODO delete perm

	int output_size;
	 float* output_tmp = predict((float*)(vertice_np->data), adj, pt_num, init_K, output_size);

	output[0][3] = output[1][3] = output[2][3] = \
		output[3][0] = output[3][1] = output[3][2] = 0;
	output[3][3] = 1;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			output[i][j] = output_tmp[3*i+j];
		}
	}
	//Py_DECREF(perms_adjs);

}

void Mesh_Processor::predict_feature(float* vertice_ori, int* adj, int pt_num, 
	int init_K, PartID part_id, float** output)
{

	PyObject* vertice_center = normalize(vertice_ori, pt_num, part_id);
	PyArrayObject* vertice_np = (PyArrayObject*)PyList_GetItem(vertice_center, 0);//TODO delete perm
	PyArrayObject* center_np = (PyArrayObject*)PyList_GetItem(vertice_center, 1);//TODO delete perm
	
	
	int ouput_size;
	float* feat_local =predict((float*)(vertice_np->data), adj, pt_num, init_K, ouput_size);
	int feat_num = ouput_size / 3;
	cout << "output feature num = " << feat_num << endl;


	PyObject* outputList=ivs_normalize(feat_local, (float*)(center_np->data), feat_num, part_id);
	PyArrayObject* feat_np = (PyArrayObject*)PyList_GetItem(outputList, 0);
	float* feat_world= (float*)(feat_np->data);

	for (int i = 0; i < feat_num; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			output[i][j] = feat_world[3 * i + j];
			//cout<< output[i][j]<<"  ";
		}
		cout << endl;

	}

}

 float* Mesh_Processor::predict(float* vertice, int* adj, int pt_num, int init_K,int& out_size) {



	//int* imNumPt = new int(1);
	vector<pair<string, Tensor>> inputs;
	vector<string> input_names;



	const int64_t tensorDims[3] = { 1,pt_num ,3 };
	TF_Tensor* v_tensor = TF_NewTensor(TF_FLOAT, tensorDims, 3,
		vertice, ((size_t)pt_num * 3) * sizeof(float),
		DeallocateTensor, NULL);


	Tensor vertice_tensor = TensorCApi::MakeTensor(v_tensor->dtype, v_tensor->shape, v_tensor->buffer);
	inputs.push_back({ "vertice",vertice_tensor });
	input_names.push_back("vertice");




	const int64_t tensorDims2[2] = { pt_num ,init_K };
	TF_Tensor* a_tensor = TF_NewTensor(TF_INT32, tensorDims2, 2,
		adj, ((size_t)pt_num * init_K) * sizeof(int),
		DeallocateTensor, NULL);


	Tensor adj_tensor = TensorCApi::MakeTensor(a_tensor->dtype, a_tensor->shape, a_tensor->buffer);
	inputs.push_back({ "adj",adj_tensor });
	input_names.push_back("adj");


	vector<Tensor> outputs;
	clock_t start0 = clock();
	Status status = sess->Run(inputs, { "output_node" }, {}, &outputs);
	clock_t end0 = clock();
	cout << "sess->Run time: " << end0 - start0 << endl;

	auto output_c = outputs[0].flat<float>();
	float*  output_test= (float*)(output_c.data());
	out_size = output_c.size();


	return output_test;

}
