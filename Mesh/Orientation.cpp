#include "pch.h"
#include "Mesh_Processor.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <time.h>

Orientation_Processor::Orientation_Processor(string graph_path,  string python_path,
	string script_name, int coarsen_times, int coarsen_level):c_times(coarsen_times), c_level(coarsen_level),
	Mesh_Processor(graph_path)
{
	init_python(python_path, script_name);

}
Orientation_Processor::~Orientation_Processor()
{
	Py_DECREF(pFunc_Coarsen);

}
int Orientation_Processor::init_numpy()
{
	import_array();

}
void Orientation_Processor::init_python(string python_path, string script_name)
{
	PyObject* pModule = Mesh_Processor::init_python(python_path, script_name);
	init_numpy();
	//天坑！！！！！！！
	//脚本无论有多少返回值，一定要放到一个列表里，即使只有一个！！！！！！
	//编辑 脚本 一定要用 Pycharm 打开，不然会有看不出来的格式错误！！！！！
	//Python中的返回数据一定要显式指定类型为32bit，因为默认是64bit，而
	//C++ 中，即使是64位机器 float 和 int 也是32 位的！！！！！！
	pFunc_Coarsen = PyObject_GetAttrString(pModule, "multi_coarsen");//multi_coarsen
	if (pFunc_Coarsen == nullptr)
		cout << "no pFunc_Coarsen is load" << endl;
	pFunc_Normal = PyObject_GetAttrString(pModule, "normalize");
	if (pFunc_Normal == nullptr)
		cout << "no pFunc_Normal is load" << endl;

}


PyObject* Orientation_Processor::coarsen(int* adj, int pt_num, int init_K)
{
	clock_t start = clock();
	npy_intp Dims[2] = { pt_num, init_K };

	//PyObject* PyArray = PyArray_SimpleNewFromData(2, Dims, NPY_DOUBLE, CArrays);
	PyObject* Adj = PyArray_SimpleNewFromData(2, Dims, NPY_INT, adj);
	PyObject* ArgArray = PyTuple_New(2);
	PyTuple_SetItem(ArgArray, 0, Adj);

	PyObject* array = PyList_New(c_times);
	for (int i = 0; i < c_times; i++)
		PyList_SetItem(array, i, Py_BuildValue("i", c_level));
	PyTuple_SetItem(ArgArray, 1, array);



	//PyTuple_SetItem(ArgArray, 1, Py_BuildValue("i", c_times));
	//PyTuple_SetItem(ArgArray, 2, Py_BuildValue("i", c_level));

	PyObject* FuncOneBack = PyObject_CallObject(pFunc_Coarsen, ArgArray);

	Py_DECREF(Adj);
	Py_DECREF(ArgArray);

	clock_t end = clock();
	cout << "coarsen time: " << end - start << endl;

	return FuncOneBack;
}




static void DeallocateTensor(void* data, std::size_t, void*) {
	//std::free(data);
}




void Orientation_Processor::predict(float* vertice_ori, int* adj, int pt_num, int init_K, float** output)
{
	cout << "inside predict_orientation\n";
	PyObject* vertice_center = normalize(vertice_ori, pt_num, PartID::LD);
	PyArrayObject* vertice_np = (PyArrayObject*)PyList_GetItem(vertice_center, 0);//TODO delete perm

	int output_size;
	float* output_tmp = run_graph((float*)(vertice_np->data), adj, pt_num, init_K, output_size);

	output[0][3] = output[1][3] = output[2][3] = \
		output[3][0] = output[3][1] = output[3][2] = 0;
	output[3][3] = 1;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			output[i][j] = output_tmp[3 * i + j];
		}
	}
	//Py_DECREF(perms_adjs);

}


float* Orientation_Processor::run_graph(float* vertice, int* adj, int pt_num, int init_K, int& out_size) {

	clock_t start = clock();
	PyObject* perms_adjs = coarsen(adj, pt_num, init_K);


	//int* imNumPt = new int(1);
	vector<pair<string, Tensor>> inputs;
	vector<string> input_names;



	const int64_t tensorDims[3] = { 1,pt_num ,3 };
	TF_Tensor* tftensor = TF_NewTensor(TF_FLOAT, tensorDims, 3,
		vertice, ((size_t)pt_num * 3) * sizeof(float),
		DeallocateTensor, NULL);


	Tensor vertice_tensor = TensorCApi::MakeTensor(tftensor->dtype, tftensor->shape, tftensor->buffer);
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
	clock_t start0 = clock();
	Status status = sess->Run(inputs, { "output_node" }, {}, &outputs);
	clock_t end0 = clock();
	cout << "sess->Run time: " << end0 - start0 << endl;

	auto output_c = outputs[0].flat<float>();
	float* output_test = (float*)(output_c.data());
	out_size = output_c.size();
	clock_t end = clock();
	cout << "predict time: " << end - start << endl;


	return output_test;

}
