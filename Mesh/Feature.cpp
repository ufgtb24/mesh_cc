#include "pch.h"
#include "Mesh_Processor.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <time.h>
Feature_Processor::Feature_Processor(string graph_path, string python_path,
	string script_name):Mesh_Processor(graph_path)

{
	init_python(python_path, script_name);

}
Feature_Processor::~Feature_Processor()
{

}
int Feature_Processor::init_numpy()
{
	import_array();

}
void Feature_Processor::init_python(string python_path, string script_name)
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

	//��ӣ�������������
	//�ű������ж��ٷ���ֵ��һ��Ҫ�ŵ�һ���б����ʹֻ��һ��������������
	//�༭ �ű� һ��Ҫ�� Pycharm �򿪣���Ȼ���п��������ĸ�ʽ���󣡣�������
	//Python�еķ�������һ��Ҫ��ʽָ������Ϊ32bit����ΪĬ����64bit����
	//C++ �У���ʹ��64λ���� float �� int Ҳ��32 λ�ģ�����������

#if defined(AI_DEBUG)

	pFunc_Loss = PyObject_GetAttrString(pModule, "loss_debug");//multi_coarsen
	if (pFunc_Loss == nullptr)
		cout << "no pFuncLoss is load" << endl;
#endif

	pFunc_Coarsen = PyObject_GetAttrString(pModule, "multi_coarsen");//multi_coarsen
	if (pFunc_Coarsen == nullptr)
		cout << "no pFunc_Coarsen is load" << endl;


	pFunc_Normal = PyObject_GetAttrString(pModule, "normalize");
	if (pFunc_Normal == nullptr)
		cout << "no pFunc_Normal is load" << endl;

	
	pFunc_iNormal = PyObject_GetAttrString(pModule, "ivs_normalize");
	if (pFunc_iNormal == nullptr)
		cout << "no pFunc_iNormal is load" << endl;

	
	clock_t end = clock();
	cout << "init_python time: " << end - start << endl;

}

PyObject* Feature_Processor::coarsen(int* adj, int pt_num, int init_K)
{
	npy_intp Dims[2] = { pt_num, init_K };

	//PyObject* PyArray = PyArray_SimpleNewFromData(2, Dims, NPY_DOUBLE, CArrays);
	PyObject* Adj = PyArray_SimpleNewFromData(2, Dims, NPY_INT, adj);
	PyObject* ArgArray = PyTuple_New(3);
	PyTuple_SetItem(ArgArray, 0, Adj);
	PyTuple_SetItem(ArgArray, 1, Py_BuildValue("i", 2));
	PyTuple_SetItem(ArgArray, 2, Py_BuildValue("i", 1));

	PyObject* FuncOneBack = PyObject_CallObject(pFunc_Coarsen, ArgArray);


	return FuncOneBack;
}



PyObject* Feature_Processor::normalize(float* x, int pt_num, int part_id)
{

	npy_intp Dims[2] = { pt_num, 3 };

	//PyObject* PyArray = PyArray_SimpleNewFromData(2, Dims, NPY_DOUBLE, CArrays);
	PyObject* X = PyArray_SimpleNewFromData(2, Dims, NPY_FLOAT, x);
	PyObject* ArgArray = PyTuple_New(2);
	PyTuple_SetItem(ArgArray, 0, X);
	PyTuple_SetItem(ArgArray, 1, Py_BuildValue("i", part_id));

	PyObject* FuncOneBack = PyObject_CallObject(pFunc_Normal, ArgArray);

	return FuncOneBack;
}

PyObject* Feature_Processor::ivs_normalize(float* feature, float* center, int feature_num, int part_id)
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



void Feature_Processor::predict(float* vertice_ori, int* adj, int pt_num,
	int init_K, PartID part_id, float** output)
{

	PyObject* vertice_center = normalize(vertice_ori, pt_num, part_id);
	PyArrayObject* vertice_np = (PyArrayObject*)PyList_GetItem(vertice_center, 0);//TODO delete perm
	PyArrayObject* center_np = (PyArrayObject*)PyList_GetItem(vertice_center, 1);//TODO delete perm

	PyObject* perm_adj_map = coarsen(adj, pt_num, init_K);

	int ouput_size;
	float* feat_local = run_graph((float*)(vertice_np->data), pt_num, perm_adj_map, ouput_size);
	int feat_num = ouput_size / 3;
	cout << "output feature num = " << feat_num << endl;


	PyObject* outputList = ivs_normalize(feat_local, (float*)(center_np->data), feat_num, part_id);

	PyArrayObject* feat_np = (PyArrayObject*)PyList_GetItem(outputList, 0);
	float* feat_world = (float*)(feat_np->data);

	for (int i = 0; i < feat_num; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			output[i][j] = feat_world[3 * i + j];
			//cout<< output[i][j]<<"  ";
		}
		//cout << endl;

	}

}

float* Feature_Processor::run_graph(float* vertice, int pt_num, PyObject* perm_adj_map, int& out_size) {



	//int* imNumPt = new int(1);
	vector<pair<string, Tensor>> inputs;



	const int64_t tensorDims[3] = { 1,pt_num ,3 };
	TF_Tensor* v_tensor = TF_NewTensor(TF_FLOAT, tensorDims, 3,
		vertice, ((size_t)pt_num * 3) * sizeof(float),
		DeallocateTensor, NULL);


	Tensor vertice_tensor = TensorCApi::MakeTensor(v_tensor->dtype, v_tensor->shape, v_tensor->buffer);
	inputs.push_back({ "vertice",vertice_tensor });
	//////////////
	for (int i = 0; i < 2; i++) {
		PyArrayObject* perm = (PyArrayObject*)PyList_GetItem(perm_adj_map, i);//TODO delete perm

		const int64_t perm_dims[1] = { perm->dimensions[0] };

		TF_Tensor* tftensor = TF_NewTensor(TF_INT32, perm_dims, 1,
			(int*)(perm->data), (perm->dimensions[0]) * sizeof(int),
			DeallocateTensor, NULL);

		Tensor perm_tensor = TensorCApi::MakeTensor(tftensor->dtype, tftensor->shape, tftensor->buffer);

		ostringstream ostr;
		ostr << "perm_" << i;
		string node_name = ostr.str();

		inputs.push_back({ node_name,perm_tensor });


	}

	for (int i = 0; i < 3 ; i++) {
		PyArrayObject* adj = (PyArrayObject*)PyList_GetItem(perm_adj_map, 2+i);//TODO delete adj

		const int64_t adj_dims[2] = { adj->dimensions[0],adj->dimensions[1] };

		TF_Tensor* tftensor = TF_NewTensor(TF_INT32, adj_dims, 2,
			(int*)(adj->data), (adj->dimensions[0] * adj->dimensions[1]) * sizeof(int),
			DeallocateTensor, NULL);

		Tensor adj_tensor = TensorCApi::MakeTensor(tftensor->dtype, tftensor->shape, tftensor->buffer);

		ostringstream ostr;
		ostr << "adj_" << i;
		string node_name = ostr.str();

		inputs.push_back({ node_name,adj_tensor });


	}
	for (int i = 0; i < 2; i++) {
		PyArrayObject* map = (PyArrayObject*)PyList_GetItem(perm_adj_map, 5+i);//TODO delete perm

		const int64_t map_dims[1] = { map->dimensions[0] };

		TF_Tensor* tftensor = TF_NewTensor(TF_INT32, map_dims, 1,
			(int*)(map->data), (map->dimensions[0]) * sizeof(int),
			DeallocateTensor, NULL);

		Tensor map_tensor = TensorCApi::MakeTensor(tftensor->dtype, tftensor->shape, tftensor->buffer);

		ostringstream ostr;
		ostr << "poolmap_" << i;
		string node_name = ostr.str();

		inputs.push_back({ node_name,map_tensor });


	}


////////////////
	//const int64_t tensorDims2[2] = { pt_num ,init_K };
	//TF_Tensor* a_tensor = TF_NewTensor(TF_INT32, tensorDims2, 2,
	//	adj, ((size_t)pt_num * init_K) * sizeof(int),
	//	DeallocateTensor, NULL);


	//Tensor adj_tensor = TensorCApi::MakeTensor(a_tensor->dtype, a_tensor->shape, a_tensor->buffer);
	//inputs.push_back({ "adj",adj_tensor });
	//input_names.push_back("adj");


	vector<Tensor> outputs;
	clock_t start0 = clock();
	Status status = sess->Run(inputs, { "output_node" }, {}, &outputs);
	clock_t end0 = clock();
	cout << "sess->Run time: " << end0 - start0 << endl;

	auto output_c = outputs[0].flat<float>();
	float* output_test = (float*)(output_c.data());
	out_size = output_c.size();


	return output_test;

}


#if defined(AI_DEBUG)
float Feature_Processor::get_loss(const char* label_path, int f_num, float* predict)
{
	npy_intp Dims[2] = { f_num, 3 };

	PyObject* Predict = PyArray_SimpleNewFromData(2, Dims, NPY_FLOAT, predict);
	PyObject* ArgArray = PyTuple_New(2);
	PyTuple_SetItem(ArgArray, 0, Predict);
	PyTuple_SetItem(ArgArray, 1, Py_BuildValue("s", label_path));
	PyObject* FuncOneBack = PyObject_CallObject(pFunc_Loss, ArgArray);


	PyArrayObject* loss_np = (PyArrayObject*)PyList_GetItem(FuncOneBack, 0);
	//cout<< *(float*)(loss_np->data)<<endl;
	float loss = *(float*)(loss_np->data);
	return loss;
}

#endif