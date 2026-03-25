#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *stochastic_ops(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *v_list_obj = NULL;
    PyObject *py_obj = NULL;
    PyObject *vinvz_obj = NULL;
    PyObject *z_obj = NULL;
    PyObject *u_obj = Py_None;
    PyObject *v_seq = NULL;
    PyArrayObject *py_arr = NULL;
    PyArrayObject *vinvz_arr = NULL;
    PyArrayObject *z_arr = NULL;
    PyArrayObject *u_arr = NULL;
    PyArrayObject *first_v_arr = NULL;
    PyArrayObject **v_arrays = NULL;
    PyArrayObject *kpy_out = NULL;
    PyArrayObject *traces_out = NULL;
    PyObject *result = NULL;
    Py_ssize_t v_count = 0;
    npy_intp n = 0;
    npy_intp s = 0;
    int typenum;

    static char *kwlist[] = {"k_list", "py", "vinvz", "z", "u", NULL};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOO|O", kwlist,
            &v_list_obj, &py_obj, &vinvz_obj, &z_obj, &u_obj)) {
        return NULL;
    }

    v_seq = PySequence_Fast(v_list_obj, "k_list must be a sequence");
    if (v_seq == NULL) {
        return NULL;
    }
    v_count = PySequence_Fast_GET_SIZE(v_seq);
    if (v_count < 1) {
        PyErr_SetString(PyExc_ValueError, "k_list must contain at least one matrix");
        goto fail;
    }

    first_v_arr = (PyArrayObject *)PyArray_FROM_OTF(
        PySequence_Fast_GET_ITEM(v_seq, 0), NPY_NOTYPE, NPY_ARRAY_CARRAY_RO);
    if (first_v_arr == NULL) {
        goto fail;
    }

    typenum = PyArray_TYPE(first_v_arr);
    if (typenum != NPY_FLOAT32 && typenum != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "Only float32 and float64 arrays are supported");
        goto fail;
    }
    if (PyArray_NDIM(first_v_arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "Covariance matrices must be 2-D");
        goto fail;
    }
    n = PyArray_DIM(first_v_arr, 0);
    if (PyArray_DIM(first_v_arr, 1) != n) {
        PyErr_SetString(PyExc_ValueError, "Covariance matrices must be square");
        goto fail;
    }

    py_arr = (PyArrayObject *)PyArray_FROM_OTF(py_obj, typenum, NPY_ARRAY_CARRAY_RO);
    vinvz_arr = (PyArrayObject *)PyArray_FROM_OTF(vinvz_obj, typenum, NPY_ARRAY_CARRAY_RO);
    z_arr = (PyArrayObject *)PyArray_FROM_OTF(z_obj, typenum, NPY_ARRAY_CARRAY_RO);
    if (py_arr == NULL || vinvz_arr == NULL || z_arr == NULL) {
        goto fail;
    }

    if (u_obj != Py_None) {
        u_arr = (PyArrayObject *)PyArray_FROM_OTF(u_obj, typenum, NPY_ARRAY_CARRAY_RO);
        if (u_arr == NULL) {
            goto fail;
        }
    }

    if (PyArray_NDIM(py_arr) != 1 || PyArray_DIM(py_arr, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "py must have shape (n,)");
        goto fail;
    }
    if (PyArray_NDIM(vinvz_arr) != 2 || PyArray_DIM(vinvz_arr, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "vinvz must have shape (n, s)");
        goto fail;
    }
    if (PyArray_NDIM(z_arr) != 2 ||
        PyArray_DIM(z_arr, 0) != n ||
        PyArray_DIM(z_arr, 1) != PyArray_DIM(vinvz_arr, 1)) {
        PyErr_SetString(PyExc_ValueError, "z must have shape (n, s) matching vinvz");
        goto fail;
    }
    if (u_arr != NULL &&
        (PyArray_NDIM(u_arr) != 1 || PyArray_DIM(u_arr, 0) != n)) {
        PyErr_SetString(PyExc_ValueError, "u must have shape (n,)");
        goto fail;
    }
    s = PyArray_DIM(vinvz_arr, 1);

    v_arrays = (PyArrayObject **)PyMem_Calloc((size_t)v_count, sizeof(PyArrayObject *));
    if (v_arrays == NULL) {
        PyErr_NoMemory();
        goto fail;
    }
    v_arrays[0] = first_v_arr;
    first_v_arr = NULL;
    for (Py_ssize_t k = 1; k < v_count; ++k) {
        v_arrays[k] = (PyArrayObject *)PyArray_FROM_OTF(
            PySequence_Fast_GET_ITEM(v_seq, k), typenum, NPY_ARRAY_CARRAY_RO);
        if (v_arrays[k] == NULL) {
            goto fail;
        }
        if (PyArray_NDIM(v_arrays[k]) != 2 ||
            PyArray_DIM(v_arrays[k], 0) != n ||
            PyArray_DIM(v_arrays[k], 1) != n) {
            PyErr_SetString(PyExc_ValueError, "All covariance matrices must have shape (n, n)");
            goto fail;
        }
    }

    npy_intp kpy_dims[2] = {n, (npy_intp)v_count};
    npy_intp trace_dims[1] = {(npy_intp)v_count};
    kpy_out = (PyArrayObject *)PyArray_SimpleNew(2, kpy_dims, typenum);
    traces_out = (PyArrayObject *)PyArray_SimpleNew(1, trace_dims, NPY_FLOAT64);
    if (kpy_out == NULL || traces_out == NULL) {
        goto fail;
    }

    Py_BEGIN_ALLOW_THREADS
    if (typenum == NPY_FLOAT32) {
        const float *py_data = (const float *)PyArray_DATA(py_arr);
        const float *vinvz_data = (const float *)PyArray_DATA(vinvz_arr);
        const float *z_data = (const float *)PyArray_DATA(z_arr);
        float *kpy_data = (float *)PyArray_DATA(kpy_out);
        double *traces_data = (double *)PyArray_DATA(traces_out);

        for (Py_ssize_t k = 0; k < v_count; ++k) {
            const float *v_data = (const float *)PyArray_DATA(v_arrays[k]);
            double trace_sum = 0.0;

            for (npy_intp i = 0; i < n; ++i) {
                double acc = 0.0;
                for (npy_intp j = 0; j < n; ++j) {
                    acc += (double)v_data[i * n + j] * (double)py_data[j];
                }
                kpy_data[i * v_count + k] = (float)acc;
            }

            for (npy_intp i = 0; i < n; ++i) {
                for (npy_intp probe = 0; probe < s; ++probe) {
                    double acc = 0.0;
                    for (npy_intp j = 0; j < n; ++j) {
                        acc += (double)v_data[i * n + j] * (double)vinvz_data[j * s + probe];
                    }
                    trace_sum += acc * (double)z_data[i * s + probe];
                }
            }
            traces_data[k] = trace_sum / (double)s;
        }
    } else {
        const double *py_data = (const double *)PyArray_DATA(py_arr);
        const double *vinvz_data = (const double *)PyArray_DATA(vinvz_arr);
        const double *z_data = (const double *)PyArray_DATA(z_arr);
        double *kpy_data = (double *)PyArray_DATA(kpy_out);
        double *traces_data = (double *)PyArray_DATA(traces_out);

        for (Py_ssize_t k = 0; k < v_count; ++k) {
            const double *v_data = (const double *)PyArray_DATA(v_arrays[k]);
            double trace_sum = 0.0;

            for (npy_intp i = 0; i < n; ++i) {
                double acc = 0.0;
                for (npy_intp j = 0; j < n; ++j) {
                    acc += v_data[i * n + j] * py_data[j];
                }
                kpy_data[i * v_count + k] = acc;
            }

            for (npy_intp i = 0; i < n; ++i) {
                for (npy_intp probe = 0; probe < s; ++probe) {
                    double acc = 0.0;
                    for (npy_intp j = 0; j < n; ++j) {
                        acc += v_data[i * n + j] * vinvz_data[j * s + probe];
                    }
                    trace_sum += acc * z_data[i * s + probe];
                }
            }
            traces_data[k] = trace_sum / (double)s;
        }
    }
    Py_END_ALLOW_THREADS

    result = Py_BuildValue("NNN", (PyObject *)kpy_out, (PyObject *)traces_out, Py_None);
    Py_INCREF(Py_None);
    kpy_out = NULL;
    traces_out = NULL;

fail:
    Py_XDECREF(v_seq);
    Py_XDECREF(py_arr);
    Py_XDECREF(vinvz_arr);
    Py_XDECREF(z_arr);
    Py_XDECREF(u_arr);
    Py_XDECREF(first_v_arr);
    if (v_arrays != NULL) {
        for (Py_ssize_t k = 0; k < v_count; ++k) {
            Py_XDECREF(v_arrays[k]);
        }
        PyMem_Free(v_arrays);
    }
    Py_XDECREF(kpy_out);
    Py_XDECREF(traces_out);
    return result;
}

static PyMethodDef module_methods[] = {
    {"stochastic_ops", (PyCFunction)stochastic_ops, METH_VARARGS | METH_KEYWORDS,
     "Compute KPy and Hutchinson trace terms for stochastic AI-REML."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_reml_accel",
    NULL,
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit__reml_accel(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
