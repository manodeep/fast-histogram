#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <sys/time.h>

/* Define docstrings */
static char module_docstring[] = "Fast histogram functioins";
static char _histogram1d_docstring[] = "Compute a 1D histogram";
static char _histogram1d_wrapper_docstring[] = "Compute a 1D histogram (wraps multiple functions)";
static char _histogram2d_docstring[] = "Compute a 2D histogram";

/* Declare the C functions here. */
static PyObject *_histogram1d(PyObject *self, PyObject *args);
static PyObject *_histogram1d_wrapper(PyObject *self, PyObject *args);
static PyObject *_histogram2d(PyObject *self, PyObject *args);

/* Define the methods that will be available on the module. */
static PyMethodDef module_methods[] = {
    {"_histogram1d", _histogram1d, METH_VARARGS, _histogram1d_docstring},
    {"_histogram1d_wrapper", _histogram1d_wrapper, METH_VARARGS, _histogram1d_wrapper_docstring},
    {"_histogram2d", _histogram2d, METH_VARARGS, _histogram2d_docstring},
    {NULL, NULL, 0, NULL}
};

/* This is the function that is called on import. */

#if PY_MAJOR_VERSION >= 3
  #define MOD_ERROR_VAL NULL
  #define MOD_SUCCESS_VAL(val) val
  #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
          static struct PyModuleDef moduledef = { \
            PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
          ob = PyModule_Create(&moduledef);
#else
  #define MOD_ERROR_VAL
  #define MOD_SUCCESS_VAL(val)
  #define MOD_INIT(name) void init##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
          ob = Py_InitModule3(name, methods, doc);
#endif

MOD_INIT(_histogram_core)
{
    PyObject *m;
    MOD_DEF(m, "_histogram_core", module_docstring, module_methods);
    if (m == NULL)
        return MOD_ERROR_VAL;
    import_array();
    return MOD_SUCCESS_VAL(m);
}


static PyObject *_histogram1d(PyObject *self, PyObject *args)
{

    long i, n;
    int ix, nx;
    double xmin, xmax, tx, fnx, normx;
    PyObject *x_obj, *x_array, *count_array;
    npy_intp dims[1];
    double *x;
    long *count;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "Oidd", &x_obj, &nx, &xmin, &xmax)) {
        PyErr_SetString(PyExc_TypeError, "Error parsing input");
        return NULL;
    }

    /* Interpret the input objects as `numpy` arrays. */
    x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an `Exception`. */
    if (x_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
        Py_XDECREF(x_array);
        return NULL;
    }

    /* How many data points are there? */
    n = (long)PyArray_DIM(x_array, 0);

    /* Build the output array */
    dims[0] = nx;
    count_array = PyArray_SimpleNew(1, dims, NPY_INT64);
    if (count_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't build output array");
        Py_DECREF(x_array);
        Py_XDECREF(count_array);
        return NULL;
    }

    PyArray_FILLWBYTE(count_array, 0);

    /* Get pointers to the data as C-types. */

    x = (double*)PyArray_DATA(x_array);
    count = (long *)PyArray_DATA(count_array);

    fnx = nx;
    normx = 1. / (xmax - xmin);

    for(i = 0; i < n; i++) {

      tx = x[i];

      if (tx >= xmin && tx < xmax) {
          ix = (tx - xmin) * normx * fnx;
          count[ix] += 1.;
      }

    }

    /* Clean up. */
    Py_DECREF(x_array);

    return count_array;

}


static int _hist1d_base(double * restrict x, const long n, const double xmin, const double xmax, const long nx, long * restrict count)
{
    long i;
    const double fnx = nx;
    const double normx = 1. / (xmax - xmin);
    for(i = 0; i < n; i++) {
        double tx = x[i];
        if (tx >= xmin && tx < xmax) {
            const long ix = (long) ((tx - xmin) * normx * fnx);
            count[ix]++;
        }
    }
    return EXIT_SUCCESS;
}

static int _hist1d_pointer_ops(double * restrict x, const long n, const double xmin, const double xmax, const long nx, long * restrict count)
{
    long i;
    const double fnx = nx;
    const double normx = 1. / (xmax - xmin);
    for(i = 0; i < n; i++) {
        if (*x >= xmin && *x < xmax) {
            const long ix = (long) ((*x - xmin) * normx * fnx);
            count[ix]++;
        }
        x++;
    }
    return EXIT_SUCCESS;
}

static int _hist1d_no_if(double * restrict x, const long n, const double xmin, const double xmax, const long nx, long * restrict count)
{
    long i;
    const double fnx = nx;
    const double normx = 1. / (xmax - xmin);
    for(i = 0; i < n; i++) {
        double tx = x[i];
        const long cond = (tx >= xmin) && (tx < xmax);
        const long ix = ((long) ((tx - xmin) * normx * fnx))  * cond;
        count[ix] += cond;
    }
    return EXIT_SUCCESS;
}

static int _hist1d_no_if_pointer_ops(double * restrict x, const long n, const double xmin, const double xmax, const long nx, long * restrict count)
{
    long i;
    const double fnx = nx;
    const double normx = 1. / (xmax - xmin);
    for(i = 0; i < n; i++) {
        const long cond = (*x >= xmin) && (*x < xmax);
        const long ix = ((long) ((*x - xmin) * normx * fnx))  * cond;
        count[ix] += cond;
        x++;
    }
    return EXIT_SUCCESS;
}

#define GETCOND(n) const long cond##n = (*(x + n) >= xmin && *(x + n) < xmax) ? 1:0
#define GETIDX(n)  const long ix##n = cond##n ? ((long) ((*(x + n) - xmin) * postfac)):0
#define INCR(n)   count[ix##n] += cond##n

static int _hist1d_no_if_pointer_ops_unroll_2(double * restrict x, const long n, const double xmin, const double xmax, const long nx, long * restrict count)
{
    const long unroll_fac=2;
    long i;
    const double fnx = nx;
    const double normx = 1. / (xmax - xmin);
    const double postfac = fnx * normx;
    
    for(i = 0; i <= (n-unroll_fac); i+=unroll_fac) {
        GETCOND(0);
        GETIDX(0);
        INCR(0);
        
        GETCOND(1);
        GETIDX(1);
        INCR(1);

        x += unroll_fac;
    }

    if(i < n) {
        int ret = _hist1d_no_if_pointer_ops(x, n-i, xmin, xmax, nx, count);
    }


    return EXIT_SUCCESS;
}

static int _hist1d_no_if_pointer_ops_unroll_4(double * restrict x, const long n, const double xmin, const double xmax, const long nx, long * restrict count)
{
    const long unroll_fac=4;
    long i;
    const double fnx = nx;
    const double normx = 1. / (xmax - xmin);
    const double postfac = fnx * normx;
    
    for(i = 0; i <= (n-unroll_fac); i+=unroll_fac) {
        GETCOND(0);
        GETIDX(0);
        INCR(0);
        
        GETCOND(1);
        GETIDX(1);
        INCR(1);

        GETCOND(2);
        GETIDX(2);
        INCR(2);

        GETCOND(3);
        GETIDX(3);
        INCR(3);
        
        x += unroll_fac;
    }

    if(i < n) {
        int ret = _hist1d_no_if_pointer_ops(x, n-i, xmin, xmax, nx, count);
    }


    return EXIT_SUCCESS;
}

#undef GETCOND
#undef GETIDX
#undef GETINCR
#undef INCR

#define ALLOCATEHIST_n(n, nx) long *count##n = calloc(nx, sizeof(*(count##n)))
#define CHECKHIST_AND_RETURN_ERROR(n) {if(count##n == NULL) return -1;}
#define GETCOND_n(n) const long cond##n = (*(x + n) >= xmin && *(x + n) < xmax) 
#define GETIDX_n(n)  const long ix##n = ((long) ((*(x + n) - xmin) * postfac))  * cond##n
#define INCR_n(n)   count##n[ix##n] += cond##n
#define FREEHIST_n(n) free(count##n)

static int _hist1d_no_if_pointer_ops_unroll_2_indp_counters(double * restrict x, const long n, const double xmin, const double xmax, const long nx, long * restrict count)
{
    const long unroll_fac=2;
    long i;
    const double fnx = nx;
    const double normx = 1. / (xmax - xmin);
    const double postfac = normx * fnx;

    ALLOCATEHIST_n(0, nx);
    ALLOCATEHIST_n(1, nx);

    CHECKHIST_AND_RETURN_ERROR(0);
    CHECKHIST_AND_RETURN_ERROR(1);
    
    for(i = 0; i <= (n-unroll_fac); i+=unroll_fac) {
        GETCOND_n(0);
        GETIDX_n(0);
        INCR_n(0);
        
        GETCOND_n(1);
        GETIDX_n(1);
        INCR_n(1);

        x += unroll_fac;
    }

    if(i < n) {
        int ret = _hist1d_no_if_pointer_ops(x, n-i, xmin, xmax, nx, count);
    }

    for(i=0;i<nx;i++) {
        count[i] += count0[i] + count1[i];
    }

    FREEHIST_n(0);
    FREEHIST_n(1);
    
    return EXIT_SUCCESS;
}


static int _hist1d_no_if_pointer_ops_unroll_4_indp_counters(double * restrict x, const long n, const double xmin, const double xmax, const long nx, long * restrict count)
{
    const long unroll_fac=4;
    long i;
    const double fnx = nx;
    const double normx = 1. / (xmax - xmin);
    const double postfac = normx * fnx;
    
    ALLOCATEHIST_n(0, nx);
    ALLOCATEHIST_n(1, nx);
    ALLOCATEHIST_n(2, nx);
    ALLOCATEHIST_n(3, nx);

    CHECKHIST_AND_RETURN_ERROR(0);
    CHECKHIST_AND_RETURN_ERROR(1);
    CHECKHIST_AND_RETURN_ERROR(2);
    CHECKHIST_AND_RETURN_ERROR(3);
    
    for(i = 0; i <= (n-unroll_fac); i+=unroll_fac) {
        GETCOND_n(0);
        GETIDX_n(0);
        INCR_n(0);
        
        GETCOND_n(1);
        GETIDX_n(1);
        INCR_n(1);

        GETCOND_n(2);
        GETIDX_n(2);
        INCR_n(2);

        GETCOND_n(3);
        GETIDX_n(3);
        INCR_n(3);
        
        x += unroll_fac;
    }

    if(i < n) {
        int ret = _hist1d_no_if_pointer_ops(x, n-i, xmin, xmax, nx, count);
    }

    for(i=0;i<nx;i++) {
        count[i] += count0[i] + count1[i] + count2[i] + count3[i];
    }
    FREEHIST_n(0);
    FREEHIST_n(1);
    FREEHIST_n(2);
    FREEHIST_n(3);
    
    return EXIT_SUCCESS;
}


#undef GETCOND_n
#undef GETIDX_n
#undef INCR_n
#undef CHECKHIST_AND_RETURN_ERROR
#undef ALLOCATEHIST_n

#define GETVAL(n)  const double tx##n = *(x + n)
#define UPDATEHIST(n) {                                                 \
        if(tx##n >= xmin && tx##n < xmax) {                             \
            const long ix = (long) ((tx##n - xmin) * normx * fnx);      \
            count[ix]++;                                                \
        }                                                               \
    }

static int _hist1d_base_unroll_2(double * restrict x, const long n, const double xmin, const double xmax, const long nx, long * restrict count)
{
    long i;
    const long unroll_fac=2;
    const double fnx = nx;
    const double normx = 1. / (xmax - xmin);
    for(i = 0; i <= (n-unroll_fac); i+=unroll_fac) {
        GETVAL(0);
        UPDATEHIST(0);

        GETVAL(1);
        UPDATEHIST(1);

        x += unroll_fac;
    }

    if(i < n) {
        int ret = _hist1d_base(x, n-i, xmin, xmax, nx, count);
    }

    return EXIT_SUCCESS;
}

static int _hist1d_base_unroll_4(double * restrict x, const long n, const double xmin, const double xmax, const long nx, long * restrict count)
{
    long i;
    const long unroll_fac=4;
    const double fnx = nx;
    const double normx = 1. / (xmax - xmin);
    for(i = 0; i <= (n-unroll_fac); i+=unroll_fac) {
        GETVAL(0);
        UPDATEHIST(0);

        GETVAL(1);
        UPDATEHIST(1);

        GETVAL(2);
        UPDATEHIST(2);

        GETVAL(3);
        UPDATEHIST(3);

        x+= unroll_fac;        
    }

    if(i < n) {
        int ret = _hist1d_base(x, n-i, xmin, xmax, nx, count);
    }

    return EXIT_SUCCESS;
}

#undef UPDATEHIST
#undef GETVAL



#define ALLOCATEHIST(n, nx) long *count##n = calloc(nx, sizeof(*(count##n)))
#define CHECKHIST_AND_RETURN_ERROR(n) {if(count##n == NULL) return -1;}
#define GETVAL_n(n)  const double tx##n = *(x + n)
#define UPDATEHIST_n(n) {                                               \
        if(tx##n >= xmin && tx##n < xmax) {                             \
            const long ix = (long) ((tx##n - xmin) * postfac);          \
            count##n[ix]++;                                             \
        }                                                               \
    }
#define FREEHIST(n) free(count##n)


static int _hist1d_base_unroll_4_indp_hist(double * restrict x, const long n, const double xmin, const double xmax, const long nx, long * restrict count)
{
    long i;
    const long unroll_fac=4;
    const double fnx = nx;
    const double normx = 1. / (xmax - xmin);
    const double postfac = fnx * normx;
    
    ALLOCATEHIST(0, nx);
    ALLOCATEHIST(1, nx);
    ALLOCATEHIST(2, nx);
    ALLOCATEHIST(3, nx);

    CHECKHIST_AND_RETURN_ERROR(0);
    CHECKHIST_AND_RETURN_ERROR(1);
    CHECKHIST_AND_RETURN_ERROR(2);
    CHECKHIST_AND_RETURN_ERROR(3);

    for(i = 0; i <= (n-unroll_fac); i+=unroll_fac) {
        GETVAL_n(0);
        UPDATEHIST_n(0);

        GETVAL_n(1);
        UPDATEHIST_n(1);

        GETVAL_n(2);
        UPDATEHIST_n(2);

        GETVAL_n(3);
        UPDATEHIST_n(3);

        x += unroll_fac;
    }

    if(i < n) {
        int ret = _hist1d_base(x, n-i, xmin, xmax, nx, count);
    }

    for(i=0;i<nx;i++) {
        count[i] += count0[i] + count1[i] + count2[i] + count3[i];
    }
    
    FREEHIST(0);
    FREEHIST(1);
    FREEHIST(2);
    FREEHIST(3);
    
    return EXIT_SUCCESS;
}

static PyObject *_histogram1d_wrapper(PyObject *self, PyObject *args)
{

    long n;
    long nx;
    double xmin, xmax;
    PyObject *x_obj, *x_array, *count_array;
    npy_intp dims[1];
    double *x;
    long *count;

    int (*allfunctions[]) (double * restrict x, const long n, const double xmin, const double xmax, const long nx, long * restrict count) =
        {_hist1d_base,
         _hist1d_base_unroll_2,
         _hist1d_base_unroll_4,
         _hist1d_base_unroll_4_indp_hist,
         _hist1d_pointer_ops,
         _hist1d_no_if,
         _hist1d_no_if_pointer_ops,
         _hist1d_no_if_pointer_ops_unroll_2,
         _hist1d_no_if_pointer_ops_unroll_2_indp_counters,
         _hist1d_no_if_pointer_ops_unroll_4,
         _hist1d_no_if_pointer_ops_unroll_4_indp_counters
        };
#define MAXLEN 1024    
    const char function_names[][MAXLEN] = {"base",
                                           "base (unroll 2)",
                                           "base (unroll 4)",
                                           "base (unroll 4, indp)",
                                           "pointers",
                                           "No branching",
                                           "No branching, pointers",
                                           "No branching, pointers (unroll 2)",
                                           "No branching, pointers (unroll 2, indp))",
                                           "No branching, pointers (unroll 4)",
                                           "No branching, pointers (unroll 4, indp)"};
                                     
    const int numfunctions = sizeof(allfunctions)/sizeof(void *);
    assert(numfunctions == 9);
    
    
    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "Oldd", &x_obj, &nx, &xmin, &xmax)) {
        PyErr_SetString(PyExc_TypeError, "Error parsing input");
        return NULL;
    }

    /* Interpret the input objects as `numpy` arrays. */
    x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an `Exception`. */
    if (x_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
        Py_XDECREF(x_array);
        return NULL;
    }

    /* How many data points are there? */
    n = (long)PyArray_DIM(x_array, 0);

    /* Build the output array */
    dims[0] = nx;
    count_array = PyArray_SimpleNew(1, dims, NPY_INT64);
    if (count_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't build output array");
        Py_DECREF(x_array);
        Py_XDECREF(count_array);
        return NULL;
    }

    PyArray_FILLWBYTE(count_array, 0);

    /* Get pointers to the data as C-types. */

    x = (double*)PyArray_DATA(x_array);
    count = (long*)PyArray_DATA(count_array);

    long *goodhist = calloc(nx, sizeof(*goodhist));
    assert(goodhist != NULL);

    volatile int ret = _hist1d_base(x, n, xmin, xmax, nx, goodhist);

#define ADD_DIFF_TIME(t0, t1)  (t1.tv_sec - t0.tv_sec) + 1e-6*(t1.tv_usec - t0.tv_usec) 
    
    /* Now call the actual function based on the 'key' passed */
    long i;
    int j, k;

    const int maxtries=10;

    fprintf(stdout,"#################################################################################################################\n");
    fprintf(stdout,"#          FUNCTION                         Npoints (x10^6)     Avg_time           Sigma_time         Best_time  \n");
    fprintf(stdout,"#################################################################################################################\n");
    for(j=0; j<numfunctions;j++) {
        double avg_time=0.0, best_time=1e48, squared_time=0.0;
        for(i=0;i<nx;i++) {
            count[i] = 0;
        }

        int pass=0;
        volatile int ret = (allfunctions[j])(x, n, xmin, xmax, nx, count);
        int failedbin=0;
        for(i=0;i<nx;i++) {
            if(goodhist[i] != count[i]) {
                fprintf(stderr,"Error: In %s> goodhist[%ld] = %ld count[%ld] = %ld\n",
                        function_names[j], i, goodhist[i], i, count[i]);
                failedbin++;
            }
        }
            
        if(failedbin == 0) {
            pass = 1;
        }

        pass=1;
        if(pass == 1) {
            for(k=0;k<maxtries;k++) {
                struct timeval t0;
                gettimeofday(&t0, NULL);
                volatile int ret = (allfunctions[j])(x, n, xmin, xmax, nx, count);
                struct timeval t1;
                gettimeofday(&t1, NULL);
                if(ret != EXIT_SUCCESS) {
                    PyErr_SetString(PyExc_TypeError, "Couldn't compute histogram");
                    Py_DECREF(count_array);
                    return NULL;
                }
                double this_iter_time = ADD_DIFF_TIME(t0, t1);
                avg_time += this_iter_time;
                squared_time += (this_iter_time*this_iter_time);
                best_time = (this_iter_time < best_time) ? this_iter_time:best_time;
            }
            avg_time /= maxtries;
            const double sigma_time = sqrt(squared_time/maxtries - avg_time*avg_time); //sigma_x = sqrt( mean(x^2)  - (mean(x))^2)
            fprintf(stdout, "%-40s   %10d    %14.4lf     %14.4lf     %14.4lf\n", function_names[j], n/1000000, avg_time, sigma_time, best_time);
        } else {
            fprintf(stderr,"Error: Function = %s output did not match the expected correct output...skipping timing tests\n", function_names[j]);
        }
    }

    /* Clean up. */
    Py_DECREF(x_array);

    free(goodhist);
    
    return count_array;

}


static PyObject *_histogram2d(PyObject *self, PyObject *args)
{

    long i, n;
    int ix, iy, nx, ny;
    double xmin, xmax, tx, fnx, normx, ymin, ymax, ty, fny, normy;
    PyObject *x_obj, *y_obj, *x_array, *y_array, *count_array;
    npy_intp dims[2];
    double *x, *y, *count;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOiddidd", &x_obj, &y_obj, &nx, &xmin, &xmax, &ny, &ymin, &ymax)) {
        PyErr_SetString(PyExc_TypeError, "Error parsing input");
        return NULL;
    }

    /* Interpret the input objects as `numpy` arrays. */
    x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    y_array = PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an `Exception`. */
    if (x_array == NULL || y_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        return NULL;
    }

    /* How many data points are there? */
    n = (long)PyArray_DIM(x_array, 0);

    /* Check the dimensions. */
    if (n != (long)PyArray_DIM(y_array, 0)) {
        PyErr_SetString(PyExc_RuntimeError, "Dimension mismatch between x and y");
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        return NULL;
    }

    /* Build the output array */
    dims[0] = nx;
    dims[1] = ny;

    count_array = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (count_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't build output array");
        Py_DECREF(x_array);
        Py_XDECREF(count_array);
        return NULL;
    }

    PyArray_FILLWBYTE(count_array, 0);

    /* Get pointers to the data as C-types. */
    x = (double*)PyArray_DATA(x_array);
    y = (double*)PyArray_DATA(y_array);
    count = (double*)PyArray_DATA(count_array);

    fnx = nx;
    fny = ny;
    normx = 1. / (xmax - xmin);
    normy = 1. / (ymax - ymin);

    for(i = 0; i < n; i++) {

      tx = x[i];
      ty = y[i];

      if (tx >= xmin && tx < xmax && ty >= ymin && ty < ymax) {
          ix = (tx - xmin) * normx * fnx;
          iy = (ty - ymin) * normy * fny;
          count[iy + ny * ix] += 1.;
      }

    }

    /* Clean up. */
    Py_DECREF(x_array);
    Py_DECREF(y_array);

    return count_array;

}
