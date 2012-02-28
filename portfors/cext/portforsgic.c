#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

#define C_ARRAY NPY_ALIGNED | NPY_CONTIGUOUS | NPY_FORCECAST
/*
Copyright (C) 2005-2006 Graham I Cummins
This program is free software; you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation; either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY 
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with 
this program; if not, write to the Free Software Foundation, Inc., 59 Temple 
Place, Suite 330, Boston, MA 02111-1307 USA
*/

static long
index1(double v, double *tab, long n) {
    long i;
    for (i=0;i<n;i++) {
        if (tab[i]==v) {
            return i;
        }
    }
    return -1;
}

static long
index2(double x, double y, double *tab, long n, long sl) {
    long i=-1;
    i = -1;
    for (i=0;i<n;i++) {
        if (tab[i]==x) {
            if (tab[i+sl]==y) {
                return i;
            }
        }
    }
    return -1;
}

static double
valat(PyObject *seq, int ind) {
    long v;
    double x;
    PyObject * p;
    p = PySequence_GetItem(seq, ind);
    v = PyInt_AsLong(p);
    Py_DECREF(p);
    //can't do v = PyInt..(PySeq...(seq, ind)). 
    //The internal PyObject gets increffed, and leaks mem.
    x = (double) v;
    return x;
}

static long
ptable1(PyObject *seq, double *prob, long sl) {
    long n, i, ind;
    double v, p;
    n = 0;
    for (i=0;i<sl;i++) {
        v = valat(seq, i);
        ind = index1(v, prob, n);
        if (ind == -1) {
            prob[n] = v;
            prob[n+sl] = 1.0;
            n++;
        } else {
            prob[ind+sl]++;
        }
    }
    return n;
}

static double
entropy(double *prob, long n, long sl) {
    //hx = -1.0*np.array([iprob[x]*np.log(iprob[x]) for x in np.unique(isp)]).sum()/np.log(2)
    double p;
    double mi = 0.0;
    long i;
    for (i=0;i<n;i++) {
        p = prob[i+sl]/sl;
        mi+= p*log(p);
    }
    return -1.0*mi/log(2);
}

static long
ptable2(PyObject *x, PyObject *y, double *jprob, long sl) {
    long i, n, ind, tsl;
    double xv, yv;
    tsl = 2*sl;
    n = 0;
    for (i=0;i<sl;i++) {
        xv = valat(x, i);
        yv = valat(y, i);
        ind = index2(xv, yv, jprob, n, sl);
        if (ind == -1) {
            jprob[n] = xv;
            jprob[n+sl] = yv;
            jprob[n+tsl] = 1.0;
            n++;
        } else {
            jprob[ind+tsl]++;
        }
    }
    return n;

}

static float min3(float x1, float x2, float x3)
{
    float least;
    least=x1;
    if (x2<least) least=x2;
    if (x3<least) least=x3;
    return least;
}



static float
pf_victorDist(long dl1, npy_float *d1, long dl2, npy_float *d2, float cost)
{
    float dist=0.0;

    float last=1.0;
    float *lasti;
    int i,j;
    lasti=(float *)malloc((dl2+1)*sizeof(float));
    for (i=0;i<dl2+1;i++) {
        lasti[i]=i;
    }
    for (i=1;i<dl1+1; i++) {
        if (i>1) lasti[dl2]=last;
        last=i;
        for (j=1;j<dl2+1;j++) {
            dist=min3(lasti[j]+1, last+1, lasti[j-1]+cost*abs(d1[i-1]-d2[j-1]));
            lasti[j-1]=last;
            last=dist;
        }
    }	
    free(lasti);
    return dist;
}

static PyObject *
gicpf_spikeD(PyObject *self, PyObject *args)
{
    PyObject *idata, *idata2;
    PyArrayObject *data, *data2;
    float distance, cost;
    if (!PyArg_ParseTuple(args, "OOf", &idata, &idata2, &cost))
        return NULL;
    data=PyArray_FROM_OTF(idata, NPY_FLOAT32, C_ARRAY);
    data2=PyArray_FROM_OTF(idata2, NPY_FLOAT32, C_ARRAY);
    if (data == NULL ||  data2== NULL) return NULL;
    if (data->nd != 1)
        {
        return PyErr_Format(PyExc_StandardError,
            "Conv pca: input array must have 1 dimension.");
        goto _fail;	
        }
    distance=pf_victorDist(data->dimensions[0], PyArray_DATA(data),
        data2->dimensions[0],PyArray_DATA(data2), cost);
    Py_XDECREF(data);
    Py_XDECREF(data2);
    return Py_BuildValue("f", distance);
    _fail:
        Py_XDECREF(data);
        Py_XDECREF(data2);			
        return NULL;
}

static PyObject *
gicpf_evttrans(PyObject *self, PyObject *args)
{
    PyObject *e1p, *e2p;
    PyArrayObject *e1, *e2;
    PyListObject *trans;
    float cost, ist, jst;
    int i,j, ni, nj;
    float *tmat;	
    if (!PyArg_ParseTuple(args, "OOf", &e1p, &e2p, &cost))
        return NULL;
    e1=PyArray_FROM_OTF(e1p, NPY_FLOAT32, C_ARRAY);
    e2=PyArray_FROM_OTF(e2p, NPY_FLOAT32, C_ARRAY);
    if (e1 == NULL ||  e2== NULL) return NULL;
    ni=e1->dimensions[0]+1;
    nj=e2->dimensions[0]+1;
    tmat=(float *)malloc((ni*nj)*sizeof(float));
    trans=PyList_New(0);
    for (i=0;i<ni;i++) {
        tmat[i]=i;
    }
    for (j=1;j<nj;j++) {
        tmat[j*ni]=j;
    }
    for (i=1;i<ni; i++) {
        for (j=1;j<nj;j++) {
            ist=*((float *) PyArray_GETPTR1(e1,i-1));
            jst=*((float *) PyArray_GETPTR1(e2,j-1));
            tmat[i+j*ni]=min3(tmat[i-1+j*ni]+1, tmat[i+(j-1)*ni]+1, tmat[i-1+(j-1)*ni]+cost*abs(ist-jst));
        }
    }
    i-=1;
    j-=1;
    PyList_Append(trans,Py_BuildValue("d", tmat[i+j*ni]));
    while (i>0 && j>0) {
        if (tmat[i+ni*j]==tmat[i-1+ni*j]+1) 
            i-=1;
        else if (tmat[i+ni*j]==tmat[i+ni*(j-1)]+1)
            j-=1;
        else {
            i-=1;
            j-=1;
            PyList_Append(trans,Py_BuildValue("ii", i, j));
        }
    }
    free(tmat);
    Py_XDECREF(e1);
    Py_XDECREF(e2);
    return trans;

}



static PyObject *
gicpf_mid(PyObject *self, PyObject *args)
{
    PyObject *x, *y;
    double *xprob, *yprob, *jprob;
    double mi, hx, hy, jpr, pmp, slad;
    long nj, nux, nuy, sl;
    int i;
    if (!PyArg_ParseTuple(args, "OO", &x, &y))
        return NULL;
    sl = (long) PySequence_Length(x);
    slad = (double) sl;
    xprob = calloc(2*sl, sizeof(double));
    yprob = calloc(2*sl, sizeof(double));
    jprob = calloc(3*sl, sizeof(double));
    nux = ptable1(x, xprob, sl);
    nuy = ptable1(y, yprob, sl);
    hy  = entropy(yprob, nuy, sl);
    hx  = entropy(xprob, nux, sl);
    nj  = ptable2(x, y, jprob, sl);
    mi = 0.0;
    for (i=0;i<nj; i++) {
        jpr = jprob[i+2*sl] / slad;
        //jprob[i] is the x value, jprob[i+nj] is the y value, jprob[i+2*nj] is the prob
        //none of these prob tables devide by the seqence length
        pmp = (xprob[ index1(jprob[i] ,xprob, nux)+sl]/slad)* (yprob[index1(jprob[i+sl],yprob,nuy)+sl] / slad);
        //product of marginal probabilities
        mi+= jpr * log(jpr/pmp )/log(2);
    }
    free(xprob);
    free(yprob);
    free(jprob);
    return Py_BuildValue("fff", mi, hx, hy);
}

/* module initialization */

static PyMethodDef PortforsgicMethods[] = {

    {"evttransform",  gicpf_evttrans, METH_VARARGS,
    "convert one spike train to another using victor's distance."},
    {"spikeDistance",  gicpf_spikeD, METH_VARARGS,
    "Calculate the Victor/Purpura Metric Space spike distance"},
        {"mi_direct",  gicpf_mid, METH_VARARGS,
        "Estimate the mutual information between two sequences of integers using the direct method"},
{NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC
initportforsgic(void)
{
    import_array();
    (void) Py_InitModule("portforsgic", PortforsgicMethods);
}


