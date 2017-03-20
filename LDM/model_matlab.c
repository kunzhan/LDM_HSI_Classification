#include <stdlib.h>
#include <string.h>
#include "LDM.h"

#include "mex.h"

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

#define NUM_OF_RETURN_FIELD 4

static const char *field_names[] = {
    "Parameters",
    "size",
    "alpha",
	"w",
};

const char *model_to_matlab_structure(mxArray *plhs[], struct model *model_)
{
    int i, size, out_id = 0;
    double *ptr;
    mxArray *return_model, **rhs;
    
    rhs = (mxArray **)mxMalloc(sizeof(mxArray *)*NUM_OF_RETURN_FIELD);
    
    // Parameters
    // for now, only solver_type is needed
    rhs[out_id] = mxCreateDoubleMatrix(7, 1, mxREAL);
    ptr = mxGetPr(rhs[out_id]);
    ptr[0] = model_->param.solver;
    ptr[1] = model_->param.kernel;
    ptr[2] = model_->param.degree;
    ptr[3] = (double)model_->param.gamma;
    ptr[4] = (double)model_->param.coef0;
    ptr[5] = model_->param.times;
    ptr[6] = (double)model_->param.eps;
    out_id++;
    
	size = (int)model_->size;

    // size
    rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
    ptr = mxGetPr(rhs[out_id]);
    ptr[0] = size;
    out_id++;
    
    if(model_->param.solver == CD)
    {
        // alpha
        rhs[out_id] = mxCreateDoubleMatrix(1, size, mxREAL);
        ptr = mxGetPr(rhs[out_id]);
        for(i = 0; i < size; i++)
            ptr[i]=model_->alpha[i];
        out_id++;
        
        // w
        rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
        out_id++;
    }
    else
    {
        // alpha
        rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
        out_id++;
        
        // w
        rhs[out_id] = mxCreateDoubleMatrix(1, size, mxREAL);
        ptr = mxGetPr(rhs[out_id]);
        for(i = 0; i < size; i++)
            ptr[i]=model_->w[i];
        out_id++;
    }
    
    /* Create a struct matrix contains NUM_OF_RETURN_FIELD fields */
    return_model = mxCreateStructMatrix(1, 1, NUM_OF_RETURN_FIELD, field_names);
    
    /* Fill struct matrix with input arguments */
    for(i = 0; i < NUM_OF_RETURN_FIELD; i++)
        mxSetField(return_model,0,field_names[i],mxDuplicateArray(rhs[i]));
    /* return */
    plhs[0] = return_model;
    mxFree(rhs);
    
    return NULL;
}

const char *matlab_matrix_to_model(struct model *model_, const mxArray *matlab_struct)
{
    int i, num_of_fields;
    double *ptr;
    int id = 0;
    mxArray **rhs;
    
    num_of_fields = mxGetNumberOfFields(matlab_struct);
    rhs = (mxArray **) mxMalloc(sizeof(mxArray *)*num_of_fields);
    
    for(i=0;i<num_of_fields;i++)
        rhs[i] = mxGetFieldByNumber(matlab_struct, 0, i);
    
    // Parameters
    ptr = mxGetPr(rhs[id]);
    model_->param.solver = (int)ptr[0];
    model_->param.kernel = (int)ptr[1];
    model_->param.degree = (int)ptr[2];
    model_->param.gamma = (double)ptr[3];
    model_->param.coef0 = (double)ptr[4];
    model_->param.times = (int)ptr[5];
    model_->param.eps = (double)ptr[6];
    id++;
    
    // size
    ptr = mxGetPr(rhs[id]);
    model_->size = (int)ptr[0];
    id++;
    
    if(model_->param.solver == CD)
    {
        ptr = mxGetPr(rhs[id]);
        model_->alpha=Malloc(double, model_->size);
        for(i = 0; i < model_->size; i++)
            model_->alpha[i]=ptr[i];        
        model_->w = NULL;
    }
    else
    {
        id++;
        model_->alpha = NULL;
        ptr = mxGetPr(rhs[id]);
        model_->w=Malloc(double, model_->size);
        for(i = 0; i < model_->size; i++)
            model_->w[i]=ptr[i];
    }
    mxFree(rhs);
    
    return NULL;
}

