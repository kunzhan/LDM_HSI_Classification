#include <float.h>
#include <stdarg.h>
#include <limits.h>
#include <locale.h>

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "LDM.h"

#include "mex.h"
#include "model_matlab.h"

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

#define CMD_LEN 2048

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

int print_null(const char *s,...) {}
int (*info)(const char *fmt,...);

double dot(struct feature_node *px, struct feature_node *py)
{
    double sum = 0;
    while(px->index != -1 && py->index != -1)
    {
        if(px->index == py->index)
        {
            sum += px->value * py->value;
            ++px;
            ++py;
        }
        else
        {
            if(px->index > py->index)
                ++py;
            else
                ++px;
        }
    }
    return sum;
}

double powi(double base, int times)
{
    int t;
    double tmp = base, ret = 1.0;
    
    for(t=times; t>0; t/=2)
    {
        if(t%2==1) ret*=tmp;
        tmp = tmp * tmp;
    }
    return ret;
}

static void fake_answer(mxArray *plhs[])
{
    plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

void do_predict(mxArray *plhs[], const mxArray *prhs[], struct model *model_)
{
    int i, j, k, low, high;
    int l_train, l_test;
    double *ptr_label, *samples, *ptr_predict_label, *ptr_accuracy, *ptr_dec_values;
    mwIndex *ir, *jc;
    int correct = 0;
    int total = 0;
    struct feature_node *train_space, *test_space;
    struct feature_node **training_instance, **testing_instance;
    
    l_test = (int) mxGetN(prhs[1]); // number of testing instance
    ptr_label = mxGetPr(prhs[0]); // groundtruth
    plhs[0] = mxCreateDoubleMatrix(l_test, 1, mxREAL); // predicted label
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL); // accuracy
    plhs[2] = mxCreateDoubleMatrix(l_test, 1, mxREAL); // predicted value
    ptr_predict_label = mxGetPr(plhs[0]);
    ptr_accuracy = mxGetPr(plhs[1]);
    ptr_dec_values = mxGetPr(plhs[2]);
    
    testing_instance = Malloc(struct feature_node*, l_test);
    samples = mxGetPr(prhs[1]);
    ir = mxGetIr(prhs[1]);
    jc = mxGetJc(prhs[1]);
    test_space = Malloc(struct feature_node, (int) mxGetNzmax(prhs[1]) + l_test);

    j = 0;
    for(i=0;i<l_test;i++)
    {
        testing_instance[i] = &test_space[j];
        low = (int) jc[i], high = (int) jc[i+1];
        for(k=low;k<high;k++)
        {
            test_space[j].index = (int) ir[k]+1;
            test_space[j].value = samples[k];
            j++;
        }
        test_space[j++].index = -1;
    }
    
    if(model_->param.solver == CD)
    {
        l_train = (int) mxGetN(prhs[2]);
        training_instance = Malloc(struct feature_node*, l_train);
        
        samples = mxGetPr(prhs[2]);
        ir = mxGetIr(prhs[2]);
        jc = mxGetJc(prhs[2]);
        train_space = Malloc(struct feature_node, (int) mxGetNzmax(prhs[2]) + l_train);
        j = 0;
        for(i=0;i<l_train;i++)
        {
            training_instance[i] = &train_space[j];
            low = (int) jc[i], high = (int) jc[i+1];
            for(k=low;k<high;k++)
            {
                train_space[j].index = (int) ir[k]+1;
                train_space[j].value = samples[k];
                j++;
            }
            train_space[j++].index = -1;
        }
    }
   
    for(i=0;i<l_test;i++)
    {
        double dec_values = 0;
        
        if(model_->param.solver == CD)
        {       
			if(model_->param.kernel == LINEAR)
                for(j=0;j<l_train;j++)
                    dec_values += dot(testing_instance[i],training_instance[j])*model_->alpha[j];

            if(model_->param.kernel == POLY)
                for(j=0;j<l_train;j++)
                    dec_values += powi(model_->param.gamma*dot(testing_instance[i],training_instance[j])+model_->param.coef0,model_->param.degree)*model_->alpha[j];
            
            if(model_->param.kernel == RBF)
                for(j=0;j<l_train;j++)
                    dec_values += exp(-model_->param.gamma*(dot(testing_instance[i],testing_instance[i])+dot(training_instance[j],training_instance[j])-2*dot(testing_instance[i],training_instance[j])))*model_->alpha[j];
            
            if(model_->param.kernel == SIGMOID)
                for(j=0;j<l_train;j++)
                    dec_values += tanh(model_->param.gamma*dot(testing_instance[i],training_instance[j])+model_->param.coef0)*model_->alpha[j];
        }
        else
        {
            while(testing_instance[i]->index!=-1)
            {
                dec_values += model_->w[testing_instance[i]->index-1] * testing_instance[i]->value;
                testing_instance[i]++;
            }
        }
        
        ptr_dec_values[i] = dec_values;
        if(dec_values > 0)
            ptr_predict_label[i] = 1;
        else
            ptr_predict_label[i] = -1;
        
        if(ptr_predict_label[i] == ptr_label[i])
            ++correct;
        ++total;
    }
    
    info("Accuracy = %g%% (%d/%d)\n", (double) correct/total*100,correct,total);
    
    ptr_accuracy[0] = (double)correct/total*100;
    
    if(model_->param.solver == CD)
    {
        free(training_instance);
        free(train_space);
    }
    free(testing_instance);
    free(test_space);
}

void exit_with_help()
{
    mexPrintf(
            "Usage: [predicted_label, accuracy, decision_values] = predict(testing_label, testing_instance, training_instance, model)\n"
            "Returns:\n"
            "  predicted_label: prediction output vector.\n"
            "  accuracy: accuracy.\n"
            "  decision_values: prediction value vector.\n"
            );
}

void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[] )
{
    struct model *model_;
    char cmd[CMD_LEN];
    info = &mexPrintf;
    
    if(nrhs != 4)
    {
        exit_with_help();
        fake_answer(plhs);
        return;
    }
    
    if(!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]) || !mxIsDouble(prhs[2])) {
        mexPrintf("Error: label vector, testing instance and training instance must be double\n");
        fake_answer(plhs);
        return;
    }
    
    if(mxIsStruct(prhs[3]))
    {
		int i;
        const char *error_msg;
        model_ = Malloc(struct model, 1);
        error_msg = matlab_matrix_to_model(model_, prhs[3]);
        if(error_msg)
        {
            mexPrintf("Error: can't read model: %s\n", error_msg);
            free_and_destroy_model(&model_);
            fake_answer(plhs);
            return;
        }    
        do_predict(plhs, prhs, model_);
		free_and_destroy_model(&model_);
    }
    else
    {
        mexPrintf("model file should be a struct array\n");
        fake_answer(plhs);
    }
    
    return;
}
