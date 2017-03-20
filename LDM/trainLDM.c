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
#define INF HUGE_VAL

void print_null(const char *s) {}
void print_string_matlab(const char *s) {mexPrintf(s);}

void exit_with_help()
{
    mexPrintf(
            "Usage: model = trainLDM(label, instance, [C, lambda1, lambda2], 'options');\n"
            "label: a l*1 vector\n"
            "instance: a d*l sparse matrix, one column is an example\n"
            "C, lambda1, lambda2: parameters of LDM\n"
            "options:\n"
            "-s solver_type: set type of solver (default 0)\n"
            "	 0 -- Coordinate Descent (dual)\n"
            "	 1 -- Average Stochastic Gradient Descent (primal)\n"
            "-k kernel_type: set type of kernel function (default 2)\n"
            "	0 -- linear: u'*v\n"
            "	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
            "	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
            "	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
            "-d degree : set degree in kernel function (default 3)\n"
            "-g gamma : set gamma in kernel function (default 1)\n"
            "-c coef0 : set coef0 in kernel function (default 0)\n"
            "-t times : set the times to scan data for ASGD\n"
            );
}

// liblinear arguments
struct parameter param;		// set by parse_command_line
struct problem prob;		// set by read_problem
struct model *model_;
struct feature_node *x_space;

int parse_command_line(int nrhs, const mxArray *prhs[], char *model_file_name)
{
    int i, argc = 1;
    char cmd[CMD_LEN];
    char *argv[CMD_LEN/2];
    void (*print_func)(const char *) = print_string_matlab;	// default printing to matlab display
    
    // default values
    param.solver = CD;
    param.kernel = RBF;
    param.degree = 3;
    param.gamma = 1;
    param.coef0 = 0;
    param.times = 5;
    param.average = 1;
    param.eps = 0.01;
    
	if(nrhs == 3) // no options
		return 0;

    if(nrhs == 4)
    {
        mxGetString(prhs[3], cmd,  mxGetN(prhs[3]) + 1);
        if((argv[argc] = strtok(cmd, " ")) != NULL)
            while((argv[++argc] = strtok(NULL, " ")) != NULL)
                ;
    }
    
    // parse options
    for(i=1;i<argc;i++)
    {
        if(argv[i][0] != '-') break;
        ++i;
        if(i>=argc && argv[i-1][1] != 'q')
            return 1;
        switch(argv[i-1][1])
        {
            case 's':
                param.solver = atoi(argv[i]);
                break;
            case 'k':
                param.kernel = atoi(argv[i]);
                break;
            case 'd':
                param.degree = atoi(argv[i]);
                break;
            case 'g':
                param.gamma = atof(argv[i]);
                break;
            case 'c':
                param.coef0 = atof(argv[i]);
                break;
            case 't':
                param.times = atoi(argv[i]);
                break;
            case 'a':
                param.average = atoi(argv[i]);
                break;
            case 'q':
                print_func = &print_null;
                i--;
                break;
            default:
                mexPrintf("unknown option\n");
                return 1;
        }
    }
    
    set_print_string_function(print_func);
    
    if(param.solver == ASGD)
    {
        param.kernel = LINEAR;
        if(param.average == 1 && param.times < 3)
            param.times = 3;
    }
    return 0;
}

static void fake_answer(mxArray *plhs[])
{
    plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

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
    double tmp = base, ret = 1;
    
    for(t=times; t>0; t/=2)
    {
        if(t%2==1)
            ret*=tmp;
        tmp = tmp * tmp;
    }
    return ret;
}

void compute_G()
{
	int i, j;

    prob.G = Malloc(double*, prob.num);
    
    for(i=0;i<prob.num;i++)
    {
        prob.G[i] = Malloc(double, prob.num);
        
        for(j=0;j<prob.num;j++)
        {
			if (param.kernel == LINEAR)
				prob.G[i][j] = dot(prob.x[i],prob.x[j]);

			if (param.kernel == POLY)
				prob.G[i][j] = powi(param.gamma*dot(prob.x[i],prob.x[j])+param.coef0,param.degree);

			if (param.kernel == RBF)
				prob.G[i][j] = exp(-param.gamma*(dot(prob.x[i],prob.x[i])+dot(prob.x[j],prob.x[j])-2*dot(prob.x[i],prob.x[j])));
                    
			if (param.kernel == SIGMOID)
				prob.G[i][j] = tanh(param.gamma*dot(prob.x[i],prob.x[j])+param.coef0);
        }
    }
}

void compute_Gy()
{
    int i, j;

    prob.Gy = Malloc(double, prob.num);
    for(i=0;i<prob.num;i++)
    {
        prob.Gy[i] = 0;
        for(j=0;j<prob.num;j++)
            prob.Gy[i] += prob.G[i][j] * prob.y[j];
    }
}

int compute_invQGY()
{
	int i, j, k;
    double *Q, *invQ;
    mxArray *invQ_mat;
    mxArray *prhs[1], *plhs[1];

    prhs[0] = mxCreateDoubleMatrix(prob.num, prob.num, mxREAL);
    Q = mxGetPr(prhs[0]);
    for(i=0;i<prob.num;i++)
    {
        for(j=0;j<prob.num;j++)
        {
            Q[i*prob.num+j] = 0;
            for(k=0;k<prob.num;k++)
                Q[i*prob.num+j] += prob.G[i][k] * prob.G[k][j];
            Q[i*prob.num+j] *= prob.num;
            Q[i*prob.num+j] -= prob.Gy[i] * prob.Gy[j];
            Q[i*prob.num+j] *= 4 * prob.lambda1 / (prob.num * prob.num);
            Q[i*prob.num+j] += prob.G[i][j];
        }
    }

	if(mexCallMATLAB(1, plhs, 1, prhs, "pinv"))
	{
		mexPrintf("Error: cannot compute the inverse matrix of Q\n");
		return 1;
	}
    invQ_mat = plhs[0];
    invQ = mxGetPr(invQ_mat);
    mxDestroyArray(prhs[0]);

    prob.invQGY = Malloc(double*, prob.num);
    for(i=0;i<prob.num;i++)
    {
        prob.invQGY[i] = Malloc(double, prob.num);
        for(j=0;j<prob.num;j++)
        {
            prob.invQGY[i][j] = 0;
            for(k=0;k<prob.num;k++)
                prob.invQGY[i][j] += invQ[i*prob.num+k] * prob.G[k][j];
            prob.invQGY[i][j] *= prob.y[j];
        }
    }
	mxDestroyArray(plhs[0]);
    return 0;
}

void compute_H()
{
    int i, j, k;
	
    prob.H = Malloc(double*, prob.num);
    for(i=0;i<prob.num;i++)
    {
        prob.H[i] = Malloc(double, prob.num);
        for(j=0;j<prob.num;j++)
        {
            prob.H[i][j] = 0;
            for(k=0;k<prob.num;k++)
                prob.H[i][j] += prob.G[i][k] * prob.invQGY[k][j];
            prob.H[i][j] *= prob.y[i];
        }
    }
}

int read_problem_sparse(const mxArray *label_in, const mxArray *instance_in, const mxArray *lambdaC_in)
{
    int i, j, k, low, high;
    mwIndex *ir, *jc;
    double *instance, *label, *lambdaC;
    
    lambdaC = mxGetPr(lambdaC_in);
    prob.C = lambdaC[0];
    prob.lambda1 = lambdaC[1];
    prob.lambda2 = lambdaC[2];
    
    label = mxGetPr(label_in);
    
    instance = mxGetPr(instance_in);
    ir = mxGetIr(instance_in);
    jc = mxGetJc(instance_in);
    prob.dim = (int) mxGetM(instance_in);
    prob.num = (int) mxGetN(instance_in);
    prob.y = Malloc(double, prob.num);
    prob.x = Malloc(struct feature_node*, prob.num);
    x_space = Malloc(struct feature_node, (int) mxGetNzmax(instance_in) + prob.num);
    
    j = 0;
    for(i=0;i<prob.num;i++)
    {
        prob.x[i] = &x_space[j];
        prob.y[i] = label[i];
        low = (int) jc[i], high = (int) jc[i+1];
        for(k=low;k<high;k++)
        {
            x_space[j].index = (int) ir[k]+1;
            x_space[j].value = instance[k];
            j++;
        }
        x_space[j++].index = -1;
    }
    
    if(param.solver == CD)
    {       
        compute_G(); // compute G       
        compute_Gy(); // compute G y      
        if(compute_invQGY())
            return 1; // compute Q^{-1} G Y       
        compute_H(); // compute H = Y G Q^{-1} G Y    
		return 0;
    }
	return 0;
}

// Interface function of matlab
// now assume prhs[0]: label prhs[1]: features
void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[] )
{
    if(nrhs > 2 && nrhs < 5) // parameter number must be 3(no option) or 4
    {
        int i, j;
        int err=0;
        
        if(!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1])) {
            mexPrintf("Error: label vector and instance matrix must be double\n");
            fake_answer(plhs);
            return;
        }
        
        if(parse_command_line(nrhs, prhs, NULL))
        {
            exit_with_help();
            fake_answer(plhs);
            return;
        }
              
        if(read_problem_sparse(prhs[0], prhs[1], prhs[2]))
        {
            fake_answer(plhs);
            return;
        }
        
        model_ = train(&prob, &param);
        
        model_to_matlab_structure(plhs, model_);
        free_and_destroy_model(&model_);
        free(prob.y);
        free(prob.x);
        if(param.solver == CD)
        {
            free(prob.Gy);
            for(i = 0; i < prob.num; i++)
            {
				free(prob.G[i]);
                free(prob.invQGY[i]);
                free(prob.H[i]);
            }
			free(prob.G);
            free(prob.invQGY);
            free(prob.H);
        }
        free(x_space);
    }
    else
    {
        exit_with_help();
        fake_answer(plhs);
        return;
    }
}
