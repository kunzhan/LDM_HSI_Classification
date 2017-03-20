#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include <algorithm>
#include "LDM.h"
using std::random_shuffle;

typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
    dst = new T[n];
    memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static void print_string_stdout(const char *s)
{
    fputs(s,stdout);
    fflush(stdout);
}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt,...)
{
    char buf[BUFSIZ];
    va_list ap;
    va_start(ap,fmt);
    vsprintf(buf,fmt,ap);
    va_end(ap);
    (*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

static void solve_cd(const problem *prob, double *alpha, double eps)
{
    int num = prob->num;
    int dim = prob->dim;
    double C = prob->C;
    double lambda1 = prob->lambda1;
    double lambda2 = prob->lambda2;
    
    int i, j, s, iter = 0;
    double d, G;
    int max_iter = 1000;
    int *index = new int[num];
    double *beta = new double[num];
    schar *y = new schar[num];
    int active_size = num;
    
    // PG: projected gradient, for shrinking and stopping
    double PG;
    double PGmax_old = INF;
    double PGmin_old = -INF;
    double PGmax_new, PGmin_new;
    
    for(i=0; i<num; i++)
    {
        if(prob->y[i] > 0)
            y[i] = +1;
        else
            y[i] = -1;
        index[i] = i;
    }
    
    //initialize alpha = 0
    for(i=0; i<num; i++)
        beta[i] = 0;
    
    //initialize alpha = (lambda2 / num) invQGY e
    d = lambda2 / num;
    for(i=0; i<num; i++)
    {
        alpha[i] = 0;
        for(j=0; j<num; j++)
            alpha[i] += prob->invQGY[i][j];
        alpha[i] *= d;
    }
    
    while(iter < max_iter)
    {
        PGmax_new = -INF;
        PGmin_new = INF;
        
        for(i=0; i<active_size; i++)
        {
            int j = i+rand()%(active_size-i);
            swap(index[i], index[j]);
        }
        
        for(s=0; s<active_size; s++)
        {
            i = index[s];
            G = 0;
            schar yi = y[i];
            
            // calculate gradient g[i] = y_i G_(i,:) alpha - 1
            for(j=0; j<num; j++)
                G += prob->G[i][j] * alpha[j];
            G *= yi;
            G -= 1;
            
            PG = 0;
            if (beta[i] == 0)
            {
                if (G > PGmax_old)
                {
                    active_size--;
                    swap(index[s], index[active_size]);
                    s--;
                    continue;
                }
                else if (G < 0)
                    PG = G;
            }
            else if (beta[i] == C/num)
            {
                if (G < PGmin_old)
                {
                    active_size--;
                    swap(index[s], index[active_size]);
                    s--;
                    continue;
                }
                else if (G > 0)
                    PG = G;
            }
            else
                PG = G;
            
            PGmax_new = max(PGmax_new, PG);
            PGmin_new = min(PGmin_new, PG);
            
            if(fabs(PG) > 1.0e-12)
            {
                double beta_old = beta[i];
                beta[i] = min(max(beta[i] - G/prob->H[i][i], 0.0), C/num);
                d = beta[i] - beta_old;
                for(int j=0; j<num; j++)
                    alpha[j] += prob->invQGY[j][i] * d; // update alpha
            }
        }
        
        iter++;
        
        if(PGmax_new - PGmin_new <= eps)
        {
            if(active_size == num)
                break;
            else
            {
                active_size = num;
                PGmax_old = INF;
                PGmin_old = -INF;
                continue;
            }
        }
        PGmax_old = PGmax_new;
        PGmin_old = PGmin_new;
        if (PGmax_old <= 0)
            PGmax_old = INF;
        if (PGmin_old >= 0)
            PGmin_old = -INF;
    }
    
    delete [] beta;
    delete [] y;
    delete [] index;
}

class asgd
{
public:
    asgd(int dim, double lambda3, double lambda1, double lambda2, int times, int average, int t0, double eta0=0);
    ~asgd();
    void renorm();
    double get_norm();
    void reset_u();
    void reset_v();
    double hingeLoss(double a, double y);
    double dHingeLoss(double a, double y);
    double get_value(const struct feature_node *x);
    void update_u(const struct feature_node *x, double alpha);
    void update_v(const struct feature_node *x, double alpha);
    void determine_eta0(const struct problem *prob, int sample); // ѡȡeta0
    double evaluate_eta(const struct problem *prob, int sample, double eta);
    void train_one(const struct feature_node *x, const double y, const feature_node *x2, const double y2, double eta, double mu);
    void test_one(const struct feature_node *x, const double y, double *ploss);
    void train_asgd(const struct problem *prob);
    double* get_w();
private:
    int  dim;
    double  lambda3;
    double  lambda1;
    double  lambda2;
    int times;
    int average;
    int t0;
    double  eta0;
    double  mu0;
    double *u;
    double *v;
    double  a;
    double  b;
    double  c;
    int  t;
};

asgd::asgd(int dim, double lambda3, double lambda1, double lambda2, int times, int average, int t0, double eta0)
: dim(dim), lambda3(lambda3), lambda1(lambda1), lambda2(lambda2), times(times), average(average), t0(t0), eta0(0), a(1), b(1), c(1), t(0)
{
    u = new double[dim];
    for(int i = 0; i < dim; i++)
        u[i] = 0;
    if(average == 1)
    {
        v = new double[dim];
        for(int i = 0; i < dim; i++)
            v[i] = 0;
    }
    else
        v = NULL;
}

asgd::~asgd()
{
    delete[] u;
    if(average == 1)
        delete[] v;
}

void asgd::reset_u()
{
    for(int i = 0; i < dim; i++)
        u[i] = 0;
    a = 1;
}

void asgd::reset_v()
{
    for(int i = 0; i < dim; i++)
        v[i] = 0;
}

void asgd::renorm()
{
    if (average == 1)
    {
        if (c != 1.0)
        {
            for (int i =0;i<dim;i++)
            {
                v[i] = (v[i] + b * u[i])/c;
                u[i] /= a;
            }
            c = 1;
            b = 0;
            a = 1;
        }
    }
    else
    {
        if (a != 1.0)
        {
            for (int i =0;i<dim;i++)
                u[i] /= a;
            a = 1;
        }
    }
}

double asgd::get_norm()
{
    double norm = 0;
    for (int i =0;i<dim;i++)
        norm += u[i]*u[i];
    norm /= (a*a);
    return norm;
}

double asgd::hingeLoss(double a, double y)
{
    double z = 1 - a * y;
    if (z < 0)
        return 0;
    return z;
}

double asgd::dHingeLoss(double a, double y)
{
    double z = 1 - a * y;
    if (z < 0)
        return 0;
    return -y;
}

double asgd::get_value(const struct feature_node *x)
{
    double wtx = 0;
    while(x->index!=-1)
    {
        wtx += u[x->index-1] * x->value;
        x++;
    }
    wtx /= a;
    return wtx;
}

void asgd::update_u(const struct feature_node *x, double alpha)
{
    while(x->index!=-1)
    {
        u[x->index-1] += alpha * x->value;
        x++;
    }
}

void asgd::update_v(const struct feature_node *x, double alpha)
{
    while(x->index!=-1)
    {
        v[x->index-1] += alpha * x->value;
        x++;
    }
}

void asgd::determine_eta0(const problem *prob, int sample)
{
    const double factor = 2.0;
    double loEta = 1;
    double loCost = evaluate_eta(prob, sample, loEta);
    double hiEta = loEta * factor;
    double hiCost = evaluate_eta(prob, sample, hiEta);
    if(loCost < hiCost)
        while(loCost < hiCost)
        {
        hiEta = loEta;
        hiCost = loCost;
        loEta = hiEta / factor;
        loCost = evaluate_eta(prob, sample, loEta);
        eta0 = hiEta;
        }
    else if(hiCost < loCost)
        while(hiCost < loCost)
        {
        loEta = hiEta;
        loCost = hiCost;
        hiEta = loEta * factor;
        hiCost = evaluate_eta(prob, sample, hiEta);
        eta0 = loEta;
        }
}

double asgd::evaluate_eta(const problem *prob, int sample, double eta)
{
    feature_node **x = prob->x;
    int num = prob->num;
    double *y = prob->y;
    int *index = new int[sample];
    int *index2 = new int[sample];
    reset_u();
    
    for(int i=0; i<sample; i++)
    {
        index[i] = rand()%num;
        index2[i] = rand()%num;
        double etai = eta / (1 + lambda3 * eta * i);
        train_one(x[index[i]], y[index[i]], x[index2[i]], y[index2[i]], etai, 1.0);
    }
    double ploss = 0;
    for(int i=0; i<sample; i++)
        test_one(x[i], y[i], &ploss);
    ploss /= sample;
    
    double yTXTw = 0;
    double wTXXTw = 0;
    for(int i=0; i<sample; i++)
    {
        double XTwi = get_value(x[index[i]]);
        yTXTw += XTwi * y[index[i]];
        wTXXTw += XTwi * XTwi;
    }
    double cost = ploss + 0.5 * lambda3 * get_norm() + 2 * lambda1 * wTXXTw / sample - 2 * lambda1 * yTXTw * c / (sample * sample) - lambda2 * yTXTw / sample;
    delete [] index;
    delete [] index2;
    reset_u();
    return cost;
}

void asgd::train_one(const feature_node *x, const double y, const feature_node *x2, const double y2, double eta, double mu)
{
    double s = get_value(x);
    a = a / (1 - lambda3 * eta);
    
    double d = - 4 * lambda1 * s + 4 * lambda1 * y * y2 * get_value(x2) + lambda2 * y - dHingeLoss(s, y);
    d *= a * eta;
    if (d != 0)
        update_u(x, d);
    
    if (average == 1)
    {
        if (mu >= 1)
        {
            reset_v();
            b = 1;
            c = a;
        }
        else if (mu < 1)
        {
            c = c / (1 - mu);
            if (d != 0)
                update_v(x, - b * d);
            b = b + mu * c / a;
        }
    }
    if(a > 1e5 || c > 1e5)
        renorm();
}

void asgd::test_one(const feature_node *x, const double y, double *ploss)
{
    *ploss += hingeLoss(get_value(x), y);
}

void asgd::train_asgd(const problem *prob)
{
    feature_node **x = prob->x;
    double *y = prob->y;
    int num = prob->num;
    int *index = new int[num];
    int *index2 = new int[num];
    for(int i=0; i < num; i++)
    {
        index[i] = i;
        index2[i] = i;
    }
    random_shuffle(index, index+num);
    random_shuffle(index2, index2+num);
    for(int i=0; i < num; i++)
    {
        if(average == 1)
        {
            double eta = eta0 / pow(1 + lambda3 * eta0 * t, 0.75);
            double mu = (t <= t0) ? 1.0 : mu0 / (1 + mu0 * (t + 1 - t0));
            train_one(x[index[i]], y[index[i]], x[index2[i]], y[index2[i]], eta, mu);
        }
        else
        {
            double eta = eta0 / (1 + lambda3 * eta * t);
            train_one(x[index[i]], y[index[i]], x[index2[i]], y[index2[i]], eta, 1);
        }
        t++;
    }
    delete [] index;
    delete [] index2;
}

double* asgd::get_w()
{
    renorm();
    if(average == 1)
        return v;
    return u;
}

static void solve_asgd(const problem *prob, int times, int average, double *w)
{
    int dim = prob->dim;
    double C = prob->C;
    double lambda1 = prob->lambda1;
    double lambda2 = prob->lambda2;
    asgd md(dim, 1.0/C, lambda1/C, lambda2/C, times, average, prob->num*2);
    int sample = min(1000, prob->num);
    md.determine_eta0(prob, sample);
    md.reset_u();
    for(int i=0; i<times; i++)
        md.train_asgd(prob);
    memcpy(w, md.get_w(), sizeof(double)*dim);
}

//
// Interface functions
//
model* train(const problem *prob, const parameter *param)
{
    model *model_ = Malloc(model,1);
    model_->param = *param;   
    if(param->solver == CD)
    {
        model_->size = prob->num;
        model_->alpha = Malloc(double, model_->size);
        model_->w = NULL;
        solve_cd(prob, &model_->alpha[0], param->eps);
    }
    else
    {
        model_->size = prob->dim;
        model_->alpha = NULL;
        model_->w = Malloc(double, model_->size);
        solve_asgd(prob, param->times, param->average, &model_->w[0]);
    }    
    return model_;
}

void free_model_content(struct model *model_ptr)
{
    if(model_ptr->alpha != NULL)
        free(model_ptr->alpha);
    if(model_ptr->w != NULL)
        free(model_ptr->w);
}

void free_and_destroy_model(struct model **model_ptr_ptr)
{
    struct model *model_ptr = *model_ptr_ptr;
    if(model_ptr != NULL)
    {
        free_model_content(model_ptr);
        free(model_ptr);
    }
}

void set_print_string_function(void (*print_func)(const char*))
{
    if (print_func == NULL)
        liblinear_print_string = &print_string_stdout;
    else
        liblinear_print_string = print_func;
}

