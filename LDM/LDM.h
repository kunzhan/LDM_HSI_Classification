#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H

#ifdef __cplusplus
extern "C" {
#endif

struct feature_node
{
	int index;
	double value;
};

struct problem
{
	int num, dim;
    double C, lambda1, lambda2;
	double *y;
	struct feature_node **x;
	double **G;
	double *Gy;
	double **invQGY;
    double **H;
};

enum { CD, ASGD }; /* solver */
enum { LINEAR, POLY, RBF, SIGMOID }; /* kernel */

struct parameter
{
	int solver;
    int kernel;
	int degree;	/* for poly */
	double gamma;	/* for poly/rbf/sigmoid */
	double coef0;	/* for poly/sigmoid */
    int times;
	int average;
	double eps;
};

struct model
{
	struct parameter param;
	int size;
	double *alpha;
	double *w;
};

struct model* train(const struct problem *prob, const struct parameter *param);
void free_model_content(struct model *model_ptr);
void free_and_destroy_model(struct model **model_ptr_ptr);
void set_print_string_function(void (*print_func) (const char*));

#ifdef __cplusplus
}
#endif

#endif /* _LIBLINEAR_H */

