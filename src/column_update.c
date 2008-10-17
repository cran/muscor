#include <R.h>
#include <Rmath.h>
#include <math.h>

enum loss_t {LS=1,LOGISTIC=2,TRUNC_LS=3,TRUNC_HUBER=4};

/**
 * @param n  number of data items
 * @param x      feature value array
 * @param x2     precomputed quadratic upper bound term 
 *               (<0 means require computation)
 * @param r      residue array
 * @param w_ptr   weight
 * @param lambda_ridge ridge regularization parameter
 * @param lambda_L1 L1 regularization parameter
 * @param loss_type loss type (1=LS, 2=Logistic, 3=truncated-LS, 4=truncated-Huber)
 * @param eval_obj  whether to evaluate (eval_obj<0) or to update (eval_obj>=0)
 *               if update, up on return:
 *                  r and w_ptr will be modified, eval_obj will be
 *                  approximate regularized cost reduction (quadratic approx)
 *               if evaluate, eval_obj should be the regularized loss reduction
 *                  at the beginning, and up on return:
 *                  r and w_ptr are not modified, eval_obj will be regularized cost reduction
 *                  
 */
static void column_update(int n, double *x, double x2, double *r, double *w_ptr, double lambda_ridge, double lambda_L1, int loss_type, double *eval_obj)
{

  double w=*w_ptr;

  /* form quadratic approximation of loss + ridge-reg  as
   * a dw^2 + 2b dw 
   */
  double a=lambda_ridge+1e-14;
  double b=lambda_ridge*w;
  int i;
  if (x2>=0) {
    /* use pre-computed quadratic upper bound for a */
    a=a+x2;
    switch((enum loss_t)loss_type) {
    case LOGISTIC:
      /* logistic */
      for (i=0; i<n; i++) {
	double ep=exp(r[i]);
	double iep=0.5*ep/(1+ep);
	b += x[i]*iep;
      }
      break;
    case TRUNC_LS:
      /* truncated LS */
      for (i=0; i<n; i++) {
	b += (r[i]<=0)?0: (x[i]*r[i]);
      }
      break;
    case TRUNC_HUBER:
      /* truncated Huber */
      for (i=0; i<n; i++) {
	b += (r[i]<=0)?0: (((r[i]>2)?2:r[i])*x[i]);
      }
      break;
    default:
      /* LS */
      for (i=0; i<n; i++) {
	b += x[i]*r[i];
      }
    }
  }
  else {
    /* compute quadratic term a using current Hessian */
    switch((enum loss_t)loss_type) {
    case LOGISTIC:
      /* logistic */
      for (i=0; i<n; i++) {
	double ep=exp(r[i]);
	double iep=0.5*ep/(1+ep);
	a += x[i]*x[i]*ep*iep;
	b += x[i]*iep;
      }
      break;
    case TRUNC_LS:
      /* truncated LS */
      for (i=0; i<n; i++) {
	a += (r[i]<-0.05)? 0: (x[i]*x[i]);
	b += (r[i]<=0)?0: (x[i]*r[i]);
      }
      break;
    case TRUNC_HUBER:
      /* truncated Huber */
      for (i=0; i<n; i++) {
	a += ((r[i]<-0.05)||(r[i]>1.05))? 0: (x[i]*x[i]);
	b += (r[i]<=0)?0: (((r[i]>2)?2:r[i])*x[i]);
      }
      break;
    default:
      /* LS */
      for (i=0; i<n; i++) {
	a += x[i]*x[i];
	b += x[i]*r[i];
      }
    }
  }

  w -=b/a;
  double dw;
  dw=lambda_L1/(2*a);
  if (w>dw) w-=dw;
  else if (w<-dw) w +=dw;
  else w=0;
  
  dw=w- *w_ptr;

  if ((dw<1e-10) && (dw>-1e-10)) {
    *eval_obj=0;
    return;
  }

  /* whether to update the residue and w */
  int is_update= (*eval_obj<0); 

  if ((!is_update) && (x2<0)) {
    /* evaluate the true loss */
    double myr;
    double loss=0;
    switch((enum loss_t)loss_type) {
    case LOGISTIC:
      /* logistic */
      for (i=0; i<n; i++) {
	myr=r[i]+x[i]*dw;
	loss += log(1+exp(myr));
      }
      break;
    case TRUNC_LS:
      /* truncated LS */
      for (i=0; i<n; i++) {
	myr=r[i]+x[i]*dw;
	loss += (myr<=0)? 0: (myr*myr);
      }
      break;
    case TRUNC_HUBER:
      /* truncated Huber */
      for (i=0; i<n; i++) {
	myr=r[i]+x[i]*dw;
	loss += (myr<=0)? 0: ((myr>2)?(4*myr-4):(myr*myr));
      }
      break;
    default:
      /* LS */
      for (i=0; i<n; i++) {
	myr=r[i]+x[i]*dw;
	loss += myr*myr;
      }
    }
    loss += lambda_L1*(fabs(w)-fabs(w-dw)) + lambda_ridge*(w*w-(w-dw)*(w-dw));
    *eval_obj= *eval_obj-loss;
  }
  else {
    *eval_obj=-(a*dw*dw+2*b*dw + lambda_L1*(fabs(w)-fabs(w-dw)));
  }

  if (is_update) {
    /* update  */
    for (i=0; i<n; i++) {
      r[i] += x[i]*dw;
    }
    *w_ptr=w;
  }
}

/* R interface to update columns (indicated by col_ind)
 * upon return: r will be updated, together with col_ind columns of w
 */
void multi_column_update(int *n_ptr, double *x, double *x2_ptr, double *r, double *w, double *lambda_ridge_ptr, double *lambda_L1_ptr, int *loss_type_ptr, double *eval_obj, int * col_ind, int *ncols_ptr)
{
  int ncols=*ncols_ptr;
  int n=*n_ptr;
  int loss_type=*loss_type_ptr;

  for (int j=0; j<ncols; j++) {
    int c=col_ind[j]-1;
    eval_obj[c]=-1;
    column_update(n, x+n*c, x2_ptr[c], r, w+c, lambda_ridge_ptr[c], lambda_L1_ptr[c], loss_type, eval_obj+c);
  }
}

/* R interface to evaluate loss 
 * upon return: cur_obj_ptr is the current objective value
 *              col_ind columns of eval_obj contains objective reductions
 */
void loss_eval(int *n_ptr, double *x, double *x2_ptr, double *r, double *w, double *lambda_ridge_ptr, double *lambda_L1_ptr, int *loss_type_ptr, double *cur_obj_ptr, double *eval_obj, int * col_ind, int *ncols_ptr) {
  int n= *n_ptr;
  int ncols=*ncols_ptr;
  int loss_type=*loss_type_ptr;

  double loss=0;

  int i;
  switch((enum loss_t)loss_type) {
  case LOGISTIC:
    /* logistic */
    for (i=0; i<n; i++) {
      loss += log(1+exp(r[i]));
    }
    break;
  case TRUNC_LS:
    /* truncated LS */
    for (i=0; i<n; i++) {
      loss += (r[i]<=0)? 0: (r[i]*r[i]);
    }
    break;
  case TRUNC_HUBER:
    /* truncated Huber */
    for (i=0; i<n; i++) {
      loss += (r[i]<=0)? 0: ((r[i]>2)?(4*r[i]-4):(r[i]*r[i]));
    }
    break;
  default:
    /* LS */
    for (i=0; i<n; i++) {
      loss += r[i]*r[i];
    }
  }
  *cur_obj_ptr=loss;


  for (i =0; i<ncols; i++) {
    int c=col_ind[i]-1;

    eval_obj[c]=loss;
    
    column_update(n,x+n*c,x2_ptr[c], r,w+c,lambda_ridge_ptr[c],lambda_L1_ptr[c],
		  loss_type,eval_obj+c);
  }
  return;
}

/**
 * R interaface to initialize x, x2, and r
 * x: can be modified
 * x2: if <0 before call, unchanged (computing Hessian on the fly)
 *     otherwise, set to pre-computed quadratic upperbound
 * r: should be initialized as x*w before the call
 */
void column_init(int *n_ptr, int *d_ptr, double *x, double *x2_ptr, double *y, double *r, int *loss_type_ptr)
{
  int n= *n_ptr;
  int d=*d_ptr;
  int loss_type=*loss_type_ptr;

  double loss=0;

  int i,j;
  double *xp;

  switch((enum loss_t)loss_type) {
  case LOGISTIC:
    for (i=0; i<n; i++) {
      r[i]=-y[i]*r[i];
      loss += log(1+exp(r[i]));
    }
    for (j=0; j<d; j++) {
      xp=x+j*n;
      for (i=0; i<n; i++) {
	xp[i]=-y[i]*xp[i];
      }
    }
    break;
  case TRUNC_LS:
  case TRUNC_HUBER:
    for (i=0; i<n; i++) {
      r[i]=1-y[i]*r[i];
    }
    for (j=0; j<d; j++) {
      xp=x+j*n;
      for (i=0; i<n; i++) {
	xp[i]=-y[i]*xp[i];
      }
    }
    break;
  default:
    /* LS */
    for (i=0; i<n; i++) {
      r[i]=r[i]-y[i];
    }
  }

  /* quadratic upper bound of Hessian */
  for (j=0; j<d; j++) {
    xp=x+j*n;
    if (x2_ptr[j]>=0) {
	for (i=0; i<n; i++) {
	  x2_ptr[j] +=xp[i]*xp[i];
	}
      if (loss_type==LOGISTIC)  x2_ptr[j] /=8;
    }
  }
  return;
}
