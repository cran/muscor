\name{muscor.model}
\alias{muscor.model}
\title{
Create a muscor model.
}
\description{
Create a muscor model. To be used for muscor().
}

\usage{
muscor.model (coefficients=NULL,
             loss.type=c("Least.Squares","Logistic.Regression",
           	                 "Modified.LS","Modified.Huber"),
             lambda.ridge=0, lambda.sparse=0,
             sparse.type=c("capped.L1","Lp"),
             sparse.param=0,
             intercept=FALSE)
}
\arguments{
\item{coefficients}{The computed coefficients.
If intercept=TRUE, it has d+1 components, with the last component being the intercept parameter. Otherwise, it has d components. Default is NULL.}
\item{loss.type}{
One of "Least.Squares","Logistic.Regression",
       "Modified.LS","Modified.Huber".
The names can be abbreviated to any unique substring.  Least.Squares is for regression with real-valued y. The other loss functions are for binary classification, assuming y to be {+1,-1} valued, and described in [Tong Zhang (2004)].
Default is "Least.Squares".}
\item{lambda.ridge}{Ridge (L2) regularization parameter. Default is zero.}
\item{lambda.sparse}{
Sparse regularization parameter: it can be a vector of size "stages" to specific different regularization strength for different stages.
Default is zero.}
\item{sparse.type}{One of "capped.L1" or "Lp". Default is "capped.L1".}
\item{sparse.param}{
If sparse.type="Lp", 
\eqn{R(w)=\|w\|_p^p}, and sparse.param is the p in (0,1].
If sparse.type=capped.L1, \eqn{R(w)=\sum_j \max(|w_j|,g)}; sparse.param=g if it has positive value, and g is choen such as no more than |sparse.param| number of |w| is larger than g if it has value <= 0. Default is 0.}
\item{intercept}{
If TRUE, an intercept is included in the model (and not penalized);
otherwise no intercept is included. Default is FALSE.}
}

\value{
A muscor model is returned. It is a list containing the same components
as the arguments.
}

\details{
The Multi-stage Convex Relaxation approach is described in
[Tong Zhang (2008)].
It relaxes the non-convex problem into L1 regularization problems in stages,
and each L1 regularization problem is solved using the function opt.L1().
}
\references{
Tong Zhang (2004) 
"Statistical Behavior and Consistency of Classification Methods
 based on Convex Risk Minimization", Annals of Statistics, 32:56--85, 2004.

Tong Zhang (2008)
"Multi-stage Convex Relaxation for Learning with Sparse Regularization",
NIPS'08.
}

\author{Tong Zhang}
\seealso{
muscor predict.muscor and opt.L1
}

\keyword{methods}





