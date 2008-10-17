\name{muscor}
\alias{muscor}
\title{
Multi-stage Convex Relaxation Solver for sparse regularized convex risk minimization
}
\description{
Using multi-stage convex relaxation to solve the following problem
with sparse regularization term R(w):
\deqn{\sum_{i=1}^n \mathrm{loss}(x_i^T w,y_i) + \mathrm{lambda.ridge} \|w\|_2^2 + \mathrm{lambda.sparse} R(w)}{sum_i loss(x[i,]\%*\% w,y[i]) + lambda.ridge * w^2 + lambda.sparse * R(w)}}
\usage{
muscor (x,y, model0=NULL, stages=1, precompute.quadratic=TRUE,
        max.iters=100, epsilon=0, verbose=0)
}
\arguments{
\item{x}{
n x d matrix of predictors
}
\item{y}{
response of size n
}
\item{model0}{
The starting muscor model specified in muscor.model(), containing information of loss, regularization, and initial coefficients.
Default is NULL (Use the default muscor model muscor.model()).
}
\item{stages}{
The number of optimization stages.
Default is 1.
}
\item{precompute.quadratic}{
If TRUE, precompute quadratic upperbounds (for the Hessian) to save computation, but yields a less accurate approximation for non-least-squares losses; otherwise, compute on the fly. Should always set to be TRUE for least squares.
Default is TRUE.
}
\item{max.iters}{
Maximum number of iterations until stopping. Each iteration is considered the equivalence of optimizing d columns, with computational cost O(nd).
Default is 100.
}
\item{epsilon}{
Stopping criterion: optimize until the risk reduction is smaller than epsilon for all variables.
Default is 0.
}
\item{verbose}{
Verbose level: the larger the more printouts. Default is 0 (no printouts).
}
}

\value{
A muscor model is returned. It is a list containing the following components:
\item{coefficients}{The computed coefficients.
If intercept=TRUE, it has d+1 components, with the last component being the intercept parameter. Otherwise, it has d components.}
\item{loss.type}{
One of "Least.Squares","Logistic.Regression",
       "Modified.LS","Modified.Huber".
The names can be abbreviated to any unique substring.  Least.Squares is for regression with real-valued y. The other loss functions are for binary classification, assuming y to be {+1,-1} valued, and described in [Tong Zhang (2004)].}
\item{lambda.ridge}{Ridge (L2) regularization parameter. }
\item{lambda.sparse}{
Sparse regularization parameter: it can be a vector of size "stages" to specific different regularization strength for different stages.}
\item{sparse.type}{One of "capped.L1" or "Lp".}
\item{sparse.param}{
If sparse.type="Lp", 
\eqn{R(w)=\|w\|_p^p}, and sparse.param is the p in (0,1].
If sparse.type=capped.L1, \eqn{R(w)=\sum_j \max(|w_j|,g)}; sparse.param=g if it has non-negative value, and g is choen such as no more than |sparse.param| number of |w| is larger than g if it has negative value. }
\item{intercept}{
If TRUE, an intercept is included in the model (and not penalized);
otherwise no intercept is included.}
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
"Some Sharp Performance Bounds for Least Squares Regression with L1 Regularization", Annals of Statistics, to appear.

Tong Zhang (2008)
"Multi-stage Convex Relaxation for Learning with Sparse Regularization",
NIPS'08.
}

\author{Tong Zhang}
\seealso{
muscor.model predict.muscor and opt.L1
}

\examples{
data(simulation)

lambda.ridge=0
lambda.L1=1

x=data.matrix(simulation$x)
y=simulation$y

mod0=muscor.model(lambda.sparse=1)

mod1=muscor(x,y,mod0, max.iters=100, verbose=1)
err=mean((predict(mod1,x)-y)^2)
print(paste("Stage 1 training mean squared error=", err))
err=sum((mod1$coef - simulation$true.coeff)^2)
print(paste("Stage 1 parameter estimation err=", err))

mod1$sparse.param=-5
mod2=muscor(x,y,mod1, max.iters=100, verbose=1)
err=mean((predict(mod2,x)-y)^2)
print(paste("Stage 2 training mean squared error=", err))
err=sum((mod2$coef - simulation$true.coeff)^2)
print(paste("Stage 2 parameter estimation error=",err))

mod2$sparse.param=-5
mod3=muscor(x,y,mod2, max.iters=100, verbose=1)
err=mean((predict(mod3,x)-y)^2)
print(paste("Stage 3 training mean squared error=", err))
err=sum((mod3$coef - simulation$true.coeff)^2)
print(paste("Stage 3 parameter estimation error=",err))

}

\keyword{optimize}
\keyword{models}
\keyword{regression}





