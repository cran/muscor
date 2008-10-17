\name{opt.L1}
\alias{opt.L1}
\title{
One stage convex risk minimization solver with L1 and L2 regularization
}
\description{
Using sparse Gauss-Seidel method to solve convex risk minimization 
problems with L1 and L2 regularization:
\deqn{\sum_{i=1}^n \mathrm{loss}(x_i^T w,y_i) + \mathrm{lambda.ridge} \|w\|_2^2 + \mathrm{lambda.L1} \|w\|_1}{sum_i loss(x[i,]\%*\% w,y[i]) + lambda.ridge * w^2 + lambda.L1 * |w|}}
\usage{
opt.L1 (x,y, loss.type= c("Least.Squares","Logistic.Regression",
                  "Modified.LS","Modified.Huber"),
        lambda.ridge=0,lambda.L1=0, w0=NULL, intercept=FALSE,
        precompute.quadratic=TRUE, max.iters=100, epsilon=0, verbose=0)
}
\arguments{
\item{x}{
n x d matrix of predictors
}
\item{y}{
response of size n
}
\item{loss.type}{
One of "Least.Squares","Logistic.Regression",
       "Modified.LS","Modified.Huber".
The names can be abbreviated to any unique substring.  Least.Squares is for regression with real-valued y. The other loss functions are for binary classification, assuming y to be {+1,-1} valued, and described in [Tong Zhang (2004)].
Default is "Least.Squares".
}
\item{lambda.ridge}{
Ridge (L2) regularization parameter: it can be a vector of size d to specific different regularization strength for different variables.
Default is zero.
}
\item{lambda.L1}{
L1 regularization parameter: it can be a vector of size d to specific different regularization strength for different variables.
Default is zero.
}
\item{w0}{
Initial coefficients for vector w, used as starting values for optimization. If intercept is true, then the vector should be of size d+1, whith the last component being the intercept parameter. Otherwise, it should be a vector of size d.
Default is zero. 
}
\item{intercept}{
If TRUE, an intercept is included in the model (and not penalized);
otherwise no intercept is included. Default is FALSE.
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
A coefficient vector is returned.
If intercept=TRUE, it has d+1 components, with the last component being the intercept parameter.
Otherwise, it has d components.
}
\details{
The algorithm is similar to that described in [Zhang and Oles (2001)], 
with L1 regularization added.

The different loss functions are described in [Tong Zhang (2004)].

The Multi-stage Convex Relaxation approach is described in
[Tong Zhang (2008)].

}
\references{
Tong Zhang and Frank J. Oles
"Text Categorization based on regularized linear classification methods",
Information Retrieval, 4:5--31, 2001.

Tong Zhang (2004) 
"Statistical Behavior and Consistency of Classification Methods
 based on Convex Risk Minimization", Annals of Statistics, 32:56--85, 2004.

Tong Zhang (2008)
"Multi-stage Convex Relaxation for Learning with Sparse Regularization",
NIPS'08.
}

\author{Tong Zhang}
\seealso{
muscor and predict.muscor
}

\examples{
data(simulation)

lambda.ridge=0
lambda.L1=1

x=data.matrix(simulation$x)
y=(as.vector(simulation$y, mode="numeric")<0)*2.0-1.0

w=opt.L1(x,y,"Least",lambda.ridge,lambda.L1, max.iters=100, verbose=1)
err=sum((x\%*\%w)*y<=0)/length(y)
print(paste("Least.Squares training err=", err))

w=opt.L1(x,y,"Logi",lambda.ridge,lambda.L1, max.iters=100, verbose=1)
err=sum((x\%*\%w)*y<=0)/length(y)
print(paste("Logistic.Regression training error=",err))

}

\keyword{optimize}
\keyword{models}
\keyword{regression}





