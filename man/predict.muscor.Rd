\name{predict.muscor}
\alias{predict.muscor}
\title{
Make predictions from a fitted muscor model
}
\description{
Make predictions from a fitted muscor model.
}
\usage{
predict.muscor(object,newx,...)
}
\arguments{
\item{object}{The computed muscor model.}
\item{newx}{ newx to predict the response}
\item{...}{ additional arguments}
}

\value{
The prediction score.
}

\details{
The Multi-stage Convex Relaxation approach is described in
[Tong Zhang (2008)].
It relaxes the non-convex problem into L1 regularization problems in stages,
and each L1 regularization problem is solved using the function opt.L1().
}
\references{

Tong Zhang (2008)
"Multi-stage Convex Relaxation for Learning with Sparse Regularization",
NIPS'08.
}

\author{Tong Zhang}
\seealso{
muscor and opt.L1
}

\keyword{methods}





