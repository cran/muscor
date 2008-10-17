\name{simulation}
\alias{simulation}
\title{Simulated high dimensional regression problem with sparse target}
\description{
 This data is generated according to the simulation experiments described 
 in [Tong Zhang (2008)].
}
\format{
  The list \code{simulation} contains the following components:
  \describe{
    \item{x}{a 100 x 500 matrix}
    \item{y}{a 100 dimensional vector}
    \item{true.coeff}{a 500 dimension vector of the true coefficients}
    \item{true.set}{a 5 dimension vector indicating the true feature set}
  }
}
\details{
 The data contains n=100 training examples, with p=500 dimensions. The true
 regression coefficient vector is sparse, with only five nonzero coefficients.
}
\references{
Tong Zhang (2008) "Adaptive Forward-Backward Greedy Algorithm for Learning Sparse Representations", Rutgers Technical Report (long version). 

Tong Zhang (2008) "Adaptive Forward-Backward Greedy Algorithm for Sparse Learning with Linear Models", NIPS'08 (short version).
}
\keyword{datasets}
