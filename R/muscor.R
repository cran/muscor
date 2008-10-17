
loss.type.convert <- function(loss.type) {
  lt= switch(loss.type,
    Least.Squares = 1,
    Logistic.Regression =2,
    Modified.LS = 3,
    Modified.Huber =4
    )
  return(lt)
}

opt.L1 <-function(x,y,
                  loss.type= c("Least.Squares","Logistic.Regression",
                    "Modified.LS","Modified.Huber"),
                  lambda.ridge=0,lambda.L1=0, 
                  w0=NULL,
                  intercept=FALSE,
                  precompute.quadratic=TRUE,
                  max.iters=100, epsilon=0,
                  verbose=0) {

  loss.type=match.arg(loss.type)
  loss.t=as.integer(loss.type.convert(loss.type))

  eps=as.double(epsilon)


  if (!is.matrix(x)) {
    print("x has to be a matrix");
    return(NULL);
  }
  if (!is.double(x)) {
    x=matrix(as.vector(x,mode="numeric"),nrow=dim(x)[1])
  }

  n=as.integer(dim(x)[1])
  d=as.integer(dim(x)[2])
  
  if ((!is.vector(y)) || (!is.double(y))) {
    y=as.vector(y,mode="numeric")
  }
  if (n != length(y)) {
    print("mismatching dimension of x and y");
    return(NULL);
  }
  
  lambda.ridge=as.vector(lambda.ridge,mode="numeric")
  if (length(lambda.ridge)<d) {
    lambda.ridge=as.vector(rep(lambda.ridge,dim(x)[2]),mode="numeric")
  }
  lambda.L1=as.vector(lambda.L1,mode="numeric")
  if (length(lambda.L1)<d) {
    lambda.L1=as.vector(rep(lambda.L1,dim(x)[2]),mode="numeric")
  }

  if (intercept) {
    d=d+1
    x=cbind(x,rep(1.0,n))
    lambda.ridge=cbind(lambda.ridge,0.0)
    lambda.L1=cbind(lambda.L1,0.0)
  }
  else {
    if (loss.t != 1) {
      # this duplicates the x matrix, which
      # prevents the original argument x being modified
      x=x+0.0
    }
  }

  if (is.null(w0)) {
    w=as.vector(rep(0.0,d),mode="numeric")
  }
  else {
    w=as.vector(w0,mode="numeric")
    if (length(w) != d) {
      print(paste("length of w0 should equal ", d));
      return(NULL)
    }
  }


  if (verbose>0) {
    print(paste("loss=",loss.type, "(",loss.t,")"," data-size=",n,"x",d));
  }

  # initialization
  #
  if (precompute.quadratic==TRUE) {
    x2=as.vector(rep(1e-8,d),mode="numeric")    
  }
  else {
    x2=as.vector(rep(-1.0,d),mode="numeric")    
  }
  # compute residue
  r=x%*%w;
  # initialize: update x, x2, and r
  .C("column_init", n, d, x, x2, y, r, loss.t, DUP=FALSE, PACKAGE="muscor");

  #start Gauss-Seidel iterations
  #
  iters=0;
  col.all=as.vector(1:d,mode="integer")
  iters=0
  col.ind=c()
  while (TRUE) {
    obj.eval=rep(as.double(0.0),d);
    obj.cur=as.double(0.0)
    # compute the current objective value obj.cur
    # and potential reductions at every column in obj.eval
    .C("loss_eval", n, x, x2, r, w, lambda.ridge, lambda.L1,
       loss.t, obj.cur, obj.eval, col.all, d, DUP=FALSE, PACKAGE="muscor");
    obj.cur=obj.cur+ sum(lambda.L1*abs(w))+sum(lambda.ridge*w*w)

    # choose the active set to perform column updates
    mv=max(obj.eval)

    obj.eval[col.ind]=mv
    col.ind=which(obj.eval>=(mv/2-(1e-10)))
    if (length(col.ind)>d/2) col.ind=col.all

    if (verbose>1) {
      print(sprintf("iters=%d ; current-loss=%g ; max-reduction=%g ; active-set-size=%d",as.integer(iters/d),obj.cur,mv,length(col.ind)));
    }

    # stopping criterion satisfied
    if (mv<eps) break;

    mv=max(mv/2,eps)
    cv=mv;
    citers=0;
    while ((cv>=mv) && (citers<d)) {
      # optimize with respect to the active set
      k=length(col.ind)
      iters=iters+k
      citers=citers+k

      # update columns in active set
      # modifies r and active columns of w
      .C("multi_column_update", n, x, x2, r, w, lambda.ridge, lambda.L1,
         loss.t, obj.eval, col.ind, k, DUP=FALSE,
         PACKAGE="muscor");

      # stopping criterion satisfied
      if (iters>=max.iters*d) break;
      # compute the current maximum reduction to check convergence
      cv=max(obj.eval[col.ind])
    }

    # stopping criterion satisfied
    if (iters>=max.iters*d) break;
  }
  return(w);
}


muscor.model <- function (coefficients=NULL,
                          loss.type=c("Least.Squares","Logistic.Regression",
                            "Modified.LS","Modified.Huber"),
                          lambda.ridge=0, lambda.sparse=0,
                          sparse.type=c("capped.L1","Lp"),
                          sparse.param=0,
                          intercept=FALSE)
{
  loss.type=match.arg(loss.type)
  sparse.type=match.arg(sparse.type)
  obj=list(coefficients=coefficients,loss.type=loss.type,
    lambda.ridge=lambda.ridge,lambda.sparse=lambda.sparse,
    sparse.type=sparse.type, sparse.param=sparse.param,
    intercept=intercept)
  class(obj) <- "muscor"
  return(obj)
}
             

predict.muscor <- function(object,newx,...) {
  if (object$intercept) {
    d=length(object$coef)
    scr=newx%*%object$coef[1:(d-1)] + coef[d]
  }
  else {
    scr=newx%*%object$coef
  }
  return(scr)
}

muscor <- function(x,y, model0=NULL,
                   stages=1, precompute.quadratic=TRUE,
                   max.iters=100, epsilon=0,
                   verbose=0)
{
  if (is.null(model0)) {
    model0=muscor.model()
  }
  w=model0$coef;
  if (length(max.iters)==1) {
    max.iters=rep(max.iters,stages)
  }
  if (length(max.iters) != stages) {
    print(paste("length of max.iters should be", stages))
    return(NULL)
  }
  if (length(epsilon)==1) {
    epsilon=rep(epsilon,stages)
  }
  if (length(epsilon) != stages) {
    print(paste("length of epsilon should be", stages))
    return(NULL)
  }

  sparse.type=model0$sparse.type

  sparse.param=model0$sparse.param
  if (length(sparse.param)==1) {
    sparse.param=rep(sparse.param,stages)
  }
  lambda.sparse=model0$lambda.sparse
  if (length(lambda.sparse)==1) {
    lambda.sparse=rep(lambda.sparse,stages)
  }
  
  k=1
  if (is.null(w)) {
    lambda.L1=lambda.sparse[1]
    w=opt.L1(x,y,model0$loss.type, model0$lambda.ridge,lambda.L1,
      w, model0$intercept, precompute.quadratic,max.iters[1],epsilon[1],verbose)
    k=2
  }

  d=dim(x)[2]

  s=k;
  while (s <=stages) {
    if (sparse.type=="Lp") {
      p=sparse.param[s]
      p=max(min(p,1),0.001)
      lambda.L1=p*abs(w[1:d]+1e-10)^(p-1)
    }
    else {
      # capped-L1
      if (sparse.param[s]>0) {
        eps0=sparse.param[s]
        lambda.L1= (abs(w[1:d])<eps0)
      }
      else {
        lambda.L1=rep(1.0,d)
        kk=abs(sparse.param[s])
        if (kk>0) {
          ind=order(abs(w[1:d]),decreasing=TRUE)
          lambda.L1[ind[1:kk]]=0.0
        }
      }
    }
    lambda.L1=lambda.sparse[s] * lambda.L1
    w=opt.L1(x,y,model0$loss.type, model0$lambda.ridge,lambda.L1,
      w, model0$intercept, precompute.quadratic,max.iters[s],epsilon[s],verbose)

    s=s+1;
  }
  
  model=model0
  model$coef=w
  return(model)
}


.onLoad <- function (lib, pkg) {
  library.dynam("muscor", pkg, lib)
}
