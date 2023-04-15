# -*- coding: utf-8 -*-
import numpy as np
from time import time
import sys,argparse
from pyspark import SparkContext
from operator import add
from pyspark.mllib.random import RandomRDDs


def swap(x):
    """ Swap the elements of a pair tuple.
    """ 
    return (x[1],x[0])
   
def predict(u,v):
    """ Given a user profile uprof and an item profile vprof, predict the rating given by the user to this item

	Inputs are:
	   -u: user profile, in the form of a numpy array
	   -v: item profile, in the form of a numpy array

	The return value is
        - the inner product <u,v>
    """
    # From the theory of matrix factorization this can be found (U^T*V)
    return np.dot(np.transpose(u),v)

def pred_diff(r,u,v):
    """ Given a rating, a user profile u and an item profile v, compute the difference between the prediction and actual rating

	Inputs are:
	   -r: the rating a user gave to an item
	   -u: user profile, in the form of a numpy array
	   -v: item profile, in the form of a numpy array

	The return value is the difference
        - δ =  <u,v> - r 
    """
    #using the formula - δ =  <u,v> - r and utilising predict function
    p = predict(u, v) - r
    return p

def gradient_u(delta,u,v):
    """ Given a user profile u and an item profile v, and the difference in rating predictions δ, compute the gradient

	      ∇_u l(u,v)   = 2 (<u,v> - r ) v 

             of the square error loss:
         
              l(u,v) = (<u,v> - r)^2

	Inputs are:
	   -δ: the difference  <u,v> - r 
	   -u: user profile, in the form of a numpy array
	   -v: item profile, in the form of a numpy array

	The return value is 
        - The gradient w.r.t. u
    """
    # using this formula ∇_u l(u,v)   = 2 (<u,v> - r ) v to get gradient
    gu = 2 * delta * v
    return gu

def gradient_v(delta,u,v):
    """ Given a user profile u and an item profile v, and the difference in rating predictions δ, compute the gradient

	      ∇_v l(u,v)= 2 (<u,v> - r) u 

        of the square error loss:

              l(u,v) = (<u,v> - r)^2

	Inputs are:
	   -δ: the difference  <u,v> - r 
	   -u: user profile, in the form of a numpy array
	   -v: item profile, in the form of a numpy array

	The return value is 
        - the gradient w.r.t. v  
    """
    # using this ∇_v l(u,v)= 2 (<u,v> - r) u  to get gradient 
    gv = 2 * delta * u
    return gv

def readRatings(f,sparkContext):
    """ Read the ratings from a file and store them in an rdd containing tuples of the form:
		(i,j,rij)

	where i,j are integers and rij is a floating number.

	Inputs are:
	     -f: The name of a file that contains the ratings, in form:
		   i,j,rij 

           per line
	     -sparContext: A Spark context

	The return value is the constructed rdd
    """
    return sparkContext.textFile(f).map(lambda x: tuple(x.split(','))).map(lambda inp: (int(inp[0]),int(inp[1]),float(inp[2])))
    
def generateUserProfiles(R,d,seed,sparkContext,N):
    """ Generate the user profiles from rdd R and store them in an RDD containing tuples of the form 
	    (i,ui)
	where u is a random np.array of dimension d.

	The random uis are generated using normalVectorRDD(), a function in RandomRDDs.
        
	Inputs are:
	     - R: an RDD that contains the ratings in (user, item, rating) form
	     - d: the dimension of the user profiles
	     - seed: a seed to be used for in generating the random vectors
         - sparkContext: a spark context
	     - N: the number of partitions to be used during joins, etc.

	The return value is an RDD containing the user profiles
    """
    
    U = R.map(lambda inp:inp[0]).distinct(numPartitions = N)
    numUsers = U.count()
    randRDD = RandomRDDs.normalVectorRDD(sparkContext, numUsers, d,numPartitions=N, seed=seed)
    U = U.zipWithIndex().map(swap)
    randRDD = randRDD.zipWithIndex().map(swap)
    return U.join(randRDD,numPartitions = N).values()

def generateItemProfiles(R,d,seed,sparkContext,N):
    """ Generate the item profiles from rdd R and store them in an RDD containing tuples of the form 
	    (j,vj)
	where v is a random np.array of dimension d.

	The random uis are generated using normalVectorRDD(), a function in RandomRDDs.
        
	Inputs are:
	     - R: an RDD that contains the ratings in (user, item, rating) form
	     - d: the dimension of the user profiles
	     - seed: a seed to be used for in generating the random vectors
         - sparkContext: a spark context
	     - N: the number of partitions to be used during joins, etc.

	The return value is an RDD containing the item profiles
    """
    # By taking the reference from the generateUserProfiles code mentioned above 
    V = R.map(lambda inp: inp[1]).distinct(numPartitions = N)
    numItems = V.count()
    randRDD = RandomRDDs.normalVectorRDD(sparkContext, numItems, d, numPartitions=N,seed=seed)
    V = V.zipWithIndex().map(swap)
    randRDD = randRDD.zipWithIndex().map(swap)
    return V.join(randRDD, numPartitions=N).values()

def joinAndPredictAll(R,U,V,N):
    """ Receives as inputs the ratings R, the user profiles U, and the items V, and constructs a joined RDD.

    Inputs are:
         - R: an RDD containing tuples of the form (i,j,rij)
         - U: an RDD containing tuples of the form (i,ui)
         - V: an RDD containing tuples of the form (j,vj)
         - N: the number of partitions to be used during joins, etc.

	The output is a joined RDD containing tuples of the form:

	(i,j,δij,ui,vj)

	where 
	  δij = <ui,vj>-rij
        is the prediction difference.
    """
    #converting (i,j,rij) in "j, (i, rij)"" format in rdd
    rdd1 = R.map(lambda x: (x[0],(x[1],x[2]))).partitionBy(N)
    # Join the above rdd with USER profile to get a format as "i, (j, rij, vj)" using map transform
    rdd2 = rdd1.join(U, numPartitions=N).map(lambda x :(x[1][0][0],(x[0],x[1][0][1],x[1][1]))).partitionBy(N)
    # Join the above rdd with Item Profile to get a format like  "i, j, ui, vj, rij" using map transform
    rdd3 = rdd2.join(V, numPartitions=N)
    rdd4 = rdd3.map(lambda x : (x[1][0][0],x[0],x[1][0][1],x[1][0][2],x[1][1]))
    # δij = <ui,vj>-rij is delta and using the above to return in the format as "i,j,δij,ui,vj"
    #used the pred_diff form "i, j, ui, vj, rij" to get δij
    finalrdd =  rdd4.map(lambda x: (x[0], x[1], pred_diff(x[2], x[3], x[4]), x[3], x[4])) 
    return finalrdd

def SE(joinedRDD):
    """ Receives as input a joined RDD and computes the MSE:

        SE(R,U,V) = Σ_{i,j in data} (<ui,vj>-rij)^2 

	The input is
        -joinedRDD: an RDD with tuples of the form (i,j,δij,ui,vj), where δij = <ui,vj> - rij is the prediction difference.

	The output is the SE.
    """
   # using the tuple indexing to calculate (<ui,vj>-rij)^2 from the finalrdd which was "i,j,δij,ui,vj"
    se=joinedRDD.map(lambda x : x[2]**2).sum()
    return se
 
def normSqRDD(profileRDD,param):
    """ Receives as input an RDD of profiles (e.g., U) as well as a parameter (e.g., λ) and computes the square of norms:
        λ Σ_i ||ui||_2^2 	

	The input is:
	  -profileRDD: an RDD of the form (i,u), where i is an index and u is a numpy array
	  -param: a scalar λ>0

	The return value is:
        λ Σ_i ||ui||_2^2 	
    """
    # taking rdd that has U and lamda as param and doing calculations using norm 2 to get λ Σ_i ||ui||_2^2 
    nsrdd = profileRDD.map(lambda x: np.linalg.norm(x[1], 2)**2).sum()*param
    return nsrdd
 

def adaptU(joinedRDD,gamma,lam,N):
    """ Receives as input a joined RDD 
	as well as a gain γ, and regularization parameter λ, and constructs a new RDD of user profiles of the form (i,ui) where

        ui = ui - γ * ∇_ui[RegSE(R,U,V)]

	and

        RegSE(R,U,V) = Σ_{i,j in R} (<ui,vj>-rij)^2 + λ Σ_i ||ui||_2^2 + μ Σ_j ||vj||_2^2		
		
	Inputs are
         -joinedRDD: an RDD with tuples of the form (i,j,δij,ui,vj), where δij = <ui,vj> - rij
         -gamma: the gain γ
         -lam: the regularization parameter λ
         -N: the number of partitions to be used in reduceByKey operations

	The return value is an RDD with tuples of the form (i,ui). The returned rdd contains exactly N partitions.
    """
    
    gsum = lambda y1, y2: (y1[0], y1[1] + y2[1])
    # using the formula  RegSE(R,U,V) = Σ_{i,j in R} (<ui,vj>-rij)^2 + λ Σ_i ||ui||_2^2 + μ Σ_j ||vj||_2^2
    # and tuple indexing from joinedrdd in form (i,j,δij,ui,vj) by using gradient_u function as defined before 
    rdd10= joinedRDD.map(lambda x: (x[0], (x[3], gradient_u(x[2], x[3], x[4])))).reduceByKey(gsum,N)
    # using the formula ui = ui - γ * ∇_ui[RegSE(R,U,V)]
    # use reguralization parameter lambda for user profiles
    finalrdd = rdd10.mapValues(lambda x: x[0] - gamma * (x[1] + 2 * lam * x[0]))
    return finalrdd

def adaptV(joinedRDD,gamma,mu,N):
    """ Receives as input a joined RDD 
	as well as a gain γ, and regularization parameter μ,  and constructs a new RDD of user profiles of the form 
 
        vj = vj - γ *  ∇_vj[RegSE(R,U,V)]

	where 

        RegSE(R,U,V) = Σ_{i,j in R} (<ui,vj>-rij)^2 + λ Σ_i ||ui||_2^2 + μ Σ_j ||vj||_2^2		
		
	Inputs are
         -joinedRDD: an RDD with tuples of the form (i,j,δij,ui,vj), where δij = <ui,vj> - rij
         -gamma: the gain γ
         -mu: the regularization parameter μ
         -N: the number of partitions to be used in reduceByKey operations

	The return value  is an RDD with tuples of the form (j,vj). The returned rdd contains exactly N partitions.
    """
    
    gsum = lambda y1, y2: (y1[0], y1[1] + y2[1])
    # using the formula  RegSE(R,U,V) = Σ_{i,j in R} (<ui,vj>-rij)^2 + λ Σ_i ||ui||_2^2 + μ Σ_j ||vj||_2^2
    # and tuple indexing from joinedrdd in form (i,j,δij,ui,vj) by using gradient_u function as defined before 
    rdd11=joinedRDD.map(lambda x: (x[1], (x[4], gradient_v(x[2], x[3], x[4])))).reduceByKey(gsum,N)
    # using the formula vj = vj - γ *  ∇_vj[RegSE(R,U,V)]
    # use reguralization parameter mu for item profiles
    final1rdd = rdd11.mapValues(lambda x: x[0] - gamma * (x[1] + 2 * mu * x[0]))
    return final1rdd


if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Parallele Matrix Factorization.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data',help = 'Directory containing folds. The folds should be named fold0, fold1, ..., foldK.')
    parser.add_argument('folds',type = int,help = 'Number of folds')
    parser.add_argument('--gain',default=0.001,type=float,help ="Gain")
    parser.add_argument('--power',default=0.2,type=float,help ="Gain Exponent")
    parser.add_argument('--epsilon',default=1.e-99,type=float,help ="Desired objective accuracy")
    parser.add_argument('--lam',default=0,type=float,help ="Regularization parameter for user features")
    parser.add_argument('--mu',default=0,type=float,help ="Regularization parameter for item features")
    parser.add_argument('--d',default=10,type=int,help ="Number of latent features")
    parser.add_argument('--outputfile',help = 'Output file')
    parser.add_argument('--maxiter',default=20,type=int, help='Maximum number of iterations')
    parser.add_argument('--N',default=20,type=int, help='Parallelization Level')
    parser.add_argument('--seed',default=1234567,type=int, help='Seed used in random number generator')
    parser.add_argument('--output',default=None, help='If not None, cross validation is skipped, and U,V are trained over entire dataset and store it in files output_U and output_V')

    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
    parser.set_defaults(verbose=False)
 
    args = parser.parse_args()

    sc = SparkContext(appName='Parallel MF')
    
    if not args.verbose :
        sc.setLogLevel("ERROR")        

    folds = {}

    if args.output is None:
        for k in range(args.folds):
            folds[k] = readRatings(args.data+"/fold"+str(k),sc)
    else:
        folds[0] = readRatings(args.data,sc)
    
    
    cross_val_rmses = []
    for k in folds:
        train_folds = [folds[j] for j in folds if j is not k]

        if len(train_folds)>0:
            train = train_folds[0]
            for fold in  train_folds[1:]:
                train=train.union(fold)
            train.repartition(args.N).cache()
            test = folds[k].repartition(args.N).cache()
            Mtrain = train.count()
            Mtest = test.count()
            
            print("Initiating fold %d with %d train samples and %d test samples" % (k,Mtrain,Mtest))
        else:
            train = folds[k].repartition(args.N).cache()
            test = train
            Mtrain = train.count()
            Mtest = test.count()
            print("Running single training over training set with %d train samples. Test RMSE computes RMSE on training set" % Mtrain )
            
        i = 0
        change = 1.e99
        obj = 1.e99
        #rd.seed(args.seed)

        #Generate user profiles
        U = generateUserProfiles(train,args.d,args.seed,sc,args.N).cache()
        V = generateItemProfiles(train,args.d,args.seed,sc,args.N).cache()

        print("Training set contains %d users and %d items" %(U.count(),V.count()))
        
        start = time()
        gamma = args.gain

        while i<args.maxiter and change > args.epsilon:
            i += 1

            joinedRDD = joinAndPredictAll(train,U,V,args.N).cache()
        
            oldObjective = obj
            obj = SE(joinedRDD) + normSqRDD(U,args.lam) + normSqRDD(V,args.mu) 	
            change = np.abs(obj-oldObjective) 

            testRMSE = np.sqrt(1.*SE(joinAndPredictAll(test,U,V,args.N))/Mtest)

            gamma = args.gain / i**args.power
            
            U.unpersist()
            V.unpersist()
            U = adaptU(joinedRDD,gamma,args.lam,args.N).cache()
            V = adaptV(joinedRDD,gamma,args.mu,args.N).cache()

            now = time()-start
            print("Iteration: %d\tTime: %f\tObjective: %f\tTestRMSE: %f" % (i,now,obj,testRMSE))
            joinedRDD.unpersist()
        
        cross_val_rmses.append(testRMSE)

        train.unpersist()
        test.unpersist()

    if args.output is None:
       print("%d-fold cross validation error is: %f " % (args.folds, np.mean(cross_val_rmses)))
    else:
       print("Saving U and V RDDs")
       U.saveAsTextFile(args.output+'_U')
       V.saveAsTextFile(args.output+'_V')
