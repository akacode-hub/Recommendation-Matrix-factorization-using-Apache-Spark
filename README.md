# Recommendation-Matrix-factorization-using-Apache-Spark
This covers the usage of the spark in making of the matrix factorization which is used in building recommendation systems 
Also to make it work for a larger data set ( when a single cpu can't process it we can use mutiple workers ) 
This shows the usage of parallelsim and how well it effect in building it for an app. 
Below are the steps to use for mutiple workers using express partition
1) cd /home/$USER/spark
2) sbatch spark-master.slurm : this tells how many jobs are submitted
3) sq : gives us th details of the number of nodes
4) traceroute node : this give the http address that we use to give input to program
5) sbatch -N 2 --partition express --exclusive --mem 60Gb --time=01:00:00 spark-workers.slurm spark://10.99.251.191:7077 : tells us again new job batch submitted for 2 workers
6) spark-submit --master spark://10.99.250.114:7077 --executor-memory 100G --driver-memory 100G MFspark.py big_data 5 --N 20 --gain 0.0001 --pow 0.2 --lam 400 --mu 400 --maxiter 20 --d 20 : command to see output
