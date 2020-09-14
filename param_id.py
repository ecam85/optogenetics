"""
ecam Apr18

Id. length of pendulum
"""

import time #For seeding
import sys #Handles command line arguments
import numpy as np #Load param if needed.

import pmh #Parallel Metropolis Hastings

from prior3 import prior #Prior
from logl import logl
from path import path #Path to save the results.

from param import param_array

SEED = int(time.time())+np.random.randint(1,10000) #Random seed
BASENAME = "myos" #Prefix file name
MAXT = 24*3600 #Maximum running time (2 hour)
TARGETIT = 1000000 #Target samples 

p0 = None #Define initial parameter here if required. (None generates randomly)

def get_chain_name(method,init_type,extra_name):
	return BASENAME+"_"+extra_name+"_"+method+"_"+init_type

def run_param_id(method,extra_name,beta=.1,cores=8):
	#Set seed
	pmh.setSeed(SEED)

	#Initital state
	if not isinstance(p0,np.ndarray):
		init_type = "random"
		state0 = prior(0)
	else:
		init_type = "fixed"
		state0 = p0

	#pmh
	if method == "is":
		pa = pmh.pa(method,prior)
	else:
		pa = pmh.pa(method,prior,beta) #wp, others.
		method = method+"b_"+str(beta)	

	#Markov chain
	mc = pmh.mc(basename=path(get_chain_name(method,init_type,extra_name),"mcmc"))

	#Sampler. K is samples per step
	mh = pmh.pmh(pa,mc,state0,Npool=cores,K=50,maxT=MAXT,targetIt=TARGETIT)

	#Log the seed
	pmh.logging.info("Seed: "+str(SEED))

	mh.run(logl)

if __name__ == "__main__":
	if len(sys.argv) > 1:
		method = sys.argv[1]
	else:
		method = "is"	

	if len(sys.argv) > 2:
		extra_name = sys.argv[2]
	else:
		extra_name = ""

	print "Running",method,extra_name

	run_param_id(method,extra_name)

