import os
import numpy as np
import matplotlib.pyplot as plt
from SGA import SGA
from dejong import *

class CHCGA(SGA):
    def __init__(self,
                 pop_size,      #Should be even number
                 sol_rng,
                 genes_per_pt,  #Should be odd number
                 xover_rate,
                 mut_rate,
                 evaluate,
                 optFit=1.,
                 max_gen=100,
                 statsFile=None,
                 quality_checks=[0.99,0.95,0.9]):

        super().__init__(pop_size,sol_rng,genes_per_pt,xover_rate,mut_rate,evaluate,optFit,max_gen,statsFile,quality_checks)

    def selection(self):
        if(self.generation==1):
            self.parentPop = np.zeros((self.pop_size, self.num_genes))
            self.parentFits = np.zeros(self.pop_size)
        
        self.objs = np.array([np.max([self.objective(sol),0]) for sol in self.population])
        self.fits = np.array([1./(obj+1) for obj in self.objs])
        
        if(!self.reliability_checks[0]):
            self.numEvals += self.pop_size

        childParent = np.append(self.population, self.parentPop)
        cpFits = np.append(self.fits, self.parentFits)
        
        #Sort based on cpFits
        srtd = np.argsort(cpFits)
        cpFits = cpFits[srtd]
        childParent = childParent[srtd]

        self.population = childParent[self.pop_size:]   #Take only pop_size greatest fits

        #Preserve a copy in parentPop & parentFits that survives recombination
        self.parentPop = self.population
        self.parentFits = cpFits[self.pop_size:]

if __name__=="__main__":
    #DeJong 1 parameters
#    pop_size =      100
#    sol_rng =       [-5.12,5.12]
#    genes_per_pt =  11
#    xover_rate =    0.7
#    mut_rate =      0.001
#    evaluate =      dejong1
#    max_gen =       50

    #DeJong 2 parameters
#    pop_size =      100
#    sol_rng =       [-5.12,5.12]
#    genes_per_pt =  11
#    xover_rate =    0.9
#    mut_rate =      0.01
#    evaluate =      dejong2
#    max_gen =       100

    #DeJong 3 parameters
#    pop_size =      100
#    sol_rng =       [-5.12,5.12]
#    genes_per_pt =  11
#    xover_rate =    0.7
#    mut_rate =      0.001
#    evaluate =      dejong3
#    max_gen =       20

    #DeJong 4 parameters
#    pop_size =      20
#    sol_rng =       [-1.28,1.28]
#    genes_per_pt =  11
#    xover_rate =    0.7
#    mut_rate =      0.0005
#    evaluate =      dejong4
#    max_gen =       20

    #DeJong 5 parameters
    pop_size =      200
    sol_rng =       [-65.536,65.532]
    genes_per_pt =  31
    xover_rate =    0.7
    mut_rate =      0.03
    evaluate =      dejong5
    max_gen =       100

    numRuns = 30
    
    bestFits = np.zeros(max_gen)
    avgFits = np.zeros(max_gen)
    worstFits = np.zeros(max_gen)
    bestObjs = np.zeros(max_gen)
    avgObjs = np.zeros(max_gen)
    worstObjs = np.zeros(max_gen)

    avgEvals = 0
    reliab = [0,0,0]

    for i in range(numRuns):
        print("Run:", i+1)

        GA = SGA(pop_size, sol_rng, genes_per_pt, xover_rate, mut_rate, evaluate, max_gen=max_gen, statsFile="stats.txt")

        GA.run()

        print("Best fit:", np.max(GA.bestFit))

        bestFits += GA.bestFit
        worstFits += GA.worstFit
        avgFits += GA.avgFit
        bestObjs += GA.bestObj
        worstObjs += GA.worstObj
        avgObjs += GA.avgObj

        avgEvals += GA.numEvals

        for i in range(len(reliab)):
            if(GA.reliability_checks[i]): reliab[i] +=1

    #Average out
    bestFits /= numRuns
    worstFits /= numRuns
    avgFits /= numRuns
    bestObjs /= numRuns
    worstObjs /= numRuns
    avgObjs /= numRuns
    avgEvals /= float(numRuns)
    for i in range(len(reliab)): reliab[i] /= float(numRuns)

    print("Reliability:")
    print("\t1%:\t", reliab[0])
    print("\t5%:\t", reliab[1])
    print("\t10%:\t", reliab[2])

    stats = open("fullStatsCHC-deJong5.txt",'w')
    stats.write("Best fits:"+str(bestFits))
    stats.write("\nWorst fits:"+str(worstFits))
    stats.write("\nAvg fits:"+str(avgFits))
    stats.write("\nBest objs:"+str(bestObjs))
    stats.write("\nWorst objs:"+str(worstObjs))
    stats.write("\nAvg objs:"+str(avgObjs))
    stats.write("\nAvg Evals:"+str(avgEvals))
    stats.write("\nReliability:")
    stats.write("\n\t0.99\t"+str(reliab[0]))
    stats.write("\n\t0.95\t"+str(reliab[1]))
    stats.write("\n\t0.90\t"+str(reliab[2]))
    stats.close()

    #Plotting
    fig, ax = plt.subplots(ncols=2, figsize=(12,4))
    ax[0].set(title="Fitness", xlim=(0,max_gen))
    ax[0].plot(bestFits, label="Best")
    ax[0].plot(worstFits, label="Worst")
    ax[0].plot(avgFits, label = "Average")
    ax[0].legend(loc="lower right")
    ax[1].set(title="Objective", xlim=(0,max_gen))
    ax[1].plot(bestObjs, label="Best")
    ax[1].plot(worstObjs, label="Worst")
    ax[1].plot(avgObjs, label = "Average")
    ax[1].legend(loc="upper right")
    fig.savefig("fullRunCHC-deJong5.png")
