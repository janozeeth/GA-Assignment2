import os
import numpy as np
import matplotlib.pyplot as plt
from dejong import *

class SGA:
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

        self.pop_size = pop_size
        self.sol_rng = sol_rng
        self.num_genes = 2*genes_per_pt
        self.xover_rate = xover_rate
        self.mut_rate = mut_rate
        self.objective = lambda x : evaluate(self.decode(x))
        self.max_gen = max_gen
        if(statsFile!=None): self.statsFile = open(statsFile,'w')
        else: self.statsFile=None

        self.quality_checks = quality_checks
        self.reliability_checks = [False]*len(quality_checks)
        self.optFit = optFit
        self.numEvals = 0

        self.bestFit = np.zeros(max_gen)
        self.worstFit = np.zeros(max_gen)
        self.avgFit = np.zeros(max_gen)
        self.bestObj = np.zeros(max_gen)
        self.worstObj = np.zeros(max_gen)
        self.avgObj = np.zeros(max_gen)

        self.findPrecision()
        self.initialize()

    def decode(self,genes):
        if(len(genes) != self.num_genes):
            exit("Soln genotype has incorrect length "+str(len(genes))+". Should have length "+str(self.num_genes)+".")

        phen = np.zeros(2)
        phen[0] = np.sum([2.**i*genes[self.num_genes//2-1-i] for i in range(self.num_genes//2)])
        phen[1] = np.sum([2.**i*genes[self.num_genes-1-i] for i in range(self.num_genes//2)])

        #Integer corresponding to all 1's. Used for linear transform below
        max_sol = np.sum([2.**i for i in range(self.num_genes//2)], dtype=float)

        #Linear transformation to desired solution range
        for i in range(2):
            phen[i] = self.sol_rng[0] + (self.sol_rng[1]-self.sol_rng[0])/max_sol*phen[i]

        return phen

    def findPrecision(self):
        rng = np.abs(self.sol_rng[1]-self.sol_rng[0])
        self.sol_prec = rng / (self.num_genes/2 -1)

    def initialize(self):
        self.population = np.random.randint(0,2,size=(self.pop_size, self.num_genes))

    def selectParent(self):
        #np.random.random picks from uniform distribution on [0,1)
        rw_int = np.random.random()*self.sumFits

        i = 0
        parent = self.population[i]
        total = self.fits[i]
        while(total < rw_int):
            i+=1
            parent = self.population[i]
            total += self.fits[i]

        return parent

    def selection(self):
        self.objs = np.array([np.max([self.objective(sol),0]) for sol in self.population])
        self.fits = np.array([1./(obj+1) for obj in self.objs])
        self.numEvals += self.pop_size

        self.sumFits = np.sum(self.fits)

        new_pop = np.array([self.selectParent() for _ in range(self.pop_size)])

        self.population = new_pop

    def crossoverParents(self, parent1, parent2):
        xover_pt = np.random.randint(1,self.num_genes)
        xover_str = parent1[xover_pt:]
        
        parent1[xover_pt:] = parent2[xover_pt:]
        parent2[xover_pt:] = xover_str

        return parent1, parent2

    def crossover(self):
        for i in range(self.pop_size//2):
            self.population[2*i], self.population[2*i+1] = self.crossoverParents(self.population[2*i], self.population[2*i+1])

    def mutate(self, chrom):
        mut_mask = np.random.binomial(1, self.mut_rate, len(chrom))

        chrom = (chrom + mut_mask) % 2

        return chrom

    def mutation(self):
        for i in range(self.pop_size):
            self.population[i] = self.mutate(self.population[i])

    def reliabilityChecks(self):
        for i in range(len(self.quality_checks)):
            if(self.reliability_checks[i]): continue    #If we already hit this window, no need to check again

            best = self.bestFit[self.generation-1]
            #Are we within epsilon-window of optimum?
            if(best >= self.quality_checks[i]*self.optFit and \
               best <= (2.-self.quality_checks[i])*self.optFit):
                self.reliability_checks[i] = True

    def trackStats(self):
        self.bestFit[self.generation-1] = np.max(self.fits)
        self.worstFit[self.generation-1] = np.min(self.fits)
        self.avgFit[self.generation-1] = np.average(self.fits)

        self.bestObj[self.generation-1] = np.min(self.objs)
        self.worstObj[self.generation-1] = np.max(self.objs)
        self.avgObj[self.generation-1] = np.average(self.objs)

        self.reliabilityChecks()

        update = "----Generation: "+str(self.generation)+"---------\n"+\
                 "\tBest Fit:\t"+str(self.bestFit[self.generation-1])+\
                 "\n\tWorst Fit:\t"+str(self.worstFit[self.generation-1])+\
                 "\n\tAvg Fit:\t"+str(self.avgFit[self.generation-1])+"\n"

        if(self.statsFile==None):
            print(update)
        else:
            self.statsFile.write(update)

    def plotFitsOverTime(self, savefile=None):
        fig, ax = plt.subplots(figsize=(6,4))
        ax.set( xlabel="Generation",
                ylabel="Fitness",
                xlim=(0,self.max_gen))

        ax.plot(self.bestFit,label="Best")
        ax.plot(self.avgFit,label="Average")
        ax.plot(self.worstFit,label="Worst")

        ax.legend(title="Population Fitness", loc="lower right")

        if(savefile==None):
            fig.show()
        else:
            fig.savefig(os.getcwd()+os.sep+savefile)

    def run(self):
        for i in range(self.max_gen):
            self.generation = i+1

            self.selection()
            self.crossover()
            self.mutation()
            
            self.trackStats()

            #Stop if within 1% of optimum
            if(self.reliability_checks[0]): break
        
        if(self.statsFile!=None): self.statsFile.close()

def dejong1(x):
    return np.sum([x[i]**2 for i in range(len(x))])

def dejong2(x):
    return np.sum([100*(x[i+1]-x[i]**2)**2 + (x[i]-1)**2 for i in range(len(x)-1)])

def dejong3(x):
    return 6*len(x) + np.sum([np.floor(x[i]) for i in range(len(x))])

def dejong4(x):
    return np.sum([(i+1)*x[i]**4 for i in range(len(x))]) + np.random.normal(0,1)

def dejong5(x):
    a = np.zeros((2,25))
    a[0] = [-32,-16,0,16,32]*5
    a[1] = [-32]*5 + [-16]*5 + [0]*5 + [16]*5 + [32]*5
    return 1./(1./500 + np.sum([1./(i + (x[0]-a[0,i])**6 + (x[1]-a[1,i])**6) for i in range(25)])) 

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

    stats = open("fullStats-deJong5.txt",'w')
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
    fig.savefig("fullRun-deJong5.png")
