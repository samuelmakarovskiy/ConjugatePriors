#Samuel Makarovskiy, Bayesian ML HW1 (Bernoulli Simulation)

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.special import gamma
import imageio

def main():
    
    #Parameters
    p = 0.8                 #True mean                             
    steps = 2000            #Max Number of samples
    interval = 4            #interval at which samples are calculated for plots (to save time)
    xbar = np.random.binomial(1,p,steps)        #generate 'steps' number of normal random variables for samples
    agood = 60              #good conjugate prior guess   
    bgood = 25
    abad = 20               #bad conjugate prior guess
    bbad = 100

    #Plotting Constants/Definitions 
    x = np.linspace(0,1,1000)            #x-coordinates for plots of pdfs later
    y = np.full(np.size(x),None)         #define a size-matching y for plotting conformity
    plt.ion()                            #Turns on interactive mode for allowing updatable plots   
    fig = plt.figure(figsize=(12,8))     #Define Size Constraints   
        
    ax1 = fig.add_subplot(121)                       #define first subplot (will be posterior dist)
    linePosteriorGood, = ax1.plot(x, y,'g-')         #plot color and variable defintions for posterior plots
    linePosteriorML, = ax1.plot(x,y, 'b-')
    linePosteriorBad, = ax1.plot(x,y,'r-')
    ax1.set_xlim(0,1)                               #xlimit defined by range of bernoulli which is [0,1]
    ax1.set_ylim(0,5)
    fig.legend((linePosteriorGood,linePosteriorBad,linePosteriorML), ('Good Beta Conjugate Prior','Bad Beta Conjugate Prior','Max Likelyhood'))
    ax1.set_title('Posterior Probability Distribitions for Mu',fontsize = 16)
    ax1.set_xlabel('x\u0304')
    ax1.set_ylabel('Probability Density')

    ax2 = fig.add_subplot(122)                          #define second subplot (will be SE plot)
    N = np.arange(1,steps+1,interval)                   #x-axis of SE plot
    StoreMSEgood = np.full(int(steps/interval),None)    #MSE is an array that fills up as data is collected during plotting, so default to NONE (size of number of calc steps taken)
    StoreMSEML = np.full(int(steps/interval),None) 
    StoreMSEbad = np.full(int(steps/interval),None)
    ax2.set_xlim(1,steps)
    ax2.set_ylim(0,.5)                                  #Arbitrary ylim, by experiment, showed best performance
    ax2.set_title('Squared Error for Mu',fontsize = 16)
    ax2.set_xlabel('N')
    ax2.set_ylabel('Squared Error')
    

    lineMseGood, = ax2.plot(N, StoreMSEgood, 'g-')       #plot color and variable defintions for SE plots
    lineMseML, = ax2.plot(N, StoreMSEML, 'b-')
    lineMseBad, = ax2.plot(N, StoreMSEbad, 'r-')

    filenames = []              #gif files storage

    #Loop Calc and Animation
    for end in range(0,steps,interval):                     #Note end = ((#samples)-1)
        #Update Terms

        #ML Terms Update
        muML = np.mean(xbar[0:end])                         #Bernoulli ML mean: Eqn 2.7               
        varML = muML*(1-muML)/(end+1)                       #variance is eqn 2.4, but for multiple obs take sample variance   
        StoreMSEML[int(end/interval)] = (muML - p)**2       #Def of square error
        m = np.sum(xbar[0:end])                             #number of bernoulli "1's"
        el = end + 1 - m                                    #number of bernoulli "0's"

        #Good Terms Update
        muNgood = (m + agood)/(m + agood + el + bgood)      #Eqn 2.20 for mean of posterior
        StoreMSEgood[int(end/interval)] = (muNgood - p)**2  #Def of square error
        #Bad Terms Update
        
        muNbad = (m + abad)/(m + abad + el + bbad)          #Same eqns as for 'good' terms
        StoreMSEbad[int(end/interval)] = (muNbad - p)**2

        #Plotting: posteriors were updated as beta distributions with updated parameters, SE was updated with new squaared error 
        linePosteriorGood.set_ydata(scipy.stats.beta.pdf(x,agood+m,bgood+el))   
        lineMseGood.set_ydata(StoreMSEgood)
        linePosteriorBad.set_ydata(scipy.stats.beta.pdf(x,abad+m,bbad+el))
        lineMseBad.set_ydata(StoreMSEbad)
        linePosteriorML.set_ydata(scipy.stats.norm(muML,np.sqrt(varML)).pdf(x))
        lineMseML.set_ydata(StoreMSEML)
        fig.canvas.draw()                   #update the updateable figure

        #Saving plots at intervals for gif creation (comment out if you dont want saved images/gif)
        #if(end%(interval*4) == 0):           #choose gif image save timing
        #    filetitle = "Bernoulli_Step_" + str(end) + ".png"   
        #    fig.savefig(filetitle)
        #    filenames.append(filetitle)       #Append to list of file names for gif

    #Compile Images into gif and save (comment out if you dont want gif)
    #with imageio.get_writer('Bernoulli.gif', mode='I') as writer:
    #    for filename in filenames:
    #        image = imageio.imread(filename)
    #        writer.append_data(image)
        

if __name__ == '__main__':
    main()
