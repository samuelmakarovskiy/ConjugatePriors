#Samuel Makarovskiy, Bayesian ML HW1 (Gaussian Simulation)

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import imageio

def main():
    
    #Parameters
    mu = 0                      #True Mean
    var = 8                     #True Variance
    steps = 2000                #Max Number of samples
    interval = 4                #interval at which samples are calculated for plots (to save time)
    xbar = np.random.normal(mu,np.sqrt(var),steps)      #generate 'steps' number of normal random variables for samples
    mu0good = 5                #good conjugate prior guess
    var0good = .2               
    mu0bad = 50                #bad conjugate prior guess
    var0bad = .25

    #Plotting Constants/Definitions 
    x = np.linspace(mu-np.sqrt(var)*5,mu+np.sqrt(var)*5, 500*int(np.sqrt(var)))     #x-coordinates for plots of pdfs later
    y = np.full(np.size(x),None)                                                    #define a size-matching y for plotting conformity
    plt.ion()                                           #Turns on interactive mode for allowing updatable plots                                          
    fig = plt.figure(figsize=(12,8))                    #Define Size Constraints
    
    ax1 = fig.add_subplot(121)                          #define first subplot (will be posterior dist)
    linePosteriorGood, = ax1.plot(x, y,'g-')            #plot color and variable defintions for posterior plots
    linePosteriorML, = ax1.plot(x,y, 'b-')
    linePosteriorBad, = ax1.plot(x,y,'r-')
    ax1.set_xlim(mu-np.sqrt(var)*5,mu+np.sqrt(var)*5)   #set limits to those of variable x and y
    ax1.set_ylim(0,1)
    fig.legend((linePosteriorGood,linePosteriorBad,linePosteriorML), ('Good Gaussian Conjugate Prior','Bad Gaussian Conjugate Prior','Max Likelyhood'))
    ax1.set_title('Posterior Probability Distribitions for Mu',fontsize = 16)
    ax1.set_xlabel('x\u0304')
    ax1.set_ylabel('Probability Density')

    ax2 = fig.add_subplot(122)                              #define second subplot (will be SE plot)
    N = np.arange(1,steps+1,interval)                       #x-axis of SE plot
    StoreMSEgood = np.full(int(steps/interval),None)        #MSE is an array that fills up as data is collected during plotting, so default to NONE (size of number of calc steps taken)
    StoreMSEML = np.full(int(steps/interval),None) 
    StoreMSEbad = np.full(int(steps/interval),None)
    ax2.set_xlim(1,steps)
    ax2.set_ylim(0,2.5*var)                                 #Arbitrary ylim, by experiment, showed best performance
    ax2.set_title('Squared Error for Mu',fontsize = 16)
    ax2.set_xlabel('N')
    ax2.set_ylabel('SE')
    

    lineMseGood, = ax2.plot(N, StoreMSEgood, 'g-')      #plot color and variable defintions for MSE plots
    lineMseML, = ax2.plot(N, StoreMSEML, 'b-')
    lineMseBad, = ax2.plot(N, StoreMSEbad, 'r-')

    filenames = []                                      #gif files storage
    
    #Loop Calc and Animation
    for end in range(0,steps,interval):                 #Note end = ((#samples)-1)
        #Update Terms

        #ML Terms Update
        muML = np.mean(xbar[0:end])                         #Eqn 2.121 (Could've used 2.126 for better complexity)
        StoreMSEML[int(end/interval)] = (muML - mu)**2      #Def of squared error
        varML = np.var(xbar[0:end])/(end+1)                 #Sample variance assumed as variance

        #Good Terms Update
        
        muNgood = (var*mu0good + (end+1)*var0good*muML)/((end + 1)*var0good + var)  #Eqn 2.141
        varNgood = (1/var0good + (end+1)/var)**(-1)                                 #Eqn 2.142
        StoreMSEgood[int(end/interval)] = (muNgood - mu)**2                         #Def squared error

        #Bad Terms Update
        
        muNbad = (var*mu0bad + (end+1)*var0bad*muML)/((end + 1)*var0bad + var)      #Same Formulas for 'good'
        varNbad = (1/var0bad + (end+1)/var)**(-1)
        StoreMSEbad[int(end/interval)] = (muNbad - mu)**2


        
        #Plotting: update all ydata to normal pdfs with appropriate parameters (or to appropriate Squared Errors) as calculated earlier 
        linePosteriorGood.set_ydata(scipy.stats.norm(muNgood,np.sqrt(varNgood)).pdf(x)) 
        lineMseGood.set_ydata(StoreMSEgood)
        linePosteriorBad.set_ydata(scipy.stats.norm(muNbad,np.sqrt(varNbad)).pdf(x))
        lineMseBad.set_ydata(StoreMSEbad)
        linePosteriorML.set_ydata(scipy.stats.norm(muML,np.sqrt(varML)).pdf(x))
        lineMseML.set_ydata(StoreMSEML)
        fig.canvas.draw()

        #Saving Plots at intervals for gif creation (comment out if you dont want saved images/gif)
        #if(end%(interval*4) == 0):                    #choose gif image save timing
        #    filetitle = "Gaussian_Step_" + str(end) + ".png"       
        #   fig.savefig(filetitle)
        #    filenames.append(filetitle)               #Append to list of file names for gif

    #Compile Images into gif and save (comment out if you dont want gif)
    #with imageio.get_writer('Gaussian.gif', mode='I') as writer:
    #    for filename in filenames:
    #        image = imageio.imread(filename)
    #        writer.append_data(image)        

        
      

if __name__ == '__main__':
    main()
