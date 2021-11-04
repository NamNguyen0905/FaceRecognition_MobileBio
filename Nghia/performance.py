import numpy as np
import matplotlib.pyplot as plt
import math

############ functions #############################################################

def dprime(gen_scores, imp_scores):
    x = math.sqrt(2) * abs(np.mean(gen_scores) - np.mean(imp_scores)) # replace 1 with the numerator
    y = math.sqrt(pow(np.std(gen_scores), 2) + pow(np.std(imp_scores), 2)) # replace 1 with the denominator
    return x / y

def plot_scoreDist(gen_scores, imp_scores, plot_title, eer, threshold_score):
    plt.figure()
    
    if threshold_score == True:
        plt.axvline(x = eer, ymin = 0, ymax = 0.5, linestyle = '--', label = 'Threshold')
    else:
        plt.axvline(x = 0, ymin = 0, ymax = 0.5, linestyle = '--', label = 'Threshold')
    
    
    plt.hist(gen_scores, color = 'green', lw = 2,
             histtype= 'step', hatch = '//', label = 'Genuine Scores')
    plt.hist(imp_scores, color = 'red', lw = 2,
             histtype= 'step', hatch = '\\', label = 'Impostor Scores')
    plt.xlim([-0.05,1.05])
    plt.legend(loc = 'best')
    dp = dprime(gen_scores, imp_scores)
    plt.title(plot_title + '\nD-prime = %.2f' % dp)    
    
    plt.show()
    return

def get_EER(far, frr):
    eer = 0
    '''
        Use the FARs and FRRs to return the error
        in which they are approximately equal.
        
    '''
    min = (far[0] + frr[0]) /2
    for i in range(1, len(far)):
        if far[i] == frr[i]:
            threshold_score = False
            return far[i], threshold_score
        
        elif abs(far[i] - frr[i]) < min:
            min = abs(far[i]-frr[i])
            eer = (far[i] + frr[i]) / 2
            threshold_score = True
            
              
    return eer, threshold_score

#Detection Error Tradeoff 
def plot_det(far, frr, far2, frr2, far3, frr3, plot_title1, plot_title2, plot_title3):
    title = 'DET'
    eer, threshold_score = get_EER(far, frr)
    eer2, threshold_score2 = get_EER(far2, frr2)   
    eer3, threshold_score3 = get_EER(far3, frr3)
              
    plt.figure()
    '''
        Refer back to lecture for DET curve
    '''
    
    plt.plot(far,frr, lw = 2, label = plot_title1)
    plt.plot(far2,frr2, lw = 2, label = plot_title2)
    plt.plot(far3,frr3, lw = 2, label = plot_title3)
    plt.legend(loc= 'best')
    
    plt.plot([0,1], [0,1], lw = 1, color = 'black')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FAR')
    plt.ylabel('FRR')
    plt.title(title + '\nEER of %s = %.3f \n%s = %.3f, %s = %.3f' 
              % (plot_title1, eer, plot_title2, eer2, plot_title3, eer3))
    
    plt.show()
    return eer, eer2, eer3, threshold_score, threshold_score2, threshold_score3

#Receiver Operating Characteristic
def plot_roc(far, tpr, far2, tpr2, far3, tpr3, plot_title1, plot_title2, plot_title3):
    title = 'ROC'
    plt.figure()
    '''
        Refer back to lecture for ROC curve
    '''
    plt.plot(far, tpr, lw = 2, label = plot_title1)
    plt.plot(far2, tpr2, lw = 2, label = plot_title2)
    plt.plot(far3, tpr3, lw = 2, label = plot_title3)
    
    plt.legend(loc= 'best')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FAR')
    plt.ylabel('TAR')
    
    plt.title(title)
    
    plt.show()
    return

# Function to compute TPR, FAR, FRR
def compute_rates(gen_scores, imp_scores, num_thresholds):
    # start at 0 to 1, number of num = 100
    thresholds =  np.linspace(0.0, 1.0, num_thresholds)
    # use np.linspace to create n threshold values 
                 # between 0 and 1
    far = [] #False Positive Rate
    frr = [] #False Negative Rate
    tpr = [] #True Positive Rate
    # tnr ? where True Negative rate?
    
    for t in thresholds:
        '''
            Initialize tp, fp, tn, fn            
        '''
        tp, fp, tn, fn = 0,0,0,0
        
        
        for g_s in gen_scores:
            '''
                Count tp and fn
            '''
            if g_s >= t:
                tp += 1
            else:
                fn += 1
                
        for i_s in imp_scores:
            '''
                Count tn and fp
            '''
            if i_s >= t:
                fp += 1
            else:
                tn += 1
                    
        far.append(fp / (fp + tn)) #equation for far
        frr.append(fn / (fn + tp)) #equation for frr
        tpr.append(tp / (tp + fn)) #equation for tpr
    return far, frr, tpr

############ main code #############################################################

def performance(gen_scores, imp_scores, gen_scores2, imp_scores2, 
                gen_scores3, imp_scores3, plot_title1, plot_title2, plot_title3, num_thresholds):    
            
    far, frr, tpr = compute_rates(gen_scores, imp_scores, num_thresholds) #parameters
    far2, frr2, tpr2 = compute_rates(gen_scores2, imp_scores2, num_thresholds)
    far3, frr3, tpr3 = compute_rates(gen_scores3, imp_scores3, num_thresholds)
    
    plot_roc(far, tpr, far2, tpr2, far3, tpr3, 
             plot_title1, plot_title2, plot_title3) #parameters
    eer, eer2, eer3, threshold_score, threshold_score2, threshold_score3 = plot_det(far, frr, 
             far2, frr2, far3, frr3, plot_title1, plot_title2, plot_title3) #parameters
    
    plot_scoreDist(gen_scores, imp_scores, plot_title1, eer, threshold_score) #parameters
    plot_scoreDist(gen_scores2, imp_scores2, plot_title2, eer2, threshold_score2) #parameters
    plot_scoreDist(gen_scores3, imp_scores3, plot_title3, eer3,  threshold_score3) #parameters

