import numpy as np
import pickle
import bz2
import os

class fp_myo :
    def __init__(self) :
        #---------- explored phase space
        self.Bbars = np.linspace(0,10.0,1000)
        self.Tbars = np.linspace(0,10.0,1000)
        self.x = np.linspace(0,1,1000) #don't change anything except the last parameter which defines the root tracking accuracy

        self.M0 = np.NaN*np.ones((len(self.Tbars),len(self.Bbars)))
        self.M1 = np.NaN*np.ones((len(self.Tbars),len(self.Bbars)))
        self.M2 = np.NaN*np.ones((len(self.Tbars),len(self.Bbars)))

        if not os.path.isfile('fpdata.pkl.bz2') :
            self.Bbars,self.Tbars,self.M0,self.M1,self.M2 = self.precompute()

        else :
            with bz2.BZ2File('fpdata.pkl.bz2', 'r') as f :
                self.Bbars,self.Tbars,self.M0,self.M1,self.M2 = pickle.load(f)
    

    def find_intersect(self,tbar,bbar) :

        f = 1/(1+np.exp(-bbar*self.x+tbar)) #generate a parametrized function
        idx = np.argwhere(np.diff(np.sign(f-self.x))).flatten()
        m_eq = f[idx] #1 or 3 sols
        M0,M1,M2 = np.NaN,np.NaN,np.NaN
        if(len(m_eq)) == 3 :
            M0 = f[idx[2]] #higher branch
            M1 = f[idx[1]] #middle 
            M2 = f[idx[0]] #lower branch

        else :
            if m_eq[0] > 0.5: 
                M0 = m_eq[0]                
            else: 
                M2 = m_eq[0]

        return M0,M1,M2


    def precompute(self) :

        for i,tbar in enumerate(self.Tbars) :        
            for j,bbar in enumerate(self.Bbars):
                self.M0[i,j] , self.M1[i,j], self.M2[i,j] = self.find_intersect(tbar=tbar,bbar=bbar)
                
        with bz2.BZ2File('fpdata.pkl.bz2', 'w') as f :
            pickle.dump((self.Bbars,self.Tbars,self.M0,self.M1,self.M2),f)

        return self.Bbars,self.Tbars,self.M0,self.M1,self.M2

