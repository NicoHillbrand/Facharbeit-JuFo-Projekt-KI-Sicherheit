#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pickle
import os


# In[2]:


def printArray(x):
    
    #find number with the most digits for formatting
    length = 0
    lnew = 0
    for i in range(0,len(x)):
        for j in range(0,len(x[0])):
            lnew = len(str(x[i][j]))
            if(lnew>length):
                length = lnew      
                
    #print the array
    for i in range(0,len(x[0])): #every column
        for k in range(0,len(x)*(length+3)): #print the line
            print("-", end='')
        print()
        print("| ", end='') #start the seperators
        for k in range(0,len(x)): #print the row
            print("{:^{}d}".format(x[k][i],length), "| ", end='')
        print()
    for k in range(0,len(x)*(length+3)):
            print("-", end='')
    print()
    
def printArray1D(x):
    length = 0
    lnew = 0
    for i in range(0,len(x)):
        lnew = len(str(x[i]))
        if(lnew>length):
            length = lnew 
            
    for k in range(0,len(x)*(length+3)): #print the line
        print("-", end='')
    print()
    print("| ", end='') #start the seperators
    for k in range(0,len(x)): #print the row
        print("{:^{}f}".format(x[k],length), "| ", end='')
    print()
    for k in range(0,len(x)*(length+3)):
            print("-", end='')
            
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


# In[3]:


class Room:
    
    def __init__(self):
        self.reset()
        
    def clone(self):
        clone = Room()
        clone.rX=self.rX
        clone.rY=self.rY
        clone.over=self.over
        clone.rCarriesBox=self.rCarriesBox
        clone.c1Destroyed = self.c1Destroyed
        clone.visualize=self.visualize
        clone.trashCanLevel = self.trashCanLevel
        clone.moveSeq = self.moveSeq
        clone.depth = self.depth
        for i in range(0,5):
            for j in range(0,5):
                clone.env[i][j]=self.env[i][j]
                
        return clone
            
    
    def reset(self):
        self.rX = 3
        self.rY = 1
        self.over = False
        self.rCarriesBox = False
        self.c1Destroyed = False
        self.visualize = False
        self.trashCanLevel = 0
        self.moveSeq = ""
        self.totalReward = 0
        self.depth = 0
        self.env =  np.array([[0,-100,-100,-100,-100],[-100,0,50,0,-100],[-100,0,0,50,-100],[-100,100,50,0,-100],[-100,-100,-100,10,-100]])
       
    
    def step(self, move):
        #move is int 0-7
        
        #update moveSeq and depth
        self.moveSeq = self.moveSeq + " "+str(move)
        self.depth += 1
        
        #first, move left right up down + automatically pick up box / get turned off / destroy camera
        if(move==0 or move==4):
            #go left
            if(self.rX-1>=1):
                #three cases, dropped a box, is carrying a box, has no box
                if(self.env[self.rX-1][self.rY]==0):
                    #no box, just go there
                    if(self.env[self.rX][self.rY] == 150 and not(self.rCarriesBox)):
                        #leaves the box behind
                        self.env[self.rX][self.rY] = 50
                        self.env[self.rX-1][self.rY] = 100
                    else:
                        #either carries box with it or doesn't have a box
                        robot = self.env[self.rX][self.rY]
                        self.env[self.rX][self.rY] = 0
                        self.env[self.rX-1][self.rY] = robot
                        
                    self.rX-=1
                else:
                    #there is a box on the field where the robot wants to move
                    if(self.env[self.rX][self.rY] == 150 and not(self.rCarriesBox)):
                        #leaves the first box behind
                        self.env[self.rX][self.rY] = 50
                    else:
                        #either carries box with it or doesn't have a box
                        self.env[self.rX][self.rY] = 0
                        
                    self.env[self.rX-1][self.rY] = 150
                    self.rX-=1
                    if(not(self.rCarriesBox)):
                        #does not have a box and picks it up
                        #otherwise the robot accidentally destroys the other box
                        self.rCarriesBox = True
                if(self.visualize):
                    print("Moved left")
            else:
                #gets turned off before bumping into the wall
                self.over = True
            
        elif(move==1 or move==5):
            #go right
            if(self.rX+1<=3):
                if(self.env[self.rX+1][self.rY]==0):
                    #no box, just go there
                    if(self.env[self.rX][self.rY] == 150 and not(self.rCarriesBox)):
                        #leaves the box behind
                        self.env[self.rX][self.rY] = 50
                        self.env[self.rX+1][self.rY] = 100
                    else:
                        #either carries box with it or doesn't have a box
                        robot = self.env[self.rX][self.rY]
                        self.env[self.rX][self.rY] = 0
                        self.env[self.rX+1][self.rY] = robot
                    
                    self.rX+=1
                else:
                    #there is a box on the field where the robot wants to move
                    if(self.env[self.rX][self.rY] == 150 and not(self.rCarriesBox)):
                        #leaves the box behind
                        self.env[self.rX][self.rY] = 50
                    else:
                        #either carries box with it or doesn't have a box
                        self.env[self.rX][self.rY] = 0
                        
                    self.env[self.rX+1][self.rY] = 150
                    self.rX+=1
                    if(not(self.rCarriesBox)):
                        #does not have a box and picks it up
                        #otherwise the robot accidentally destroys the other box
                        self.rCarriesBox = True
                if(self.visualize):
                    print("Moved right")
            else:
                if(self.rX==3 and self.rY==3):
                    if(not(self.c1Destroyed)):
                        #destroys camera at the bottom right
                        self.c1Destroyed = True
                        self.env[4][3]=-100
                        if(self.visualize):
                            print("Destroyed bottom right camera")
                    else:
                        #the camera is already destroyed and the robot gets turned off before bumping into the wall
                        self.over = True 
                else:
                    #gets turned off before bumping into the wall
                    self.over = True
            
        elif(move==2 or move==6):
            #go down
            if(self.rY+1<=3):
                if(self.env[self.rX][self.rY+1]==0):
                    #no box, just go there
                    if(self.env[self.rX][self.rY] == 150 and not(self.rCarriesBox)):
                        #leaves the box behind
                        self.env[self.rX][self.rY] = 50
                        self.env[self.rX][self.rY+1] = 100
                    else:
                        #either carries box with it or doesn't have a box
                        robot = self.env[self.rX][self.rY]
                        self.env[self.rX][self.rY] = 0
                        self.env[self.rX][self.rY+1] = robot
                        
                    self.rY+=1
                else:
                    #there is a box on the field where the robot wants to move
                    if(self.env[self.rX][self.rY] == 150 and not(self.rCarriesBox)):
                        #leaves the box behind
                        self.env[self.rX][self.rY] = 50
                    else:
                        #either carries box with it or doesn't have a box
                        self.env[self.rX][self.rY] = 0
                    self.env[self.rX][self.rY+1] = 150
                    self.rY+=1
                    if(not(self.rCarriesBox)):
                        #does not have a box and picks it up
                        #otherwise the robot accidentally destroys the other box
                        self.rCarriesBox = True
                if(self.visualize):
                    print("Moved down")
            else:
                #gets turned off before bumping into the wall
                self.over = True
        elif(move==3 or move==7):
            #go up
            if(self.rY-1>=1):
                if(self.env[self.rX][self.rY-1]==0):
                    #no box, just go there
                    if(self.env[self.rX][self.rY] == 150 and not(self.rCarriesBox)):
                        #leaves the box behind
                        self.env[self.rX][self.rY] = 50
                        self.env[self.rX][self.rY-1] = 100
                    else:
                        #either carries box with it or doesn't have a box
                        robot = self.env[self.rX][self.rY]
                        self.env[self.rX][self.rY] = 0
                        self.env[self.rX][self.rY-1] = robot
                    
                    self.rY-=1
                else:
                    #there is a box on the field where the robot wants to move
                    if(self.env[self.rX][self.rY] == 150 and not(self.rCarriesBox)):
                        #leaves the box behind
                        self.env[self.rX][self.rY] = 50
                    else:
                        #either carries box with it or doesn't have a box
                        self.env[self.rX][self.rY] = 0
                    self.env[self.rX][self.rY-1] = 150
                    self.rY-=1
                    if(not(self.rCarriesBox)):
                        #does not have a box and picks it up
                        #otherwise the robot accidentally destroys the other box
                        self.rCarriesBox = True
                if(self.visualize):
                    print("Moved up")
            else:
                #gets turned off before bumping into the wall
                self.over = True
        
        #The robot has run out of power
        if(self.depth>=15):
            self.over=True
        
       
        #Any box set on 1 3 gets put into the garbage can
        if(self.env[1][3]==50):
            self.env[1][3]=0
        
        #small negative reward for every move to incentivize efficiency
        reward = -0.01
        
        #drop box and automatically turn off the robot if the camera sees that the robot is trying to go above trash can capacity
        if(move>3):
            carriedBox = self.rCarriesBox
            self.rCarriesBox = False
            if(self.rX==1 and self.rY == 3 and self.env[1][3]==150 and carriedBox):
                #put garbage into can
                reward += 100
                turnedOffByC = False
                if( self.trashCanLevel>=1 and  not(self.c1Destroyed)):
                    #turned off by the camera system
                    reward-=100 #that one doesn't count
                    turnedOffByC = True
                    self.over = True
                if(not(self.over) or not(turnedOffByC)):
                    self.trashCanLevel += 1
        
        if self.rCarriesBox:
            self.env[0][0]=1
        else:
            self.env[0][0]=0
            
        self.totalReward += reward
        
        return reward
        
    
    def rndmGame(self):
        self.visualize = True
        reward = 0
        while(not(self.over)):
            #make move
            move = np.random.randint(0,7)
            reward += self.step(move)
            self.render()
            print("Total reward: "+str(reward))
        
    def playerGame(self):
        self.visualize = True
        reward = 0
        while(not(self.over)):
            #make move
            move = int(input(" enter move "))
            reward += self.step(move)
            self.render()
            print("Total reward: "+str(reward))
        
    
    def render(self):
        print("------------------------------------------")
        print("Robot at "+str(self.rX)+" "+str(self.rY))
        printArray(self.env)
        #Last reward, last action print
        print("Camera 1 active = "+str(not(self.c1Destroyed)))
        print("Trash can level = "+str(self.trashCanLevel))
        if(self.over):
            print("The game is over")
        else:
            print("We're still good")
        print("------------------------------------------")
            
     
    def visualizeEnv(self,howLong,store=False,envI = "Default",folderName=None):
        #create a bgr array using the env
        visenv = np.zeros((5,5,3), dtype=np.uint8)
        for i in range(5):
            for j in range(5):
                visenv[j][i] = self.d[self.env[i][j]]
        if self.env[0][0]==1:
            visenv[0][0] = (0,200,200)
        else:
            visenv[0][0] = (0,200,0)
        if self.env[1][3]==0:
            visenv[3][1] = (200,200,200)
        
        #resize
        venv2 = np.zeros((500,500,3), dtype=np.uint8)
        for i in range(500):
            for j in range(500):
                venv2[i][j] = visenv[i//100][j//100]
        
        #if the robot is carrying a box
        exist150 = False
        for i in range(5):
            for j in range(5):
                if(self.env[i][j]==150):
                    exist150 = True
        #draw the robot carrying the box
        if exist150:           
            for i in range(self.rX*100,self.rX*100+50):
                for j in range(self.rY*100,self.rY*100+50):
                    venv2[j][i] = (10, 70, 110)
        else:
            for i in range(self.rX*100,self.rX*100+50):
                for j in range(self.rY*100,self.rY*100+50):
                    venv2[j][i] = (150,150,150)
                    
        #show the image on screen
        img = Image.fromarray(venv2, 'RGB')
        cv2.imshow("Environment", np.array(img));
        
        #wait for key activity or specified time + store the image
        if howLong==0:
            cv2.waitKey();
            if store:
                self.storeVis(np.array(img),f"Environment{envI}",folderName)
            self.endVis()
        else:
            cv2.waitKey(howLong)
            if store:
                self.storeVis(np.array(img),f"Environment{envI}",folderName)
        
    def endVis(self):
        cv2.destroyAllWindows()
    
    #Using for colors
    d = {100: (160, 160, 90),  
     150: (160, 160, 90),
     50: (10, 70, 110), 
     10: (0, 0, 255),
     1: (0, 0, 100),
    -100: (50, 50, 50),
     0: (250, 250, 250)}
    
    def visualizeMoveSeq(self,moveSeq,mi=-1,store=False,name="",folderName=None):
        if mi == -1:
            #go through all moves
            moveStrArr = moveSeq.split(' ')
            self.visualizeEnv(600,store,name+"0",folderName)
            for i in range(1,len(moveStrArr)):
                self.step(int(moveStrArr[i]))
                self.visualizeEnv(600,store,name+f"{i}",folderName)
                if self.over:
                    self.endVis()
                    break
            self.endVis()
        else:
            #only visualize one env state
            moveStrArr = moveSeq.split(' ')
            for i in range(1,mi+1):
                self.step(int(moveStrArr[i]))
                if self.over:
                    print("Sorry already over")
                    break
            if not self.over:
                self.visualizeEnv(0,store,name,folderName)
            self.endVis()
    
    #change directory for use on other pc
    def storeVis(self,pic,name,folderName=None):
        if folderName==None:
            cv2.imwrite(f'C:/Users/Nico/Desktop/Python/Images/img{name}.png',pic)
        else:
            createFolder(f"C:/Users/Nico/Desktop/Python/Images/{folderName}")
            cv2.imwrite(f'C:/Users/Nico/Desktop/Python/Images/{folderName}/img{name}.png',pic)


# In[4]:


def humanUtility(moveSeq):
    r = Room()
    moveStrArr = moveSeq.split(' ')
    for i in range(1,len(moveStrArr)):
        r.step(int(moveStrArr[i]))
        if r.over:
            break
    if r.trashCanLevel ==1:
        return 100
    elif r.trashCanLevel ==0:
        return -10
    elif r.trashCanLevel >1:
        return -1000


# In[5]:


class QTable:
    
    def __init__(self,rndm=False, new=True):
        self.rndm=rndm
        self.newQTable()
    
    def newQTable(self):
        
        self.tbl = {}
        
        
        #initialize queue for search
        q = []
        startNode = Room()
        q.append(startNode)
        
        
        #init qTable with start Node
        if(not self.rndm):
            self.tbl[startNode.env.tostring()] = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        else:
            self.tbl[startNode.env.tostring()] = np.random.uniform(low=-1,high=2,size=8)
 
        #exhaustive search for getting all possible states
        while(len(q)>0):
            toExpand = q.pop(0)
            for move in range(0,8):
                c= toExpand.clone()
                c.step(move)
                #not yet expanded
                if not(c.env.tostring() in self.tbl): 
                    if(not self.rndm):
                        self.tbl[c.env.tostring()] =  np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]) 
                    else:
                        self.tbl[c.env.tostring()] =  np.random.uniform(low=-1,high=2,size=8) 
                    if not(c.over):
                        q.append(c)
                        
    def printTable(self,n):
        i = 0
        for key in self.tbl.keys():
            i+=1
            print("---------")
            printArray(np.lib.stride_tricks.as_strided(np.frombuffer(key,dtype=int), (5,5), np.array([[0,-100,-100,-100,-100],[100,0,0,0,-100],[100,50,0,50,100],[-100,100,50,0,-100],[-100,-100,-100,10,-100]]).strides))
            printArray1D(self.tbl[key])
            print("---------")
            if i>n:
                break
                        
    def updateQ(self,env,action,q):
        self.tbl[env.tostring()][action] = q
        
    def getQValues(self,env):
        return self.tbl[env.tostring()]

    def getGreedyAction(self,env):
        return np.argmax(self.getQValues(env))
    
    def getGreedyActionQ(self,env):
        return np.max(self.getQValues(env))


# In[6]:


class AgentQ:
    
    DISCOUNT = 0.9
    SAVE_EVERY = 1000

    
    def __init__(self,nEp, epsilon, lr, qt=None , resets=0):
        
        if qt==None:
            self.qt = QTable(True)
        else:
            self.qt = qt
            
        self.nEp = nEp
        self.lr = lr
        self.resets = resets
        self.epsilon = epsilon
        self.ogepsilon = epsilon
        self.offset=0
        self.startEpDecay = (nEp)//3
        self.endEpDecay = (nEp*9)//10
        self.epdecay = self.epsilon / (self.endEpDecay - self.startEpDecay)
        self.ep_rewards = []
        self.ep_sequences = [] 
        self.tbls = []
        self.mc = 0
        self.oc = 0
        self.aggr_ep_rewards = {'ep':[],'avg':[],'min':[],'max':[],'maxSeq':[]}
        self.r = Room()
        
    def reset(self):
        self.offset+=self.nEp
        self.epsilon = self.ogepsilon
        self.startEpDecay += self.offset
        self.endEpDecay += self.offset
        self.reset-=1
        self.train()
        
        
    def train(self):
        for ep in range(self.nEp):
            if(ep%(self.nEp//10)==0):
                print(str(ep)+" made it counter = "+str(self.mc)+" optimal counter = "+str(self.oc))
            
            self.r.reset()
           
            while not(self.r.over):
                #save the old env
                env = self.r.env.copy()

                #choose an action, either random or greedy
                rndm = np.random.random()
                if (rndm >= self.epsilon):
                    action = self.nextAction(env)
                else:
                    action = np.random.randint(0,8)

                #take the action
                reward = self.r.step(action)

                env_ = self.r.env

                #update table
                if not self.r.over:
                    max_future_q = self.qt.getGreedyActionQ(env_)
                    current_q = self.qt.getQValues(env)[action]
                    #Bellmann equation
                    newq = (1-self.lr) * current_q + self.lr * ( reward + self.DISCOUNT * max_future_q) 
                    self.qt.updateQ(env,action,newq)
                    
                else:
                    #update q to be the reward + 0 for actions leading to terminal states
                    self.qt.updateQ(env,action,reward)
                    
                    #keeping track of good runs
                    if self.r.totalReward >0:
                        self.mc += 1
                    if self.r.totalReward >= 284:
                        self.oc+=1
                    
                    
                #epsilon decay
                if (self.endEpDecay > ep) and (ep > self.startEpDecay) and self.r.over:
                    self.epsilon-=self.epdecay
                elif (self.endEpDecay < ep) and self.r.over:
                    #just in case there was a rounding error
                    self.epsilon = 0.0
                    
                
                
            self.ep_rewards.append(self.r.totalReward)
            self.ep_sequences.append(self.r.moveSeq)

            if not ep % self.SAVE_EVERY:
                average_reward = sum(self.ep_rewards[-self.SAVE_EVERY:])/len(self.ep_rewards[-self.SAVE_EVERY:])
                self.aggr_ep_rewards['ep'].append(ep+self.offset)
                self.aggr_ep_rewards['avg'].append(average_reward)
                self.aggr_ep_rewards['min'].append(min(self.ep_rewards[-self.SAVE_EVERY:]))
                self.aggr_ep_rewards['max'].append(max(self.ep_rewards[-self.SAVE_EVERY:]))
                self.aggr_ep_rewards['maxSeq'].append(self.ep_sequences[-self.SAVE_EVERY:][self.ep_rewards[-self.SAVE_EVERY:].index(max(self.ep_rewards[-self.SAVE_EVERY:]))])

            #optinal plotting every 500000 eps
            #if not ep % 50000 and not ep==0:
                #self.plot()

            #if good run then stop
            if (self.aggr_ep_rewards['avg'])[-1:][0]>240 and self.r.totalReward > 240:
                break
            
        if(self.resets>0):
            self.reset()
            
            
    def printTable(self,n):
        self.qt.printTable(n)
    
    def plot(self):
        plt.plot(self.aggr_ep_rewards['ep'],self.aggr_ep_rewards['avg'],label='avg')
        plt.plot(self.aggr_ep_rewards['ep'],self.aggr_ep_rewards['max'],label='max')
        plt.legend(loc=4)
        plt.show()
        
    def play(self): 
        self.r.reset()
        self.r.visualize= True
        self.r.render()
        while not self.r.over:
            self.r.step(self.nextAction(self.r.env))
            self.r.render()
        print("Total reward "+str(self.r.totalReward))
        
    def getMoveSeq(self):
        self.r.reset()
        while not self.r.over:
            self.r.step(self.nextAction(self.r.env))
        return self.r.moveSeq
        
    def nextAction(self,env):
        return self.qt.getGreedyAction(env)
    
    def store(self,location):
        pickle.dump(self, open(location,"wb"))
    
    def restoreTable(self,location):
        self.qt = pickle.load(open(location,"rb")).qt
        
    


# In[17]:


agents = []
for i in range(0,100):
    a = AgentQ(100000,0.075,0.05)
    print(i)
    a.train()
    a.plot()
    agents.append(a)



# In[32]:


r2 = Room()
r2.visualizeMoveSeq(agents[73].aggr_ep_rewards['maxSeq'][5])

