import pandas as pd
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

def Signum(x):
    if x>0:
        return 1
    elif x<0:
        return -1
    elif x==0:
        return 0

def getdata(filename):
    data = np.loadtxt(open("dataset.txt", "rb"), delimiter=",", skiprows=0)
    data = np.delete(data,0,axis=1)
    # data = ( data - np.mean(data, axis= 0) )/np.std(data, axis= 0)
    # data = ( data - np.min(data, axis= 0) )/(np.max(data, axis= 0)-np.min(data,axis= 0))

    # #print(data, [...])
    return data;
#data is in the numpy array

#have to split into train and test
def TrainTestSplit(data):
    train_size = int(data.shape[0]*80/100)
    test_size = int(data.shape[0]*20/100)
    #print(train_size,test_size, [...])
    train_data = data[0:train_size,:]
    #print(train_data,train_data.shape, [...])
    test_data = data[train_size:,:]
    #print(test_data,test_data.shape, [...])
    return train_data,test_data

def GetTrainPoints(data):
    return data[:,0],data[:,1],data[:,2]

# here split to make error for lasso and ridge
def GetLossAndGradient(w0,w1,w2,x_train,y_train,z_train,lamda,l2=False):

    zcap = w0 + x_train * w1 + y_train * w2
    #print('z_train ', z_train, z_train.shape)
    #print('zcap ', zcap,zcap.shape)
    err =  -z_train + zcap
    #print( np.sum( err**2 ) )
    #print('err ',np.sum(err),err.shape)
    #input()
    if l2:
        dw0 = (np.sum(err) + lamda*w0)
        #print(dw0, [...])
        dw1 = (np.sum(np.multiply(err,x_train)) + lamda*w1)
        #print(dw1)
        dw2 = (np.sum(np.multiply(err,y_train)) + lamda*w2)
        #print(dw2)
        errsq = np.square(err)
        #print(errsq,errsq.shape)
        E = 0.5*np.sum(errsq) + 0.5*lamda*(w0*w0+w1*w1+w2*w2)
        #print(E, [...])
        return E,dw0,dw1,dw2 #half sum of erors of all the points
    else:
        dw0 = (np.sum(err) + lamda*Signum(w0))
        #print(dw0, [...])
        dw1 = (np.sum(np.multiply(err,x_train)) + lamda*Signum(w1))
        #print(dw1)
        dw2 = (np.sum(np.multiply(err,y_train)) + lamda*Signum(w2))
        #print(dw2)
        errsq = np.square(err)
        #print(errsq,errsq.shape)
        E = 0.5*np.sum(errsq) + lamda*(abs(w0)+abs(w1)+abs(w2))
        #print(E, [...])
        return E,dw0,dw1,dw2 #half sum of erors of all the points

#doing for lasso regression
def GD(x_train,y_train,z_train,lamda,l2=False):
    #what is the stopping criterion
    w0,w1,w2 = np.random.normal(),np.random.normal(),np.random.normal()
    #learnign  rate
    eta = 0.000001
    numberOfEpochs = 10000
    # range lamda from 1 to 1000000
    prev_loss = float('inf')
    for x in range(numberOfEpochs):
        Eold,dw0,dw1,dw2 = GetLossAndGradient(w0,w1,w2,x_train,y_train,z_train,lamda,l2)
        # print(Eold)
        # loss.append(Eold)
        #print('grad_w', dw0, dw1, dw2)
        w0 = w0 - eta * dw0
        w1 = w1 - eta * dw1
        w2 = w2 - eta * dw2
        print(lamda,"\t",Eold,w0,w1,w2)

        if abs(Eold-prev_loss)<1e-7:
            break;
        prev_loss = Eold

    return w0, w1, w2,Eold

def NormalEquation(data):
    x = data[:,:2]
    y = data[:,2]

    x = np.hstack((np.ones((x.shape[0],1)),x))

    w = np.linalg.inv((x.T).dot(x)).dot(x.T).dot(y)
    return w.tolist()

def GetLoss(w0,w1,w2,data):
    x_test,y_test,z_test = data[:,0],data[:,1],data[:,2]
    zcap = w0 + x_test * w1 + y_test * w2
    err = zcap - z_test
    errsq = np.square(err)
    E = np.sum(errsq)
    return 0.5*E

def GetFullError(data):
    z_test = data[:,2]
    deviation = z_test-np.mean(z_test)
    devsq = np.square(deviation)
    E = np.sum(devsq)
    return 0.5*E

def main():
    data = getdata("dataset.txt")
    #print(data,data.shape)
    #to randomly pick the train and test
    np.random.shuffle(data)
    #print(data,data.shape)
    train_data,test_data = TrainTestSplit(data)
    #save the training data and test data and use it for all the cases
    print('traindata',train_data)
    print('testdata',test_data)
    x_train,y_train,z_train = GetTrainPoints(train_data)
    #print(x_train,y_train,z_train)
    #once the poitns are there z_poitns is the true value

    w = NormalEquation(train_data)
    print('NormalEquation',w[0],w[1],w[2])
    test_loss = GetLoss(w[0],w[1],w[2],test_data)
    train_loss = GetLoss(w[0],w[1],w[2],train_data)
    print('Test Loss for this data ',test_loss)
    print('Train Loss:',train_loss)

    input()
    #prit from:10000
    #high:3000000
    #low:
    lamda_list = []
    test_err = []
    train_err = []
    min_test_loss=float('inf')
    min_w0=0
    min_w1=0
    min_w2=0
    min_lamda = 0

    for iter in range(-10,0,1):
        param = 1*np.exp(iter)
        # param = iter
        print('With Lamda: ',param,'iter: ',iter)
        w0_l2,w1_l2,w2_l2,err_l2 = GD(x_train,y_train,z_train,param,l2=True)
        # print("non stochastic",w0_ns,w1_ns,w2_ns)
        test_loss = GetLoss(w0_l2,w1_l2,w2_l2,test_data)
        train_loss = GetLoss(w0_l2,w1_l2,w2_l2,train_data)
        print('Test Loss: ',iter,'is ',test_loss)
        if test_loss<min_test_loss:
            min_test_loss = test_loss
            min_w0,min_w1,min_w2,min_lamda = w0_l2,w1_l2,w2_l2,param
        print('Train Loss:',iter,'is',err_l2)
        print(w0_l2,w1_l2,w2_l2)
        print("\n")
        test_err.append(test_loss)
        train_err.append(train_loss)
        lamda_list.append(iter)
        # plt.plot(lamda_list,test_err,'r')
        # plt.plot(lamda_list,train_err,'b')
        # # plt.savefig('Pics/'+str(iter)+'.png')
        # plt.show()
    #r^2 calculation
    print(min_test_loss,min_w0,min_w1,min_w2,min_lamda)
    print("Full error:",GetFullError(test_data))
    print('r^2:',min_test_loss/GetFullError(test_data))



    plt.plot(lamda_list,test_err,'r')
    plt.show()


    lamda_list = []
    test_err = []
    train_err = []
    min_test_loss=float('inf')
    min_w0=0
    min_w1=0
    min_w2=0
    min_lamda = 0



    # values = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,0,1,10,20,30,100]
    for iter in range(-10,0,1):
        param = 1*np.exp(iter)
        # param = iter
        print('With Lamda: ',param,'iter: ',iter)
        w0_l2,w1_l2,w2_l2,err_l2 = GD(x_train,y_train,z_train,param,l2=False)
        # print("non stochastic",w0_ns,w1_ns,w2_ns)
        test_loss = GetLoss(w0_l2,w1_l2,w2_l2,test_data)
        train_loss = GetLoss(w0_l2,w1_l2,w2_l2,train_data)
        print('Test Loss: ',iter,'is ',test_loss)

        if test_loss<min_test_loss:
            min_test_loss = test_loss
            min_w0,min_w1,min_w2,min_lamda = w0_l2,w1_l2,w2_l2,param

        print('Train Loss:',iter,'is',err_l2)
        print(w0_l2,w1_l2,w2_l2)
        print("\n")
        test_err.append(test_loss)
        train_err.append(train_loss)
        lamda_list.append(iter)
        # plt.plot(lamda_list,test_err,'r')
        # plt.plot(lamda_list,train_err,'b')
        # plt.savefig('Pics/'+str(iter)+'.png')
        # plt.show()
    #
    print(min_test_loss,min_w0,min_w1,min_w2,min_lamda)
    print("Full error:",GetFullError(test_data))
    print('r^2:',min_test_loss/GetFullError(test_data))
    #
    #
    plt.plot(lamda_list,test_err,'b')
    print(test_err)
    # plt.plot(lamda_list,train_err,'b')
    plt.show()

if __name__=="__main__":
    main()
