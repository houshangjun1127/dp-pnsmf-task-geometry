"""Independent mathematical dense oracle; imports no prototype functions."""
import numpy as np

def indicator(histories,nf):
    A=np.zeros((len(histories),nf))
    for i,history in enumerate(histories): A[i,list(history)]=1
    return A

def users(items,histories,ridge,gravity,radius=None):
    A=indicator(histories,len(items)); answers=[]
    for i in range(len(A)):
        Hessian=items.T@np.diag(A[i]+gravity)@items+ridge*np.identity(items.shape[1])
        b=items.T@A[i]
        sol=np.linalg.inv(Hessian)@b
        if radius is not None and np.linalg.norm(sol)>radius: sol=radius*sol/np.linalg.norm(sol)
        answers.append(sol)
    return np.asarray(answers)

def psd(raw):
    eigenvalue,Q=np.linalg.eigh(.5*raw+.5*raw.T)
    D=np.diag([max(0,float(x)) for x in eigenvalue])
    return Q@D@Q.T

def items(U,histories,nf,ridge,gravity,radius,sigmas,draws,private):
    A=indicator(histories,nf); r=U.shape[1]
    H=U.T@U
    if private: H=H+sigmas[2]*radius**2*draws['shared']
    answers=[]; systems=[]
    for j in range(nf):
        G=U.T@np.diag(A[:,j])@U; b=U.T@A[:,j]
        if private:
            G=G+sigmas[0]*radius**2*draws['gram'][j]
            b=b+sigmas[1]*radius*draws['rhs'][j]
        system=psd(G)+gravity*psd(H)+ridge*np.identity(r)
        answers.append(np.linalg.inv(system)@b); systems.append(system)
    return np.asarray(answers),np.asarray(systems)

def user_objective(u,items,mask,ridge,gravity):
    prediction=items@u
    return np.sum(mask*(1-prediction)**2)+gravity*np.dot(prediction,prediction)+ridge*np.dot(u,u)

def item_objective(v,U,mask,ridge,gravity):
    prediction=U@v
    return np.sum(mask*(1-prediction)**2)+gravity*np.dot(prediction,prediction)+ridge*np.dot(v,v)
