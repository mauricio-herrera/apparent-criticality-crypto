import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import SplineTransformer

OUT=Path('/mnt/data/jcn_final_calc')
OUT.mkdir(exist_ok=True)
RNG=np.random.default_rng(20260713)


def spectral_radius(B):
    return float(np.max(np.abs(np.linalg.eigvals(B))))

def perron(B):
    vals, vecs=np.linalg.eig(B)
    j=np.argmax(np.abs(vals)); v=np.abs(np.real(vecs[:,j])); return v/(np.linalg.norm(v)+1e-12)

def make_network(K, degree=4, target_rho=.65, seed=0):
    rng=np.random.default_rng(seed)
    B=np.zeros((K,K))
    groups=np.arange(K)%2
    for j in range(K):
        same=np.where(groups==groups[j])[0]
        other=np.where(groups!=groups[j])[0]
        ns=min(degree-1,len(same))
        no=1 if len(other) else 0
        sel=list(rng.choice(same,size=ns,replace=False))+list(rng.choice(other,size=no,replace=False))
        for i in sel: B[i,j]=rng.uniform(.05,.18)
    np.fill_diagonal(B,np.maximum(np.diag(B),rng.uniform(.04,.10,size=K)))
    B*=target_rho/max(spectral_radius(B),1e-12)
    return B

def simulate_counts(B,T,drive_strength=.16,seed=0):
    rng=np.random.default_rng(seed); K=B.shape[0]
    c=rng.uniform(.6,1.4,K); c/=np.mean(c)
    # smooth nonstationary common drive with pulses + oscillation, always nonnegative
    t=np.arange(T)
    f=.35+.18*np.sin(2*np.pi*t/700)+.10*np.sin(2*np.pi*t/173)
    for center in [int(.25*T),int(.55*T),int(.78*T)]:
        f += .8*np.exp(-0.5*((t-center)/(35+center%17))**2)
    f=np.maximum(f,0)
    mu=np.full(K,.015)
    Y=np.zeros((T,K),dtype=float); H=np.zeros((T,K),dtype=float)
    decay=np.exp(-np.log(2)/8.0)
    for tt in range(1,T):
        H[tt]=decay*H[tt-1]+(1-decay)*Y[tt-1]
        lam=mu+B@H[tt]+drive_strength*c*f[tt]
        lam=np.clip(lam,1e-6,1.5)
        Y[tt]=rng.poisson(lam)
    return Y,H,f,c

def fit_rows(Y,H,extra=None,positive=True,ridge=1e-3):
    T,K=Y.shape
    cols=[np.ones(T),H]
    if extra is not None:
        E=extra if extra.ndim==2 else extra[:,None]
        cols.append(E)
    X=np.column_stack(cols)
    p=X.shape[1]
    R=ridge*np.eye(p); R[0,0]=0
    coef=np.linalg.solve(X.T@X+R, X.T@Y)
    B=coef[1:1+K,:].T
    if positive: B=np.maximum(B,0.0)
    pred=X@coef
    return B,pred

def metrics(B0,Bhat):
    support=(B0.ravel()>1e-10).astype(int); score=Bhat.ravel()
    auc=roc_auc_score(support,score); ap=average_precision_score(support,score)
    err=np.linalg.norm(Bhat-B0,'fro')/(np.linalg.norm(B0,'fro')+1e-12)
    rho=spectral_radius(Bhat); rho0=spectral_radius(B0)
    align=float(abs(np.dot(perron(B0),perron(Bhat))))
    # top-k precision only as descriptive ranking, clearly oracle-cardinality
    n=int(support.sum()); idx=np.argsort(score)[::-1][:n]; pred=np.zeros_like(support); pred[idx]=1
    tp=np.sum((pred==1)&(support==1)); prec=tp/max(np.sum(pred),1); rec=tp/max(np.sum(support),1)
    f1=2*prec*rec/max(prec+rec,1e-12)
    return dict(rho=rho,rho_error=abs(rho-rho0),adj_rel_error=err,roc_auc=auc,average_precision=ap,perron_alignment=align,oracle_cardinality_f1=f1)

def run_one(K,T,rep):
    B0=make_network(K,seed=1000*K+rep)
    Y,H,f,c=simulate_counts(B0,T,seed=9000*K+rep)
    burn=max(300,T//20); Y=Y[burn:]; H=H[burn:]; f=f[burn:]
    # naive constant baseline
    Bn,_=fit_rows(Y,H)
    # flexible baseline: cubic splines of scaled time (does not observe f)
    x=np.linspace(0,1,len(Y))[:,None]
    spl=SplineTransformer(n_knots=10,degree=3,include_bias=False).fit_transform(x)
    Bflex,_=fit_rows(Y,H,extra=spl)
    # observed common-drive adjustment
    Badj,_=fit_rows(Y,H,extra=f)
    rows=[]
    for name,B in [('constant_baseline',Bn),('flexible_spline_baseline',Bflex),('observed_drive_adjusted',Badj)]:
        m=metrics(B0,B); m.update(K=K,T=T,rep=rep,method=name,rho_true=spectral_radius(B0))
        rows.append(m)
    return rows

rows=[]
# Feasible end-to-end benchmark: medium and larger networks.
for K in [20,50,100]:
    for T in [5000,15000]:
        reps=4 if K<100 else 2
        for rep in range(reps):
            print('run',K,T,rep,flush=True)
            rows.extend(run_one(K,T,rep))

df=pd.DataFrame(rows)
df.to_csv(OUT/'end_to_end_hawkes_results.csv',index=False)
summary=df.groupby(['K','T','method']).agg(['mean','std']).reset_index()
# flatten
summary.columns=['_'.join([str(x) for x in c if x!='']).rstrip('_') for c in summary.columns]
summary.to_csv(OUT/'end_to_end_hawkes_summary.csv',index=False)
print(summary.to_string(index=False))
