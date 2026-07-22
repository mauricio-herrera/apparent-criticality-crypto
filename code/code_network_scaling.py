import numpy as np, pandas as pd, matplotlib.pyplot as plt, networkx as nx
from pathlib import Path
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
OUT=Path('/mnt/data/jcn_submission_final/figures'); OUT.mkdir(parents=True,exist_ok=True)
TAB=Path('/mnt/data/jcn_submission_final/tables'); TAB.mkdir(parents=True,exist_ok=True)

def rho(A): return float(np.max(np.abs(np.linalg.eigvals(A))))
def scale(A,target):
 r=rho(A); return A*target/r

def make_net(K,rng,deg=4,target=.65):
 grp=np.r_[np.zeros(K//2,int),np.ones(K-K//2,int)]
 A=np.zeros((K,K)); p=deg/(K-1)
 for i in range(K):
  for j in range(K):
   if i!=j and rng.random() < p*(1.5 if grp[i]==grp[j] else .5): A[i,j]=rng.gamma(1.8,.22)
 for j in range(K): A[(j+1)%K,j]=max(A[(j+1)%K,j],.04)
 return scale(A,target),grp

def perron(A):
 vals,vecs=np.linalg.eig(A); v=np.abs(np.real(vecs[:,np.argmax(np.real(vals))])); return v/(np.linalg.norm(v)+1e-15)
def auc(y,s):
 y=np.asarray(y,int); s=np.asarray(s,float); r=rankdata(s); n1=y.sum(); n0=len(y)-n1
 return float((r[y==1].sum()-n1*(n1+1)/2)/(n1*n0))
def top_f1(B0,B):
 m=(B0>1e-12).ravel(); n=m.sum(); pred=np.zeros_like(m); pred[np.argpartition(B.ravel(),-n)[-n:]]=1
 tp=np.sum(pred&m); return float(2*tp/(2*n+1e-15))
def g_est(B,c,frac):
 K=len(c); out=np.zeros(K); q=max(2,int(frac*K))
 for k in range(K):
  rr=B[:,k]/np.maximum(c,1e-9); out[k]=max(0,np.median(np.partition(rr,q-1)[:q]))
 return out

rows=[]
for K in [10,25,50,100]:
 for T in [1000,5000,20000]:
  for rep in range(5):
   rng=np.random.default_rng(20260713+K*1000+T+rep)
   B0,grp=make_net(K,rng)
   c=rng.uniform(.6,1.4,K); c/=np.linalg.norm(c)
   g=rng.uniform(.6,1.4,K); g/=np.linalg.norm(g)
   # fixed low-rank contamination, rescaled so the population naive radius is 0.91
   D=np.outer(c,g)
   lo,hi=0.,2.
   for _ in range(12):
    mid=(lo+hi)/2
    if rho(B0+mid*D)<.91: lo=mid
    else: hi=mid
   Bhat=B0+((lo+hi)/2)*D
   Bobs=np.maximum(Bhat+rng.normal(0,.18/np.sqrt(T),(K,K)),0)
   gh=g_est(Bobs,c,max(.5,1-6/K)); Bcor=np.maximum(Bobs-np.outer(c,gh),0)
   r0,rn,rc=rho(B0),rho(Bobs),rho(Bcor)
   v0,vn,vc=perron(B0),perron(Bobs),perron(Bcor)
   support=(B0>1e-12).ravel().astype(int)
   P=np.eye(K)-np.outer(c,c)/(c@c)
   rows.append(dict(K=K,T=T,rep=rep,density=np.mean(B0>0),rho_true=r0,rho_naive=rn,rho_corrected=rc,
    radius_error_naive=abs(rn-r0),radius_error_corrected=abs(rc-r0),
    rel_frob_naive=np.linalg.norm(Bobs-B0)/np.linalg.norm(B0),rel_frob_corrected=np.linalg.norm(Bcor-B0)/np.linalg.norm(B0),
    support_auc_naive=auc(support,Bobs.ravel()),support_auc_corrected=auc(support,Bcor.ravel()),
    support_ap_naive=average_precision_score(support,Bobs.ravel()),support_ap_corrected=average_precision_score(support,Bcor.ravel()),
    support_f1_oracle_naive=top_f1(B0,Bobs),support_f1_oracle_corrected=top_f1(B0,Bcor),
    perron_cos_naive=abs(v0@vn),perron_cos_corrected=abs(v0@vc),
    projection_error=np.linalg.norm(P@Bobs-P@B0)/(np.linalg.norm(P@B0)+1e-15)))

df=pd.DataFrame(rows); df.to_csv(TAB/'network_scaling_results.csv',index=False)

# size scaling at T=5000
g=df[df['T'].eq(5000)].groupby('K').mean(numeric_only=True).reset_index()
fig,ax=plt.subplots(2,2,figsize=(10.5,8))
items=[('radius_error','Absolute Perron-radius error',(None,None)),('rel_frob','Relative network error',(None,None)),('support_ap','Edge average precision',(0,1.02)),('perron_cos','Perron-centrality cosine',(0,1.02))]
for a,(stem,lab,ylim) in zip(ax.flat,items):
 a.plot(g.K,g[f'{stem}_naive'],'o-',label='Naive')
 a.plot(g.K,g[f'{stem}_corrected'],'s-',label='Deconfounded')
 a.set_xscale('log'); a.set_xlabel('Number of nodes K'); a.set_ylabel(lab); a.grid(alpha=.2)
 if ylim[0] is not None:a.set_ylim(*ylim)
ax[0,0].legend(frameon=False); fig.tight_layout(); fig.savefig(OUT/'fig_network_scaling.pdf',bbox_inches='tight'); plt.close(fig)

# sample scaling K=50
g=df[df.K.eq(50)].groupby('T').mean(numeric_only=True).reset_index()
fig,ax=plt.subplots(1,3,figsize=(11.2,3.4))
for a,stem,lab in zip(ax,['radius_error','rel_frob','support_ap'],['Perron-radius error','Relative network error','Edge average precision']):
 a.plot(g['T'],g[f'{stem}_naive'],'o-',label='Naive'); a.plot(g['T'],g[f'{stem}_corrected'],'s-',label='Deconfounded')
 a.set_xscale('log'); a.set_xlabel('Effective sample size T'); a.set_ylabel(lab); a.grid(alpha=.2)
ax[0].legend(frameon=False); fig.tight_layout(); fig.savefig(OUT/'fig_sample_scaling.pdf',bbox_inches='tight'); plt.close(fig)

# network visualization
rng=np.random.default_rng(4551); K=25; B0,grp=make_net(K,rng); c=rng.uniform(.6,1.4,K); c/=np.linalg.norm(c); gg=rng.uniform(.6,1.4,K); gg/=np.linalg.norm(gg); D=np.outer(c,gg)
lo,hi=0,2
for _ in range(12):
 mid=(lo+hi)/2
 if rho(B0+mid*D)<.91:lo=mid
 else:hi=mid
Bhat=B0+((lo+hi)/2)*D; Bobs=np.maximum(Bhat+rng.normal(0,.18/np.sqrt(5000),(K,K)),0); gh=g_est(Bobs,c,max(.5,1-6/K)); Bcor=np.maximum(Bobs-np.outer(c,gh),0)
G=nx.from_numpy_array(B0.T,create_using=nx.DiGraph); pos=nx.spring_layout(G,seed=9,k=.62,iterations=250); cent=perron(B0)
fig,axs=plt.subplots(1,3,figsize=(12.2,4.2))
for a,A,title in zip(axs,[B0,Bobs,Bcor],['True influence network','Naive inferred network','Deconfounded estimate']):
 vals=A[A>0]; thr=np.quantile(vals,.78); ed=[(j,i) for i in range(K) for j in range(K) if i!=j and A[i,j]>=thr]
 widths=[.3+2.5*A[i,j]/(A.max()+1e-12) for j,i in ed]
 nx.draw_networkx_nodes(G,pos,node_size=80+700*cent,node_color=grp,cmap='viridis',edgecolors='white',linewidths=.4,ax=a)
 nx.draw_networkx_edges(G,pos,edgelist=ed,width=widths,alpha=.42,arrows=True,arrowsize=6,connectionstyle='arc3,rad=.06',ax=a)
 a.set_title(title); a.axis('off')
fig.tight_layout(); fig.savefig(OUT/'fig_network_visualization.pdf',bbox_inches='tight'); plt.close(fig)

summary=df[df['T'].eq(5000)].groupby('K').mean(numeric_only=True).reset_index()
summary.to_csv(TAB/'network_scaling_T5000.csv',index=False)
print(summary[['K','density','rho_true','rho_naive','rho_corrected','rel_frob_naive','rel_frob_corrected','support_ap_naive','support_ap_corrected','perron_cos_naive','perron_cos_corrected']].round(3).to_string(index=False))
