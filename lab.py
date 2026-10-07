"""Pooling independent within-event scatter: precision versus class mismatch."""
from pathlib import Path
import json
import numpy as np
import optics as o
import clock_model as c

SEED=20261007
CLASSES=np.arange(len(c.SYNC))%2
MISMATCH=(1.,2.,4.,8.)


def estimate(repeats, labels=None):
    """Center EACH event separately; never pool different timestamp means.

    Pool residual scatter with its degrees of freedom. Equal repeat budgets
    make this the mean of event sample covariances, not covariance of raw data.
    """
    z=np.asarray(repeats,float)
    if z.ndim!=3 or z.shape[2]!=2 or z.shape[1]<3 or not np.all(np.isfinite(z)):
        raise ValueError('Require finite (events,m>=3,2) repeats')
    r=z-z.mean(axis=1,keepdims=True)
    s=np.einsum('nki,nkj->nij',r,r)/(z.shape[1]-1)
    if labels is not None:
        labels=np.asarray(labels)
        if labels.shape!=(len(z),) or labels.dtype.kind not in 'iu':
            raise ValueError('Require one integer quality class per event')
        s=np.stack([s[labels==g].mean(axis=0) for g in labels])
    if np.any(np.linalg.eigvalsh(s)[:,0]<=0):
        raise ValueError('Covariance is singular')
    return s


def covariances(mismatch):
    """Within each known quality class, half have q times larger noise SD."""
    if not np.isfinite(mismatch) or mismatch<1:
        raise ValueError('Require finite mismatch >=1')
    scale=np.where((np.arange(len(CLASSES))//2)%2,mismatch,1.)
    return c.covariances(8)*scale[:,None,None]**2


def predicted_relative_mse(cov,m=4):
    """Exact marginal variance-estimate MSE versus each event's true variance.

    Independent Gaussian scatter: Var(S_i,xx)=2*C_i,xx^2/(m-1).
    Pooling unequal covariances adds squared bias; Wishart common-scale
    interpretation applies ONLY when all covariances in the class agree.
    """
    v=cov[:,0,0];out=[]
    for i,g in enumerate(CLASSES):
        vg=v[CLASSES==g]
        out.append(((vg.mean()-v[i])**2+2*np.sum(vg**2)/((m-1)*len(vg)**2))/v[i]**2)
    return float(np.mean(out))


def summary(e):
    e=np.asarray(e)
    if not len(e):return dict(successful_trials=0)
    return dict(rmse_ms=float(np.sqrt(np.mean(e**2))),bias_ms=float(e.mean()),
                sd_ms=float(e.std(ddof=1)),mean_error_mcse_ms=float(e.std(ddof=1)/np.sqrt(len(e))),successful_trials=len(e))


def experiment(trials=600,seed=SEED):
    image_rng,sync_rng,repeat_rng=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    truth=np.column_stack((c.SYNC,c.RATE*c.SYNC+c.OFFSET))
    center=o.ORIGIN+o.VELOCITY*(c.RATE*(c.FRAME+c.DELAY)+c.OFFSET)
    centers=np.array([o.fit_center(o.image(center)+image_rng.normal(0,o.READ_SD,len(o.PIXELS))) for _ in range(trials)])
    # Common standard normals pair the mismatch levels; calibration and evaluation
    # streams remain independent. Every trial obtains fresh repeat calibration.
    eval_noise=sync_rng.normal(size=(trials,len(truth),2))
    repeat_noise=repeat_rng.normal(size=(trials,len(truth),4,2))
    rows=[]
    for q in MISMATCH:
        cov=covariances(q);chol=np.linalg.cholesky(cov)
        errors={k:[] for k in ['oracle','per_event','class_pool']};failed={k:0 for k in errors};mse=[]
        for t,u in enumerate(centers):
            z=truth+np.einsum('nij,nj->ni',chol,eval_noise[t])
            repeats=truth[:,None,:]+np.einsum('nij,nkj->nki',chol,repeat_noise[t])
            pooled=estimate(repeats,CLASSES)
            mse.append(np.mean((pooled[:,0,0]/cov[:,0,0]-1)**2))
            for name,assumed in [('oracle',cov),('per_event',estimate(repeats)),('class_pool',pooled)]:
                try:errors[name].append(1000*(c.delay_from_center(u,c.calibrate(*z.T,assumed))-c.DELAY))
                except ValueError:failed[name]+=1
        rows.append(dict(mismatch_sd_factor=q,pooled_relative_variance_mse=float(np.mean(mse)),
                         predicted_mse=predicted_relative_mse(cov),
                         **{k:dict(**summary(e),failed_trials=failed[k]) for k,e in errors.items()}))
    return dict(seed=seed,trials=trials,repeats_per_event=4,events_per_class=20,
                sigma_px=o.SIGMA,center_sd_px=float(centers.std(ddof=1)),true_delay_ms=40.,experiment=rows)


def demo(out='outputs'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=experiment();rows=report['experiment'];out=Path(out);out.mkdir(exist_ok=True,parents=True)
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    center=o.ORIGIN+o.VELOCITY*(c.RATE*(c.FRAME+c.DELAY)+c.OFFSET)
    ax[0].plot(o.PIXELS,o.image(center)-o.BACKGROUND,'o-')
    ax[0].set(xlim=(-1,5),xlabel='Sensor coordinate (px)',ylabel='Integrated signal (ADU)',title='Diffraction -> pixels -> optical time')
    for key,label in [('pooled_relative_variance_mse','Simulation'),('predicted_mse','Exact bias + variance')]:
        ax[1].loglog(MISMATCH,[r[key] for r in rows],'o-',label=label)
    from matplotlib.ticker import NullLocator
    ax[1].set_xticks(MISMATCH, [str(int(q)) for q in MISMATCH])
    ax[1].xaxis.set_minor_locator(NullLocator())
    ax[1].set(xlabel='Within-class noise SD ratio q',ylabel='Relative variance-estimate MSE',title='Pooling assumes shared covariance');ax[1].legend(fontsize=8)
    for key,label in [('oracle','Known covariance'),('per_event','Four repeats per event'),('class_pool','Pool within two classes')]:
        ax[2].plot(MISMATCH,[r[key]['rmse_ms'] for r in rows],'o-',label=label)
    ax[2].set(xlabel='Within-class noise SD ratio q',ylabel='Camera-delay RMSE (ms)',title='Precision benefit can reverse');ax[2].legend(fontsize=8)
    for a in ax:a.grid(alpha=.2)
    fig.savefig(out/'experiment.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))
    if any(r[k]['failed_trials'] for r in rows for k in ['oracle','per_event','class_pool']):
        raise RuntimeError('Fit failures recorded; reported RMSE conditions on success')

if __name__=='__main__':demo()
