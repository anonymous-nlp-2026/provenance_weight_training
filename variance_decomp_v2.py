import json, numpy as np, sys, os

# Pure numpy versions of alpha_eff functions (avoid torch overhead on tiny arrays)
def alpha_eff_np(q, b):
    omq = np.clip(1.0 - q, 1e-30, 1.0)
    if b == 0.0:
        return float(omq.mean())
    log_w = b * np.log(omq)
    log_w -= log_w.max()
    w = np.exp(log_w)
    return float((w * omq).sum() / w.sum())

def ess_np(q, b):
    if b == 0.0:
        return float(len(q))
    omq = np.clip(1.0 - q, 1e-30, 1.0)
    log_w = b * np.log(omq)
    log_w -= log_w.max()
    w = np.exp(log_w)
    sw = w.sum()
    return float(sw**2 / (w**2).sum())

def find_b_adaptive_tau_np(q, tau=0.7, n_min=2, tau_delta=0.3, tau_min=0.5, b_max=20.0, res=200):
    a0 = alpha_eff_np(q, 0.0)
    tau_b = max(tau_min, min(tau, a0 + tau_delta))
    if a0 >= tau_b:
        return 0.0, tau_b
    n = len(q)
    if n_min > n:
        n_min = n
    best_b, best_a = 0.0, a0
    thresh_b = None
    for bv in np.linspace(0, b_max, res + 1):
        e = ess_np(q, bv)
        if e < n_min:
            break
        a = alpha_eff_np(q, bv)
        if a > best_a:
            best_a = a
            best_b = bv
        if a >= tau_b and thresh_b is None:
            thresh_b = bv
    if thresh_b is not None:
        lo, hi = max(0, thresh_b - b_max/res), thresh_b
        for _ in range(20):
            mid = (lo + hi) / 2
            if alpha_eff_np(q, mid) >= tau_b:
                hi = mid
            else:
                lo = mid
        if ess_np(q, hi) >= n_min:
            return hi, tau_b
    return best_b, tau_b

# Part 1: Empirical
print('=== Part 1: Empirical alpha_eff variance ===')
def load_sm(p):
    with open(p) as f:
        return [json.loads(l) for l in f]

b4p = ['output/models/tau07_seed42/step_metrics.jsonl',
       'output/models/tau07_seed123/step_metrics.jsonl',
       'output/models/tau07_seed456/step_metrics.jsonl']
b16p = ['output/models/bs16_adaptive_tau07_seed42/step_metrics.jsonl',
        'output/models/bs16_adaptive_tau07_seed123/step_metrics.jsonl',
        'output/models/bs16_adaptive_tau07_seed456/step_metrics.jsonl']

emp = {}
for lab, ps in [('B=4', b4p), ('B=16', b16p)]:
    aa, amq, ab = [], [], []
    for p in ps:
        if not os.path.exists(p): continue
        m = load_sm(p)
        a = [x['alpha_eff'] for x in m]
        aa.extend(a)
        amq.extend([x['mean_q'] for x in m])
        ab.extend([x['b'] for x in m])
        print(f'  {os.path.basename(os.path.dirname(p))}: n={len(m)} a_mean={np.mean(a):.4f} a_var={np.var(a):.6f}')
    arr = np.array(aa)
    emp[lab] = {'a_var': float(arr.var()), 'a_mean': float(arr.mean()), 'a_std': float(arr.std()),
                'mq_var': float(np.var(amq)), 'mean_b': float(np.mean(ab)), 'n': len(arr)}
    print(f'  {lab} total: var={arr.var():.6f} std={arr.std():.4f} mean={arr.mean():.4f}')
vr = emp['B=4']['a_var'] / emp['B=16']['a_var']
print(f'Var ratio B4/B16: {vr:.2f}')

# Part 2: Simulation
print('
=== Part 2: Simulation ===')
data_q, data_r = [], []
with open('/root/provenance_weight_training/data/scored_data.jsonl') as f:
    for l in f:
        d = json.loads(l)
        data_q.append(d['q_score'])
        data_r.append(1 if d['depth'] == 0 else 0)
qa, ra = np.array(data_q), np.array(data_r)
print(f'N={len(qa)}, real_frac={ra.mean():.4f}')

np.random.seed(42)
NS = 2000
tau, nm = 0.7, 2
sr = {}

for B in [4, 16]:
    tf, aad, ab0, abn, bs = [], [], [], [], []
    for i in range(NS):
        idx = np.random.choice(len(qa), size=B, replace=False)
        bq, br = qa[idx], ra[idx]
        bst, _ = find_b_adaptive_tau_np(bq, tau=tau, n_min=nm)
        tf.append(float(br.mean()))
        aad.append(alpha_eff_np(bq, bst))
        ab0.append(alpha_eff_np(bq, 0.0))
        abn.append(float((bq < 0.5).mean()))
        bs.append(bst)
        if (i+1) % 500 == 0:
            print(f'  B={B}: {i+1}/{NS} done', flush=True)
    tf, aad, ab0, abn, bs = [np.array(x) for x in [tf, aad, ab0, abn, bs]]
    
    def r2f(x, y):
        c = np.corrcoef(x, y)[0, 1]
        return float(c), float(c**2)
    rb0, r2b0 = r2f(ab0, tf)
    rbn, r2bn = r2f(abn, tf)
    rad, r2ad = r2f(aad, tf)
    gap = 1.0 - r2ad
    dn = 1.0 - r2b0
    be = r2b0 - r2ad
    
    print(f'B={B}: R2_soft={r2b0:.4f} R2_bin={r2bn:.4f} R2_adp={r2ad:.4f}')
    print(f'  gap={gap:.4f} det_noise={dn:.4f}({dn/gap*100:.1f}%) b_eff={be:.4f}({be/gap*100:.1f}%)')
    print(f'  a_adp: mean={aad.mean():.4f} var={aad.var():.6f}')
    print(f'  true:  mean={tf.mean():.4f} var={tf.var():.6f}')
    print(f'  b*: mean={bs.mean():.2f} std={bs.std():.2f} b0={float((bs==0).mean()):.3f}')
    sr[f'B{B}'] = {'r2_soft':r2b0,'r2_bin':r2bn,'r2_adp':r2ad,'gap':gap,
                   'dn_frac':dn/gap if gap>0 else 0,'be_frac':be/gap if gap>0 else 0}

# Part 3: Detector
print('
=== Part 3: Detector ===')
rq, sq = qa[ra==1], qa[ra==0]
fp = float((rq > 0.5).mean())
fn = float((sq < 0.5).mean())
acc = float(((qa < 0.5) == (ra == 1)).mean())
print(f'Acc={acc:.4f} FP={fp:.4f}({(rq>0.5).sum()}/{len(rq)}) FN={fn:.4f}({(sq<0.5).sum()}/{len(sq)})')
for lo, hi in [(0.3,0.7),(0.4,0.6),(0.45,0.55)]:
    mk = (qa>=lo)&(qa<=hi)
    n = mk.sum()
    ac = float(((qa[mk]<0.5)==(ra[mk]==1)).mean()) if n>0 else 0
    print(f'  q[{lo},{hi}]: {n}({n/len(qa)*100:.1f}%) acc={ac:.3f}')

# Part 4: Theory
print('
=== Part 4: Theoretical ===')
ps = 1-ra.mean()
from scipy.stats import binom as bn
for B in [4,8,16,32]:
    bv = ps*(1-ps)/B
    line = f'  B={B:2d}: Var={bv:.6f} Std={np.sqrt(bv):.4f}'
    if B <= 16:
        line += f' P(all_s)={bn.pmf(B,B,ps):.6f} P(all_r)={bn.pmf(0,B,ps):.6f}'
    print(line)

# Save
out = {'empirical':emp,'var_ratio':vr,'simulation':sr,
       'detector':{'acc':acc,'fp':fp,'fn':fn,'bnd_04_06':int(((qa>=0.4)&(qa<=0.6)).sum())},
       'data':{'N':len(qa),'real':int(ra.sum()),'synth':int((1-ra).sum()),'real_frac':float(ra.mean())}}
with open('/root/provenance_weight_training/alpha_eff_variance_decomposition.json','w') as f:
    json.dump(out,f,indent=2)
print('
Saved JSON')
