
import json, numpy as np, torch, sys, os
sys.path.insert(0, '/root/provenance_weight_training')
from weighting.alpha_eff import compute_alpha_eff, compute_ess, find_optimal_b_adaptive_tau

def load_sm(path):
    m = []
    with open(path) as f:
        for l in f:
            m.append(json.loads(l))
    return m

print('Part 1: Empirical alpha_eff variance')
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
        if not os.path.exists(p):
            print(f'MISSING: {p}')
            continue
        m = load_sm(p)
        aa.extend([x['alpha_eff'] for x in m])
        amq.extend([x['mean_q'] for x in m])
        ab.extend([x['b'] for x in m])
        print(f'  {os.path.basename(os.path.dirname(p))}: n={len(m)} alpha_mean={np.mean([x["alpha_eff"] for x in m]):.4f} var={np.var([x["alpha_eff"] for x in m]):.6f}')
    a = np.array(aa)
    emp[lab] = {'alpha_var': float(a.var()), 'alpha_mean': float(a.mean()), 'alpha_std': float(a.std()),
                'mq_var': float(np.var(amq)), 'mean_b': float(np.mean(ab)), 'n': len(a)}
    print(f'  {lab}: var={a.var():.6f} std={a.std():.4f} mean={a.mean():.4f}')

vr = emp['B=4']['alpha_var'] / emp['B=16']['alpha_var']
print(f'Var ratio B4/B16: {vr:.2f}')
print(f'mean_q var ratio: {emp["B=4"]["mq_var"]/emp["B=16"]["mq_var"]:.2f}')

print()
print('Part 2: Simulation')
data = []
with open('/root/provenance_weight_training/data/scored_data.jsonl') as f:
    for l in f:
        d = json.loads(l)
        data.append({'q': d['q_score'], 'r': 1 if d['depth'] == 0 else 0})
qa = np.array([d['q'] for d in data])
ra = np.array([d['r'] for d in data])
print(f'N={len(data)}, real_frac={ra.mean():.4f}')

np.random.seed(42)
NS = 5000
tau, nm = 0.7, 2
sr = {}

for B in [4, 16]:
    tf, aad, ab0, abn, bs = [], [], [], [], []
    for _ in range(NS):
        idx = np.random.choice(len(data), size=B, replace=False)
        bq, br = qa[idx], ra[idx]
        qt = torch.tensor(bq, dtype=torch.float32)
        bst, _ = find_optimal_b_adaptive_tau(qt, tau=tau, n_min=nm, tau_delta=0.3, tau_min=0.5)
        tf.append(float(br.mean()))
        aad.append(compute_alpha_eff(qt, bst))
        ab0.append(compute_alpha_eff(qt, 0.0))
        abn.append(float((bq < 0.5).mean()))
        bs.append(bst)
    tf, aad, ab0, abn, bs = [np.array(x) for x in [tf, aad, ab0, abn, bs]]
    
    def r2(x, y):
        c = np.corrcoef(x, y)[0, 1]
        return float(c), float(c**2)
    rb0, r2b0 = r2(ab0, tf)
    rbn, r2bn = r2(abn, tf)
    rad, r2ad = r2(aad, tf)
    gap = 1.0 - r2ad
    dn = 1.0 - r2b0
    be = r2b0 - r2ad
    
    print(f'B={B}: R2(soft)={r2b0:.4f} R2(binary)={r2bn:.4f} R2(adaptive)={r2ad:.4f}')
    print(f'  gap={gap:.4f} detector_noise={dn:.4f}({dn/gap*100:.1f}%) b_effect={be:.4f}({be/gap*100:.1f}%)')
    print(f'  alpha_adp: mean={aad.mean():.4f} var={aad.var():.6f}')
    print(f'  true_frac: mean={tf.mean():.4f} var={tf.var():.6f}')
    print(f'  b*: mean={bs.mean():.2f} std={bs.std():.2f} b0_frac={float((bs==0).mean()):.3f}')
    sr[f'B{B}'] = {'r2_soft': r2b0, 'r2_binary': r2bn, 'r2_adaptive': r2ad,
                   'gap': gap, 'dn_frac': dn/gap, 'be_frac': be/gap,
                   'alpha_var': float(aad.var()), 'true_var': float(tf.var())}

print()
print('Part 3: Detector errors')
rq, sq = qa[ra==1], qa[ra==0]
fp = float((rq > 0.5).mean())
fn = float((sq < 0.5).mean())
acc = float(((qa < 0.5) == (ra == 1)).mean())
print(f'Acc={acc:.4f} FP={fp:.4f}({(rq>0.5).sum()}/{len(rq)}) FN={fn:.4f}({(sq<0.5).sum()}/{len(sq)})')
for lo, hi in [(0.3,0.7),(0.4,0.6),(0.45,0.55)]:
    mk = (qa>=lo)&(qa<=hi)
    n = mk.sum()
    ac = float(((qa[mk]<0.5)==(ra[mk]==1)).mean()) if n>0 else 0
    print(f'  q[{lo},{hi}]: {n}({n/len(data)*100:.1f}%) acc={ac:.3f}')

print()
print('Part 4: Theoretical batch noise')
ps = 1-ra.mean()
from scipy.stats import binom as bn
for B in [4,8,16,32]:
    bv = ps*(1-ps)/B
    print(f'  B={B}: Var={bv:.6f} Std={np.sqrt(bv):.4f}', end='')
    if B <= 16:
        print(f' P(all_synth)={bn.pmf(B,B,ps):.6f} P(all_real)={bn.pmf(0,B,ps):.6f}', end='')
    print()

out = {'empirical': emp, 'var_ratio': vr, 'simulation': sr,
       'detector': {'acc': acc, 'fp': fp, 'fn': fn,
                    'boundary_04_06': int(((qa>=0.4)&(qa<=0.6)).sum()),
                    'boundary_pct': float(((qa>=0.4)&(qa<=0.6)).mean())},
       'data': {'N': len(data), 'real': int(ra.sum()), 'synth': int((1-ra).sum()), 'real_frac': float(ra.mean())}}
with open('/root/provenance_weight_training/alpha_eff_variance_decomposition.json', 'w') as f:
    json.dump(out, f, indent=2)
print('Saved JSON')
