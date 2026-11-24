"""CPU-only derivations from EXISTING HW12 artifacts. No training or model inference.
Usage (from the hw12 folder):  python3 src/derive_metrics.py
Reads  results/knowledge_distillation/teacher_logits/teacher_all_120000_logits.npy
Writes results/knowledge_distillation/derived_metrics.json"""
import json, math, sys
from pathlib import Path
import numpy as np
root = Path(__file__).resolve().parents[1]; KD = root / "results/knowledge_distillation"; out = {}

# 1) Teacher soft-target statistics from the saved (120000,4) logit cache.
z = np.load(KD / "teacher_logits/teacher_all_120000_logits.npy").astype(np.float64)
assert z.shape == (120000, 4) and np.isfinite(z).all()
soft = {}
for T in [1, 2, 5, 10]:
    s = z / T; s -= s.max(1, keepdims=True); p = np.exp(s); p /= p.sum(1, keepdims=True)
    H = -(p * np.log(p)).sum(1)
    ps = np.sort(p, 1)
    soft[str(T)] = {"mean_entropy_nats": H.mean(), "mean_entropy_over_log4": H.mean()/math.log(4),
                    "mean_max_prob": ps[:, -1].mean(), "mean_second_prob": ps[:, -2].mean(),
                    "mean_nontarget_mass": 1 - ps[:, -1].mean()}
out["teacher_soft_targets"] = soft
out["teacher_pred_class_counts_120k"] = np.bincount(z.argmax(1), minlength=4).tolist()
out["teacher_logit_margin_median"] = float(np.median(np.sort(z,1)[:,-1]-np.sort(z,1)[:,-2]))

# 2) Teacher per-class P/R/F1 from the confusion matrix printed in the executed teacher evaluation
#    (notebooks/supporting/DATA266_HW12_Distillation_working_notebook_executed.ipynb, cell 23).
cm = np.array([[1804,10,50,36],[9,1878,6,7],[33,6,1736,125],[28,9,102,1761]])
assert cm.trace() == 7179 and cm.sum() == 7600
names = ["World","Sports","Business","Sci/Tech"]; pr = {}
for i,n in enumerate(names):
    P = cm[i,i]/cm[:,i].sum(); R = cm[i,i]/cm[i].sum(); pr[n] = {"precision":P,"recall":R,"f1":2*P*R/(P+R)}
out["teacher_per_class"] = pr
out["teacher_macro_f1_from_cm"] = float(np.mean([v["f1"] for v in pr.values()]))

# 3) Paired teacher/student contingency from recorded counts + exact McNemar test.
t_corr, s_corr, n = 7179, round(0.946579*7600), 7600
b = 92                                 # teacher correct, student wrong (recorded)
both = t_corr - b; c = s_corr - both   # student correct, teacher wrong (derived)
k, m = min(b,c), b+c
p_exact = min(1.0, 2*sum(math.comb(m,i) for i in range(k+1))/2**m)
out["paired"] = {"student_correct": s_corr, "both_correct": both, "teacher_only": b,
                 "student_only": c, "both_wrong": n-both-b-c, "mcnemar_exact_p": p_exact,
                 "mcnemar_chi2_cc": (abs(b-c)-1)**2/m}
# 95% normal-approx half-width for a 12,000-example validation accuracy near 0.948
out["val_acc_95ci_halfwidth_pp"] = 1.96*math.sqrt(0.948*0.052/12000)*100

# 4) Student per-class correct counts from recorded per-class accuracies.
out["student_per_class_correct"] = {nm: round(a/100*1900) for nm,a in zip(names,[95.4211,98.8947,91.2105,93.1053])}

# 5) Equivalent unscaled weight: aCE+(1-a)T^2KL is proportional to a'CE+(1-a')KL.
out["equivalent_unscaled_alpha"] = {f"T{T}_a{a}": a/(a+(1-a)*T*T) for T,a in
     [(1,.5),(2,.5),(5,.5),(10,.5),(5,.25),(5,.75)]}

# 6) Analytical OPT-125M storage estimate for 4-bit GPTQ (g=128) with embeddings kept FP16.
d, L, V, P, ff = 768, 12, 50272, 2050, 3072
lin = L*(4*d*d + 2*d*ff); emb = V*d + P*d; total = 125239296
other = total - lin - emb
bpw = 4 + (16+4)/128  # per-group FP16 scale + 4-bit packed zero point (approx.)
est = (lin*bpw/8 + (emb+other)*2)/2**20
out["gptq_size_model"] = {"quantized_linear_params": lin, "embedding_params": emb, "other_params": other,
   "quantized_fraction": lin/total, "approx_bits_per_weight": bpw, "estimated_MiB": est,
   "fp16_all_MiB": total*2/2**20}
json.dump(out, open(KD / "derived_metrics.json", "w"), indent=2, default=float)
print(json.dumps(out, indent=1, default=float))
