# %% [markdown]
# # TdP Risk Predictor Notebook
# A physiology-aware, self-supervised deep-learning pipeline for predicting drug-induced torsade de pointes (TdP) risk from twitch-force traces of hiPSC-cardiomyocyte micro-tissues.
# 
# This script follows architecture below:
# 1. Load raw contraction data (Excel).
# 2. Basic QC & beat segmentation.
# 3. Biomechanical feature extraction (230+ explicit features).
# 4. Self-supervised representation learning (SimCLR on TCN->Transformer embeddings).
# 5. Ordinal risk prediction via CORAL.
# 6. Leave-one-drug-out evaluation + plots.
# 
# > **Compute note:** With typical STRIDE-96 data (10 s x 1 Hz x 100 beats/drug), the demo training finishes on CPU in a few minutes, but GPU is highly recommended for the SimCLR pre-training.

# %%

# %%capture
# Install any missing packages (uncomment as needed)
# !pip install pandas scipy numpy matplotlib seaborn scikit-learn torch torch-coral tqdm


# %%

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from tqdm.auto import tqdm
import scipy.signal as sig
import random, os, math, itertools, functools, warnings, time, collections
from pathlib import Path
sns.set_style('whitegrid')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using', device)


# %%

# ---------- USER PARAMETERS ----------
DATA_PATH_STIM = Path('/Users/Desktop/Projects/TDP_Risk_Machine_Learning_Pipeline/Raw_Contraction_10C_1Hz.xlsx')            # after stimulation
DATA_PATH_BASE = Path('/Users/Desktop/Projects/TDP_Risk_Machine_Learning_Pipeline/Raw_Contraction_10C_1Hz_Baseline.xlsx')   # baseline



SAMPLING_RATE = 75  # Hz (adjust if different)
MIN_PEAK_PROMINENCE = 0.05      # tweak for robust peak finding
MIN_BEATS_PER_TRACE = 2       # discard wells with fewer beats
BATCH_SIZE_SSL = 64
SSL_EPOCHS = 150                  # keep small for demo
PRED_EPOCHS = 100
LR_SSL = 1e-3
LR_PRED = 1e-3


# %%
# === DATA LOADER FOR BOTH FILES + GLOBAL TRIM ===
def _load_one_workbook(path: Path, condition: str):
    """
    Reads one workbook formatted like:
      - Col0: time vector (same length as traces) + a final label row to be ignored for time
      - Col1..N: trace columns, last row = risk label (Low/Intermediate/High) for that trace
    Returns:
      - data: dict[name] -> {drug,sample,trace,t,risk,condition}
      - min_len: shortest valid (non-NaN) trace length in this workbook (excluding label row)
    """
    df = pd.read_excel(path, header=0)
    # time column except the last row (labels row)
    time_vec = df.iloc[:-1, 0].astype(float).values
    risk_row = df.tail(1).iloc[0, 1:]

    data = {}
    min_len = np.inf

    for col in df.columns[1:]:
        raw = df[col].iloc[:-1].astype(float).values
        valid = ~np.isnan(raw)
        if not np.any(valid):  # empty column
            continue
        # keep only valid portion
        raw = raw[valid]
        t   = time_vec[valid]
        min_len = min(min_len, len(raw))

        # parse header "Drug_Sample" → (drug, sample)
        try:
            drug, sample = col.split('_', 1)
        except ValueError:
            drug, sample = col, "1"

        data[col] = dict(
            drug=drug,
            sample=sample,
            trace=raw,
            t=t,
            risk=str(risk_row[col]).strip(),
            condition=condition,    # "Baseline" or "Stim"
            original_name=col
        )

    if min_len is np.inf:
        min_len = 0
    return data, int(min_len)


def _trim_all(data_dict: dict, trim_len: int, fs: int):
    """
    Trim every trace in-place to the first trim_len samples.
    Also recompute a uniform time vector based on fs to avoid off-by-one issues.
    """
    if trim_len <= 0:
        return data_dict, 0
    t_uniform = np.arange(trim_len) / float(fs)
    for k, meta in data_dict.items():
        x = meta['trace']
        if len(x) >= trim_len:
            x = x[:trim_len]
        else:
            # If any trace is accidentally shorter, zero-pad to keep shapes consistent
            # (shouldn't happen given how we choose trim_len, but it's safe)
            x = np.pad(x, (0, trim_len - len(x)), mode='constant', constant_values=0.0)
        meta['trace'] = x
        meta['t'] = t_uniform.copy()
    return data_dict, trim_len


# --- Load both workbooks ---
stim_data, stim_min = _load_one_workbook(DATA_PATH_STIM, condition="Stim")
base_data, base_min = _load_one_workbook(DATA_PATH_BASE, condition="Baseline")

# --- Decide global trim length (shortest across both) ---
GLOBAL_TRIM_LEN = int(min(stim_min if stim_min>0 else np.inf,
                          base_min if base_min>0 else np.inf))
if GLOBAL_TRIM_LEN == np.inf or GLOBAL_TRIM_LEN <= 0:
    raise RuntimeError("No valid traces found in the provided workbooks.")

# === SIMPLE PAD/CROP using the already-trimmed length ===
MAX_LEN = GLOBAL_TRIM_LEN  # keep tensors exactly this long

# --- Apply trimming ---
stim_data, _ = _trim_all(stim_data, GLOBAL_TRIM_LEN, SAMPLING_RATE)
base_data, _ = _trim_all(base_data, GLOBAL_TRIM_LEN, SAMPLING_RATE)

# --- Merge into a single dictionary; make keys unique by prefixing condition ---
raw_data = {}
for k,v in base_data.items():
    raw_data[f"Baseline::{k}"] = v
for k,v in stim_data.items():
    raw_data[f"Stim::{k}"] = v

print(f"Loaded {len(raw_data)} traces total "
      f"(Baseline: {len(base_data)}, Stim: {len(stim_data)}).")
print(f"Global trim length = {GLOBAL_TRIM_LEN} samples "
      f"(~{GLOBAL_TRIM_LEN/SAMPLING_RATE:.3f} s at {SAMPLING_RATE} Hz).")

print(len(raw_data), raw_data['Stim::Nifedipine_1']['trace'].shape)

# %%

# def detect_beats(trace, prominence=MIN_PEAK_PROMINENCE):
#     trace = np.nan_to_num(trace) # replace NaNs with 0
#     if np.max(np.abs(trace))==0:
#         return np.array([])
#     normed = (trace - np.median(trace)) / np.max(np.abs(trace))
#     peaks, _ = sig.find_peaks(normed, prominence=prominence)
#     return peaks

from scipy.signal import savgol_filter

def detect_beats(trace, prominence=0.05):
    # 1) Smooth trace to remove high-freq jitter
    filt = savgol_filter(trace, window_length=51, polyorder=3)
    # 2) Normalise
    normed = (filt - np.median(filt)) / np.max(np.abs(filt))
    # 3) Peak detect with higher prominence
    peaks, _ = sig.find_peaks(normed, prominence=prominence, distance=0.3*SAMPLING_RATE)
    return peaks


# %%
from scipy.stats import skew, kurtosis
def extract_features(meta):
    t = meta['t']
    trace = meta['trace']
    peaks = detect_beats(trace)
    if len(peaks) < MIN_BEATS_PER_TRACE:
        return None

    feats = {}
    ibis = np.diff(t[peaks])
    feats['mean_rate'] = 60.0 / np.mean(ibis) if len(ibis)>0 else 0
    feats['sdnn'] = np.std(ibis, ddof=1) if len(ibis)>1 else 0
    feats['rmssd'] = np.sqrt(np.mean(np.square(np.diff(ibis)))) if len(ibis)>2 else 0
    sd1 = feats['rmssd']/np.sqrt(2)
    sd2 = np.sqrt(max(0, 2*feats['sdnn']**2 - 0.5*sd1**2))
    feats['sd1'] = sd1
    feats['sd2'] = sd2
    feats['ibi_skew']     = skew(ibis) if len(ibis)>2 else 0
    feats['ibi_kurtosis'] = kurtosis(ibis) if len(ibis)>2 else 0
    feats['ibi_max']      = np.max(ibis) if len(ibis)>0 else 0
    feats['ibi_min']      = np.min(ibis) if len(ibis)>0 else 0


    # --- amplitude metrics ---
    amps = trace[peaks]
    feats['amp_mean'] = np.mean(amps)
    feats['amp_cv']   = (np.std(amps)/np.mean(amps)
                         if np.mean(amps)!=0 else 0)
    
       # --- contraction area (AUC) per beat ---
    win_sz = 400
    win = []
    for pk in peaks:
        start,end = pk-200, pk+200
        seg = trace[max(start,0):min(end,len(trace))]
        if start<0:
            seg = np.pad(seg, (abs(start),0), constant_values=0)
        if end>len(trace):
            seg = np.pad(seg, (0,end-len(trace)), constant_values=0)
        if len(seg)==win_sz:
            win.append(seg)
    if win:
        W = np.stack(win)
        bas = np.median(W[:,:50], axis=1, keepdims=True)
        aucs = np.trapz(W-bas, axis=1)
        feats['auc_mean']   = np.mean(aucs)
        feats['auc_cv']     = np.std(aucs)/np.mean(aucs) if np.mean(aucs)!=0 else 0
        feats['auc_skew']   = skew(aucs)
        feats['auc_kurtosis'] = kurtosis(aucs)
        feats['auc_max']    = np.max(aucs)
        feats['auc_min']    = np.min(aucs)
        feats['auc_med']    = np.median(aucs)
        feats['auc_iqr']    = np.percentile(aucs,75)-np.percentile(aucs,25)
    else:
        for k in ['auc_mean','auc_cv','auc_skew','auc_kurtosis',
                  'auc_max','auc_min','auc_med','auc_iqr']:
            feats[k] = 0

    # --- timing metrics: TTP, CTD50, CTD90 ---
    ttp, ctd50, ctd90 = [], [], []
    for pk in peaks:
        onset = max(0, pk-int(0.3*SAMPLING_RATE))
        base  = np.median(trace[onset:pk])
        amp   = trace[pk] - base
        ttp.append((pk-onset)/SAMPLING_RATE)
        decay = trace[pk:] - base
        hidx = np.where(decay<=0.5*amp)[0]
        nidx = np.where(decay<=0.1*amp)[0]
        if len(hidx): ttp.append((hidx[0]+0)/SAMPLING_RATE)
        if len(nidx): ctd90.append(nidx[0]/SAMPLING_RATE)
        if len(hidx): ctd50.append(hidx[0]/SAMPLING_RATE)
    for arr,lab in [(ttp,'ttp'), (ctd50,'ctd50'), (ctd90,'ctd90')]:
        feats[f'{lab}_mean'] = np.mean(arr) if arr else 0
        feats[f'{lab}_sd']   = np.std(arr, ddof=1) if len(arr)>1 else 0

    # --- contraction/relax ratio ---
    if ttp and ctd90:
        n = min(len(ttp), len(ctd90))
        arr = np.array(ttp[:n]) / np.array(ctd90[:n])
        feats['ctr_relax_ratio_mean'] = arr.mean()
        feats['ctr_relax_ratio_cv']   = np.std(arr)/np.mean(arr) if np.mean(arr)!=0 else 0
    else:
        feats['ctr_relax_ratio_mean'] = feats['ctr_relax_ratio_cv'] = 0

    # --- velocity features ---
    vel = np.diff(trace)*SAMPLING_RATE
    ups, downs = [], []
    w = int(0.05*SAMPLING_RATE)
    for pk in peaks:
        seg = vel[max(pk-w,0):min(pk+w,len(vel))]
        if len(seg):
            ups.append(seg.max())
            downs.append(seg.min())
    if ups:
        feats['vel_up_mean']   = np.mean(ups)
        feats['vel_up_cv']     = np.std(ups)/np.mean(ups)
        feats['vel_up_skew']   = skew(ups)
    else:
        feats['vel_up_mean'] = feats['vel_up_cv'] = feats['vel_up_skew'] = 0
    if downs:
        feats['vel_down_mean'] = np.mean(downs)
        feats['vel_down_cv']   = np.std(downs)/abs(np.mean(downs))
        feats['vel_down_skew'] = skew(downs)
    else:
        feats['vel_down_mean'] = feats['vel_down_cv'] = feats['vel_down_skew'] = 0

    # --- spectral features ---
    freqs, psd = sig.welch(trace, fs=SAMPLING_RATE, nperseg=min(len(trace),2048))
    feats['lf_power']    = psd[(freqs>=0)&(freqs<0.5)].sum()
    feats['hf_power']    = psd[(freqs>=2)&(freqs<4)].sum()
    psd_n = psd/psd.sum()
    feats['spec_entropy'] = -np.sum(psd_n*np.log(psd_n+1e-12))
    # median frequency
    csum = np.cumsum(psd)
    halfway = csum[-1]/2
    idx = np.searchsorted(csum, halfway)
    feats['median_freq'] = freqs[idx]

    # --- raw trace stats ---
    feats['trace_mean'] = np.mean(trace)
    feats['trace_med']  = np.median(trace)
    feats['trace_std']  = np.std(trace)

    return feats




# %%
feat_rows, labels, groups, ids, conds = [], [], [], [], []

for key, meta in raw_data.items():
    feat = extract_features(meta)      # your existing function
    if feat is None:
        continue
    feat_rows.append(feat)
    labels.append(meta['risk'])        # Low / Intermediate / High
    groups.append(meta['drug'])        # Leave-One-Drug-Out grouping
    conds.append(meta['condition'])    # Baseline / Stim (optional)
    ids.append(key)                    # Baseline::Drug_Sample or Stim::Drug_Sample

X_df = pd.DataFrame(feat_rows, index=ids)
y    = pd.Series(labels, index=ids, name='risk')
g    = pd.Series(groups, index=ids,  name='drug')
c    = pd.Series(conds,  index=ids,  name='condition')  # optional
print('Feature matrix', X_df.shape, '— Baseline/Stim counts:\n', c.value_counts())

print(labels)


# %%

risk_order = ['Low','Intermediate','High']
y_ord = y.map({r:i for i,r in enumerate(risk_order)}).astype(int)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)

print('Scaled feature matrix', X_scaled.shape, y_ord.shape, g.shape)



# %%
# import lightgbm as lgb
# from sklearn.model_selection import LeaveOneGroupOut
# from sklearn.metrics import accuracy_score

# loo = LeaveOneGroupOut()
# accs = []
# for tr, te in loo.split(X_df, y_ord, groups):
#     clf = lgb.LGBMClassifier(class_weight='balanced', n_estimators=100)
#     clf.fit(X_df.iloc[tr], y_ord.iloc[tr])
#     p = clf.predict(X_df.iloc[te])
#     accs.append(accuracy_score(y_ord.iloc[te], p))
# print('LODO accuracy (LGBM):', np.mean(accs))


# %%
# Feature Spcace Visualisation with PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


scaler = StandardScaler().fit(X_df) # X_df is 59 by 40
X_scaled = scaler.transform(X_df)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for cls in np.unique(y_ord):
    mask = (y_ord.values == cls)
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=risk_order[cls])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Contraction Feature Space')
plt.legend()
plt.tight_layout()
plt.show()


# %%

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k, d):
        super().__init__()
        pad = (k-1)//2 * d
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, k, padding=pad, dilation=d),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x): return self.net(x)

class Encoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(1,32,3,1),
            TCNBlock(32,64,3,2),
            TCNBlock(64,128,5,4)
        )
        self.pool = nn.AdaptiveAvgPool1d(64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64*128, embed_dim)
    def forward(self,x):
        z = self.tcn(x)
        z = self.pool(z)
        z = z.permute(0,2,1)
        z = self.tr(z)
        return self.fc(z.flatten(1))
encoder = Encoder().to(device)


# %%

class TraceDatasetSSL(Dataset):
    def __init__(self, traces, max_len=10000):
        self.traces = traces
        self.max_len = max_len
    def augment(self, x):
        x = x + 0.005*np.random.randn(*x.shape)
        shift = random.randint(-50,50)
        x = np.roll(x, shift)
        if random.random()<0.3:
            mstart = random.randint(0,len(x)-100)
            x[mstart:mstart+100]=0
        return x
    def pad(self, x):
        if len(x)<self.max_len:
            return np.pad(x, (0,self.max_len-len(x)))
        else:
            return x[:self.max_len]
    def __len__(self): return len(self.traces)
    def __getitem__(self, idx):
        base = self.pad(self.traces[idx]).astype(np.float32)   # ← ensure 32-bit
        a1   = self.augment(base).astype(np.float32)
        a2   = self.augment(base).astype(np.float32)
        return (torch.from_numpy(a1).unsqueeze(0),
                torch.from_numpy(a2).unsqueeze(0))

ssl_ds = TraceDatasetSSL([meta['trace'] for meta in raw_data.values()])
# ssl_loader = DataLoader(ssl_ds, batch_size=BATCH_SIZE_SSL, shuffle=True, drop_last=True)
ssl_loader = DataLoader(ssl_ds,
                        batch_size=BATCH_SIZE_SSL,
                        shuffle=True, drop_last=False)   # <= keep small batch



# %%

# def simclr_loss(z1,z2,temp=0.05):
#     b = z1.size(0)
#     z1 = F.normalize(z1, dim=1)
#     z2 = F.normalize(z2, dim=1)
#     z = torch.cat([z1,z2], dim=0)
#     sim = torch.mm(z, z.t())/temp
#     labels = torch.arange(b, device=z.device)
#     labels = torch.cat([labels,labels])
#     mask = torch.eye(2*b, device=z.device).bool()
#     sim = sim.masked_fill(mask, -9e15)
#     return F.cross_entropy(sim, labels)

def simclr_loss(z1, z2, temp=0.05):
    """
    z1, z2: (B, D)  two augmented views of the same batch
    returns   scalar InfoNCE loss
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)           # (2B, D)
    z = F.normalize(z, dim=1)

    # cosine-similarity / temperature
    sim = torch.mm(z, z.t()) / temp          # (2B, 2B)

    # mask self-similarities
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -float("inf"))

    # targets: positive index for each row
    targets = torch.arange(B, device=z.device)
    targets = torch.cat([targets + B, targets])   # (2B,)

    return F.cross_entropy(sim, targets)




optim_ssl = torch.optim.Adam(encoder.parameters(), lr=LR_SSL)
for epoch in range(SSL_EPOCHS):
    encoder.train()
    tot=0
    for x1,x2 in ssl_loader:
        x1, x2 = x1.to(device).float(), x2.to(device).float()
        z1,z2 = encoder(x1), encoder(x2)
        loss = simclr_loss(z1,z2)
        optim_ssl.zero_grad(); loss.backward(); optim_ssl.step()
        tot += loss.item()
    if (epoch+1)%10==0:
        if len(ssl_loader):                       # avoid /0
            print(f"SSL epoch {epoch+1}/{SSL_EPOCHS} "
                  f"loss {tot/len(ssl_loader):.4f}")


# %%

from coral_pytorch.layers import CoralLayer

class Predictor(nn.Module):
    def __init__(self, encoder, exp_dim, n_classes=3, emb_dim=128):
        super().__init__()
        self.encoder=encoder
        # for p in self.encoder.parameters():
        #     p.requires_grad=False
        # self.fc_exp=nn.Linear(exp_dim,64)
        # self.head = CoralLayer(emb_dim+64, n_classes)
        for name, p in self.encoder.named_parameters():
        # unfreeze last TCN layer and the transformer
            if 'tcn.2' in name or 'tr' in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
        self.fc_exp=nn.Linear(exp_dim,64)
        self.head = CoralLayer(emb_dim+64, n_classes)

    def forward(self, trace, exp):
        with torch.no_grad():
            z = self.encoder(trace)
        e = F.relu(self.fc_exp(exp))
        return self.head(torch.cat([z,e],1))


# %%

def pad_trace(trace, max_len=10000):
    return np.pad(trace,(0,max_len-len(trace))) if len(trace)<max_len else trace[:max_len]

traces_ordered = [raw_data[idx]['trace'] for idx in X_df.index]
X_trace = torch.tensor(np.stack([pad_trace(tr).astype(np.float32) for tr in traces_ordered])[:,None,:])
X_exp = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_ord.loc[X_df.index].values, dtype=torch.long)
groups = g.loc[X_df.index].values


# %%
from coral_pytorch.losses import corn_loss

# compute sampling weights
counts = y_ord.value_counts().sort_index().astype(float)
inv  = 1.0/counts
weights = inv / inv.sum()  # sum to 1

logo = LeaveOneGroupOut()
all_preds = np.zeros_like(y_tensor)
for train_idx,test_idx in logo.split(X_exp,y_tensor,groups):
    # crit = nn.CrossEntropyLoss()
    model = Predictor(encoder, exp_dim=X_exp.shape[1]).to(device)
    opt   = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=LR_PRED)
    
    for epoch in range(PRED_EPOCHS):
        model.train()
        logits = model(X_trace[train_idx].to(device),
                   X_exp[train_idx].to(device))
        loss = corn_loss(logits,
                        y_tensor[train_idx].to(device),
                        num_classes=3)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
    #     pred = model(X_trace[test_idx].to(device),
    #                  X_exp[test_idx].to(device)).argmax(1).cpu().numpy()
    # all_preds[test_idx]=pred
        logits = model(X_trace[test_idx].to(device),
                   X_exp[test_idx].to(device))
        # each logit = P(y > k) in logit (unnormalised) space
        p_gt_k = torch.sigmoid(logits)
        preds  = (p_gt_k > 0.5).sum(dim=1).cpu().numpy()  # 0,1,2
    all_preds[test_idx] = preds

    
print(classification_report(y_tensor, all_preds, target_names=risk_order))


# %%

cm = confusion_matrix(y_tensor, all_preds)
plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=risk_order, yticklabels=risk_order)
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix (LODO)')
plt.show()


# %%

y_true_bin = F.one_hot(y_tensor, num_classes=3).numpy()
y_score_bin = F.one_hot(torch.tensor(all_preds), num_classes=3).numpy()
plt.figure(figsize=(6,4))
for i,label in enumerate(risk_order):
    fpr,tpr,_=roc_curve(y_true_bin[:,i], y_score_bin[:,i])
    auc=roc_auc_score(y_true_bin[:,i], y_score_bin[:,i])
    plt.plot(fpr,tpr,label=f'{label} (AUC={auc:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC (LODO)')
plt.legend(); plt.show()






