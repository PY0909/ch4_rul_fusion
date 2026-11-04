# Probabilistic baseline provenance

## Six first-round external baselines — stage 0 literature verification (2026-11-04, complete)

Scope: extend the baseline panel to B3–B6 per the chapter plan §8.5. B1 (HNN)
and B2 (EDL-NIG) are already implemented and registered above. All four new
papers were verified against full texts (B4 open access; B3/B5/B6 PDFs
supplied on 2026-11-04 into `references/pdfs/`). Paper-reported values below
are separated from project adaptations; anything the paper does not state is
registered as a reconstruction choice, never as a paper value.

### B4 — dual-channel TCN + Transformer with MC dropout

Paper: “A Parallel Prognostic Method Integrating Uncertainty Quantification for
Probabilistic Remaining Useful Life Prediction of Aero-Engine,” *Processes*
12(12):2925, 2024, DOI [10.3390/pr12122925](https://doi.org/10.3390/pr12122925)
(open access; full text read from the MDPI XML version).

Paper-reported configuration: parallel TCN and Transformer channels over the
raw window. TCN: 3 residual blocks with filters 8/6/4, kernel size 4, dilations
1/2/4, spatial-dropout inside blocks. Transformer encoder: d_model 14,
2 heads, 2 encoder layers, dropout 0.2, sinusoidal positional encoding.
Training: Adam lr 1e-3, batch 128, 50 epochs, window length 30, stride 1,
min–max normalization, piecewise RUL cap 125, 14 non-constant sensors. Inference:
MC dropout with k forward passes → predictive mean/std → Gaussian 95% CI
(mean ± 1.96σ). The paper evaluates FD001/FD003 only; k is not stated
numerically.

Planned project adaptations: window 30 → 50 and FD001/FD003 →
FD002/FD004 + DS03/DS08a per unified protocol; interval band 95% Gaussian →
nominal 80% (q10/q90 empirical quantiles of the draws) to match the ranking
protocol; k unspecified → 30 (existing `mc_dropout` default). Each is a
deviation from the paper and will be reported as such.

### B3 — LSTM with probabilistic layer (full text supplied)

Paper: A. K. B. Mohd Nor, S. R. Pedapati, M. B. Muhammad, M. A. A. Majid,
“Demonstrating Aleatoric Uncertainty in Remaining Useful Life Prediction Using
LSTM with Probabilistic Layer,” *Lecture Notes in Mechanical Engineering*
(ICPER 2020 proceedings), Springer, 2022, DOI
[10.1007/978-981-19-1939-8_41](https://doi.org/10.1007/978-981-19-1939-8_41).
Full text PDF `references/pdfs/978-981-19-1939-8_41.pdf` read on 2026-11-04.

Paper-reported configuration: single-layer LSTM (635 hidden units) → Dense
(331) → Dropout (0.03) → **Normal-distribution output layer** producing the
mean and standard deviation of a health-index prediction; aleatoric-only
uncertainty. Hyperparameters are the optimum of a Bayesian optimization over
500 evaluations (search spaces given in the paper's Table 1): mini-batch 43,
max epoch 14, learning rate 5.613e-4. Data: C-MAPSS FD001 only; inputs are the
3 operating settings plus the 14 monotonic sensors (2,3,4,7,8,9,11,12,13,14,
15,17,20,21); the training target is a health index obtained from a piecewise
RUL label with cap 100 normalized to [0,1]; degradation start points are
detected per sensor by CUSUM; validation is a random 20% split of training
data. Uncertainty is reported by drawing 1000 samples from the predicted
Normal per window and plotting μ±2σ bounds. Reported FD001 RMSE 24.1,
score 1095 (not copied).

Planned project adaptations: direct RUL target with the chapter cap 125
instead of the HI/cap-100 formulation; window 50; the chapter's registered
engine split, fixed-cuts validation and one-shot official test instead of a
random 20% split; FD002/FD004 + DS03/DS08a instead of FD001; predictive
interval from the analytic Normal quantiles (q10/q90 for the 80% band) instead
of sampled μ±2σ bounds; the 1000-sample draw is redundant given the analytic
form and is not reproduced. The paper's BayesOpt architecture/training values
(635/331/0.03/43/14 epochs/5.613e-4) are kept as reported.

### B5 — Bayesian cooperative probabilistic Transformer (full text supplied)

Paper: S. Xie, W. Cheng, Z. Nie, J. Xing, X. Chen, Q. Huang, R. Zhang,
Y. Yang, “Bayesian cooperative probabilistic Transformer for remaining useful
life prediction with uncertainty estimation in industrial equipment,”
*Advanced Engineering Informatics* 67:103515, 2025, DOI
[10.1016/j.aei.2025.103515](https://doi.org/10.1016/j.aei.2025.103515).
Full text PDF `references/pdfs/1-s2.0-S1474034625004082-main.pdf` read on
2026-11-04.

Paper-reported configuration: each sensor channel is modelled separately by a
Transformer encoder (window 30, stride 1, sinusoidal PE, 5 attention heads,
feed-forward structure 30-64-30, prediction head FC1/FC2 = 20/10, dropout
0.3). Deterministic attention logits are replaced by the probabilistic
attention model (PAM): α̃ij = μij + τσij with τ~N(0,1), μij/σij = MLP(Qi,Kj)
with MLP structure 2-10-6-2. The loss combines a heteroscedastic NLL with a
noise channel η², the attention NLL (γ = 0.2, grid-searched) and an L2
penalty on η² (λ = 0.1). Cognitive uncertainty comes from MC-dropout with
M = 30 samples on C-MAPSS; total uncertainty U = σ²PU + H(α) + mean η².
UGMIF trains two models on all channels with channel losses weighted by
(1 − softmax(Uc)) (LU-model) and (1 + softmax(Uc)) (HU-model) and fuses at
inference by uncertainty-softmax weights (Eqs. 22–26). Training: Adam,
lr 0.005→0.001, batch 256, 200 epochs; min–max normalization; piecewise cap
125; 14 sensors; intervals reported at a 95% confidence level (β ∈ {0.75,
0.9, 0.95} in the NCWP study). The paper repeats each experiment 10 times;
reported FD002 RMSE 14.25 / PICP 0.956 and FD004 17.52 / 0.945 (not copied).

Not stated in the paper: the number of encoder layers, the exact embedding
dimension, and the explicit interval formula (a Gaussian assumption around
the fused mean with variance U is implied). These become registered
reconstruction choices.

Planned project adaptations: window 50; FD002/FD004 + DS03/DS08a under the
chapter's registered split, fixed-cuts validation and one-shot official test
(the paper's own split protocol is unspecified); 80% band via Normal quantiles
of the fused mean/√U instead of the 95% level; single seed 0 run instead of
10 repetitions; M = 30 and the UGMIF/LU/HU machinery kept as reported.

### B6 — multi-scale self-attention + approximate Bayesian + KDE (full text supplied)

Paper: J. Du, J. Fu, “A multi-scale self-attention method with uncertainty
quantification for probabilistic remaining useful life prediction of rotating
machinery,” *Engineering Research Express* 7:035543, 2025, DOI
[10.1088/2631-8695/adeeea](https://doi.org/10.1088/2631-8695/adeeea).
Full text PDF `references/pdfs/Du_2025_Eng._Res._Express_7_035543.pdf` read
on 2026-11-04.

Paper-reported configuration: MSSA unit = N = 2 multi-scale convolution
modules (parallel Conv1d kernels {3,5,7} → BatchNorm → ReLU → Dropout,
concatenate, 1×1-conv residual projection when channels mismatch, ReLU)
followed by sinusoidal PE and one self-attention layer (LayerNorm + Dropout +
residual); predictor = global average pooling → Linear → ReLU → Linear;
80,225 parameters in total. Preprocessing: the 14 standard sensors, Z-score
normalization grouped by operating condition (train-set statistics) followed
by exponential smoothing with α = 0.2, sliding window L = 30 stride 1,
piecewise cap 125. Training: Adam, 50 epochs, best weights by minimum RMSE.
Uncertainty: MC dropout with G = 100 forward passes, kernel density estimation
with a Gaussian kernel over the draws, normalized PDF, 95% CIs; the point
estimate is the draw mean. The paper's sensitivity studies choose scales
{3,5,7}, N = 2 and α = 0.2; reported FD002 RMSE 14.19 / score 1173.83 and
FD004 13.85 / 1222.37 (not copied).

Not stated in the paper: the dropout rate, per-layer channel widths (the
80,225-parameter total is the constraint), the learning rate, the batch size
and the KDE bandwidth h. These become registered reconstruction choices
(channel widths sized to match the parameter count; KDE bandwidth by Scott's
rule).

Planned project adaptations: window 50; FD002/FD004 + DS03/DS08a under the
chapter's registered split, fixed-cuts validation and one-shot official test;
best-epoch selection by validation RMSE instead of training RMSE; 80% band
from the KDE/draw quantiles instead of 95% CIs; the paper's grouped Z-score +
α = 0.2 exponential smoothing preprocessing is kept as part of the method
(other baselines keep their own paper preprocessing, as B2 does with min-max).

### Reserve decision (recorded 2026-11-04, not triggered)

B7 (“Degradation-aware contrastive Mamba with dual uncertainty,” *Meas. Sci.
Technol.* 2026, DOI 10.1088/1361-6501/ae6e95) remains paywalled at the
abstract level. B8 (IEEE SDPC 2025, DOI 10.1109/SDPC68151.2025.11347806)
combines **snapshot ensemble** with MC dropout as its core mechanism, which
conflicts with the project's no-ensemble constraint and therefore cannot
serve as a like-for-like substitute. With B3/B5/B6 full texts supplied, the
swap rule is no longer relevant; the original six-baseline list stands.

## HNN and Gaussian Deep Ensemble

The reference implementation is [BayesRUL](https://github.com/arthurviens/bayesrul),
revision `5116b156d378cd03a2153f6ecdce0b2a7b79e861`. The implementation was
checked in `bayesrul/models/inception.py`,
`bayesrul/lightning_wrappers/frequentist.py`, and
`bayesrul/inference/deep_ensemble.py`.

The HNN uses `InceptionModel` with `out_size=2`; its head applies softplus to
both outputs, and `DnnWrapper` minimizes Gaussian NLL using the squared second
output as variance. The compact Inception blocks, channel sizes, initialization
(Xavier normal for Conv1d, Kaiming normal for Linear), Adam optimizer defaults,
and no-dropout default are ported. Deep Ensemble uses the same member network,
cycles the active member across training batches, aggregates the Gaussian
mixture by its total variance, and early-stops on validation mixture NLL.

Reference paper: L. Basora, A. Viens, M. Arias Chao, and X. Olive, “A benchmark
on uncertainty quantification for deep learning prognostics,” *Reliability
Engineering & System Safety*, 2025, DOI
[10.1016/j.ress.2024.110513](https://doi.org/10.1016/j.ress.2024.110513).
The reference benchmark is N-CMAPSS and BayesRUL's default feature representation
is not identical to this project. This implementation concatenates the
chapter's sensor and operating-setting windows and uses the frozen FD002/FD004
engine-level split and train-only grouped scaling. Those are explicit target
protocol adaptations; published benchmark scores are not copied or claimed.

## EDL-NIG

The method reference is S. Ben Ayed, R. Sadeghian Broujeny, and R. T. Hamza,
“Remaining Useful Life Prediction with Uncertainty Quantification Using
Evidential Deep Learning,” *Journal of Artificial Intelligence and Soft
Computing Research*, 15(1), 37–55 (2025), DOI
[10.2478/jaiscr-2025-0003](https://doi.org/10.2478/jaiscr-2025-0003).
The supplied full text, `10.2478_jaiscr-2025-0003.pdf`, was read directly.
It specifies a five-layer MLP whose output is `(gamma, nu, alpha, beta)`,
dropout `0.3`, Adam, and the evidential NIG objective. The paper reports
FD002/FD004 at 50 epochs and batch size 32 (Table 2), selects parameters to
minimize RMSE, and describes min-max normalization of the selected sensors.
The implementation uses five `Linear` transformations in total, ReLU after
each hidden transformation, and dropout after each hidden layer. The article
does not provide hidden-layer widths, learning rate, or the evidence penalty
coefficient; code defaults (hidden width 128, learning rate 1e-3, coefficient
1e-2) are explicit reconstruction choices, not reported paper values.

The paper's preprocessing text says seven sensor columns are removed, but its
list repeats `P2`; Figure 4 shows the selected input sensors. We follow the
sensor names visible in Figure 4 (11 names, despite the figure text saying
“10 sensors”) and use the final cycle in each existing history window as the
cycle-wise MLP input. Operation settings are excluded from this model. The NIG
loss uses the marginal likelihood and absolute-error evidence regularizer;
test output persists `(gamma, nu, alpha, beta)`, predictive Student-t values,
and aleatoric/epistemic variance components for recomputation.

No author-maintained code repository was found by searching GitHub repository
metadata for the DOI/title/authors or in the HAL record. This is not proof that
private or unindexed code does not exist. The article's reported evaluation
uses an 80/20 train/test split and ten-fold cross-validation; this project
instead keeps the registered engine split, fixed endpoint validation and
official test set to compare all chapter baselines consistently. The paper
mentions a piecewise RUL target but does not state its cap value; the local
runner uses the chapter's locked cap of 125 for training labels, with raw-RUL
validation and official-test labels. The model architecture and reported
hyperparameters are followed where specified; split/validation, RUL cap and
unspecified MLP widths remain protocol/reconstruction adaptations.

## N-CMAPSS matched comparison

`scripts/run_ncmapss_probabilistic.py` reuses the probabilistic architectures and
losses above with DS03/DS08a registered split, fixed-cut validation, history 16,
batch 32, Adam at 1e-3 and 300 epochs/patience. HNN and ensemble use dropout 0
and weight decay 1e-3; EDL uses dropout 0.3, no weight decay and evidence weight
1e-2. For N-CMAPSS EDL all available physical sensors are used rather than
applying the C-MAPSS paper sensor-ID filter. Evaluation defaults deferred until
explicit `--eval_test`; official-test metrics exclude NASA score, a C-MAPSS
convention.



HNN and Deep Ensemble use the chapter's grouped train-only scaler. EDL-NIG
uses train-only per-feature min-max scaling as described by its paper. All
three use the registered FD002/FD004 engine split and endpoint validation;
HNN and Ensemble select by validation NLL, while EDL-NIG selects by validation
RMSE in line with the paper's use of RMSE to select configurations. The
resolved criterion, scaler mode, selected sensor IDs and corresponding input
positions are recorded in run artifacts. Test is evaluated once. The shared
split and local metrics support controlled comparison; published scores are
not presented as directly comparable.
