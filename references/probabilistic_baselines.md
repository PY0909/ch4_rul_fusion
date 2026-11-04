# Probabilistic baseline provenance

## Six first-round external baselines — stage 0 literature verification (2026-11-04, paused pending full-text collection)

Scope: extend the baseline panel to B3–B6 per the chapter plan §8.5. B1 (HNN)
and B2 (EDL-NIG) are already implemented and registered above. Stage 0
verification status below distinguishes paper-reported values from project
adaptations; anything not obtainable from the paper is registered as a
reconstruction choice, never as a paper value.

### B4 — dual-channel TCN + Transformer with MC dropout (full text secured)

Paper: “A Parallel Prognostic Method Integrating Uncertainty Quantification for
Probabilistic Remaining Useful Life Prediction of Aero-Engine,” *Processes*
12(12):2925, 2024, DOI [10.3390/pr12122925](https://doi.org/10.3390/pr12122925)
(open access; full text read from the MDPI XML version on 2026-11-04).

Paper-reported configuration: parallel TCN and Transformer channels over the
raw window. TCN: 3 residual blocks with filters 8/6/4, kernel size 4, dilations
1/2/4, spatial-dropout inside blocks. Transformer encoder: d_model 14,
2 heads, 2 encoder layers, dropout 0.2, sinusoidal positional encoding.
Training: Adam lr 1e-3, batch 128, 50 epochs, window length 30, stride 1,
min–max normalization, piecewise RUL cap 125, 14 non-constant sensors (same
drop list convention as this project). Inference: MC dropout with k forward
passes → predictive mean/std → Gaussian 95% CI (mean ± 1.96σ). The paper
evaluates FD001/FD003 only; k is not stated numerically in the text or
Table 4.

Planned project adaptations (to be finalized at implementation): window 30 →
50 and FD001/FD003 → FD002/FD004 + DS03/DS08a per unified protocol; interval
band 95% Gaussian → nominal 80% (q10/q90 empirical quantiles of the draws) to
match the ranking protocol; k unspecified → 30 (existing `mc_dropout`
default). Each is a deviation from the paper and will be reported as such.

### B3 — LSTM with probabilistic layer (closed access, abstract only)

Paper: A. K. B. Mohd Nor, S. R. Pedapati, M. B. Muhammad, M. A. A. Majid,
“Demonstrating Aleatoric Uncertainty in Remaining Useful Life Prediction Using
LSTM with Probabilistic Layer,” *Lecture Notes in Mechanical Engineering*
(ICPER 2020 proceedings), Springer, 2022, DOI
[10.1007/978-981-19-1939-8_41](https://doi.org/10.1007/978-981-19-1939-8_41).

Known from abstract/metadata: LSTM backbone whose output layer is a
probabilistic layer capturing aleatoric (data) uncertainty; demonstrated on
NASA C-MAPSS turbofan data. Full text paywalled at Springer; no author code
found. Pending collection: LSTM depth/units, the exact probabilistic layer
(TFP-style Gaussian output vs variational dense), loss, FD subsets, window,
optimizer/epochs, and how intervals are reported. Until the full text
arrives, any implementation numbers are registered reconstruction choices.

### B5 — Bayesian cooperative probabilistic Transformer (paywalled, abstract secured)

Paper: S. Xie, W. Cheng, Z. Nie, J. Xing, X. Chen, Q. Huang, R. Zhang,
Y. Yang, “Bayesian cooperative probabilistic Transformer for remaining useful
life prediction with uncertainty estimation in industrial equipment,”
*Advanced Engineering Informatics* 67:103515, 2025, DOI
[10.1016/j.aei.2025.103515](https://doi.org/10.1016/j.aei.2025.103515).

Known from abstract (via the Xi'an Jiaotong University scholar page): a
probabilistic Transformer integrating attentional uncertainty (NLL loss),
cognitive/epistemic uncertainty (Bayesian approach), and aleatoric
uncertainty (noise channel), plus an uncertainty-guided multi-source fusion
strategy; validated on C-MAPSS and NCWP data. Full text paywalled; no code
link in the abstract. Pending collection: architecture sizes, the Bayesian
approximation form, noise-channel definition, fusion details, hyperparameters,
C-MAPSS subsets. Fallback (pre-authorized): mechanism-level reconstruction —
Transformer encoder + approximate-Bayesian (MC-dropout-style) epistemic term +
Gaussian NLL aleatoric head, fusion module dropped as dataset-specific — with
every reconstruction parameter registered as non-paper values.

### B6 — multi-scale self-attention + approximate Bayesian + KDE (paywalled, abstract secured)

Paper: “A multi-scale self-attention method with uncertainty quantification
for probabilistic remaining useful life prediction of rotating machinery,”
*Engineering Research Express*, 2025, DOI
[10.1088/2631-8695/adeeea](https://doi.org/10.1088/2631-8695/adeeea).

Known from abstract: multi-scale attention module in the encoder + approximate
Bayesian framework + kernel density estimation producing the predictive
density (PDF), evaluated with probabilistic metrics. Full text paywalled
(hybrid journal); no code found. Pending collection: multi-scale encoder
structure, Bayesian approximation details, KDE bandwidth selection, datasets,
hyperparameters. Fallback (pre-authorized): mechanism-level reconstruction —
multi-scale self-attention encoder + MC-dropout approximate Bayesian + KDE
predictive density over draws (Scott bandwidth) — registered as
reconstruction choices.

### Reserve decision (recorded 2026-11-04)

B7 (“Degradation-aware contrastive Mamba with dual uncertainty,” *Meas. Sci.
Technol.* 2026, DOI 10.1088/1361-6501/ae6e95) is also paywalled at the
abstract level. B8 (IEEE SDPC 2025, DOI 10.1109/SDPC68151.2025.11347806)
combines **snapshot ensemble** with MC dropout as its core mechanism, which
conflicts with the project's no-ensemble constraint and therefore cannot
serve as a like-for-like substitute. Per the swap rule, B5/B6 stay in the six;
if their full texts cannot be collected, they are implemented as
mechanism-level reconstructions with registered deviations rather than
replaced.

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
