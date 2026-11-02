# Probabilistic baseline provenance

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

## Local evaluation protocol

HNN and Deep Ensemble use the chapter's grouped train-only scaler. EDL-NIG
uses train-only per-feature min-max scaling as described by its paper. All
three use the registered FD002/FD004 engine split and endpoint validation;
HNN and Ensemble select by validation NLL, while EDL-NIG selects by validation
RMSE in line with the paper's use of RMSE to select configurations. The
resolved criterion, scaler mode, selected sensor IDs and corresponding input
positions are recorded in run artifacts. Test is evaluated once. The shared
split and local metrics support controlled comparison; published scores are
not presented as directly comparable.
