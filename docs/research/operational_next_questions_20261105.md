# Next questions, recorded before B1/P1 final scores

The current batch tests 24M shared fields, smaller single-seed controls and four
separate 124/125M language models. It does not train a unified GPT-2-scale
multimodal model. This note records the next hypotheses from code inspection and
train/selection-only diagnostics; none is a conclusion inferred from final test
speakers. Complete the registered comparisons before adding more variants.

## Spend the next GPU hours on information, not just weights

The train-only supervision audit finds 8.23 paired person-track hours but only
about 2.05 hours of independent common meeting time across seven meetings.
66.7% of sampled AMI windows contain no annotated words from the tracked speaker;
this is not a claim of acoustic silence. Extra paired data should add people,
meetings, scene conditions and active speech, rather than repeating synchronized
views of the same event. Stratify active speech while retaining quiet intervals
as a separate, measured robustness condition. Freeze participant/session splits
and timing controls before fitting. Do not discard difficult final cases.

The low CTC-frame-rate hypothesis is not supported: fewer than 0.1% of eligible
sampled training targets are impossible at the present packet rate, and none of
the sampled selection targets is impossible. Increasing the output clock alone
is therefore not the first intervention. Current phone targets spread a CMU
first pronunciation uniformly within aligned words; the downloaded MFA source
also contains real phone alignments. Compare those targets on the same people
and exposure before claiming an architectural fix for phonetic accuracy.

## Locate the content bottleneck

Report actual field CTC separately from recognition of rendered waveforms and
compare source/oracle-renderer recognition before judging a generator. A useful
follow-up is a matched content-bottleneck control: predicted discrete speech
units or phones feeding an explicitly counted renderer, versus direct mel
prediction. Any external speech teacher, pretrained codec or TTS system must be
listed separately; it cannot become an uncounted source of the model's knowledge.

Test a train-only temporal mouth encoder against the current 64-dimensional
face PCA edge. The oracle video-code reconstruction is a codec ceiling, not a
learned generation result. A fixed appearance can dominate pixel error, so use
motion, timing, within-person content retrieval and fresh-person tests together.
[AV-HuBERT](https://arxiv.org/abs/2201.02184) supports the research direction of
masked, matched audio/visual content prediction; its reported results are not a
performance estimate for our much smaller paired corpus or current video edge.

Repair gives the network coarse packet visibility and a chunk-local
observed-only baseline, while the practical interpolation control retains fine
mel visibility over the clip. A pre-test secondary readout now conserves known
fine mel context at the renderer, without changing hidden predictions. The
network's finer observation interface still needs a separate revision against the
strong baseline. Do not weaken that baseline to manufacture a gain.

## Separate representation from signal parameterization

The B1 parallel local/Fourier/long branches are exactly one linear causal
31-tap filter in real arithmetic. The inference-only folding experiment tests
that equality on trained weights; it does not establish identical optimization.
Register a single-filter training control with matched effective initial kernel,
data, optimizer budget and parameter accounting. Learned recurrent memory slots
are not regularly spaced historical observations; convolving over their prefix
mixes state slots with physical time. Compare a clock-consistent convolution
cache against this design before attributing long-horizon benefits to a physical
spectral prior. T1 uses sequentially normalized paths and does not have this same
parallel-branch folding identity.

Both B1 architectures already contain shared adapters and recurrent fields.
An advantage for both over late fusion would support a representation design;
only paired SLWM-versus-Transformer effects support an extra signal-path claim.
The same distinction applies to decision utility: compare the field actor with
the matched adaptive direct-sensor actor as well as all fixed sensing rules.

## Language scale needs token and compute curves

The present endpoint is 150,994,944 training tokens per 124/125M model, about
1.2 tokens per parameter. The prepared English corpus contains roughly 2.6B
tokens, so data exists for a substantially longer next stage. Do not call this
endpoint convergence or fit a scaling law from two seeds and one main size.
Report token-matched quality and total measured compute, including the actual
parent checkpoint cost, separately. Use unobserved test windows for the next
confirmatory experiment after inspecting this batch's test results.
[Compute-optimal language training](https://arxiv.org/abs/2203.15556) motivates
scaling training data with parameters; extrapolating its numerical optimum to
this architecture or multimodal packet counts would require new evidence.

## From recorded sensing to physical decisions

P1 changes which recorded sensor becomes visible. It does not learn how a
physical action changes the world. The next minimal transition should include
an action embedding, proprioception, goal/reward and multi-step latent rollouts,
with recorded actions and episode/environment-disjoint evaluation. Compare a
simple behavior-cloning policy, a Transformer policy, a frozen-field actor and
action-conditioned planning under the same observation/action budget. Test
whether shuffling actions or removing memory destroys the claimed benefit.
[DROID](https://droid-dataset.github.io/) provides a relevant candidate source of
camera, robot-action and language episodes (76k trajectories, about 350 hours);
dataset access, storage and a small deterministic episode subset should be
validated before any large download. The full dataset is not currently used.

An operational loop also needs timestamped asynchronous capture, bounded queues,
dropout handling, uncertainty thresholds and action deadlines. Current fields
wait for a 1.365s chunk plus FFT analysis support. Offline waveform rendering
adds 512ms of future predicted mel context. Neither is evidence of a fast motor
controller. Keep a slower semantic memory/planning cycle separate from a fast
control layer, and establish the transition first in a bounded recorded or
simulated environment. Actual drone or humanoid control is not demonstrated.
