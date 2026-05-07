# exploration.md

# Exploring Output Heads and the Model's Latent World View

## 0. Purpose

This document defines how to explore, visualize, and evaluate the **output heads** of the Signal Latent World Model (SLWM). The goal is not only to measure whether the model can decode text, audio, video, or actions, but to inspect how the shared latent signal field maps between modalities.

The central question is:

> What does the model's latent world state contain, and how does that state become text, visual imagination, audio prediction, action proposals, or silence?

This document focuses on **diagnostic exploration**, not normal agent behavior. In normal inference, the learned policy/commitment mechanism decides which heads are allowed to produce external output. In exploration mode, researchers may explicitly activate diagnostic heads to inspect the model's internal representation.

---

## 1. Core Idea

The model receives compressed multimodal signals and processes them into an updated latent world field:

```text
input signals
    ↓
modality encoders
    ↓
shared latent signal field
    ↓
latent processor
    ↓
processed latent world field
    ↓
policy + output heads
```

The output heads are not just user-facing decoders. They are also **probes** into the model's internal world view.

Examples:

```text
latent world field → text head      → description / answer / explanation
latent world field → visual head    → image/video reconstruction or imagination
latent world field → audio head     → predicted sound / speech / ambience
latent world field → action head    → movement proposal / affordance
latent world field → uncertainty    → confidence, source, observed vs inferred
```

In exploration mode, we ask:

```text
What visual scene does the latent field imply?
What text description does the same latent field imply?
What audio does the same latent field imply?
What action does the same latent field afford?
Where does the model confuse observed, inferred, and imagined information?
```

---

## 2. Important Distinction: Behavior vs Exploration

Normal operation:

```text
latent field → policy → committed output heads only
```

Exploration operation:

```text
latent field → researcher-selected diagnostic heads
```

This distinction is critical.

The model may internally predict visual, audio, or action states, but that does not mean it should always externally output them. Exploration mode intentionally overrides normal behavior selection so researchers can inspect what the model appears to represent.

Diagnostic decoding should be clearly marked as:

```text
observed
reconstructed
predicted
imagined
inferred
uncertain
```

Never treat all decoded outputs as factual observations.

---

## 3. Output Head Types

### 3.1 Text Head

Purpose:

```text
latent world field → English text / code / explanation / answer
```

Exploration uses:

- Describe a visual/audio latent state in English.
- Explain what the model thinks is happening.
- Summarize cross-modal context.
- Translate latent action proposals into human-readable plans.
- Express uncertainty and source attribution.

Example probes:

```text
Describe the scene represented by this latent state.
What sounds are likely present?
What is uncertain?
What was observed vs inferred?
What action would be appropriate?
```

---

### 3.2 Visual Head

Purpose:

```text
latent world field → image / video / visual latent / segmentation-like map
```

Exploration uses:

- Reconstruct what the model visually understood.
- Generate predicted future frames.
- Visualize imagined scenes from text/audio.
- Inspect object permanence and spatial structure.
- Reveal hallucinated visual content.

Possible outputs:

```text
static image reconstruction
future frame prediction
short video rollout
depth-like map
object/region heatmap
uncertainty map
observed-vs-inferred visual map
```

---

### 3.3 Audio Head

Purpose:

```text
latent world field → audio waveform / spectrogram / audio latent
```

Exploration uses:

- Decode predicted sound from visual context.
- Reconstruct speech, ambient sound, or event audio.
- Inspect whether visual scenes imply consistent audio.
- Compare audio imagination against text descriptions.

Possible outputs:

```text
audio waveform
mel spectrogram
audio event probabilities
speech-like latent
ambient sound latent
future audio prediction
```

---

### 3.4 Action Head

Purpose:

```text
latent world field → action proposal / motor trajectory / affordance map
```

Exploration uses:

- Inspect what the model thinks can be acted upon.
- Decode movement proposals from visual/audio context.
- Test whether action proposals are safe, grounded, and context-dependent.
- Separate imagined actions from committed actions.

Possible outputs:

```text
NO_OP / wait / observe
look target
reach target
grasp target
move trajectory
gesture proposal
speech-action coordination
```

The action head should always expose whether an action is:

```text
proposed
simulated
committed
blocked
unsafe
uncertain
```

---

### 3.5 Policy / Commitment Head

Purpose:

```text
latent world field + goal + constraints → behavior commitment
```

Exploration uses:

- Determine when the model chooses to speak, move, wait, ask, or stay silent.
- Inspect conflict resolution between heads.
- Test whether uncertainty suppresses overconfident output.
- Study multi-head behavior such as speaking while moving.

Example outputs:

```json
{
  "speak": 0.91,
  "move": 0.04,
  "look": 0.33,
  "wait": 0.12,
  "ask_clarification": 0.08,
  "commit": "speak",
  "reason": "question is answerable from observed context",
  "confidence": 0.86
}
```

---

### 3.6 Uncertainty / Source Head

Purpose:

```text
latent world field → uncertainty, source, observed/inferred/predicted tags
```

Exploration uses:

- Reduce hallucination by identifying unsupported decoded content.
- Separate perception from imagination.
- Check if generated text is grounded in actual input signals.

Required labels:

```text
observed
reconstructed
predicted
inferred
imagined
unknown
unsupported
```

This head is required for all serious exploration.

---

## 4. Cross-Modal Exploration Matrix

Each row is an experiment type. Each experiment tests whether the model maps one modality into another through the shared latent field.

| Input                | Diagnostic Output | Question                                                    |
| -------------------- | ----------------- | ----------------------------------------------------------- |
| Video                | Text              | Can the model describe what it sees?                        |
| Video                | Audio             | Does the visual scene imply plausible sound?                |
| Video                | Action            | What actions does the scene afford?                         |
| Audio                | Text              | Can the model describe the sound event?                     |
| Audio                | Visual            | What visual scene does the sound imply?                     |
| Audio                | Action            | Does the sound suggest movement, attention, or no-op?       |
| Text                 | Visual            | Can text induce a stable visual latent scene?               |
| Text                 | Audio             | Can text induce plausible sound/speech/ambience?            |
| Text                 | Action            | Does instruction text produce appropriate action proposals? |
| Video + Audio        | Text              | Does the description integrate both signals?                |
| Video + Text         | Audio             | Does text condition the predicted sound?                    |
| Audio + Text         | Visual            | Does text disambiguate the imagined scene?                  |
| Video + Audio + Text | Policy            | Does the model choose the correct output behavior?          |

---

## 5. Exploration Modes

### 5.1 Reconstruction Mode

Tests whether the latent field preserves information already present in the input.

```text
input signal → latent field → same-modality reconstruction
```

Examples:

```text
video → latent → video
text → latent → text
speech/audio → latent → audio
```

Useful for checking whether the latent field retains enough detail.

Failure modes:

```text
blurry visual reconstruction
lost object identity
incorrect spatial layout
wrong speaker/event audio
text paraphrase changes facts
```

---

### 5.2 Cross-Modal Translation Mode

Tests whether one modality can produce another.

```text
input modality A → latent field → output modality B
```

Examples:

```text
video → text
text → visual
visual → audio
audio → visual
text → action
```

This is the core exploration mode for world-view inspection.

---

### 5.3 Future Prediction Mode

Tests whether the latent field supports temporal dynamics.

```text
current context → future latent → decoded future signal
```

Examples:

```text
video frames 1-8 → predicted frames 9-16
audio seconds 0-2 → predicted seconds 2-4
action history → next movement proposal
text context → next semantic state
```

The output must be labeled as **predicted**, not observed.

---

### 5.4 Imagination Mode

Tests whether the model can construct plausible latent states from partial or abstract prompts.

Examples:

```text
text prompt → imagined visual scene
sound prompt → imagined source object
partial video → completed scene
partial audio → completed ambience
```

Imagination mode is useful, but dangerous for factuality. All outputs must be tagged as imagined or inferred unless grounded in input signals.

---

### 5.5 Counterfactual Mode

Tests whether the latent field supports intervention.

Examples:

```text
What changes if the object is heavier?
What sound changes if the glass breaks?
What action changes if the obstacle moves?
What answer changes if one fact is removed?
```

Counterfactual tests are useful for determining whether the model has causal structure or only surface correlation.

---

### 5.6 Policy Inspection Mode

Tests whether the policy chooses appropriate output commitments.

Examples:

```text
Should the model speak or stay silent?
Should it move or observe?
Should it ask for more information?
Should it output uncertainty instead of a claim?
Should it speak while acting?
```

Policy inspection must log both selected and rejected proposals.

---

## 6. Latent World View Visualization

The model's world view cannot be directly observed. It can only be probed through decoders, interventions, and consistency tests.

Recommended tools:

### 6.1 Diagnostic Decoders

Train frozen-core probes:

```text
latent field → text description
latent field → visual reconstruction
latent field → audio reconstruction
latent field → action affordance
latent field → uncertainty/source map
```

Use frozen-core probes to avoid confusing the probe's learning with the core model's representation.

---

### 6.2 Latent Traversals

Modify one latent direction at a time and decode the results.

Examples:

```text
increase motion direction → visual output shows more movement?
increase impact direction → audio output shows louder collision?
increase uncertainty direction → text output becomes more cautious?
increase object permanence direction → hidden object remains represented?
```

---

### 6.3 Modality Consistency Checks

Decode multiple heads from the same latent field and compare them.

Example:

```text
latent field → visual: dog barking in park
latent field → audio: engine noise
latent field → text: quiet library
```

This is inconsistent and should be flagged.

Consistency dimensions:

```text
object consistency
event consistency
spatial consistency
temporal consistency
audio-visual consistency
text-grounding consistency
action-affordance consistency
```

---

### 6.4 Observed vs Inferred Maps

Every diagnostic output should optionally include source attribution:

```text
which parts came from input?
which parts were reconstructed?
which parts were predicted?
which parts were guessed?
which parts are unsupported?
```

For visual outputs, this may be a heatmap.

For text outputs, this may be span-level attribution.

For audio outputs, this may be a time-frequency confidence map.

For actions, this may be an affordance and risk map.

---

## 7. Evaluation Questions

### 7.1 Textual Readout

Questions:

```text
Can the text head accurately describe visual/audio context?
Does it separate observed facts from inferred facts?
Does it avoid unsupported details?
Can it describe uncertainty?
Can it write code when the latent context is textual/code-based?
```

Metrics:

```text
caption accuracy
question-answer accuracy
grounded claim rate
unsupported claim rate
abstention quality
code pass@k
```

---

### 7.2 Visual Readout

Questions:

```text
Can the visual head reconstruct or imagine scenes from text/audio?
Does it preserve object identity and spatial layout?
Does it hallucinate objects not supported by input?
Does future prediction preserve temporal continuity?
```

Metrics:

```text
object consistency
scene consistency
temporal consistency
visual hallucination rate
human preference
classifier agreement
```

---

### 7.3 Audio Readout

Questions:

```text
Can the audio head reconstruct sound events?
Can video imply plausible audio?
Can text imply plausible speech/ambience?
Does audio remain synchronized with visual motion?
```

Metrics:

```text
spectral distance
event classification agreement
ASR consistency for speech
human preference
audio-visual sync score
```

---

### 7.4 Action Readout

Questions:

```text
Can the model infer affordances from visual/audio/text context?
Does it choose no-op when action is unnecessary?
Does it suppress unsafe actions?
Can it coordinate speech and movement?
```

Metrics:

```text
action success rate
unsafe proposal rate
unnecessary action rate
no-op correctness
affordance accuracy
policy regret
```

---

### 7.5 Policy / Commitment

Questions:

```text
Does the policy choose the right output modality?
Does it combine modalities when appropriate?
Does it avoid speaking when uncertain?
Does it ask when information is missing?
Does it suppress internal imagination from becoming false output?
```

Metrics:

```text
modality selection accuracy
commitment precision
commitment recall
false commitment rate
unnecessary silence rate
unsafe commitment rate
uncertainty calibration
```

---

## 8. Minimum Exploration Suite

The first version should include a small, controlled suite before large multimodal testing.

### 8.1 Video → Text

Input:

```text
short video clip
```

Outputs:

```text
caption
observed object list
event description
uncertainty notes
```

Goal:

```text
Inspect whether visual latents support grounded language.
```

---

### 8.2 Audio → Text

Input:

```text
audio clip
```

Outputs:

```text
audio event description
speaker/speech transcription if present
uncertainty notes
```

Goal:

```text
Inspect whether audio latents support grounded language.
```

---

### 8.3 Text → Visual

Input:

```text
English description or code/comment context
```

Outputs:

```text
image or visual latent reconstruction
object list
scene graph
uncertainty notes
```

Goal:

```text
Inspect how language induces visual imagination.
```

---

### 8.4 Video → Audio

Input:

```text
silent video clip
```

Outputs:

```text
predicted audio latent
spectrogram
audio event labels
```

Goal:

```text
Inspect whether visual motion implies plausible sound.
```

---

### 8.5 Audio → Visual

Input:

```text
audio event
```

Outputs:

```text
imagined visual scene
possible source objects
uncertainty notes
```

Goal:

```text
Inspect ambiguity and multimodal inference.
```

---

### 8.6 Context → Policy

Input:

```text
multimodal context + goal
```

Outputs:

```text
speak/move/look/wait/ask gates
selected commitment
rejected proposals
reason/confidence
```

Goal:

```text
Inspect behavior selection rather than raw decoding.
```

---

## 9. Logging Format

Each exploration run should be logged as structured data.

```json
{
  "run_id": "exp_000001",
  "model_id": "slwm_124m_v0",
  "checkpoint": "step_100000",
  "mode": "video_to_text",
  "input_modalities": ["video"],
  "diagnostic_heads": ["text", "uncertainty"],
  "policy_enabled": false,
  "input_refs": {
    "video": "samples/video_001.mp4"
  },
  "outputs": {
    "text": "A person is opening a door.",
    "uncertainty": {
      "observed": ["person", "door"],
      "inferred": ["opening"],
      "unsupported": []
    }
  },
  "metrics": {
    "grounded_claim_rate": 0.92,
    "unsupported_claim_rate": 0.03
  },
  "notes": "Model correctly identifies the main event but is uncertain about the room type."
}
```

---

## 10. Repository Structure

Recommended files:

```text
exploration/
  configs/
    video_to_text.yaml
    audio_to_text.yaml
    text_to_visual.yaml
    video_to_audio.yaml
    policy_inspection.yaml
  probes/
    text_probe.py
    visual_probe.py
    audio_probe.py
    action_probe.py
    uncertainty_probe.py
  scripts/
    run_exploration.py
    decode_latent.py
    latent_traversal.py
    compare_heads.py
    inspect_policy.py
  notebooks/
    latent_worldview.ipynb
    cross_modal_consistency.ipynb
    policy_commitment_maps.ipynb
  outputs/
    galleries/
    audio/
    videos/
    reports/
  logs/
    exploration_runs.jsonl
```

---

## 11. Required Controls

Every exploration experiment should include controls.

### 11.1 Baseline Controls

Compare against:

```text
single-modality model
frozen random head
same architecture without shared latent core
same architecture without spectral/signal blocks
same architecture without uncertainty/source head
```

---

### 11.2 Input Ablations

Remove or corrupt one modality:

```text
video only
video + muted audio
audio only
audio + blank video
text only
text with key noun removed
text with contradiction inserted
```

The decoded outputs should change in meaningful ways.

---

### 11.3 Policy Ablations

Compare:

```text
policy disabled
policy enabled
policy without uncertainty
policy without safety constraints
policy with forced output head
```

This tests whether the policy is genuinely making useful output decisions.

---

## 12. Failure Modes to Track

### 12.1 Cross-Modal Hallucination

The model decodes a modality-specific detail unsupported by the input.

Example:

```text
Input: audio of rain
Output visual: car crash scene
```

---

### 12.2 Overconfident Imagination

The model presents imagined information as observed.

Example:

```text
Input: dog barking audio
Text output: "A brown dog is behind a wooden fence."
```

The dog may be inferred, but color/fence are unsupported.

---

### 12.3 Decoder Dominance

A powerful decoder invents plausible output even when the latent field is weak.

Mitigation:

```text
use frozen probes
measure uncertainty
compare against random-latent decoding
```

---

### 12.4 Modality Collapse

The shared latent field becomes dominated by one modality.

Example:

```text
text dominates visual/audio interpretation
video dominates action interpretation
```

Mitigation:

```text
balanced training
modality dropout
cross-modal consistency losses
separate adapter normalization
```

---

### 12.5 Policy Over-Suppression

The policy avoids output too often.

Example:

```text
always waits
always asks clarification
never acts
```

Track usefulness as well as safety.

---

### 12.6 Policy Over-Commitment

The policy commits to speech/action without enough evidence.

Example:

```text
speaks confidently from ambiguous audio
moves when object location is uncertain
```

This is especially important for embodied systems.

---

## 13. Success Criteria

The exploration system is considered successful if it can demonstrate:

```text
1. The same latent field can support multiple coherent diagnostic decodings.
2. Textual descriptions are grounded in visual/audio evidence.
3. Visual imagination from text is plausible but marked as imagined.
4. Audio predictions from video are plausible and temporally aligned.
5. Action proposals reflect affordances, uncertainty, and no-op options.
6. The policy selects output heads appropriately under changing goals.
7. The uncertainty/source head reduces unsupported external claims.
8. Ablations show that the shared latent core matters.
```

A strong result is not merely good generation quality. A strong result is **consistent, grounded, inspectable multimodal mapping** through the latent field.

---

## 14. Research Notes

The main research risk is confusing decoded outputs with actual internal understanding. A visual decoder may produce a beautiful image from a weak latent state. A text decoder may produce a convincing explanation that is not grounded. An audio decoder may generate plausible sound that the model did not really infer.

Therefore, every exploration must combine:

```text
decoded output
source attribution
uncertainty
cross-head consistency
ablation controls
```

The purpose of `exploration.md` is to make the model's latent world view testable, not to assume that any one decoded modality reveals the truth.

---

## 15. Minimal First Implementation

Start with five diagnostic paths:

```text
video → latent → text
text → latent → visual
audio → latent → text
video → latent → audio
latent → policy gates
```

Required outputs:

```text
primary decoded output
uncertainty/source labels
cross-modal consistency score
run log
human-inspectable report
```

Recommended first milestone:

```text
A small dashboard where a researcher can input video, audio, or text, then inspect text, visual, audio, action, policy, and uncertainty heads from the same latent field.
```

This dashboard should be considered a scientific instrument for studying the model, not a product interface.
