# Conditional output gallery — X0-M1

Actual saved-model outputs, with original/codec/null controls. All synthesized WAVs use peak normalization for listening; do not infer calibrated loudness. Mouth appearance is low-resolution inverse PCA. Audio-derived identity is unknown. Recognized sentences are one of 12 known prompts, not open text generation.

## EXP-T2-301 · 1015_IEO_SAD_LO · lowest_error

Expected prompt: It's eleven o'clock. · acted emotion SAD · 1.84s.

- [Real audio](EXP-T2-301/validation_1015_IEO_SAD_LO/reference_audio.wav) / [codec audio](EXP-T2-301/validation_1015_IEO_SAD_LO/codec_audio.wav) / [model audio reconstruction](EXP-T2-301/validation_1015_IEO_SAD_LO/audio_reconstruction.wav).
- [Video→generated audio](EXP-T2-301/validation_1015_IEO_SAD_LO/video_to_audio.wav) / [no-input audio](EXP-T2-301/validation_1015_IEO_SAD_LO/null_audio.wav).
- [Audio→mouth](EXP-T2-301/validation_1015_IEO_SAD_LO/audio_to_mouth.gif) / [audio→integrated motion](EXP-T2-301/validation_1015_IEO_SAD_LO/audio_to_motion.gif).
- [Real mouth](EXP-T2-301/validation_1015_IEO_SAD_LO/reference_mouth.gif) / [codec mouth](EXP-T2-301/validation_1015_IEO_SAD_LO/codec_mouth.gif) / [model video reconstruction](EXP-T2-301/validation_1015_IEO_SAD_LO/video_reconstruction.gif) / [no-input mouth](EXP-T2-301/validation_1015_IEO_SAD_LO/null_mouth.gif).

Audio closed-set readout: “It's eleven o'clock.” (100% calibrated class probability).
Video closed-set readout: “It's eleven o'clock.” (48% calibrated class probability).

## EXP-T2-301 · 1015_DFA_DIS_XX · median_error

Expected prompt: Don't forget a jacket. · acted emotion DIS · 2.16s.

- [Real audio](EXP-T2-301/validation_1015_DFA_DIS_XX/reference_audio.wav) / [codec audio](EXP-T2-301/validation_1015_DFA_DIS_XX/codec_audio.wav) / [model audio reconstruction](EXP-T2-301/validation_1015_DFA_DIS_XX/audio_reconstruction.wav).
- [Video→generated audio](EXP-T2-301/validation_1015_DFA_DIS_XX/video_to_audio.wav) / [no-input audio](EXP-T2-301/validation_1015_DFA_DIS_XX/null_audio.wav).
- [Audio→mouth](EXP-T2-301/validation_1015_DFA_DIS_XX/audio_to_mouth.gif) / [audio→integrated motion](EXP-T2-301/validation_1015_DFA_DIS_XX/audio_to_motion.gif).
- [Real mouth](EXP-T2-301/validation_1015_DFA_DIS_XX/reference_mouth.gif) / [codec mouth](EXP-T2-301/validation_1015_DFA_DIS_XX/codec_mouth.gif) / [model video reconstruction](EXP-T2-301/validation_1015_DFA_DIS_XX/video_reconstruction.gif) / [no-input mouth](EXP-T2-301/validation_1015_DFA_DIS_XX/null_mouth.gif).

Audio closed-set readout: “Don't forget a jacket.” (74% calibrated class probability).
Video closed-set readout: “It's eleven o'clock.” (66% calibrated class probability).

Additional frozen linear-head diagnosis: [audio→mouth](EXP-T2-301/validation_1015_DFA_DIS_XX/audio_linear_mouth.gif), [video→mouth reconstruction](EXP-T2-301/validation_1015_DFA_DIS_XX/video_linear_mouth.gif), [audio reconstruction](EXP-T2-301/validation_1015_DFA_DIS_XX/audio_linear_audio.wav), [video→audio](EXP-T2-301/validation_1015_DFA_DIS_XX/video_linear_audio.wav). These separately fitted heads do not replace the original model outputs.

## EXP-T2-301 · 1072_IWW_HAP_XX · highest_error

Expected prompt: I wonder what this is about. · acted emotion HAP · 2.76s.

- [Real audio](EXP-T2-301/validation_1072_IWW_HAP_XX/reference_audio.wav) / [codec audio](EXP-T2-301/validation_1072_IWW_HAP_XX/codec_audio.wav) / [model audio reconstruction](EXP-T2-301/validation_1072_IWW_HAP_XX/audio_reconstruction.wav).
- [Video→generated audio](EXP-T2-301/validation_1072_IWW_HAP_XX/video_to_audio.wav) / [no-input audio](EXP-T2-301/validation_1072_IWW_HAP_XX/null_audio.wav).
- [Audio→mouth](EXP-T2-301/validation_1072_IWW_HAP_XX/audio_to_mouth.gif) / [audio→integrated motion](EXP-T2-301/validation_1072_IWW_HAP_XX/audio_to_motion.gif).
- [Real mouth](EXP-T2-301/validation_1072_IWW_HAP_XX/reference_mouth.gif) / [codec mouth](EXP-T2-301/validation_1072_IWW_HAP_XX/codec_mouth.gif) / [model video reconstruction](EXP-T2-301/validation_1072_IWW_HAP_XX/video_reconstruction.gif) / [no-input mouth](EXP-T2-301/validation_1072_IWW_HAP_XX/null_mouth.gif).

Audio closed-set readout: “I wonder what this is about.” (79% calibrated class probability).
Video closed-set readout: “That is exactly what happened.” (11% calibrated class probability).

## EXP-T2-301 · 1026_DFA_FEA_XX · fixed_test_person

Expected prompt: Don't forget a jacket. · acted emotion FEA · 2.08s.

- [Real audio](EXP-T2-301/test_1026_DFA_FEA_XX/reference_audio.wav) / [codec audio](EXP-T2-301/test_1026_DFA_FEA_XX/codec_audio.wav) / [model audio reconstruction](EXP-T2-301/test_1026_DFA_FEA_XX/audio_reconstruction.wav).
- [Video→generated audio](EXP-T2-301/test_1026_DFA_FEA_XX/video_to_audio.wav) / [no-input audio](EXP-T2-301/test_1026_DFA_FEA_XX/null_audio.wav).
- [Audio→mouth](EXP-T2-301/test_1026_DFA_FEA_XX/audio_to_mouth.gif) / [audio→integrated motion](EXP-T2-301/test_1026_DFA_FEA_XX/audio_to_motion.gif).
- [Real mouth](EXP-T2-301/test_1026_DFA_FEA_XX/reference_mouth.gif) / [codec mouth](EXP-T2-301/test_1026_DFA_FEA_XX/codec_mouth.gif) / [model video reconstruction](EXP-T2-301/test_1026_DFA_FEA_XX/video_reconstruction.gif) / [no-input mouth](EXP-T2-301/test_1026_DFA_FEA_XX/null_mouth.gif).

Audio closed-set readout: “Don't forget a jacket.” (75% calibrated class probability).
Video closed-set readout: “Don't forget a jacket.” (12% calibrated class probability).

## EXP-T2-301 · 1046_DFA_ANG_XX · fixed_test_person

Expected prompt: Don't forget a jacket. · acted emotion ANG · 2.28s.

- [Real audio](EXP-T2-301/test_1046_DFA_ANG_XX/reference_audio.wav) / [codec audio](EXP-T2-301/test_1046_DFA_ANG_XX/codec_audio.wav) / [model audio reconstruction](EXP-T2-301/test_1046_DFA_ANG_XX/audio_reconstruction.wav).
- [Video→generated audio](EXP-T2-301/test_1046_DFA_ANG_XX/video_to_audio.wav) / [no-input audio](EXP-T2-301/test_1046_DFA_ANG_XX/null_audio.wav).
- [Audio→mouth](EXP-T2-301/test_1046_DFA_ANG_XX/audio_to_mouth.gif) / [audio→integrated motion](EXP-T2-301/test_1046_DFA_ANG_XX/audio_to_motion.gif).
- [Real mouth](EXP-T2-301/test_1046_DFA_ANG_XX/reference_mouth.gif) / [codec mouth](EXP-T2-301/test_1046_DFA_ANG_XX/codec_mouth.gif) / [model video reconstruction](EXP-T2-301/test_1046_DFA_ANG_XX/video_reconstruction.gif) / [no-input mouth](EXP-T2-301/test_1046_DFA_ANG_XX/null_mouth.gif).

Audio closed-set readout: “Don't forget a jacket.” (24% calibrated class probability).
Video closed-set readout: “I wonder what this is about.” (16% calibrated class probability).

## EXP-T2-301 · 1081_DFA_ANG_XX · fixed_test_person

Expected prompt: Don't forget a jacket. · acted emotion ANG · 1.92s.

- [Real audio](EXP-T2-301/test_1081_DFA_ANG_XX/reference_audio.wav) / [codec audio](EXP-T2-301/test_1081_DFA_ANG_XX/codec_audio.wav) / [model audio reconstruction](EXP-T2-301/test_1081_DFA_ANG_XX/audio_reconstruction.wav).
- [Video→generated audio](EXP-T2-301/test_1081_DFA_ANG_XX/video_to_audio.wav) / [no-input audio](EXP-T2-301/test_1081_DFA_ANG_XX/null_audio.wav).
- [Audio→mouth](EXP-T2-301/test_1081_DFA_ANG_XX/audio_to_mouth.gif) / [audio→integrated motion](EXP-T2-301/test_1081_DFA_ANG_XX/audio_to_motion.gif).
- [Real mouth](EXP-T2-301/test_1081_DFA_ANG_XX/reference_mouth.gif) / [codec mouth](EXP-T2-301/test_1081_DFA_ANG_XX/codec_mouth.gif) / [model video reconstruction](EXP-T2-301/test_1081_DFA_ANG_XX/video_reconstruction.gif) / [no-input mouth](EXP-T2-301/test_1081_DFA_ANG_XX/null_mouth.gif).

Audio closed-set readout: “Don't forget a jacket.” (48% calibrated class probability).
Video closed-set readout: “It's eleven o'clock.” (24% calibrated class probability).

## EXP-T2-308 · 1015_IEO_SAD_LO · lowest_error

Expected prompt: It's eleven o'clock. · acted emotion SAD · 1.84s.

- [Real audio](EXP-T2-308/validation_1015_IEO_SAD_LO/reference_audio.wav) / [codec audio](EXP-T2-308/validation_1015_IEO_SAD_LO/codec_audio.wav) / [model audio reconstruction](EXP-T2-308/validation_1015_IEO_SAD_LO/audio_reconstruction.wav).
- [Video→generated audio](EXP-T2-308/validation_1015_IEO_SAD_LO/video_to_audio.wav) / [no-input audio](EXP-T2-308/validation_1015_IEO_SAD_LO/null_audio.wav).
- [Audio→mouth](EXP-T2-308/validation_1015_IEO_SAD_LO/audio_to_mouth.gif) / [audio→integrated motion](EXP-T2-308/validation_1015_IEO_SAD_LO/audio_to_motion.gif).
- [Real mouth](EXP-T2-308/validation_1015_IEO_SAD_LO/reference_mouth.gif) / [codec mouth](EXP-T2-308/validation_1015_IEO_SAD_LO/codec_mouth.gif) / [model video reconstruction](EXP-T2-308/validation_1015_IEO_SAD_LO/video_reconstruction.gif) / [no-input mouth](EXP-T2-308/validation_1015_IEO_SAD_LO/null_mouth.gif).

Audio closed-set readout: “It's eleven o'clock.” (100% calibrated class probability).
Video closed-set readout: “The surface is slick.” (63% calibrated class probability).

## EXP-T2-308 · 1048_IWW_HAP_XX · median_error

Expected prompt: I wonder what this is about. · acted emotion HAP · 2.28s.

- [Real audio](EXP-T2-308/validation_1048_IWW_HAP_XX/reference_audio.wav) / [codec audio](EXP-T2-308/validation_1048_IWW_HAP_XX/codec_audio.wav) / [model audio reconstruction](EXP-T2-308/validation_1048_IWW_HAP_XX/audio_reconstruction.wav).
- [Video→generated audio](EXP-T2-308/validation_1048_IWW_HAP_XX/video_to_audio.wav) / [no-input audio](EXP-T2-308/validation_1048_IWW_HAP_XX/null_audio.wav).
- [Audio→mouth](EXP-T2-308/validation_1048_IWW_HAP_XX/audio_to_mouth.gif) / [audio→integrated motion](EXP-T2-308/validation_1048_IWW_HAP_XX/audio_to_motion.gif).
- [Real mouth](EXP-T2-308/validation_1048_IWW_HAP_XX/reference_mouth.gif) / [codec mouth](EXP-T2-308/validation_1048_IWW_HAP_XX/codec_mouth.gif) / [model video reconstruction](EXP-T2-308/validation_1048_IWW_HAP_XX/video_reconstruction.gif) / [no-input mouth](EXP-T2-308/validation_1048_IWW_HAP_XX/null_mouth.gif).

Audio closed-set readout: “I wonder what this is about.” (100% calibrated class probability).
Video closed-set readout: “Maybe tomorrow it will be cold.” (14% calibrated class probability).

Additional frozen linear-head diagnosis: [audio→mouth](EXP-T2-308/validation_1048_IWW_HAP_XX/audio_linear_mouth.gif), [video→mouth reconstruction](EXP-T2-308/validation_1048_IWW_HAP_XX/video_linear_mouth.gif), [audio reconstruction](EXP-T2-308/validation_1048_IWW_HAP_XX/audio_linear_audio.wav), [video→audio](EXP-T2-308/validation_1048_IWW_HAP_XX/video_linear_audio.wav). These separately fitted heads do not replace the original model outputs.

## EXP-T2-308 · 1072_IWW_HAP_XX · highest_error

Expected prompt: I wonder what this is about. · acted emotion HAP · 2.76s.

- [Real audio](EXP-T2-308/validation_1072_IWW_HAP_XX/reference_audio.wav) / [codec audio](EXP-T2-308/validation_1072_IWW_HAP_XX/codec_audio.wav) / [model audio reconstruction](EXP-T2-308/validation_1072_IWW_HAP_XX/audio_reconstruction.wav).
- [Video→generated audio](EXP-T2-308/validation_1072_IWW_HAP_XX/video_to_audio.wav) / [no-input audio](EXP-T2-308/validation_1072_IWW_HAP_XX/null_audio.wav).
- [Audio→mouth](EXP-T2-308/validation_1072_IWW_HAP_XX/audio_to_mouth.gif) / [audio→integrated motion](EXP-T2-308/validation_1072_IWW_HAP_XX/audio_to_motion.gif).
- [Real mouth](EXP-T2-308/validation_1072_IWW_HAP_XX/reference_mouth.gif) / [codec mouth](EXP-T2-308/validation_1072_IWW_HAP_XX/codec_mouth.gif) / [model video reconstruction](EXP-T2-308/validation_1072_IWW_HAP_XX/video_reconstruction.gif) / [no-input mouth](EXP-T2-308/validation_1072_IWW_HAP_XX/null_mouth.gif).

Audio closed-set readout: “I wonder what this is about.” (86% calibrated class probability).
Video closed-set readout: “That is exactly what happened.” (13% calibrated class probability).

## EXP-T2-308 · 1026_DFA_FEA_XX · fixed_test_person

Expected prompt: Don't forget a jacket. · acted emotion FEA · 2.08s.

- [Real audio](EXP-T2-308/test_1026_DFA_FEA_XX/reference_audio.wav) / [codec audio](EXP-T2-308/test_1026_DFA_FEA_XX/codec_audio.wav) / [model audio reconstruction](EXP-T2-308/test_1026_DFA_FEA_XX/audio_reconstruction.wav).
- [Video→generated audio](EXP-T2-308/test_1026_DFA_FEA_XX/video_to_audio.wav) / [no-input audio](EXP-T2-308/test_1026_DFA_FEA_XX/null_audio.wav).
- [Audio→mouth](EXP-T2-308/test_1026_DFA_FEA_XX/audio_to_mouth.gif) / [audio→integrated motion](EXP-T2-308/test_1026_DFA_FEA_XX/audio_to_motion.gif).
- [Real mouth](EXP-T2-308/test_1026_DFA_FEA_XX/reference_mouth.gif) / [codec mouth](EXP-T2-308/test_1026_DFA_FEA_XX/codec_mouth.gif) / [model video reconstruction](EXP-T2-308/test_1026_DFA_FEA_XX/video_reconstruction.gif) / [no-input mouth](EXP-T2-308/test_1026_DFA_FEA_XX/null_mouth.gif).

Audio closed-set readout: “Don't forget a jacket.” (98% calibrated class probability).
Video closed-set readout: “Don't forget a jacket.” (14% calibrated class probability).

## EXP-T2-308 · 1046_DFA_ANG_XX · fixed_test_person

Expected prompt: Don't forget a jacket. · acted emotion ANG · 2.28s.

- [Real audio](EXP-T2-308/test_1046_DFA_ANG_XX/reference_audio.wav) / [codec audio](EXP-T2-308/test_1046_DFA_ANG_XX/codec_audio.wav) / [model audio reconstruction](EXP-T2-308/test_1046_DFA_ANG_XX/audio_reconstruction.wav).
- [Video→generated audio](EXP-T2-308/test_1046_DFA_ANG_XX/video_to_audio.wav) / [no-input audio](EXP-T2-308/test_1046_DFA_ANG_XX/null_audio.wav).
- [Audio→mouth](EXP-T2-308/test_1046_DFA_ANG_XX/audio_to_mouth.gif) / [audio→integrated motion](EXP-T2-308/test_1046_DFA_ANG_XX/audio_to_motion.gif).
- [Real mouth](EXP-T2-308/test_1046_DFA_ANG_XX/reference_mouth.gif) / [codec mouth](EXP-T2-308/test_1046_DFA_ANG_XX/codec_mouth.gif) / [model video reconstruction](EXP-T2-308/test_1046_DFA_ANG_XX/video_reconstruction.gif) / [no-input mouth](EXP-T2-308/test_1046_DFA_ANG_XX/null_mouth.gif).

Audio closed-set readout: “Don't forget a jacket.” (32% calibrated class probability).
Video closed-set readout: “Don't forget a jacket.” (12% calibrated class probability).

## EXP-T2-308 · 1081_DFA_ANG_XX · fixed_test_person

Expected prompt: Don't forget a jacket. · acted emotion ANG · 1.92s.

- [Real audio](EXP-T2-308/test_1081_DFA_ANG_XX/reference_audio.wav) / [codec audio](EXP-T2-308/test_1081_DFA_ANG_XX/codec_audio.wav) / [model audio reconstruction](EXP-T2-308/test_1081_DFA_ANG_XX/audio_reconstruction.wav).
- [Video→generated audio](EXP-T2-308/test_1081_DFA_ANG_XX/video_to_audio.wav) / [no-input audio](EXP-T2-308/test_1081_DFA_ANG_XX/null_audio.wav).
- [Audio→mouth](EXP-T2-308/test_1081_DFA_ANG_XX/audio_to_mouth.gif) / [audio→integrated motion](EXP-T2-308/test_1081_DFA_ANG_XX/audio_to_motion.gif).
- [Real mouth](EXP-T2-308/test_1081_DFA_ANG_XX/reference_mouth.gif) / [codec mouth](EXP-T2-308/test_1081_DFA_ANG_XX/codec_mouth.gif) / [model video reconstruction](EXP-T2-308/test_1081_DFA_ANG_XX/video_reconstruction.gif) / [no-input mouth](EXP-T2-308/test_1081_DFA_ANG_XX/null_mouth.gif).

Audio closed-set readout: “Don't forget a jacket.” (72% calibrated class probability).
Video closed-set readout: “I'm on my way to the meeting.” (16% calibrated class probability).

## EXP-T2-309 · 1015_IEO_SAD_LO · lowest_error

Expected prompt: It's eleven o'clock. · acted emotion SAD · 1.84s.

- [Real audio](EXP-T2-309/validation_1015_IEO_SAD_LO/reference_audio.wav) / [codec audio](EXP-T2-309/validation_1015_IEO_SAD_LO/codec_audio.wav) / [model audio reconstruction](EXP-T2-309/validation_1015_IEO_SAD_LO/audio_reconstruction.wav).
- [Video→generated audio](EXP-T2-309/validation_1015_IEO_SAD_LO/video_to_audio.wav) / [no-input audio](EXP-T2-309/validation_1015_IEO_SAD_LO/null_audio.wav).
- [Audio→mouth](EXP-T2-309/validation_1015_IEO_SAD_LO/audio_to_mouth.gif) / [audio→integrated motion](EXP-T2-309/validation_1015_IEO_SAD_LO/audio_to_motion.gif).
- [Real mouth](EXP-T2-309/validation_1015_IEO_SAD_LO/reference_mouth.gif) / [codec mouth](EXP-T2-309/validation_1015_IEO_SAD_LO/codec_mouth.gif) / [model video reconstruction](EXP-T2-309/validation_1015_IEO_SAD_LO/video_reconstruction.gif) / [no-input mouth](EXP-T2-309/validation_1015_IEO_SAD_LO/null_mouth.gif).

Audio closed-set readout: “It's eleven o'clock.” (97% calibrated class probability).
Video closed-set readout: “It's eleven o'clock.” (36% calibrated class probability).

## EXP-T2-309 · 1069_IOM_FEA_XX · median_error

Expected prompt: I'm on my way to the meeting. · acted emotion FEA · 2.08s.

- [Real audio](EXP-T2-309/validation_1069_IOM_FEA_XX/reference_audio.wav) / [codec audio](EXP-T2-309/validation_1069_IOM_FEA_XX/codec_audio.wav) / [model audio reconstruction](EXP-T2-309/validation_1069_IOM_FEA_XX/audio_reconstruction.wav).
- [Video→generated audio](EXP-T2-309/validation_1069_IOM_FEA_XX/video_to_audio.wav) / [no-input audio](EXP-T2-309/validation_1069_IOM_FEA_XX/null_audio.wav).
- [Audio→mouth](EXP-T2-309/validation_1069_IOM_FEA_XX/audio_to_mouth.gif) / [audio→integrated motion](EXP-T2-309/validation_1069_IOM_FEA_XX/audio_to_motion.gif).
- [Real mouth](EXP-T2-309/validation_1069_IOM_FEA_XX/reference_mouth.gif) / [codec mouth](EXP-T2-309/validation_1069_IOM_FEA_XX/codec_mouth.gif) / [model video reconstruction](EXP-T2-309/validation_1069_IOM_FEA_XX/video_reconstruction.gif) / [no-input mouth](EXP-T2-309/validation_1069_IOM_FEA_XX/null_mouth.gif).

Audio closed-set readout: “I'm on my way to the meeting.” (99% calibrated class probability).
Video closed-set readout: “It's eleven o'clock.” (20% calibrated class probability).

## EXP-T2-309 · 1072_IWW_HAP_XX · highest_error

Expected prompt: I wonder what this is about. · acted emotion HAP · 2.76s.

- [Real audio](EXP-T2-309/validation_1072_IWW_HAP_XX/reference_audio.wav) / [codec audio](EXP-T2-309/validation_1072_IWW_HAP_XX/codec_audio.wav) / [model audio reconstruction](EXP-T2-309/validation_1072_IWW_HAP_XX/audio_reconstruction.wav).
- [Video→generated audio](EXP-T2-309/validation_1072_IWW_HAP_XX/video_to_audio.wav) / [no-input audio](EXP-T2-309/validation_1072_IWW_HAP_XX/null_audio.wav).
- [Audio→mouth](EXP-T2-309/validation_1072_IWW_HAP_XX/audio_to_mouth.gif) / [audio→integrated motion](EXP-T2-309/validation_1072_IWW_HAP_XX/audio_to_motion.gif).
- [Real mouth](EXP-T2-309/validation_1072_IWW_HAP_XX/reference_mouth.gif) / [codec mouth](EXP-T2-309/validation_1072_IWW_HAP_XX/codec_mouth.gif) / [model video reconstruction](EXP-T2-309/validation_1072_IWW_HAP_XX/video_reconstruction.gif) / [no-input mouth](EXP-T2-309/validation_1072_IWW_HAP_XX/null_mouth.gif).

Audio closed-set readout: “I wonder what this is about.” (91% calibrated class probability).
Video closed-set readout: “That is exactly what happened.” (12% calibrated class probability).

## EXP-T2-309 · 1026_DFA_FEA_XX · fixed_test_person

Expected prompt: Don't forget a jacket. · acted emotion FEA · 2.08s.

- [Real audio](EXP-T2-309/test_1026_DFA_FEA_XX/reference_audio.wav) / [codec audio](EXP-T2-309/test_1026_DFA_FEA_XX/codec_audio.wav) / [model audio reconstruction](EXP-T2-309/test_1026_DFA_FEA_XX/audio_reconstruction.wav).
- [Video→generated audio](EXP-T2-309/test_1026_DFA_FEA_XX/video_to_audio.wav) / [no-input audio](EXP-T2-309/test_1026_DFA_FEA_XX/null_audio.wav).
- [Audio→mouth](EXP-T2-309/test_1026_DFA_FEA_XX/audio_to_mouth.gif) / [audio→integrated motion](EXP-T2-309/test_1026_DFA_FEA_XX/audio_to_motion.gif).
- [Real mouth](EXP-T2-309/test_1026_DFA_FEA_XX/reference_mouth.gif) / [codec mouth](EXP-T2-309/test_1026_DFA_FEA_XX/codec_mouth.gif) / [model video reconstruction](EXP-T2-309/test_1026_DFA_FEA_XX/video_reconstruction.gif) / [no-input mouth](EXP-T2-309/test_1026_DFA_FEA_XX/null_mouth.gif).

Audio closed-set readout: “Don't forget a jacket.” (63% calibrated class probability).
Video closed-set readout: “I'm on my way to the meeting.” (12% calibrated class probability).

## EXP-T2-309 · 1046_DFA_ANG_XX · fixed_test_person

Expected prompt: Don't forget a jacket. · acted emotion ANG · 2.28s.

- [Real audio](EXP-T2-309/test_1046_DFA_ANG_XX/reference_audio.wav) / [codec audio](EXP-T2-309/test_1046_DFA_ANG_XX/codec_audio.wav) / [model audio reconstruction](EXP-T2-309/test_1046_DFA_ANG_XX/audio_reconstruction.wav).
- [Video→generated audio](EXP-T2-309/test_1046_DFA_ANG_XX/video_to_audio.wav) / [no-input audio](EXP-T2-309/test_1046_DFA_ANG_XX/null_audio.wav).
- [Audio→mouth](EXP-T2-309/test_1046_DFA_ANG_XX/audio_to_mouth.gif) / [audio→integrated motion](EXP-T2-309/test_1046_DFA_ANG_XX/audio_to_motion.gif).
- [Real mouth](EXP-T2-309/test_1046_DFA_ANG_XX/reference_mouth.gif) / [codec mouth](EXP-T2-309/test_1046_DFA_ANG_XX/codec_mouth.gif) / [model video reconstruction](EXP-T2-309/test_1046_DFA_ANG_XX/video_reconstruction.gif) / [no-input mouth](EXP-T2-309/test_1046_DFA_ANG_XX/null_mouth.gif).

Audio closed-set readout: “I think I have a doctor's appointment.” (41% calibrated class probability).
Video closed-set readout: “The surface is slick.” (13% calibrated class probability).

## EXP-T2-309 · 1081_DFA_ANG_XX · fixed_test_person

Expected prompt: Don't forget a jacket. · acted emotion ANG · 1.92s.

- [Real audio](EXP-T2-309/test_1081_DFA_ANG_XX/reference_audio.wav) / [codec audio](EXP-T2-309/test_1081_DFA_ANG_XX/codec_audio.wav) / [model audio reconstruction](EXP-T2-309/test_1081_DFA_ANG_XX/audio_reconstruction.wav).
- [Video→generated audio](EXP-T2-309/test_1081_DFA_ANG_XX/video_to_audio.wav) / [no-input audio](EXP-T2-309/test_1081_DFA_ANG_XX/null_audio.wav).
- [Audio→mouth](EXP-T2-309/test_1081_DFA_ANG_XX/audio_to_mouth.gif) / [audio→integrated motion](EXP-T2-309/test_1081_DFA_ANG_XX/audio_to_motion.gif).
- [Real mouth](EXP-T2-309/test_1081_DFA_ANG_XX/reference_mouth.gif) / [codec mouth](EXP-T2-309/test_1081_DFA_ANG_XX/codec_mouth.gif) / [model video reconstruction](EXP-T2-309/test_1081_DFA_ANG_XX/video_reconstruction.gif) / [no-input mouth](EXP-T2-309/test_1081_DFA_ANG_XX/null_mouth.gif).

Audio closed-set readout: “I think I have a doctor's appointment.” (31% calibrated class probability).
Video closed-set readout: “I'm on my way to the meeting.” (15% calibrated class probability).
