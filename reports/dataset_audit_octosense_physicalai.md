# Dataset Audit: OctoSense and PhysicalAI-Human-1M

## 1. Executive Summary
This document audits the OctoSense and PhysicalAI-Human-1M datasets for their suitability in training Signal-Latent World Models (SLWM) as defined in AGENTS.md. Key criteria include multimodal signal integrity, causal processing, and potential for disjoint source/scene/subject splitting.

## 2. OctoSense Audit
- Source: anthonytec2/OctoSense
- Modalities: Stereo RGB, Event Cameras, LiDAR, Thermal, IMU, GPS, CAN bus.
- Scale: ~59 hours of driving, plus marine and quadruped platform data.
- Causal Integrity: Needs audit for look-ahead in sensor fusion or label pre-processing (e.g., optical flow/depth computed using future frames).
- Split Strategy:
    - Source: Use platform-specific splits (car vs. boat vs. quadruped).
    - Scene: Geographic and weather-based segmentation.
    - Subject: Driver/platform-id mapping if available.

## 3. PhysicalAI-Human-1M Audit
- Source: nvidia/PhysicalAI-Human-1M-100K
- Modalities: Image, Video, Text, Tabular, hand-tracking.
- Scale: 1M+ samples (100K subset for initial checks).
- Causal Integrity: Need to confirm action labels are not future-dependent.
- Split Strategy:
    - Source: Subject-disjoint splits are essential for generalization.
    - Scene: Environment-disjoint splits.

## 4. Proposed Research Readiness Plan
1. Infrastructure: Develop DatasetAdapter classes to convert raw data into the SLWM LatentSignalField format.
2. Causal Audit: Implement unit tests that check for temporal look-ahead in labels (target[t+1] availability at t).
3. Splitting: Create a manifest.json that locks in disjoint train/val/test splits based on recording/scene/subject.
