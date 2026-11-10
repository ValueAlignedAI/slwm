# Continuous inference examples

The same 43.69 seconds and 1,024 model ticks were used for both 116.50M architectures, seed 511, at 16,384 updates on the fresh mixture. The held recording is STARSS23 `fold4_room10_mix001` from Finland. State carries between 16 successive blocks. The small files here contain model-generated tracks. The original inputs, full audiovisual comparisons, fixed replays and optimizer states are on the Fedora checkpoint disks. [Generation measurements and file hashes](../metrics/continuous-generation.json) · [STARSS23 source](https://doi.org/10.5281/zenodo.7880637)

| Input → output | Hybrid | Dense GPT-2 |
|---|---|---|
| Sound → silent video | [Video](hybrid/audio-to-video.webm) | [Video](gpt2/audio-to-video.webm) |
| Video → sound | [Audio](hybrid/video-to-audio.ogg) | [Audio](gpt2/video-to-audio.ogg) |
| Four written prompts → sound and video | [Clip](hybrid/text-to-audio-video.webm) | [Clip](gpt2/text-to-audio-video.webm) |

The text prompts describe clapping, walking, a closing door and a knock in that order. Video has 16×16 native grayscale pixels. For the fresh hybrid, generated audio loudness correlated 0.42 with the source recording over time, compared with 0.10 for dense on the same example. Neither model's generated video motion exceeded the 95th percentile of the shifted controls on this clip. Speech content, generalization and decision success are measured separately. [Timing and speech tests](../alignment-and-control.md)
