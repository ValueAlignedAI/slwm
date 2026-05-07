"""Audio latent/feature adapter for Sprint I2."""

from models.adapters.base import ProjectedFeatureAdapter



class AudioSignalAdapter(ProjectedFeatureAdapter):
    """Map audio latents/features to ``z: FloatTensor[B,T,D]``.

    Shape contract:
        audio ``features``/``latents``: ``FloatTensor[B,T_audio,A]`` ->
        canonical packet ``z: FloatTensor[B,T,D]`` and ``mask: BoolTensor[B,T]``.

    Sprint I2 expects precomputed/provided audio latents or features; raw
    waveform/log-mel preprocessing remains out of scope.
    """

    modality = "audio"

    def __init__(
        self,
        latent_length: int = 1024,
        latent_dim: int = 768,
        *,
        audio_feature_dim: int = 80,
        seed: int = 0,
        codec_name: str = "provided_audio_latents",
    ) -> None:
        super().__init__(
            latent_length=latent_length,
            latent_dim=latent_dim,
            input_dim=audio_feature_dim,
            modality=self.modality,
            seed=seed,
            codec_name=codec_name,
        )


__all__ = ["AudioSignalAdapter"]
