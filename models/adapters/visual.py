"""Visual/video latent/feature adapter for Sprint I2."""

from models.adapters.base import ProjectedFeatureAdapter



class VisualSignalAdapter(ProjectedFeatureAdapter):
    """Map visual/video latents/features to ``z: FloatTensor[B,T,D]``.

    Shape contract:
        visual ``features``/``latents``: ``FloatTensor[B,T_visual,V]`` ->
        canonical packet ``z: FloatTensor[B,T,D]`` and ``mask: BoolTensor[B,T]``.

    Sprint I2 expects precomputed/provided patch, tubelet, or codec latents;
    raw image/video preprocessing remains out of scope.
    """

    modality = "visual_video"

    def __init__(
        self,
        latent_length: int = 1024,
        latent_dim: int = 768,
        *,
        visual_feature_dim: int = 256,
        seed: int = 0,
        codec_name: str = "provided_visual_latents",
    ) -> None:
        super().__init__(
            latent_length=latent_length,
            latent_dim=latent_dim,
            input_dim=visual_feature_dim,
            modality=self.modality,
            seed=seed,
            codec_name=codec_name,
        )


__all__ = ["VisualSignalAdapter"]
