from dataclasses import dataclass


@dataclass(frozen=True)
class StylePreset:
    key: str
    label: str
    description: str
    saturation_boost: float = 1.25
    white_threshold: int = 234
    black_threshold: int = 30
    neutral_transition: int = 42
    l_blend_alpha: float = 0.10
    guided_filter_radius: int = 2
    guided_filter_eps: float = 0.01
    chroma_warm_shift: float = 0.0
    chroma_red_shift: float = 0.0
    l_gamma: float = 1.0
    denoise_sigma: int = 15
    diffusion_steps: int = 16
    cel_flatten: float = 0.35
    neutral_fade_floor: float = 0.35


STYLE_PRESETS: dict[str, StylePreset] = {
    "neutral": StylePreset(
        key="neutral",
        label="Neutral (default)",
        description="Faithful to model output, balanced post-processing.",
    ),
    "shonen": StylePreset(
        key="shonen",
        label="Shonen Vibrant",
        description="Punchy primaries, warm skin, high saturation.",
        saturation_boost=1.7,
        white_threshold=225,
        black_threshold=25,
        l_blend_alpha=0.15,
        chroma_warm_shift=4.0,
        chroma_red_shift=2.0,
    ),
}


def get_style(key: str | None) -> StylePreset:
    if not key:
        return STYLE_PRESETS["neutral"]
    return STYLE_PRESETS.get(key.lower(), STYLE_PRESETS["neutral"])
