from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StylePreset:
    key: str
    label: str
    description: str
    saturation_boost: float = 1.0
    gamma: float = 1.0  # < 1.0 ilumina/suaviza, > 1.0 da contraste/oscurece
    shadow_lift: float = 0.0  # Eleva pedestal de negros (estética celuloide/analógica)
    tint_a: float = 0.0  # Eje Verde (-) / Magenta (+) en espacio LAB
    tint_b: float = 0.0  # Eje Azul (-) / Amarillo (+) en espacio LAB
    white_threshold: int = 250
    black_threshold: int = 26
    guided_filter_radius: int = 3
    guided_filter_eps: float = 0.02


STYLE_PRESETS: dict[str, StylePreset] = {
    "neutral": StylePreset(
        key="neutral",
        label="Neutral (Equilibrado)",
        description="Salida balanceada directa del modelo neuronal sin sobreprocesamiento.",
        saturation_boost=1.0,
        gamma=1.0,
        shadow_lift=0.0,
        tint_a=0.0,
        tint_b=0.0,
        white_threshold=250,
        black_threshold=25,
        guided_filter_radius=3,
        guided_filter_eps=0.02,
    ),
    "faithful_1to1": StylePreset(
        key="faithful_1to1",
        label="Fidelidad 1:1",
        description="Fidelidad cromática estricta con máxima nitidez de trazo.",
        saturation_boost=1.0,
        gamma=1.0,
        shadow_lift=0.0,
        tint_a=0.0,
        tint_b=0.0,
        white_threshold=250,
        black_threshold=26,
        guided_filter_radius=3,
        guided_filter_eps=0.015,
    ),
    "shonen": StylePreset(
        key="shonen",
        label="Shonen Vibrant",
        description="Primarios vivos, piel cálida y contraste dinámico de acción.",
        saturation_boost=1.35,
        gamma=1.08,
        shadow_lift=0.0,
        tint_a=9.0,
        tint_b=14.0,
        white_threshold=250,
        black_threshold=28,
        guided_filter_radius=3,
        guided_filter_eps=0.02,
    ),
    "vibrant": StylePreset(
        key="vibrant",
        label="Ultra Vibrant",
        description="Saturación de alta intensidad con realce de gama completa.",
        saturation_boost=1.55,
        gamma=1.05,
        shadow_lift=0.0,
        tint_a=6.0,
        tint_b=10.0,
        white_threshold=252,
        black_threshold=26,
        guided_filter_radius=3,
        guided_filter_eps=0.02,
    ),
    "webtoon_modern": StylePreset(
        key="webtoon_modern",
        label="Webtoon / Manhwa Moderno",
        description="Estética digital nítida: medios tonos claros, sombras recortadas y piel suave.",
        saturation_boost=1.20,
        gamma=0.94,
        shadow_lift=0.0,
        tint_a=7.0,
        tint_b=8.0,
        white_threshold=252,
        black_threshold=26,
        guided_filter_radius=4,
        guided_filter_eps=0.025,
    ),
    "shojo_pastel": StylePreset(
        key="shojo_pastel",
        label="Shojo / Romance Pastel",
        description="Tonos desaturados suaves, sombras aireadas y rubor rosado uniforme.",
        saturation_boost=0.92,
        gamma=0.88,
        shadow_lift=8.0,
        tint_a=14.0,
        tint_b=3.0,
        white_threshold=248,
        black_threshold=20,
        guided_filter_radius=4,
        guided_filter_eps=0.03,
    ),
    "seinen_dark": StylePreset(
        key="seinen_dark",
        label="Seinen / Dark Fantasy",
        description="Atmósfera fría y pesada: sombras profundas, bajo brillo y tinte azulado.",
        saturation_boost=0.80,
        gamma=1.22,
        shadow_lift=0.0,
        tint_a=-6.0,
        tint_b=-16.0,
        white_threshold=248,
        black_threshold=34,
        guided_filter_radius=3,
        guided_filter_eps=0.02,
    ),
    "retro_90s": StylePreset(
        key="retro_90s",
        label="Retro Anime 90s",
        description="Pedestal de negros levantado, calidez analógica y sombras estilo celuloide.",
        saturation_boost=1.15,
        gamma=0.96,
        shadow_lift=14.0,
        tint_a=5.0,
        tint_b=18.0,
        white_threshold=248,
        black_threshold=28,
        guided_filter_radius=4,
        guided_filter_eps=0.035,
    ),
    "cyberpunk_neon": StylePreset(
        key="cyberpunk_neon",
        label="Cyberpunk Neon",
        description="Bipolaridad cromática acentuada: sombras cian/azul y luces magenta frío.",
        saturation_boost=1.45,
        gamma=1.12,
        shadow_lift=0.0,
        tint_a=18.0,
        tint_b=-20.0,
        white_threshold=252,
        black_threshold=30,
        guided_filter_radius=3,
        guided_filter_eps=0.02,
    ),
}


def get_style(key: str | None) -> StylePreset:
    if not key:
        return STYLE_PRESETS["neutral"]
    return STYLE_PRESETS.get(key.lower(), STYLE_PRESETS["neutral"])
