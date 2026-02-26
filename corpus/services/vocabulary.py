GM_CATEGORIES = {
    range(0, 8): "piano",
    range(8, 16): "chromatic_percussion",
    range(16, 24): "organ",
    range(24, 32): "guitar",
    range(32, 40): "bass",
    range(40, 48): "strings",
    range(48, 56): "ensemble",
    range(56, 64): "brass",
    range(64, 72): "reed",
    range(72, 80): "pipe",
    range(80, 88): "synth",
    range(88, 96): "synth",
    range(96, 104): "synth_fx",
    range(104, 112): "ethnic",
    range(112, 120): "percussive",
    range(120, 128): "sound_fx",
}


def categorize_instrument(program, is_drum):
    """Map a GM program number to an instrument category string."""
    if is_drum:
        return "drums"
    for prog_range, category in GM_CATEGORIES.items():
        if program in prog_range:
            return category
    return "other"


def build_vocabulary(scan_results):
    """Build instrument vocabulary from scan results.

    Returns dict mapping category name to one-hot index.
    """
    categories = set()
    for song in scan_results:
        for inst in song.get("instruments", []):
            cat = categorize_instrument(inst["program"], inst["is_drum"])
            categories.add(cat)
    return {cat: idx for idx, cat in enumerate(sorted(categories))}
