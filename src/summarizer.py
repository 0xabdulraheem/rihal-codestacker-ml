from __future__ import annotations

from typing import Any


def generate_anomaly_summary(
    fields: dict[str, Any],
    features: dict[str, float],
    is_forged: int,
    probability: float,
) -> str:
    reasons = []
    verdict = "SUSPICIOUS" if is_forged == 1 else "GENUINE"

    ela_mean = features.get("ela_mean", 0)
    ela_max = features.get("ela_max", 0)
    ela_patch_std = features.get("ela_patch_std", 0)
    ela_patch_range = features.get("ela_patch_range", 0)
    ela_kurtosis = features.get("ela_kurtosis", 0)
    noise_std = features.get("noise_std", 0)
    gradient_skew = features.get("gradient_skew", 0)
    blur_score = features.get("blur_score", 0)
    field_completeness = features.get("field_completeness", 1.0)
    ela_diff = features.get("ela_diff_mean", 0)

    if ela_max > 200:
        reasons.append(
            f"High ELA peak detected (max={ela_max:.0f}). "
            "This suggests certain regions of the image have compression artifacts "
            "that differ significantly from the rest, which is a common indicator of "
            "copy-paste or pixel-level editing."
        )
    elif ela_max > 120:
        reasons.append(
            f"Moderate ELA peak detected (max={ela_max:.0f}). "
            "Some regions show slightly inconsistent compression patterns."
        )

    if ela_patch_std > 15:
        reasons.append(
            f"Inconsistent ELA across image regions (patch std={ela_patch_std:.1f}). "
            "Different areas of the document show varying levels of compression artifacts, "
            "suggesting parts of the image may have been modified independently."
        )

    if ela_patch_range > 30:
        reasons.append(
            f"Large ELA variation between patches (range={ela_patch_range:.1f}). "
            "The difference between the most and least compressed regions is unusually high."
        )

    if abs(ela_diff) > 10:
        reasons.append(
            "JPEG ghost artifacts detected. Comparing ELA at different quality levels "
            "reveals inconsistencies that suggest the image was saved multiple times "
            "with different compression settings, which often occurs during editing."
        )

    if ela_kurtosis > 5:
        reasons.append(
            f"Unusual ELA distribution shape (kurtosis={ela_kurtosis:.1f}). "
            "The error pattern has heavy tails, indicating isolated regions with "
            "very different compression characteristics."
        )

    if noise_std > 20:
        reasons.append(
            f"High noise variation detected (std={noise_std:.1f}). "
            "The noise pattern across the document is inconsistent, which can indicate "
            "that parts of the image originated from different sources."
        )

    if blur_score < 50 and ela_max > 100:
        reasons.append(
            f"Inconsistent sharpness detected (blur score={blur_score:.0f}). "
            "The document appears very blurry overall, but contains sharp editing "
            "artifacts. This combination can indicate tampering on a low-quality scan."
        )

    if gradient_skew > 2.0 or gradient_skew < -1.0:
        reasons.append(
            f"Unusual edge distribution (gradient skew={gradient_skew:.2f}). "
            "The distribution of edge intensities is abnormal, suggesting "
            "unnatural transitions that may result from digital editing."
        )

    if field_completeness < 0.67:
        missing = []
        if not fields.get("vendor"):
            missing.append("vendor")
        if not fields.get("date"):
            missing.append("date")
        if not fields.get("total"):
            missing.append("total")
        if missing:
            reasons.append(
                f"Missing fields: {', '.join(missing)}. "
                "Legitimate receipts typically contain all standard fields. "
                "Missing information may indicate the document was altered or is incomplete."
            )

    if not reasons and is_forged == 1:
        reasons.append(
            "The combination of multiple subtle image features triggered the forgery detector. "
            "While no single feature is strongly anomalous, the overall pattern of compression "
            "artifacts, noise levels, and texture characteristics is more consistent with "
            "a modified document than a genuine one."
        )

    if not reasons and is_forged == 0:
        reasons.append(
            "All image forensic indicators are within normal ranges. "
            "The compression artifacts are uniform, the noise pattern is consistent, "
            "and the ELA shows no signs of localized editing."
        )

    summary_lines = [f"Verdict: {verdict} (confidence: {probability:.1%})\n"]

    if is_forged == 1:
        summary_lines.append("Findings:\n")
    else:
        summary_lines.append("Analysis:\n")

    for i, reason in enumerate(reasons, 1):
        summary_lines.append(f"{i}. {reason}\n")

    if is_forged == 1:
        summary_lines.append(
            "\nRecommendation: This document should be flagged for manual review. "
            "The forensic analysis identified patterns commonly associated with "
            "digital tampering."
        )
    else:
        summary_lines.append(
            "\nRecommendation: No action required. The document appears to be authentic "
            "based on image forensic analysis."
        )

    return "\n".join(summary_lines)
