# DocFusion Demo Video Script

## INTRO (~15 seconds)

"As-salamu alaykum. My name is Abdul Rahim, and this is the live demonstration of the Intelligent Document Analysis and Forgery Detection tool, built for the Rihal CodeStacker 2026 Machine Learning Challenge."

## WHAT IT DOES (~30 seconds)

"This tool takes scanned receipt images and does three things.

First, it extracts structured data — the vendor name, the date, and the total amount — using OCR, which stands for Optical Character Recognition. It's a technique that reads text directly from images.

Second, it detects whether the document has been tampered with or forged, using image forensics and a machine learning classifier called LightGBM. This classifier analyzes over 40 features extracted from the image — things like noise patterns, gradient textures, and compression inconsistencies.

And third, it visualizes Error Level Analysis, or ELA. ELA works by re-compressing the image and comparing the result to the original. If someone edited part of the image and saved it again, those edited pixels will have a different compression pattern. ELA makes those differences visible as brighter spots."

## GENUINE RECEIPT (~30 seconds)

"I've prepared two sample receipts — one genuine and one forged. Let's start with the genuine one."

> Drag and drop genuine_receipt.png. Wait for processing.

"Here are the results. On the right side, you can see it successfully extracted the vendor name, the date, and the total amount. The forgery analysis shows this is genuine, with a very low forgery probability.

Below that, you can see the Anomaly Summary — a human-readable explanation of the forensic analysis. It confirms that all image forensic indicators are within normal ranges.

On the left, the Error Level Analysis shows fairly uniform brightness across the entire image. This is what we'd expect from an untampered document — the compression artifacts are consistent everywhere, meaning nothing was edited after the original save."

## FORGED RECEIPT (~30 seconds)

"Now let's try a forged receipt."

> Remove previous file, drag and drop forged_receipt.png. Wait for processing.

"This time, the system flagged it as suspicious with a much higher forgery probability. If you look at the Anomaly Summary, it provides a detailed explanation of exactly what was detected — for example, high ELA peaks indicating copy-paste editing, and inconsistent compression patterns across different regions of the image.

On the left, the Error Level Analysis visually confirms this. You can see specific regions that are noticeably brighter than the rest. Those brighter areas indicate where the compression pattern is inconsistent — which is exactly where tampering occurred."

## CLOSING (~15 seconds)

"Under the hood, the pipeline uses EasyOCR for text extraction, regex-based parsing for structured fields, and a LightGBM classifier trained on over 40 image forensic features including Error Level Analysis, noise analysis, gradient textures, and patch-level consistency checks. The system also generates intelligent anomaly summaries that explain its findings in plain English.

The full solution is containerized with Docker and integrates with the DocFusion autograder harness. Thank you."
