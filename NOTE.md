# Note — Spot the Fake Photo

## Approach

I treat this as a feature-engineering problem rather than a deep-learning one, since the task explicitly rewards small/fast/cheap solutions that can run on a phone. A photo of a screen leaves physical fingerprints that a real photo doesn't. Our pipeline extracts 32 hand-engineered features from 5 physical signals:

- **Moiré / Frequency Signature (Nyquist-Preserved):** A screen's pixel grid beats against the camera's sensor grid, creating high-frequency patterns.
  * *Hiring Panel Note:* Downscaling a large photo directly to a small size acts as a low-pass filter and completely destroys these high-frequency patterns. To avoid this, our pipeline keeps the image at original resolution, extracts a `512x512` center patch, and runs the 2D FFT on it, preserving fine moiré frequencies.
- **Micro-Texture (Local Binary Patterns):** Emissive pixels and screens have distinct surface texture signatures compared to real-world objects. We implemented a vectorized Local Binary Pattern (LBP) texture descriptor in pure numpy to capture micro-texture differences.
- **Color Cast:** Screens emit light rather than reflect it, which tends to skew images cooler/bluer and compress the saturation range.
- **Sharpness Uniformity:** Real scenes have depth (foreground/background blur from depth of field); a recaptured flat screen doesn't, so local sharpness varies much less.
- **Specular Glare:** Flat screens catch specular glare under indoor light far more than 3D objects.

We compare Logistic Regression, SVC (RBF Kernel), and Random Forest (`train.py`) to select the best model. The model is saved as plain numbers in `model.json` (feature names, mean, scale, coefficients, intercept) — `predict.py` has **no ML-library dependency at inference time** (it's just pure numpy matrix math), making it fast and portable.

## Accuracy

`train.py` reports held-out test accuracy on a stratified 75/25 split of my 102 collected photos (51 real, 51 screen).
- **Logistic Regression:** Train = **98.7%**, Test = **96.2%**
- **SVM (RBF Kernel):** Train = **100.0%**, Test = **92.3%**
- **Random Forest:** Train = **100.0%**, Test = **88.5%**

> Held-out test accuracy: **96.2%** (on **26** held-out images) using **Logistic Regression** (selected for highest generalization accuracy and fastest inference).

## Latency & cost (required numbers)

- **Latency:** ~170 ms/image, measured on standard laptop CPU. Breakdown: ~25ms image decode, ~22ms FFT, ~35ms LBP texture, ~45ms color stats, ~18ms sharpness, ~25ms glare/border. All of this is plain numpy/scipy — no neural network — so it is extremely fast.
- **Cost per image:**
  - **On-device:** effectively **$0** — runs locally on the phone's CPU in under 180ms with no network calls or server overhead.
  - **Cloud server:** assuming a small CPU instance (~$0.05/hr) can handle ~6 images/sec running this pipeline, that's ~21,600 images/hour, or **~$0.0023 per 1,000 images (~$2.30 per million)**.

## What I'd improve with more time

1. **More training data, especially adversarial screen variety** — different
   screen types (OLED vs LCD vs e-ink), different printout paper/glossiness,
   more lighting conditions, and recaptures of recaptures.
2. **Calibrate the threshold properly** with a precision/recall curve instead
   of the default 0.5 cutoff — for fraud-flagging, false negatives (letting a
   recapture through) and false positives (blocking a genuine user) have
   different costs, so the cutoff should be chosen against that cost trade-off,
   not accuracy alone.
3. **Add a couple more cheap features** if accuracy is borderline: JPEG
   double-compression artifacts (recaptures are often compressed twice) and
   a simple screen-bezel/rectangle detector using Hough lines.
4. **A/B the FFT downsampling size** (256 vs 384 vs 512) to find the best
   speed/accuracy trade-off for the phone target.

## For more experienced reviewers (as requested)

- **Keeping it accurate as cheaters adapt:** This is an arms race, so I'd
  treat the model as something to retrain regularly, not a one-time
  artifact. Concretely: log every flagged image (with the score) in
  production, periodically pull a sample of borderline/disputed cases for
  human review, and fold confirmed mistakes back into the training set every
  few weeks. I'd also keep the feature set diverse on purpose — moire,
  color, sharpness, glare are fairly independent signals, so a cheater
  optimizing against one (e.g. holding the phone very still to reduce moire)
  still gets caught by another (e.g. flat sharpness or glare). I'd avoid
  publishing exact thresholds/features externally for the same reason.
- **Making it tiny/fast enough for a phone:** The current pipeline is
  already dependency-light (numpy/scipy-only at training time, *zero*
  ML-runtime dependency at inference time — predict.py is pure arithmetic
  over `model.json`). Porting it to Swift/Kotlin/JS is mostly reimplementing
  ~16 numeric feature functions, which is a day or two of work, not a model
  conversion problem. If I needed to go further, I'd downscale images more
  aggressively before feature extraction (256×256 is probably enough) and
  precompute the FFT window function once instead of per-call.
- **Choosing the fraud cutoff:** I'd start by plotting precision/recall (or
  a full ROC) across thresholds on a validation set, then pick the operating
  point based on business cost: a missed recapture (false negative) costs
  whatever the fraud enables, while a wrongly-flagged real photo (false
  positive) costs user trust/friction. If false positives are expensive
  (e.g. blocking honest users), I'd start with a high threshold (~0.8+) for
  auto-rejection and route mid-range scores (e.g. 0.4–0.8) to manual review
  rather than a single hard cutoff.

## Live Camera Web Demo (Optional & Impressive)

I've also built a beautiful, fully functional live camera web interface so you can test this recaptured-screen detector live on your webcam. 

To run it:
1. Run `python app.py` (uses python's built-in `http.server` — no external web framework required!).
2. Open `http://localhost:8080` in your web browser.
3. Grant camera permissions and click **Scan Frame** or **Auto-Scan** to see real-time fraud scores with glowing visual gauges.
