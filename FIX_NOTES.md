# SwiftChain Fix Notes

## What changed

- The prediction path now calls the saved Gradient Boosting classifier and scaler.
- The previous hand-written shipping-mode rules have been removed from the prediction path.
- The app reconstructs the exact 309-column model schema using the artifact metadata.
- The 15 numeric columns are scaled using `swiftchain_scaler.pkl`.
- Model classes are mapped explicitly: `-1 = Late`, `0 = On-Time`, `1 = Early`.
- Per-class probabilities and model confidence are displayed.
- Model and scaler paths are resolved relative to `app.py`.
- Artifact schema validation runs at startup.
- The dependency versions are bounded for saved-artifact compatibility.

## Important inference limitation

The current user interface exposes six fields. The original training data contains 41 raw fields. The app fills uncollected numeric features with the scaler training means and uses reference categories for uncollected categorical features. This makes the saved model run correctly, but it is a partial-information prediction.

For a production deployment, the next upgrade should be a complete preprocessing pipeline that accepts all required raw order fields, or a complete order-record upload/API payload.

## Upload instructions

Upload the contents of this folder to the GitHub repository root. Streamlit Community Cloud should use:

```text
app.py
requirements.txt
swiftchain_delay_predictor.pkl
swiftchain_scaler.pkl
```

The `.pkl` files must remain beside `app.py`.
