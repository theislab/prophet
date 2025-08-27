# Prophet Installation

## Install

```bash
pip install prophet
```

**Complete functionality included:**
- Prophet model (training, inference, fine-tuning)
- Data processing utilities
- PyTorch & Lightning
- Visualization tools (Matplotlib, Seaborn, Plotly)
- Jupyter notebook support
- Experiment tracking (W&B, TensorBoard)

## Usage

```python
from prophet import Prophet

# Train
model = Prophet(
    iv_emb_path="interventions.csv",
    cl_emb_path="cell_lines.csv",
    model_pth="checkpoint.ckpt"
)
model.train(df, iv_col="drug", cl_col="cell_line", readout_col="response")

# Predict
predictions = model.predict(target_ivs=["DRUG1"], target_cls=["CELL1"], save=False)
```

## Verification

```python
import prophet
from prophet import Prophet
print("✅ Prophet ready!")
```

## Help

- **Issues**: https://github.com/theislab/prophet/issues
- **Tutorials**: See `tutorials/` directory
