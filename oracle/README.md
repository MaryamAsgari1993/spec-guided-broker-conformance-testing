# Oracle weights

These files are the reusable conformance oracle from the QRS paper. They are **not** retrained here.

| File | Task | Architecture |
|------|------|----------------|
| `best_cnn_binary_model.pth` | Binary conformance (compliant vs. non-compliant) | 1D CNN |
| `best_gru_multiclass_model.pth` | Rule-class attribution | Bidirectional GRU |
| `selected_hyperparameters.json` | Nested-CV summary from QRS (reference) | — |

Inference hyperparameters used in this repository match the evaluation scripts:

- Binary CNN: filters `[64, 128, 256]`, kernel 3, global max pooling, `Lmax = 100`
- Multi-class GRU: hidden size 128, 2 layers, bidirectional, `Lmax = 100`
