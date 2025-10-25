# Training NicheFlow on New Datasets

To train **NicheFlow** on a new dataset, you need to follow three main steps: dataset preprocessing, classifier training, and NicheFlow training.

---

## 1. Dataset Preprocessing

1. Load your dataset as an **Annotated Data object (`AnnData`)**.
2. Perform your preprocessing with **Scanpy**:
   - Apply **PCA** before using our preprocessor (**required**).
3. Specify the following parameters for the preprocessor:
   - **Timepoint column** - column specifying the temporal information.
   - **Cell type column** - column specifying cell types.
   - **Temporally ordered timepoints** - ensure your timepoints are in the correct order.  
     - *Note:* Sometimes timepoints are not trivially sortable (e.g., axolotl dataset). Refer to the example notebook.
   - **Coordinate standardization** - we recommend standardizing over min-max scaling.
   - **Radius** - defines the size of microenvironments.
   - **`dx` and `dy`** - discretization steps for generating the grid of **test microenvironments**.
   - **Device** - set to `cuda` (recommended) to speed up preprocessing.
   - **Chunk size** - controls memory usage. Reduce it if you encounter errors.
4. Preprocess the dataset and save it in the [`data/`](../data/) folder

**Recommendation:**  
See [`download_and_preprocess.ipynb`](../notebooks/download_and_preprocess.ipynb) for a complete example of how we downloaded and preprocessed the datasets used in our paper.

---

## 2. Classifier Training

Before training NicheFlow, you need a trained classifier for your dataset. This requires creating a **new datamodule configuration** and **experiment configuration**.

### Datamodule Config
- Define `ct_{dataset_name}` in [`configs/data/`](../configs/data/).
- Specify:
  - Number of PCA components.
  - Number of cell type classes.

### Experiment Config
- Define `experiment/classifier/{dataset_name}` in [`configs/experiment/classifier`](../configs/experiment/classifier/).
- Set:
  - The correct data override (pointing to your new datamodule).
  - A unique WandB run name.

Then, train the classifier:
```bash
python nicheflow/train.py experiment=classifier/{dataset_name}
```

## 3. NicheFlow Training

Once the classifier is trained, you can train NicheFlow.

### Datamodule Config

- Define `nicheflow_{dataset_name}` in [`configs/data/`](../configs/data/).
- Specify:
  - Dataset filepath.
  - Number of PCA components.
  - Number of cell type classes.
  - Number of timepoints/slices.
  - Path to the **trained classifier checkpoint**.

### Experiment Config

- Define `experiment/nicheflow/{cfm | gvfm | glvfm}/{dataset_name}` in [`configs/experiment/nicheflow`](../configs/experiment/nicheflow/).
- Set:
  - The correct data override.
  - A unique WandB run name.

Then, train NicheFlow:
```bash
python nicheflow/train.py experiment=nicheflow/{cfm|gvfm|glvfm}/{dataset_name}
```
