# BERT Optimizer Comparison

I wanted to see if AdamW is really the best choice for fine-tuning BERT, so I ran an experiment comparing different optimizers. 
## Experiment Setup

I tested 4 optimizers on BERT fine-tuning using the GLUE benchmark (MRPC and SST-2):

* **AdamW** – the standard choice for BERT fine-tuning
* **LAMB** – designed for large-batch training
* **SGD with warmup** – classic optimizer with a warmup schedule
* **AdaFactor** – memory-efficient alternative to AdamW

### Why MRPC and SST-2?

* **MRPC** (paraphrase detection) is small and noisy → a good test of convergence
* **SST-2** (sentiment classification) is larger and cleaner → good for generalization

## Results (Summary)

```
AdamW:     88.15% accuracy (winner, no surprise)
AdaFactor: 76.69% accuracy (solid, very memory efficient)
SGD:       61.71% accuracy (fast but underperforms)
LAMB:      60.82% accuracy (struggled on small datasets)
```

## How to Reproduce

### Install dependencies

```bash
pip install transformers datasets wandb torch-optimizer scikit-learn matplotlib seaborn
```

### (Optional) Configure Weights & Biases tracking

```bash
wandb login
```

### Run the experiment

```bash
python main.py
```

### Run a single optimizer

```python
from optimizer_experiment import OptimizationExperiment

experiment = OptimizationExperiment()
datasets = experiment.load_datasets()
prepared = experiment.prepare_datasets(datasets)

result = experiment.train_with_optimizer(
    dataset_name="mrpc",
    dataset=prepared["mrpc"],
    optimizer_name="AdamW"
)
```

## Key Takeaways

* **AdamW is still the best overall** – reliable, consistent, strong performance.
* **AdaFactor is a good option if GPU memory is limited** – about 37% less memory usage, with only moderate performance loss.
* **LAMB didn’t perform well on small datasets** – may require tuning (different LR/warmup) or larger-scale tasks.
* **SGD is very fast but accuracy suffers** – not recommended unless speed is the top priority.

## Memory Usage

```
AdaFactor: 1.37 GB (most efficient)
SGD:       1.78 GB
AdamW:     2.19 GB
LAMB:      2.19 GB
```

If you’re running on a smaller GPU, AdaFactor is worth considering.

## Limitations

* Only tested on **2 GLUE datasets** (MRPC is very small, not representative)
* No hyperparameter tuning per optimizer (same setup for fairness, but possibly suboptimal)
* LAMB likely needs different learning rates to shine
* More optimizers (RMSprop, Adagrad, etc.) could be added

## Additional Observations

* SST-2 was less sensitive to optimizer choice than MRPC
* AdamW showed the smoothest convergence
* Training times were similar (SGD was noticeably faster)
* GPU memory differences were more pronounced than expected

## Project Structure

* `OptimizationExperiment` class orchestrates training and evaluation
* All optimizers tested under identical conditions
* Memory usage tracked for each run
* Results automatically saved as CSV and plots

### Generated files

* `bert_optimizer_comparison_results.csv` – raw numbers
* `optimizer_comparison_results.png` – accuracy charts
* W\&B logs (if enabled)

## Dependencies

* [transformers](https://github.com/huggingface/transformers)
* [datasets](https://github.com/huggingface/datasets)
* [torch](https://pytorch.org/)
* [wandb](https://wandb.ai/) (optional)
* [scikit-learn](https://scikit-learn.org/)
* [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/)
* [torch-optimizer](https://github.com/jettify/pytorch-optimizer)

## Contributing

Ideas for improvement:

* Test on more GLUE tasks
* Tune learning rates per optimizer
* Try other transformer models (RoBERTa, DistilBERT, etc.)
* Add additional optimizers
* Improve hyperparameter search



## License

MIT 

---


