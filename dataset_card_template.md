# Xeno-Canto Audio Dataset

## Dataset Description

This dataset contains audio recordings from the Xeno-Canto database, a collection of bird vocalizations from around the world.

### Dataset Structure

Each example contains:
- `audio`: The audio file (WAV format)
- `file_name`: Original filename
- `xc_id`: Xeno-Canto recording ID

### Source

Audio files are sourced from [Xeno-Canto](https://xeno-canto.org/), a collaborative project dedicated to sharing bird sounds from around the world.

### Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("YOUR_USERNAME/YOUR_DATASET_NAME")

# Access an example
example = dataset["train"][0]
audio = example["audio"]
xc_id = example["xc_id"]

# Audio format
print(audio.keys())  # dict_keys(['path', 'array', 'sampling_rate'])
print(f"Sample rate: {audio['sampling_rate']}")
print(f"Audio array shape: {audio['array'].shape}")
```

### License

Please refer to the [Xeno-Canto terms of use](https://xeno-canto.org/about/terms) for licensing information on individual recordings.

### Citation

If you use recordings from Xeno-Canto, please cite:
```
Xeno-canto: Bird sounds from around the world. Available at www.xeno-canto.org
```
