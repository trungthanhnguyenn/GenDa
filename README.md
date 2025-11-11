# GenDa

GenDa is an open-source project for generating synthetic data and automating data processing, especially for AI and NLP applications.

## Features
- Generate synthetic data using flexible AI pipelines
- Integrate with HuggingFace Datasets
- Easily extensible with new pipelines and models
- Chunked data saving and HuggingFace Hub push support

## Requirements
- Python >= 10

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/trungthanhnguyenn/GenDa.git
   cd GenDa
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Clone submodule llama.cpp and build cpu version
   ```bash
   git clone https://github.com/ggml-org/llama.cpp.git
   cd llama.cpp
   ```

   - # CPU BUILD USING CMAKE #

   ```bash
   cmake -B build
   cmake --build build --config Release   
   ```

## Usage
Example to generate synthetic data:
```bash
python -m examples.make_synthetic_data
```
You can edit `examples/make_synthetic_data.py` to change the model, dataset, or pipeline parameters.

## Folder Structure
```
GenDa/
├── src/                  # Main source code (pipeline, model, ...)
├── examples/             # Usage examples
├── data_gen/             # Generated synthetic data
├── requirements.txt      # Python dependencies
├── README.md             # This documentation
└── ...
```

## Contribution
Contributions are welcome! Please create a pull request or open an issue for ideas or bug reports.

## Contact
- Author: [trungthanhnguyenn]
- Email: [sktkctman2@gmail.com]
- Github: [github.com/trungthanhnguyenn]
