# openft
A helper framework for fine tuning with OpenAI

This library is a simple wrapper around the OpenAI python SDK, with a focus on making the act
of fine tuning on their platform smooth. I've found this useful for a few different projects.

## Usage
It expects at least 3 text files:
1. prompt.txt - containing the System Prompt
2. questions.txt - containing the example questions in order
3. answers.txt - containing the example questions in order

And that's pretty much it! There are a few extra configurable
options which can be supplied at initilisation but the library
takes care of everything else:

1. Load all of the above text files from disk
2. Create the dataset in the format OpenAI expects
3. Run some checks (check message token sizes are less than the maximum, estimate costs of training)
4. Upload the dataset to OpenAI as a .jsonl file and wait for it to be processed
5. Creates a fine tuning job with that dataset
6. Polls to check the job's status during training
7. Wait for it to succeed or error and return useful information
8. Fetch the results file and write locally to disk

From there, it's as simple as
```python
    from openft import OpenFT

    conf = {
        # directory containing the 3 files
        "training_dir": "training_data/", 
        "base_model_name": "gpt-3.5-turbo",
        "num_epochs": 3,
        "fine_tune_suffix": "my-fine-tuned-model"
    }
    ft = OpenFT(conf)
    result_file_paths = ft.launch_fine_tune()
```
