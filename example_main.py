from openft import OpenFT

if __name__ == '__main__':
    conf = {
        "training_dir": "training_data/",
        "base_model_name": "gpt-3.5-turbo",
        "num_epochs": 5,
        "with_validation": False,
        "fine_tune_suffix": "example-ft", 
    }

    ft = OpenFT(conf)
    result_file_paths = ft.launch_fine_tune()