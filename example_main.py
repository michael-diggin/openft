from openft import OpenFT

if __name__ == '__main__':
    conf = {
        "training_dir": "training_data/",
    }

    ft = OpenFT(conf)
    result_file_paths = ft.launch_fine_tune()