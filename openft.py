import os
import io
import time
import openai
from openai.types import FileObject
from openai.types.fine_tuning.fine_tuning_job import FineTuningJob, Hyperparameters
import tiktoken
from data import load_from_file, create_single_ft_message, write_dataset_to_jsonl, write_dataset_to_buffer
from utils import check_all_examples_are_bounded, calc_total_tokens, calc_cost_of_training

class OpenFT:
    def __init__(self, config: dict):
        self._setup_with_config(config)
        # expect the API key to be present as OPENAI_API_KEY env variable
        # and the organisation ID present as OPENAI_ORG_ID
        self.client = openai.OpenAI(api_key = self._api_key, organization=self._org_id)


    def _setup_with_config(self, config: dict):
        self.training_dir = config.get("training_dir", "training_data/")
        self.base_model = config.get("base_model_name", "gpt-3.5-turbo")
        self.max_token_size = config.get("max_token_size", 4096)
        self.num_epochs = config.get("num_epochs", 1)
        self.batch_size = config.get("batch_size", 0)
        self.data_split = config.get("data_split", "\n\n")
        self._api_key = config.get("openai_api_key", None)
        self._org_id = config.get("openai_org_id", None)
        self.encoding = config.get("encoding", "cl100k_base")
        self.ft_suffix = config.get("fine_tune_suffix", None)
        self.poll_wait = config.get("poll_wait", 30)
        self.token_cost = config.get("training_token_cost", 0.008)
    
    def create_training_dataset(self) -> list[dict]:
        """
        create_training_dataset creates a dataset from the given question/answer text files
        in the format that OpenAI expects.
        """
        sys_prompts = load_from_file(os.path.join(self.training_dir, "system_prompt.txt"), None)
        questions = load_from_file(os.path.join(self.training_dir, "questions.txt"), self.data_split)
        answers = load_from_file(os.path.join(self.training_dir, "answers.txt"), self.data_split)

        assert len(sys_prompts) == 1, f"Found {len(sys_prompts)} number of system prompts"
        sys_prompt = sys_prompts[0]
        assert len(questions) == len(answers), f"Found a different number of questions ({len(questions)}) to answers ({len(answers)})"

        dataset = []
        for q, a in zip(questions, answers):
            dataset.append(create_single_ft_message(sys_prompt, q, a))

    def launch_fine_tune(self, validate=True, user_prompt=False, dataset_name="dataset.jsonl", write_to_disk=False, output_dir="") -> list[str]:
        """
        launch_fine_tune runs the pipeline of
        1. creating the datatset
        2. validating the dataset
        3. uploading the file(s) to OpenAI
        4. creating a fine tune job
        5. watching the fine tune job
        6. processing a succesful job
        7. fetches the results files and writes to disk
        7. returns the file paths of the results files
        """
        dataset = self.create_training_dataset()

        if validate:
            # validate that all examples are within the bound
            enc = tiktoken.get_encoding(self.encoding)
            flag, _ = check_all_examples_are_bounded(dataset, enc, self.max_token_size)
            assert flag, f"Not all examples are within the {self.max_token_size} number of tokens limit"

            total = calc_total_tokens(dataset, enc)
            cost = calc_cost_of_training(dataset, enc, self.num_epochs, self.token_cost)
            print(f"There are {total} tokens in all {len(dataset)} examples")
            print(f"This is esitimated to cost ${cost:.2f} for {self.num_epochs} epoch of training")

        if write_to_disk:
            out_path = os.path.join(self.training_dir, dataset_name)
            write_dataset_to_jsonl(dataset, out_path)
            print(f"Wrote dataset locally to {out_path}")
            file_content = open(out_path, 'rb')
        else:
            data_buffer = write_dataset_to_buffer(dataset)
            file_content = (dataset_name, data_buffer)

        if user_prompt:
            user_input = input("Are you happy to proceed? (yes/no): ")
            if user_input.lower() not in ["y", "yes"]:
                print("Exiting...")
                return []
        
        print("Uploading dataset to OpenAI...")
        file_object = self._upload_file(file_content)
        print(f"File created with ID: {file_object.id}")
        print(f"File can be viewed at https://platform.openai.com/storage/files/{file_object.id}")
        file_id = file_object.id
        file_object = self.client.files.wait_for_processing(id=file_id)
        assert file_object.status == "processed", "File was not processed correctly"

        print("Creating fine tuning job...")
        ft_job = self._create_fine_tune(file_id)
        print(f"Created fine tune job: {ft_job.id}")
        print(f"Fine tuning job ca be viewed at https://platform.openai.com/finetune/{ft_job.id}")

        while True:
            time.sleep(self.poll_wait)
            ongoing_job = self._fetch_ft_job(ft_job.id)
            match ongoing_job.status:
                case "succeeded":
                    print("Fine Tuning Job succeeded")
                    break
                case "failed":
                    print("Fine Tuning Job failed")
                    error_state = ongoing_job.error
                    print(f"Error status: {error_state.code}; {error_state.message}")
                    return []
                case "cancelled":
                    print(f"Fine Tuning Job was cancelled, exiting")
                    return []
                case _:
                    print(f"Fine Tuning Job status: {ongoing_job.status}")

        # handle post processing for succeeded FT Jobs
        results_files = ongoing_job.result_files
        billable_tokens = ongoing_job.trained_tokens
        result_model = ongoing_job.fine_tuned_model
        print("\n")
        print(f"Fine Tuned Model is {result_model}")
        cost = billable_tokens * self.token_cost / 1000.
        print(f"Total cost of training is ${cost} with {billable_tokens} tokens")

        result_files_locations = self.fetch_results_files(results_files, output_dir)
        return result_files_locations

    def fetch_results_files(self, result_files: list[str], output_dir: str = "") -> list[str]:
        """
        fetch_results_files fetches the files from OpenAI and writes them
        to the given directory, returning the file paths
        """
        output = []
        for rf in result_files:
            output.append(self._fetch_and_write_file(rf, output_dir))
        return output
    
    def _fetch_and_write_file(self, result_file: str, output_dir: str = ""):
        file = self.client.files.retrieve(file_id=result_file)
        content = self.client.files.content(file_id=result_file)
        fp = os.path.join(output_dir, file.filename)
        content.write_to_file(fp)
        return fp

    def _upload_file(self, file_content: io.BufferedIOBase) -> FileObject:
        file = self.client.files.create(
            file=file_content,
            purpose='fine-tune'
        )
        return file
    
    def _create_fine_tune(self, file_id: str) -> FineTuningJob:
        hyper_params = Hyperparameters(n_epochs=self. num_epochs)
        if self.batch_size > 0:
            hyper_params.batch_size
        return self.client.fine_tuning.jobs.create(
            training_file=file_id,
            model=self.base_model,
            suffix=self.ft_suffix,
            hyperparameters=hyper_params
        )
    
    def _fetch_ft_job(self, ft_id: str) -> FineTuningJob:
        return self.client.fine_tuning.jobs.retrieve(
            fine_tuning_job_id=ft_id
        )
