# openft
A helper framework for fine tuning with OpenAI


- /training_data/questions.txt
- /training_data/answers.txt
- /training_data/prompt.py
- Given config file?
- Given model name, (max tokens default = 4096)
- Num epochs, batch sizes etc (defaults)

- Load all of the above from disk (given a dir - default training_dat)
- Create dataset.jsonl file
- Run utils (check message token sizes, estimate costs)
- Write file to disk or to a buffer (buffer only but also configurable? io.StringIO)
- Assert it was created correctly and return info about it
- Start Fine tuning job with the details
- Poll it to check details and return if it's successful or errored
- If successful or errored, return all the relevant details
- Fetch the results file and process locally

Extras:
- dry-run with user prompt?
- add org ID
- print out links to file/ft-job
- optional validation data
- more than just QA datasets