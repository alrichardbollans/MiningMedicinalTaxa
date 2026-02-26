### Costs

For a single hparam run (01/04/25):

- Claude - \$0.59
- LLama via Fireworks - \$0.49
- gpt via openai - \$0.45
- deepseek via deepseek api - \$0.07

For evaluation (approx 9 papers), these were the following incurred costs:

- Claude - \$ 5.14
- LLama via Fireworks - \$5.03
- gpt via openai - \$3.59
- deepseek via deepseek api - \$0.22

Note that these are only approximate and extra costs may have been incurred from retrying chunks by splitting multiple times.

For finetuning gpt4o - \$ 12.06
("hyperparameters": {
"n_epochs": 4,
"batch_size": 1,
"learning_rate_multiplier": 2
}:)

- Evaluation of gpt4o_FT - \$5.33
- Deployment of gpt4o_FT on five papers - \$2.86

104 is a big chunk!
