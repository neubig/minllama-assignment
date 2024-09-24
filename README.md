# Min-Llama Assignment
by Vijay Viswanathan (based on the previous [minbert-assignment](https://github.com/neubig/minbert-assignment))

This is an exercise in developing a minimalist version of Llama2, part of Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2024/).

In this assignment, you will implement some important components of the Llama2 model to better understanding its architecture. 
You will then perform sentence classification on ``sst`` dataset and ``cfimdb`` dataset with this model.

## Assignment Details

### Your task
The code to implement can be found in `llama.py`, `classifier.py` and `optimizer.py`. You are reponsible for writing _core components_ of Llama2 (one of the leading open source language models). In doing so, you will gain a strong understanding of neural language modeling. We will load pretrained weights for your language model from `stories42M.pt`; an 8-layer, 42M parameter language model pretrained on the [TinyStories](https://arxiv.org/abs/2305.07759) dataset (a dataset of machine-generated children's stories). This model is small enough that it can be trained (slowly) without a GPU. You are encouraged to use Colab or a personal GPU machine (e.g. a Macbook) to be able to iterate more quickly.

Once you have implemented these components, you will test our your model in 3 settings:
1) Generate a text completion (starting with the sentence `"I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is"`). You should see coherent, grammatical English being generated (though the content and topicality of the completion may be absurd, since this LM was pretrained exclusively on children's stories).
2) Perform zero-shot, prompt-based sentiment analysis on two datasets (SST-5 and CFIMDB). This will give bad results (roughly equal to choosing a random target class).
3) Perform task-specific finetuning of your Llama2 model, after implementing a classification head in `classifier.py`. This will give much stronger classification results.
4) If you've done #1-3 well, you will get an A! However, since you've come this far, try implementing something new on top of your hand-written language modeling system! If your method provides strong empirical improvements or demonstrates exceptional creativity, you'll get an A+ on this assignment.

### Important Notes
* Follow `setup.sh` to properly setup the environment and install dependencies.
* There is a detailed description of the code structure in [structure.md](./structure.md), including a description of which parts you will need to implement.
* You are only allowed to use libraries that are installed by `setup.sh`, no other external libraries are allowed (e.g., `transformers`).
* The `data/cfimdb-test.txt` file provided to you does **not** contain gold-labels, and contains a placeholder negative (-1) label. Evaluating your code against this set will show lower accuracies so do not worry if the numbers don't make sense.
* We will run your code with commands below (under "Reference outputs/accuracies"), so make sure that whatever your best results are reproducible using these commands.
    * Do not change any of the existing command options (including defaults) or add any new required parameters

## Reference outputs/accuracies: 

*Text Continuation* (`python run_llama.py --option generate`)
You should see continuations of the sentence `I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is...`. We will generate two continuations - one with temperature 0.0 (which should have a reasonably coherent, if unusual, completion) and one with temperature 1.0 (which is likely to be logically inconsistent and may contain some coherence or grammar errors).

*Zero Shot Prompting*
Zero-Shot Prompting for SST:

`python run_llama.py --option prompt --batch_size 10  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-prompting-output.txt --test_out sst-test-prompting-output.txt [--use_gpu]`

Prompting for SST:
Dev Accuracy: 0.213 (0.000)
Test Accuracy: 0.224 (0.000)

Zero-Shot Prompting for CFIMDB:

`python run_llama.py --option prompt --batch_size 10  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-prompting-output.txt --test_out cfimdb-test-prompting-output.txt [--use_gpu]`

Prompting for CFIMDB:
Dev Accuracy: 0.498 (0.000)
Test Accuracy: -

*Classification Finetuning*

`python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 80  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-finetuning-output.txt --test_out sst-test-finetuning-output.txt [--use_gpu]`

Finetuning for SST:
Dev Accuracy: 0.414 (0.014)
Test Accuracy: 0.418 (0.017)

`python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 10  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-finetuning-output.txt --test_out cfimdb-test-finetuning-output.txt [--use_gpu]`

Finetuning for CFIMDB:
Dev Accuracy: 0.800 (0.115)
Test Accuracy: -

Mean reference accuracies over 10 random seeds with their standard deviation shown in brackets.

### Submission

**Code:**
You will submit a full code package, with output files, on **Canvas**. This package will be checked by the TAs in the 1-2 weeks 
   after the assignment for its correctness and executability.

**Report (optional):** Your zip file can include a pdf file, named ANDREWID-report.pdf, if (1) you've implemented something else on top of the requirements and further improved accuracy for possible extra points (see "Grading" below), and/or (2) if your best results are with some hyperparameters other than the default, and you want to specify how we should run your code. If you're doing (1), we expect your report should be 1-2 pages, but no more than 3 pages. If you're doing (2), the report can be very brief.

#### Canvas Submission

For submission via [Canvas](https://canvas.cmu.edu/),
the submission file should be a zip file with the following structure (assuming the
lowercase Andrew ID is ``ANDREWID``):
```
ANDREWID/
├── run_llama.py
├── base_llama.py
├── llama.py
├── rope.py
├── classifier.py
├── config.py
├── optimizer.py
├── sanity_check.py
├── tokenizer.py
├── utils.py
├── README.md
├── structure.md
├── sanity_check.data
├── generated-sentence-temp-0.txt
├── generated-sentence-temp-1.txt
├── [OPTIONAL] sst-dev-advanced-output.txt
├── [OPTIONAL] sst-test-advanced-output.txt
├── sst-dev-prompting-output.txt
├── sst-test-prompting-output.txt
├── sst-dev-finetuning-output.txt
├── sst-test-finetuning-output.txt
├── [OPTIONAL] cfimdb-dev-advanced-output.txt
├── [OPTIONAL] cfimdb-test-advanced-output.txt
├── cfimdb-dev-prompting-output.txt
├── cfimdb-test-prompting-output.txt
├── cfimdb-dev-finetuning-output.txt
├── cfimdb-test-finetuning-output.txt
└── setup.sh
```

`prepare_submit.py` can help to create(1) or check(2) the to-be-submitted zip file. It
will throw assertion errors if the format is not expected, and *submissions that fail
this check will be graded down*.

Usage:
1. To create and check a zip file with your outputs, run
   `python3 prepare_submit.py path/to/your/output/dir ANDREWID`
2. To check your zip file, run
   `python3 prepare_submit.py path/to/your/submit/zip/file.zip ANDREWID`

Please double check this before you submit to Canvas; most recently we had about 10/100
students lose a 1/3 letter grade because of an improper submission format.


### Grading
* A+: (Advanced implementation) You additionally implement something else on top of the requirements for A, and achieve significant accuracy improvements or demonstrate exceptional creativity. This improvement can be in either the zero-shot setting (no task-specific finetuning required) or in the finetuning setting (improving over our current finetuning implementation). Please write down the things you implemented and experiments you performed in the report. You are also welcome to provide additional materials such as commands to run your code in a script and training logs.
    * perform [continued pre-training](https://arxiv.org/abs/2004.10964) using the language modeling objective to do domain adaptation
    * enable zero-shot prompting using a more principled inference algorithm than our current implementation. For example, we did not include an attention mask despite right-padding all inputs (to enable batch prediction); this could be improved.
    * perform [prompt-based finetuning](https://arxiv.org/abs/2109.01247)
    * add [regularization](https://arxiv.org/abs/1909.11299) to our finetuning process
    * try parameter-efficient finetuning (see Section 2.2 [here](https://arxiv.org/abs/2110.04366) for an overview)
    * try alternative fine-tuning algorithms e.g. [SMART](https://www.aclweb.org/anthology/2020.acl-main.197) or [WiSE-FT](https://arxiv.org/abs/2109.01903)
    * add other model components on top of the model
* A: You implement all the missing pieces and the original ``classifier.py`` with ``--option prompt`` and ``--option finetune`` code such that coherent text (i.e. mostly grammatically well-formed) can be generated and the model achieves comparable accuracy (within 0.05 accuracy for SST or 0.15 accuracy for CFIMDB) to our reference implementation.
* A-: You implement all the missing pieces and the original ``classifier.py`` with ``--option prompt`` and ``--option finetune`` code but coherent text is not generated (i.e. generated text is not well-formed English) or accuracy is not comparable to the reference (accuracy is more than 0.05 accuracy or 0.15 accuracy from our reference scores, for for SST and CFIMDB, respectively).
* B+: All missing pieces are implemented and pass tests in ``sanity_check.py`` (llama implementation) and ``optimizer_test.py`` (optimizer implementation)
* B or below: Some parts of the missing pieces are not implemented.

If your results can be confirmed through the submitted files, but there are problems with your
code submitted through Canvas, such as not being properly formatted, not executing in
the appropriate amount of time, etc., you will be graded down 1/3 grade (e.g. A+ -> A or A- -> B+).

All assignments must be done individually and we will be running plagiarism detection
on your code. If we confirm that any code was plagiarized from that of other students
in the class, you will be subject to strict measure according to CMUs academic integrity
policy. That being said, *you are free to use publicly available resources* (e.g. papers or open-source
code), but you ***must provide proper attribution***.

### Acknowledgement
This code is based on llama2.c by Andrej Karpathy. Parts of the code are also from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
