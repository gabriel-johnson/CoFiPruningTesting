import logging
import os
import sys
import time
import random
from copy import deepcopy

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset, load_metric, DatasetDict, interleave_datasets, Features, Value, Sequence, ClassLabel
from transformers import AutoConfig, AutoTokenizer, EvalPrediction, default_data_collator, DataCollatorWithPadding
from transformers import (HfArgumentParser, TrainingArguments, PretrainedConfig,
                          glue_output_modes, set_seed)

from args import AdditionalArguments, DataTrainingArguments
from utils.cofi_utils import *
from models.l0_module import L0Module
from models.modeling_bert import CoFiBertForSequenceClassification, CoFiTeacherBertForSequenceClassification
from models.modeling_roberta import CoFiRobertaForSequenceClassification
from trainer.trainer import CoFiTrainer 
from utils.utils import *
from models.model_args import ModelArguments

import wandb

# maps labels for different tasks
import data_formater

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

task_to_teacher_models = {
    "cola" : "textattack/bert-base-uncased-CoLA",
    "mnli" : "textattack/bert-base-uncased-MNLI",
    "mrpc" : "textattack/bert-base-uncased-MRPC",
    "qnli" : "textattack/bert-base-uncased-QNLI",
    "qqp" :  "textattack/bert-base-uncased-QQP",
    "rte" :  "textattack/bert-base-uncased-RTE",
    "sst2" : "textattack/bert-base-uncased-SST-2",
    "stsb" : "textattack/bert-base-uncased-STS-B",
    "wnli" : "textattack/bert-base-uncased-WNLI",
}

glue_tasks_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst2": 2,
    "stsb": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, AdditionalArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, additional_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, additional_args = parser.parse_args_into_dataclasses()
    

    os.makedirs(training_args.output_dir, exist_ok=True)

     # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    

    # save args
    torch.save(data_args, os.path.join(
        training_args.output_dir, "data_args.bin"))
    torch.save(model_args, os.path.join(
        training_args.output_dir, "model_args.bin"))
    torch.save(additional_args, os.path.join(
        training_args.output_dir, "additional_args.bin"))

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # print all arguments
    log_all_parameters(logger, model_args, data_args,
                       training_args, additional_args)

    multiple_datasets = False

    def load_data():
        if data_args.task_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                "./glue.py", data_args.task_name.replace("-", ""), cache_dir=model_args.cache_dir)
            t_name = data_args.task_name
        elif data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
            )
            t_name = data_args.dataset_name
        else:
            # Loading a dataset from your local files.
            # CSV/JSON training and evaluation files are needed.
            t_name = data_args.t_name
            data_files = {"train": data_args.train_file,
                        "validation": data_args.validation_file}

            # Get the test dataset: you can provide your own CSV/JSON test file (see below)
            # when you use `do_predict` without specifying a GLUE benchmark task.
            if training_args.do_predict:
                if data_args.test_file is not None:
                    train_extension = data_args.train_file.split(".")[-1]
                    test_extension = data_args.test_file.split(".")[-1]
                    assert (
                        test_extension == train_extension
                    ), "`test_file` should have the same extension (csv or json) as `train_file`."
                    data_files["test"] = data_args.test_file
                else:
                    raise ValueError(
                        "Need either a GLUE task or a test file for `do_predict`.")

            for key in data_files.keys():
                logger.info(f"load a local file for {key}: {data_files[key]}")

            if data_args.train_file.endswith(".csv"):
                # Loading a dataset from local csv files
                raw_datasets = load_dataset(
                    "csv", data_files=data_files, cache_dir=model_args.cache_dir)
            elif data_args.train_file.endswith(".tsv"):
                dataset_dict = {}
                for key in data_files:
                    dataset_dict[key] = load_from_tsv(data_files[key])
                raw_datasets = DatasetDict(dataset_dict)
            else:
                # Loading a dataset from local json files
                raw_datasets = load_dataset(
                    "json", data_files=data_files, cache_dir=model_args.cache_dir)
        
        return raw_datasets
        # See more about loading any type of standard or custom dataset at
        # https://huggingface.co/docs/datasets/loading_datasets.html.

        # Labels
    

    if data_args.task_list is not None:
        raw_datasets = []

        multiple_datasets = True
        for index, task in enumerate(data_args.task_list):
            data_args.task_name = task
            raw_datasets.append(load_data())
        data_args.task_name = None
        t_name = None

    
    else:
        raw_datasets = load_data()
        data_args.task_list = None


    Model = CoFiBertForSequenceClassification if model_args.model_name_or_path.startswith(
        "bert") else CoFiRobertaForSequenceClassification

    glue_token_list = {"additional_special_tokens": [f"<{task}>" for task in data_args.task_list]}



    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        max_length=56,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer.add_special_tokens(glue_token_list)


    def create_dataset(task, dataset, label_count):
            train_dataset = data_formater.map_labels(task, dataset["train"], label_count)

            test_dataset = data_formater.map_labels(task, dataset["test"], label_count)
            print("\n\neval_dataset before map labels: ", dataset["train"][500], "\n\n")

            val_dataset = data_formater.map_labels(task, dataset["validation"],label_count )
            print("\n\ntrain_dataset after map labels: ", train_dataset[500], "\n\n")


            return (DatasetDict({
                'train': train_dataset,
                'test': test_dataset, 
                'validation': val_dataset,
            }))

    label_count = 0

   

    if data_args.task_list is not None:
        combined_raw_datasets = []    
        label_list = ["negative", "positive"]


        for (index, dataset) in enumerate(raw_datasets):
            combined_raw_datasets.append(create_dataset(data_args.task_list[index], dataset, label_count))
            # label_count = label_count + glue_tasks_num_labels[data_args.task_list[index]]
        # label_list = []
    
        # is_regression = []
        # num_labels = 0
        # for index, task in enumerate(data_args.task_list):
        #     is_regression.append(data_args.task_name == "stsb")
        #     if not is_regression[index]:
        #         for label in raw_datasets[index]["train"].features["label"].names:
        #             label_list.append(label)
        #             num_labels = num_labels + 1
        #     else:
        #         raw_datasets[index]["num_labels"] = 1
       

    elif data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in [
            "float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    is_regression = False

    num_labels = 2#len(label_list)

    label_map = {i: label for i, label in enumerate(label_list)}
    label_to_id = {label: i for i, label in enumerate(label_list)}



    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=2,
        finetuning_task=data_args.task_name if data_args.task_name is not None else data_args.task_list,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        id2label=label_map,
        label2id=label_to_id,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    config.do_layer_distill = additional_args.do_layer_distill #! True
    # set up configuration for distillation
    if additional_args.do_distill:
        config.output_attentions = True
        config.output_hidden_states = True

    
    
    # return
    TeacherModel = CoFiTeacherBertForSequenceClassification
    if additional_args.do_distill:
        if data_args.task_list is not None:
            teacher_model = []
            for i, task in enumerate(data_args.task_list):
                print(f"getting the following teaching model: {task_to_teacher_models[task]}")
                teacher_model.append(
                    TeacherModel.from_pretrained(task_to_teacher_models[task], config=deepcopy(config), task = task)
                )
                teacher_model[i].resize_token_embeddings(len(tokenizer))
                teacher_model[i].eval()

        else:
            teacher_model = Model.from_pretrained(
                additional_args.distillation_path,
                config=deepcopy(config)
            )
            teacher_model.eval()



    model = Model.from_pretrained(
            model_args.model_name_or_path,
            tokenizer,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            
        ) #! inside the function, we get the original struct  #! CofiBertForSequenceClassification

    model.resize_token_embeddings(len(tokenizer))


    for i, teacher in enumerate(teacher_model):
        task_name = f"task{i + 1}_classifier"
        getattr(model, task_name).weight.data = teacher.classifier.weight.data.clone()
        getattr(model, task_name).bias.data = teacher.classifier.bias.data.clone()


    # initialize the layer transformation matrix to be an identity matrix
    if additional_args.do_layer_distill:
        initialize_layer_transformation(model)

    logger.info(model)
    logger.info(f"Model size: {calculate_parameters(model)}")

    zs = None
    
    if additional_args.pretrained_pruned_model is not None:
        zs = load_zs(additional_args.pretrained_pruned_model)
        model = load_model(additional_args.pretrained_pruned_model, Model, zs)
        print(
            f"Model Size after pruning: {calculate_parameters(model)}")

    l0_module = None
    if additional_args.pruning_type is not None:
        l0_module = L0Module(config=config,
                             droprate_init=additional_args.droprate_init,
                             temperature=additional_args.temperature,
                             target_sparsity=additional_args.target_sparsity,
                             pruning_type=additional_args.pruning_type)

    if data_args.task_list is not None:
        sentences = []
        for index, task in enumerate(data_args.task_list):
            sentences.append(task_to_keys[task])
        
    
    elif data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None


    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    # label_to_id = label_to_id
    
    
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and (data_args.task_name is not None or data_args.task_list is not None)
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)


    for name, param in model.named_parameters():
        if("embeddings" in name):
            param.requres_grad = True
            continue
        if("classifier" in name):
            param.requires_grad = True
            continue
        param.requires_grad = False
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # return

    # for name, param in model.named_parameters():
    #     print(f"Name: {name}")
    #     print(f"Parameter: {param}")
    #     print(f"Parameter shape: {param.shape}")
    #     print("-" * 50)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    train_dataset_arr = []

    eval_dataset_arr = []
    

    if data_args.task_list is None:
        with training_args.main_process_first(desc="dataset map pre-processing"):
            raw_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            ) #! get dataset
        
        if training_args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets["train"]
            if data_args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(data_args.max_train_samples))

        if training_args.do_eval:
            if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
            if data_args.max_eval_samples is not None:
                eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

        if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
            if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                raise ValueError("--do_predict requires a test dataset")
            predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
            if data_args.max_predict_samples is not None:
                predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

        # Log a few random samples from the training set:
        if training_args.do_train:
            for index in random.sample(range(len(train_dataset)), 3):
                logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        # Get the metric function
        if data_args.task_name is not None:
            metric = load_metric("glue", data_args.task_name)
        else:
            metric = load_metric("accuracy")


    else:
        for (i, dataset) in enumerate(combined_raw_datasets):
            dataset = dataset.cast_column("label", ClassLabel(num_classes=2, names=["negative", "positive"]))
            sentence1_key, sentence2_key = task_to_keys[data_args.task_list[i]]
            with training_args.main_process_first(desc="dataset map pre-processing"):
                dataset = dataset.map(
                    preprocess_function,
                    batched=True,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                ) #! get dataset
            
            if training_args.do_train:
                if "train" not in dataset:
                    raise ValueError("--do_train requires a train dataset")
                train_dataset = dataset["train"]
                if data_args.max_train_samples is not None:
                    train_dataset = train_dataset.select(range(data_args.max_train_samples))

            if training_args.do_eval:
                if "validation" not in dataset and "validation_matched" not in dataset:
                    raise ValueError("--do_eval requires a validation dataset")
                eval_dataset = dataset["validation_matched" if data_args.task_name == "mnli" else "validation"]
                if data_args.max_eval_samples is not None:
                    eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

            if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
                if "test" not in dataset and "test_matched" not in dataset:
                    raise ValueError("--do_predict requires a test dataset")
                predict_dataset = dataset["test_matched" if data_args.task_name == "mnli" else "test"]
                if data_args.max_predict_samples is not None:
                    predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

            # Log a few random samples from the training set:
            if training_args.do_train:
                for index in random.sample(range(len(train_dataset)), 3):
                    logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
            
            train_dataset_arr.append(train_dataset.remove_columns("idx"))
            eval_dataset_arr.append(eval_dataset)



      

        eval_dataset = {l: eval_dataset_arr[i] for i, l in enumerate(data_args.task_list)}
        print("\n\n\n******eval_dataset******\n\n",eval_dataset, "\n\n\n")


    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.

    # print(f"grad: {model.classifier.weight.grad}")


    def compute_metrics(p: EvalPrediction, task=None):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        metric = load_metric("accuracy")
        result = metric.compute(predictions=preds, references=p.label_ids)
        # if len(result) > 1:
        #     result["combined_score"] = np.mean(list(result.values())).item()
        # return result
    
        # return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
        if data_args.task_name is not None:
            metric = load_metric("glue", data_args.task_name)
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif data_args.task_list is not None:
            metric = load_metric("glue", task)
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    

# Apply the new feature schema to both datasets
    

    # return
    logger.info(
        f"************* {len(train_dataset_arr)} Training Examples Loaded *************")
    logger.info(
        f"************* {len(eval_dataset)} Evaluation Examples Loaded *************")

    model.resize_token_embeddings(len(tokenizer))
    # teacher_model.resize_token_embeddings(len(tokenizer))

    # torch.set_printoptions(threshold=torch.inf)




    trainer = CoFiTrainer(
        model=model,
        args=training_args,
        additional_args=additional_args,
        train_dataset=train_dataset_arr if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        l0_module=l0_module,
        teacher_model=teacher_model,
        additional_train=data_args.additional_train,
    )

    
    # for param in model.parameters():
    #     # if name == "task1_head" or name == "task2_head" or name == "task3_head":
    #     #     continue
    #     param.requires_grad = False

    

    # # Verify that requires_grad is only True for the output layers
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
        
    # for name, param in model.named_parameters():
    #     if name == "task1_head" or name == "task2_head" or name == "task3_head":
    #         print("making task1head require grad!")
    #         param.requires_grad = True
        # print(f"Name: {name}")
        # print(f"Parameter: {param}")
        # print(f"Parameter shape: {param.shape}")
    #     # print("-" * 50)
    # model.task1_head.requires_grad = True
    # model.task2_head.requires_grad = True
    # model.task3_head.requires_grad = True
    # print(f"task1 weights: {model.task1_head.weight}")
    # print(f"task2 weights: {model.task2_head.weight}")
    # print(f"task3 weights: {model.task3_head.weight}")
    # print(f"\n\n\ntask1 req grad: {model.task1_head.requires_grad}\n\n\n")

    # return
    if training_args.do_train:

        # trainer.prelim_train(combined_raw_datasets)
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    # wandb.init(project='Cofi')
    os.environ["WANDB_DISABLED"] = "true"
    t_start = time.time()
    main()
    t_end = time.time()
    logger.info(f"Training took {round(t_end - t_start, 2)} seconds.")
