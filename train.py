from dataclasses import asdict
import transformers
from transformers import Trainer, AutoProcessor, HfArgumentParser, TrainingArguments, AutoConfig, logging

from models import ModelArguments
from data.lmm_dataset import DataArguments, LMMDataset

logger = logging.get_logger(__name__)

if __name__ == "__main__":
    training_args, model_args, data_args = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments)).parse_args_into_dataclasses()
    config = AutoConfig.from_pretrained(model_args.pretrained_model_name_or_path)
    model = getattr(transformers, config.architectures[0]).from_pretrained(
        model_args.pretrained_model_name_or_path, 
        torch_dtype="auto", attn_implementation='flash_attention_2'
    )
    for m in model_args.freeze_modules:
        logger.warning(f"Freezing module {m}")
        getattr(model, m).requires_grad_(False)
    if 'Qwen2VL' in model.config.architectures[0]:
        processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct', padding_side='right') # Qwen2vl-base processor has some bugs. otherwise we do not need this
    else:
        processor = AutoProcessor.from_pretrained(model_args.pretrained_model_name_or_path, padding_side='right')
    train_dataset = LMMDataset(**asdict(data_args), **asdict(training_args), **asdict(model_args), processor=processor)
    Trainer(
        model=model, args=training_args, 
        data_collator=train_dataset.data_collator, 
        train_dataset=train_dataset, processing_class=processor
    ).train(resume_from_checkpoint=not training_args.overwrite_output_dir)