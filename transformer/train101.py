from datasets import load_dataset #数据集处理
from transformers import TrainingArguments # 模型训练参数
from transformers import AutoTokenizer #分词器
from transformers import AutoModelForSequenceClassification #模型
from transformers import Trainer #模型训练
import evaluate #模型评估
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#加载模型和分词器
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5).to(device)
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

#加载数据集
datasets_train = load_dataset('yelp_review_full',split='train')
#数据集太大，在自家的机器上只能加载一点点记录
small_train_dataset = datasets_train.shuffle(seed=42).select(range(100)) #截取一小部分数据集用于模型训练（1000行）
small_eval_dataset = datasets_train.shuffle(seed=1003).select(range(100)) #截取一小部分数据集用于模型评估（1000行）


# 对数据集进行分词
tokenized_datasets_train = small_train_dataset.map(tokenize_function, batched=True)
tokenized_datasets_eval = small_eval_dataset.map(tokenize_function, batched=True)

# 定义训练参数
training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch",fp16=True)
metric = evaluate.load("accuracy")

# 定义训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets_train,
    eval_dataset=tokenized_datasets_eval
)

# 开始训练
trainer.train()

