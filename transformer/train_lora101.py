from datasets import load_dataset
from transformers import TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, Trainer
import evaluate
import torch
from peft import LoraConfig, get_peft_model, TaskType  # 添加TaskType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载基础模型
model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-cased", 
    num_labels=5
).to(device)

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# 配置LoRA参数 - 使用正确的任务类型
lora_config = LoraConfig(
    r=8,  # LoRA秩
    lora_alpha=32,  # 缩放因子
    target_modules=["query", "value"],  # 目标模块
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS  # 修正为正确的任务类型
)

# 将基础模型转换为LoRA模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 打印可训练参数

def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True,
        max_length=128  # 限制长度以节省内存
    )

# 加载并处理数据集
dataset = load_dataset('yelp_review_full')
small_train_dataset = dataset['train'].shuffle(seed=42).select(range(100))
small_eval_dataset = dataset['test'].shuffle(seed=42).select(range(50))  # 使用测试集

tokenized_train = small_train_dataset.map(tokenize_function, batched=True)
tokenized_eval = small_eval_dataset.map(tokenize_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="lora_trainer",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=torch.cuda.is_available(),
    learning_rate=1e-3,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="no",
    report_to="none"  # 禁用wandb报告
)

# 定义评估函数
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 创建训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics
)

# 开始训练
trainer.train()

# 保存LoRA权重
model.save_pretrained("lora_weights")
