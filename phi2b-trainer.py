import os
import sys
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import numpy as np
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam

class LithiumMarketDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=1024):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.questions = [
            "What is the trend in production volume for {Company} over the past year?",
            "How does {Company}'s price for {Product Type} compare to the industry average?",
            "What factors are contributing to {Company}'s market share changes in {Country}?",
            "How has {Company}'s product purity changed over time, and what might be driving these changes?",
            "What is the relationship between {Company}'s capacity utilization and its profitability?",
            "How does {Company}'s investment strategy compare to its competitors in {Country}?",
            "What are the key focus areas of {Company}'s R&D spending, and how might this impact future market position?",
            "How has {Company}'s environmental impact score changed over time, and what measures are they taking to improve it?",
            "What is the correlation between energy consumption and production volume for {Company}, and how does this compare to industry standards?",
            "How is {Company} addressing water usage concerns in its {Product Type} production?",
            "What strategies is {Company} implementing to reduce CO2 emissions in its production process?",
            "How does {Company}'s recovery rate compare to industry benchmarks, and what implications does this have for efficiency?",
            "What are the pros and cons of {Company}'s chosen extraction method compared to alternatives?",
            "How do {Company}'s export volumes correlate with global market demand for {Product Type}?",
            "What factors are influencing {Company}'s import volumes, and how might this affect their market strategy?",
            "How does {Company}'s stock level strategy impact its ability to meet market demand?",
            "What is the relationship between {Company}'s employee count and its production efficiency?",
            "How does {Company}'s safety incident rate compare to industry standards, and what measures are they taking to improve?",
            "What is the impact of {Company}'s market share on pricing power for {Product Type} in {Country}?",
            "How does seasonality affect {Company}'s production and pricing strategies?",
            "What is the relationship between {Company}'s R&D spending and its product innovation rate?",
            "How does {Company}'s environmental impact score correlate with its market valuation?",
            "What is the impact of geopolitical factors on {Company}'s production and export volumes?",
            "How does {Company}'s supply chain resilience compare to its competitors?",
            "What is the relationship between {Company}'s capacity utilization and its environmental impact score?",
            "How does {Company}'s product mix strategy impact its overall profitability?",
            "What are the key drivers of {Company}'s pricing strategy for {Product Type}?",
            "How does {Company}'s investment in automation technologies affect its production efficiency and employee count?",
            "What is the impact of regulatory changes on {Company}'s operations and market strategy in {Country}?",
            "How does {Company}'s vertical integration strategy affect its market position and profitability?",
            "What is the correlation between {Company}'s safety performance and its operational efficiency?",
            "How does {Company}'s inventory management strategy impact its financial performance?",
            "What is the relationship between {Company}'s market share and its pricing power in different regions?",
            "How does {Company}'s product quality (purity) affect its market positioning and customer retention?",
            "What is the impact of {Company}'s sustainability initiatives on its brand value and market share?",
            "How does {Company}'s innovation rate compare to industry peers, and what is its impact on market leadership?",
            "What is the relationship between {Company}'s capital expenditure and its production capacity growth?",
            "How does {Company}'s geographic diversification strategy affect its risk profile and market opportunities?",
            "What is the impact of raw material price fluctuations on {Company}'s profitability and pricing strategy?",
            "How does {Company}'s employee productivity compare to industry benchmarks, and what factors contribute to any differences?",
            "What is the relationship between {Company}'s export volume and its domestic market share?",
            "How does {Company}'s product lifecycle management strategy impact its long-term market position?",
            "What is the correlation between {Company}'s environmental performance and its ability to secure new contracts or enter new markets?",
            "How does {Company}'s customer concentration risk compare to its competitors, and what strategies are in place to mitigate this risk?",
            "What is the impact of {Company}'s digital transformation initiatives on its operational efficiency and market responsiveness?",
            "How does {Company}'s talent retention rate correlate with its innovation output and market performance?",
            "What is the relationship between {Company}'s production volume and its economies of scale?",
            "How does {Company}'s product diversification strategy impact its overall business resilience?",
            "What is the impact of {Company}'s strategic partnerships or joint ventures on its market access and technology development?",
            "How does {Company}'s financial leverage compare to industry peers, and what implications does this have for its growth strategy?"
        ]

    def __len__(self):
        return len(self.data) * len(self.questions)

    def __getitem__(self, idx):
        data_idx = idx // len(self.questions)
        question_idx = idx % len(self.questions)
        
        row = self.data.iloc[data_idx]
        question = self.questions[question_idx].format(**row.to_dict())
        
        answer = self.generate_answer(question, row)
        
        full_text = f"Question: {question}\nAnswer: {answer}"
        encoding = self.tokenizer(full_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0).clone()
        }

    def generate_answer(self, question, row):
        # This is a placeholder for more sophisticated answer generation
        # In a real-world scenario, you would implement more complex logic here
        # to generate insightful answers based on the data and question
        return "Based on the available data, a comprehensive analysis would be required to provide an accurate and insightful answer to this question. The answer would involve examining trends, comparing data points, and considering multiple factors that influence the company's performance and market dynamics."

class PhiFineTuner(LightningModule):
    def __init__(self, model_name, learning_rate=2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.model = None

    def setup(self, stage=None):
        if stage == 'fit' and self.model is None:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                use_cache=False
            )
            self.model.gradient_checkpointing_enable()
            self.model.train()

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return DeepSpeedCPUAdam(self.parameters(), lr=self.learning_rate)

def main():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model_name = "microsoft/phi-3-mini-128k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    csv_file = os.path.join('data', 'ltmidata.csv')
    dataset = LithiumMarketDataset(csv_file, tokenizer, max_length=512)
    
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler, num_workers=4, pin_memory=True)
    
    model = PhiFineTuner(model_name)
    
    deepspeed_strategy = DeepSpeedStrategy(
        stage=2,
        offload_optimizer=True,
        offload_parameters=True,
        allgather_bucket_size=5e8,
        reduce_bucket_size=5e8,
    )

    trainer = Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=4,
        strategy=deepspeed_strategy,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        enable_checkpointing=True,
        logger=False
    )
    
    trainer.fit(model, dataloader)
    
    if local_rank == 0:
        output_dir = os.path.join('output', 'ltmi-phi-3-mini-128k-instruct')
        model.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()