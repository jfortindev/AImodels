import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam
import random

import pandas as pd
import random
from torch.utils.data import Dataset

class LithiumMarketDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=1024):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.questions = [
            "What is the trend in production volume for {Company} over the past year?",
            "How does {Company}'s price for their main product compare to the industry average?",
            "What factors are contributing to {Company}'s market share changes in {Country}?",
            "How has {Company}'s product purity changed over time, and what might be driving these changes?",
            "What is the relationship between {Company}'s capacity utilization and its profitability?",
            "How does {Company}'s investment strategy compare to its competitors in {Country}?",
            "What are the key focus areas of {Company}'s R&D spending, and how might this impact future market position?",
            "How has {Company}'s environmental impact score changed over time, and what measures are they taking to improve it?",
            "How is {Company} addressing water usage concerns in its production?",
            "What strategies is {Company} implementing to reduce CO2 emissions in its production process?",
            "How does {Company}'s recovery rate compare to industry benchmarks, and what implications does this have for efficiency?",
            "What are the pros and cons of {Company}'s chosen extraction method compared to alternatives?",
            "How do {Company}'s export volumes correlate with global market demand for their main product?",
            "What factors are influencing {Company}'s import volumes, and how might this affect their market strategy?",
            "How does {Company}'s stock level strategy impact its ability to meet market demand?",
            "What is the relationship between {Company}'s employee count and its production efficiency?",
            "How does {Company}'s safety incident rate compare to industry standards, and what measures are they taking to improve?",
            "What is the impact of {Company}'s market share on pricing power in {Country}?",
            "How does seasonality affect {Company}'s production and pricing strategies?",
            "What is the relationship between {Company}'s R&D spending and its product innovation rate?",
            "How does {Company}'s environmental impact score correlate with its market valuation?",
            "What is the impact of geopolitical factors on {Company}'s production and export volumes?",
            "How does {Company}'s supply chain resilience compare to its competitors?",
            "What is the relationship between {Company}'s capacity utilization and its environmental impact score?",
            "How does {Company}'s product mix strategy impact its overall profitability?",
            "What are the key drivers of {Company}'s pricing strategy for their main product?",
            "How does {Company}'s investment in automation technologies affect its production efficiency and employee count?",
            "What is the impact of regulatory changes on {Company}'s operations and market strategy in {Country}?",
            "How does {Company}'s vertical integration strategy affect its market position and profitability?",
            "What is the correlation between {Company}'s safety performance and its operational efficiency?",
            "How does {Company}'s inventory management strategy impact its financial performance?",
            "What is the relationship between {Company}'s market share and its pricing power in different regions?",
            "How does {Company}'s product quality affect its market positioning and customer retention?",
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
            "How does {Company}'s financial leverage compare to industry peers, and what implications does this have for its growth strategy?",
            "What is the relationship between {Company}'s energy consumption and its production efficiency?"
        ]

    def __len__(self):
        return len(self.data) * len(self.questions)

    def __getitem__(self, idx):
        data_idx = idx // len(self.questions)
        question_idx = idx % len(self.questions)
        
        row = self.data.iloc[data_idx]
        question = self.questions[question_idx].format(
            Company=row.get('Company', 'the company'),
            Country=row.get('Country', 'their main market')
        )
        
        answer = self.generate_answer(question, row)
        
        full_text = f"Question: {question}\nAnswer: {answer}"
        encoding = self.tokenizer(full_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0).clone()
        }

    def generate_answer(self, question, row):
        company = row.get('Company', 'The company')
        product_type = row.get('Product Type', 'main product')
        country = row.get('Country', 'main market')
        
        answer = f"Based on the data for {company}, a comprehensive analysis of their {product_type} operations in {country} reveals the following insights:\n\n"
        
        if "production volume" in question.lower():
            answer += f"Production Volume: {row.get('Production Volume (tonnes)', 'N/A')} tonnes. "
            answer += "This figure should be compared to historical data to determine the trend. "
            answer += f"It's important to consider factors such as Capacity Utilization ({row.get('Capacity Utilization (%)', 'N/A')}%) and market demand.\n\n"
        
        elif "price" in question.lower():
            answer += f"Price: ${row.get('Price (USD/tonne)', 'N/A')} per tonne. "
            answer += "To determine how this compares to the industry average, we'd need additional market data. "
            answer += f"Consider the impact of factors like product purity ({row.get('Purity (%)', 'N/A')}%) and market share ({row.get('Market Share (%)', 'N/A')}%).\n\n"
        
        elif "market share" in question.lower():
            answer += f"Market Share: {row.get('Market Share (%)', 'N/A')}%. "
            answer += f"Factors potentially influencing this include production volume ({row.get('Production Volume (tonnes)', 'N/A')} tonnes), "
            answer += f"price (${row.get('Price (USD/tonne)', 'N/A')} per tonne), and product quality (purity: {row.get('Purity (%)', 'N/A')}%).\n\n"
        
        elif "purity" in question.lower():
            answer += f"Product Purity: {row.get('Purity (%)', 'N/A')}%. "
            answer += "Changes in purity could be driven by improvements in extraction methods or R&D efforts. "
            answer += f"Consider the company's R&D spending (${row.get('R&D Spending (million USD)', 'N/A')} million) and their extraction method ({row.get('Extraction Method', 'N/A')}).\n\n"
        
        elif "capacity utilization" in question.lower():
            answer += f"Capacity Utilization: {row.get('Capacity Utilization (%)', 'N/A')}%. "
            answer += "This metric can significantly impact profitability. Higher utilization often leads to better cost efficiency, "
            answer += f"but also consider factors like market demand and stock levels ({row.get('Stock Level (tonnes)', 'N/A')} tonnes).\n\n"
        
        elif "investment strategy" in question.lower():
            answer += f"Investment: ${row.get('Investment (million USD)', 'N/A')} million. "
            answer += "To compare this to competitors, we'd need industry-wide data. Consider how this investment level "
            answer += f"aligns with their market share ({row.get('Market Share (%)', 'N/A')}%) and production volume ({row.get('Production Volume (tonnes)', 'N/A')} tonnes).\n\n"
        
        elif "r&d spending" in question.lower():
            answer += f"R&D Spending: ${row.get('R&D Spending (million USD)', 'N/A')} million. "
            answer += "Key focus areas might include improving extraction methods, increasing product purity, or developing new product types. "
            answer += f"This could impact future market position by affecting product quality (current purity: {row.get('Purity (%)', 'N/A')}%) or production efficiency.\n\n"
        
        elif "environmental impact" in question.lower():
            answer += f"Environmental Impact Score: {row.get('Environmental Impact Score', 'N/A')}. "
            answer += "To assess changes over time, we'd need historical data. Improvement measures might include "
            answer += f"reducing energy consumption (currently {row.get('Energy Consumption (MWh/tonne)', 'N/A')} MWh/tonne) and "
            answer += f"CO2 emissions (currently {row.get('CO2 Emissions (kg/tonne)', 'N/A')} kg/tonne).\n\n"
        
        elif "water usage" in question.lower():
            answer += f"Water Usage: {row.get('Water Usage (m³/tonne)', 'N/A')} m³/tonne. "
            answer += f"For {product_type} production, this level of water usage should be considered in the context of local water resources in {country}. "
            answer += "Reduction strategies might include process optimization or water recycling technologies.\n\n"
        
        elif "co2 emissions" in question.lower():
            answer += f"CO2 Emissions: {row.get('CO2 Emissions (kg/tonne)', 'N/A')} kg/tonne. "
            answer += "Reduction strategies could include improving energy efficiency, using renewable energy sources, or optimizing the extraction process. "
            answer += f"Consider the relationship with energy consumption ({row.get('Energy Consumption (MWh/tonne)', 'N/A')} MWh/tonne).\n\n"
        
        else:
            # For other questions, provide a generic response using available data
            metrics = [f"{col}: {row.get(col, 'N/A')}" for col in row.index if col != 'Company']
            answer += "To address this question comprehensively, we should consider multiple factors including:\n"
            answer += "\n".join(random.sample(metrics, min(4, len(metrics))))  # Randomly select up to 4 metrics
            answer += "\n\nA detailed analysis of these factors and their interrelationships would provide more accurate insights."

        return answer

class Qwen2FineTuner(LightningModule):
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
    # Initialize the process group
    dist.init_process_group(backend='nccl')
    
    # Get the local rank from the environment variable
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Set the device
    torch.cuda.set_device(local_rank)

    # Set up the model, tokenizer, and dataset
    model_name = "KnutJaegersberg/Qwen2-Deita-500m"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    csv_file = os.path.join('data', 'ltmidata.csv')
    dataset = LithiumMarketDataset(csv_file, tokenizer, max_length=512)
    
    # Set up the distributed sampler
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler, num_workers=4, pin_memory=True)
    
    # Set up the model
    model = Qwen2FineTuner(model_name)
    
    # Set up the DeepSpeed strategy
    deepspeed_strategy = DeepSpeedStrategy(
        stage=2,
        offload_optimizer=True,
        offload_parameters=True,
        allgather_bucket_size=5e8,
        reduce_bucket_size=5e8,
    )

    # Set up the trainer
    trainer = Trainer(
        max_epochs=2,
        accelerator="gpu",
        devices="auto",  # Use all available GPUs
        strategy=deepspeed_strategy,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        enable_checkpointing=True,
        logger=False
    )
    
    trainer.fit(model, dataloader)
    
    if local_rank == 0:
        output_dir = os.path.join('output', 'ltmi-qwen2-500m')
        model.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()