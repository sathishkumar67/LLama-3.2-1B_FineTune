from model import *
from huggingface_hub import hf_hub_download
import numpy as np
from dataset import *
import lightning as L
from lightning.pytorch import Trainer

hf_hub_download(repo_id="pt-sk/ll-3.2-1B_Instruct", filename="original/consolidated.00.pth", repo_type="model", local_dir="/kaggle/working")

gin.parse_config_file('config/1B.gin')
config = ModelArgs()

torch.manual_seed(config.seed)

model = Transformer(config)


weights = torch.load("/kaggle/working/original/consolidated.00.pth", map_location="cpu", )

fp32_weights = {k: v.to(dtype=torch.float32) for k, v in weights.items()}

model.load_state_dict(fp32_weights)

hf_hub_download(repo_id="pt-sk/chatgpt-dataset", filename="conversation_tokens.npy", repo_type="dataset", local_dir="/kaggle/working")

conversation = np.load("/kaggle/working/conversation_tokens.npy")

dataset = TokenDataset(config, conversation)
dataloader = DataLoader(dataset, shuffle=True, drop_last=True, batch_size=config.batch_size)


class ModelWrapper(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.optimizer = self.configure_optimizers()

    def training_step(self, batch, batch_idx):
        self.model.train()
        optimizer = self.optimizers()
        optimizer.zero_grad()
        
        batch, label = batch
        logits, loss = self.model(batch, label)
        self.log("Train_Loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.000006)
        return optimizer
    
modelwrapper = ModelWrapper(model)
trainer = L.Trainer(devices=2,accelerator="gpu", strategy="deepspeed_stage_2", precision="bf16", max_epochs=1, gradient_clip_val=1.0)
trainer.fit(modelwrapper, dataloader)