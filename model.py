import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import lightning.pytorch as pl
import wandb



class Meso4(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super(Meso4, self).__init__()
        self.learning_rate = learning_rate
        
       
        self.validation_step_outputs = []
        
        
        self.train_accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.val_accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.f1_metric = torchmetrics.F1Score(task="binary")
        self.precision_macro_metric = torchmetrics.Precision(average="macro", task="binary")
        self.recall_macro_metric = torchmetrics.Recall(average="macro", task="binary")
        self.precision_micro_metric = torchmetrics.Precision(average="micro", task="binary")
        self.recall_micro_metric = torchmetrics.Recall(average="micro", task="binary")

       
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1) 
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, padding=1)

        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, padding=1)

        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=2, padding=1)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(kernel_size=4, padding=1)

        
        self.flatten = nn.Flatten()

        
        dummy_input = torch.zeros(1, 3, 256, 256)
        dummy_output = self._forward_conv(dummy_input)
        flattened_size = dummy_output.view(dummy_output.size(0), -1).size(1)

       
        self.fc1 = nn.Linear(flattened_size, 16)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, 1)

    def _forward_conv(self, x):
        
        x = self.pool1(self.bn1(self.conv1(x)))
        x = self.pool2(self.bn2(self.conv2(x)))
        x = self.pool3(self.bn3(self.conv3(x)))
        x = self.pool4(self.bn4(self.conv4(x)))
        return x

    def forward(self, x):
        
        x = self._forward_conv(x)
        x = self.flatten(x)
        x = self.dropout(self.leaky_relu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return torch.sigmoid(x)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1) 
        y_hat = self.forward(x)
        loss = nn.functional.mse_loss(y_hat, y.float())
        acc = ((y_hat > 0.5) == y).float().mean()
        self.log('train/loss', loss, prog_bar=True, on_epoch=True)
        self.log('train/acc', acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
       
        images, labels = batch
        labels = labels.unsqueeze(1)  

        
        outputs = self.forward(images)
        preds = torch.sigmoid(outputs) > 0.5  

        
        valid_acc = self.val_accuracy_metric(preds.int(), labels.int())
        precision_macro = self.precision_macro_metric(preds.int(), labels.int())
        recall_macro = self.recall_macro_metric(preds.int(), labels.int())
        precision_micro = self.precision_micro_metric(preds.int(), labels.int())
        recall_micro = self.recall_micro_metric(preds.int(), labels.int())
        f1 = self.f1_metric(preds.int(), labels.int())

        
        self.log("valid/loss", nn.functional.binary_cross_entropy_with_logits(outputs, labels.float()), prog_bar=True)
        self.log("valid/acc", valid_acc, prog_bar=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True)
        self.log("valid/f1", f1, prog_bar=True)

       
        self.validation_step_outputs.append({"labels": labels, "logits": outputs})
        return {"labels": labels, "logits": outputs}

    def on_validation_epoch_end(self):
       
        labels = torch.cat([x["labels"] for x in self.validation_step_outputs]).cpu().numpy()
        logits = torch.cat([x["logits"] for x in self.validation_step_outputs]).cpu().numpy()

        
        preds = (logits > 0.5).astype(int)

       
        labels = labels.flatten().tolist()
        preds = preds.flatten().tolist()

       
        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    preds=preds,
                    y_true=labels,
                    class_names=["Real", "DeepFake"]
                )
            }
        )

        
        self.validation_step_outputs.clear()

