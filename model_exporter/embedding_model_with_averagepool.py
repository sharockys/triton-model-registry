import typer
import torch
from torch import nn
from optimum.exporters.onnx import onnx_export_from_model
from transformers import AutoTokenizer, AutoModel


class ModelWithAveragePooling(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.pooler = nn.AdaptiveAvgPool1d(1)

    def forward(self, input_ids, attention_mask):
        outputs: torch.Tensor = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return self.pooler(outputs.last_hidden_state.transpose(1, 2)).squeeze(2)

    @property
    def config(self):
        return self.model.config


def main(
    model_checkpoint: str = typer.Option(
        default="intfloat/multilingual-e5-large-instruct",
        help="Model checkpoint to be used.",
    ),
    export_path: str = typer.Option(
        default="model_export", help="Path to export the model."
    ),
    device: str = typer.Option(
        default="cpu", help="Device to be used for exporting the model."
    ),
):
    model = AutoModel.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.save_pretrained(export_path)
    typer.echo(model.config)
    avg_model = ModelWithAveragePooling(model)
    onnx_export_from_model(
        model=avg_model,
        output="model_export",
        task="feature-extraction",  # output name will be`last_hidden_state`
        device=device,
    )
    typer.echo(
        "Model exported successfully. Use https://netron.app to visualize the model."
    )


if __name__ == "__main__":
    typer.run(main)
