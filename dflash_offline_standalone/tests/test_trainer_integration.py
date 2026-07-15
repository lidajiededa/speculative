import torch
from safetensors.torch import save_file
from transformers import TrainingArguments
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from dflash_offline.data import DFlashDataCollator, OfflineDFlashDataset
from dflash_offline.modeling_qwen3_dflash import QwenDFlashDraftModel
from dflash_offline.trainer import DFlashOfflineTrainer


def test_trainer_runs_one_cpu_step(tmp_path):
    target_dir = tmp_path / "target"
    hidden_dir = tmp_path / "hidden"
    output_dir = tmp_path / "output"
    target_dir.mkdir()
    hidden_dir.mkdir()

    target_config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        layer_types=["full_attention", "full_attention"],
        tie_word_embeddings=False,
    )
    target_config.save_pretrained(target_dir)
    save_file(
        {
            "model.embed_tokens.weight": torch.randn(64, 32),
            "lm_head.weight": torch.randn(64, 32),
        },
        target_dir / "model.safetensors",
    )

    draft_config = Qwen3Config(**target_config.to_dict())
    draft_config.architectures = ["QwenDFlashDraftModel"]
    draft_config.block_size = 2
    draft_config.num_anchors = 2
    draft_config.loss_decay_gamma = 7.0
    draft_config.attention_backend = "sdpa"
    draft_config.dflash_config = {"mask_token_id": 63, "target_layer_ids": [0, 1]}
    model = QwenDFlashDraftModel(draft_config)

    torch.save(
        {
            "input_ids": torch.randint(0, 63, (1, 8)),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
            "loss_mask": torch.ones(1, 8),
            "hidden_states": torch.randn(1, 8, 64),
        },
        hidden_dir / "sample.ckpt",
    )
    dataset = OfflineDFlashDataset(hidden_dir, expected_hidden_size=64)
    args = TrainingArguments(
        output_dir=output_dir,
        max_steps=1,
        per_device_train_batch_size=1,
        learning_rate=1e-4,
        report_to=[],
        use_cpu=True,
        remove_unused_columns=False,
        save_strategy="no",
        logging_steps=1,
    )
    trainer = DFlashOfflineTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DFlashDataCollator(),
        draft_config=draft_config,
        target_model_path=str(target_dir),
        target_dtype=torch.float32,
        embed_weight_key="model.embed_tokens.weight",
        lm_head_key="lm_head.weight",
        trust_remote_code=False,
        optimizer_precision="standard",
        dflash_max_grad_norm=1.0,
    )
    result = trainer.train()
    assert result.global_step == 1
