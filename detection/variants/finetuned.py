"""A3 variant (ii): fine-tuned transformer.

DeBERTa-v3-base trained end to end on the benchmark's five-class and
scope-relation labels, with no hand-authored features. The learned-representation
arm of the three-way comparison.

Fits the 6 GB floor: fp16, batch 8 with gradient accumulation to an effective
batch of 32, sequence length 384. The 8 GB profile raises the physical batch
and lowers accumulation to reach the *same* effective batch, so both machines
train the same model -- otherwise the two members would get different
checkpoints from one config and the comparison would be meaningless.

Two heads share one encoder: five-class conflict type, and four-way scope
relation. Multi-task rather than two models because the two labels are
correlated (only conditional pairs have a relation) and because one encoder is
all the VRAM budget allows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import settings
from contract.models import ConflictType, QueryRecord, ScopeRelation
from detection.stage1 import PairCandidate
from detection.variants.base import DetectorVariant, labels_from_record

TYPE_LABELS = [c.value for c in ConflictType]
RELATION_LABELS = [r.value for r in ScopeRelation]
_NO_RELATION = "__none__"
RELATION_LABELS_WITH_NULL = RELATION_LABELS + [_NO_RELATION]


@dataclass
class TrainingReport:
    epochs: int
    steps: int
    final_loss: float
    n_examples: int
    device: str
    effective_batch_size: int

    def render(self) -> str:
        return (
            f"fine-tune: {self.n_examples} examples, {self.epochs} epochs, "
            f"{self.steps} steps, final loss {self.final_loss:.4f}, "
            f"device={self.device}, effective batch={self.effective_batch_size}"
        )


class _MultiTaskModel:
    """Encoder plus two classification heads. Built lazily so importing this
    module costs nothing on a machine without torch."""

    def __init__(self, model_name: str, device: str, fp16: bool):
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.type_head = nn.Linear(hidden, len(TYPE_LABELS))
        self.relation_head = nn.Linear(hidden, len(RELATION_LABELS_WITH_NULL))

        self.device = device
        self.encoder.to(device)
        self.type_head.to(device)
        self.relation_head.to(device)
        self.fp16 = fp16

    def parameters(self):
        return (
            list(self.encoder.parameters())
            + list(self.type_head.parameters())
            + list(self.relation_head.parameters())
        )

    def forward(self, batch):
        out = self.encoder(**batch)
        # Mean-pool over real tokens. DeBERTa has no pooled [CLS] output, and
        # mean pooling over the attention mask is the standard substitute.
        mask = batch["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.type_head(pooled), self.relation_head(pooled)

    def train(self):
        self.encoder.train(); self.type_head.train(); self.relation_head.train()

    def eval(self):
        self.encoder.eval(); self.type_head.eval(); self.relation_head.eval()


class FineTunedDetector(DetectorVariant):
    name = "finetuned"
    description = "DeBERTa-v3-base fine-tuned end to end, no hand-authored features"

    def __init__(self, *, profile: str | None = None, model_name: str | None = None):
        self.hw = settings.hardware(profile)
        self.spec = self.hw.finetune
        self.model_name = model_name or self.spec.model
        self._model: _MultiTaskModel | None = None

    @property
    def requires_training(self) -> bool:
        return True

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    # -- data ------------------------------------------------------------------ #

    @staticmethod
    def _examples(records: list[QueryRecord], stage1) -> list[tuple[str, str, str, str]]:
        """Build (text_i, text_j, type_label, relation_label) tuples.

        Pairs are read in canonical id order, so the training set is identical
        under any permutation of a record's passages.
        """
        out = []
        for rec in records:
            gold = labels_from_record(rec)
            by_id = {p.id: p.text for p in rec.passages}
            for (a, b), (ctype, relation) in gold.items():
                if a not in by_id or b not in by_id:
                    continue
                out.append((
                    by_id[a], by_id[b], ctype.value,
                    relation.value if relation else _NO_RELATION,
                ))
        return out

    # -- training -------------------------------------------------------------- #

    def fit(self, records: list[QueryRecord], *, stage1=None) -> FineTunedDetector:
        import torch
        from torch.nn.functional import cross_entropy

        examples = self._examples(records, stage1)
        if not examples:
            raise ValueError("no labelled pairs to fine-tune on")

        device = self.hw.resolve_device()
        model = _MultiTaskModel(self.model_name, device, self.hw.use_fp16())
        self._model = model

        opt = torch.optim.AdamW(model.parameters(), lr=self.spec.learning_rate)
        accum = self.spec.gradient_accumulation_steps
        bs = self.spec.batch_size

        n_batches = (len(examples) + bs - 1) // bs
        total_steps = max(1, (n_batches * self.spec.epochs) // accum)
        warmup = int(total_steps * self.spec.warmup_ratio)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda s: (s / max(1, warmup)) if s < warmup
            else max(0.0, (total_steps - s) / max(1, total_steps - warmup)),
        )
        # fp16 needs loss scaling; without it small gradients flush to zero and
        # the model silently fails to learn.
        scaler = torch.amp.GradScaler("cuda", enabled=self.hw.use_fp16())

        t_idx = {v: i for i, v in enumerate(TYPE_LABELS)}
        r_idx = {v: i for i, v in enumerate(RELATION_LABELS_WITH_NULL)}

        model.train()
        step, last_loss = 0, 0.0
        for epoch in range(self.spec.epochs):
            for bi in range(n_batches):
                chunk = examples[bi * bs:(bi + 1) * bs]
                if not chunk:
                    continue
                enc = model.tokenizer(
                    [c[0] for c in chunk], [c[1] for c in chunk],
                    padding=True, truncation=True,
                    max_length=self.spec.max_seq_length, return_tensors="pt",
                ).to(device)

                y_type = torch.tensor([t_idx[c[2]] for c in chunk], device=device)
                y_rel = torch.tensor([r_idx[c[3]] for c in chunk], device=device)

                with torch.amp.autocast("cuda", enabled=self.hw.use_fp16()):
                    logits_type, logits_rel = model.forward(enc)
                    loss = cross_entropy(logits_type, y_type) + cross_entropy(logits_rel, y_rel)
                    loss = loss / accum

                scaler.scale(loss).backward()
                last_loss = float(loss.item()) * accum

                if (bi + 1) % accum == 0 or bi == n_batches - 1:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    step += 1

        model.eval()
        self.report = TrainingReport(
            epochs=self.spec.epochs, steps=step, final_loss=last_loss,
            n_examples=len(examples), device=device,
            effective_batch_size=self.spec.effective_batch_size,
        )
        return self

    # -- inference -------------------------------------------------------------- #

    def classify(self, candidate: PairCandidate) -> tuple[ConflictType, ScopeRelation | None]:
        return self.classify_many([candidate])[0]

    def classify_many(self, candidates):
        if self._model is None:
            raise RuntimeError(
                "FineTunedDetector has not been trained. Call fit() or load() first -- "
                "unlike the other two variants it has no rule-based fallback."
            )
        import torch

        model = self._model
        out = []
        bs = self.spec.batch_size
        for start in range(0, len(candidates), bs):
            chunk = candidates[start:start + bs]
            enc = model.tokenizer(
                [c.text_i for c in chunk], [c.text_j for c in chunk],
                padding=True, truncation=True,
                max_length=self.spec.max_seq_length, return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                logits_type, logits_rel = model.forward(enc)

            for t_row, r_row in zip(logits_type, logits_rel):
                ctype = ConflictType(TYPE_LABELS[int(t_row.argmax())])
                rel_label = RELATION_LABELS_WITH_NULL[int(r_row.argmax())]
                relation = None if rel_label == _NO_RELATION else ScopeRelation(rel_label)

                # Contract repairs. The two heads are independent, so they can
                # disagree; the contract cannot represent that, so resolve it
                # here rather than emitting a pair that fails validation.
                if ctype is ConflictType.CONDITIONAL and relation is None:
                    # Take the best real relation instead of downgrading the
                    # type: the type head is the one trained on the label the
                    # comparison is judged on.
                    real = r_row[:len(RELATION_LABELS)]
                    relation = ScopeRelation(RELATION_LABELS[int(real.argmax())])
                if ctype not in (ConflictType.CONDITIONAL, ConflictType.FACTUAL):
                    relation = None
                out.append((ctype, relation))
        return out

    # -- persistence ------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        import torch

        if self._model is None:
            raise RuntimeError("nothing to save; train first")
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self._model.encoder.save_pretrained(p / "encoder")
        self._model.tokenizer.save_pretrained(p / "encoder")
        torch.save(
            {"type_head": self._model.type_head.state_dict(),
             "relation_head": self._model.relation_head.state_dict()},
            p / "heads.pt",
        )
        (p / "meta.json").write_text(
            json.dumps({"model_name": self.model_name, "type_labels": TYPE_LABELS,
                        "relation_labels": RELATION_LABELS_WITH_NULL}, indent=2),
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> FineTunedDetector:
        import torch

        p = Path(path)
        device = self.hw.resolve_device()
        model = _MultiTaskModel(str(p / "encoder"), device, self.hw.use_fp16())
        heads = torch.load(p / "heads.pt", map_location=device)
        model.type_head.load_state_dict(heads["type_head"])
        model.relation_head.load_state_dict(heads["relation_head"])
        model.eval()
        self._model = model
        return self
