"""Reference synchronous executor for the online component contracts."""

from __future__ import annotations

from .contracts import RuntimeComponents, RuntimeResult


class SyncExecutor:
    def run(self, components: RuntimeComponents, *, epochs: int) -> RuntimeResult:
        if epochs <= 0:
            raise ValueError("epochs must be greater than zero")
        steps = 0
        examples = 0
        weighted_loss = 0.0
        for epoch in range(epochs):
            for seeds in components.seeds.iter_epoch(epoch):
                plan = components.sampler.sample(seeds)
                features = components.fetcher.fetch(plan)
                batch = components.assembler.assemble(features)
                result = components.trainer.train_step(batch)
                steps += 1
                examples += result.examples
                weighted_loss += result.loss * result.examples
                del seeds, plan, features, batch
        return RuntimeResult(
            steps=steps,
            examples=examples,
            mean_loss=weighted_loss / max(examples, 1),
        )
