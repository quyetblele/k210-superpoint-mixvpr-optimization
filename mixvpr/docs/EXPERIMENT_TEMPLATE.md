# Experiment ID

## Hypothesis

State one falsifiable claim.

## Controlled change

Change exactly one primary variable.

## Frozen controls

- data/split:
- seed:
- teacher:
- parent checkpoint:
- augmentation:
- loss weights:
- optimizer/schedule:
- evaluation protocol:

## Before-change evidence

- checkpoint SHA256:
- model spec:
- DEV R@1/R@5/R@10:
- margin:
- params/MACs:
- hardware status:

## Experiment

Record:
- exact command;
- exact structural change;
- pruning indices if applicable;
- task/KD losses;
- gradient norms/cosines when relevant;
- immediate damage;
- recovery curve;
- ONNX/PTQ/nncase evidence when relevant.

## Result

- best DEV checkpoint:
- checkpoint SHA256:
- DEV R@1/R@5/R@10:
- delta vs parent:
- recovery fraction:
- params/MACs:
- compiler/gencode/simulator:
- unexpected issues:

## Decision

Choose exactly one:
- ACCEPT
- REJECT
- INCONCLUSIVE

Explain the evidence and the single next experiment.
Do not use independent TEST to make this decision.
