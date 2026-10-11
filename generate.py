"""Generate text from a saved v2.py checkpoint without running training."""
import argparse
from pathlib import Path
import torch
from v2 import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path(__file__).parent / 'checkpoints' / 'best.pt')
    parser.add_argument('--prompt', default='')
    parser.add_argument('--tokens', type=int, default=500, help='Number of new characters')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'])
    args = parser.parse_args()
    if args.tokens < 0:
        parser.error('--tokens must be nonnegative')
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}. Run v2.py first.')
    model, chars, step = load_checkpoint(args.checkpoint, args.device)
    stoi = {ch: i for i, ch in enumerate(chars)}
    unknown = sorted(set(args.prompt) - set(chars))
    if unknown:
        parser.error(f'Prompt contains characters outside the training vocabulary: {unknown!r}')
    torch.manual_seed(args.seed)
    tokens = [stoi[ch] for ch in args.prompt] if args.prompt else [0]
    context = torch.tensor([tokens], dtype=torch.long, device=next(model.parameters()).device)
    output = model.generate(context, max_new_tokens=args.tokens)[0].tolist()
    print(f'Loaded checkpoint at step {step}.')
    print(''.join(chars[i] for i in output))


if __name__ == '__main__':
    main()
