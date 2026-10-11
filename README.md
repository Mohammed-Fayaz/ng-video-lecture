
# nanogpt-lecture

Code created in the [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) video lecture series, specifically on the first lecture on nanoGPT. Publishing here as a Github repo so people can easily hack it, walk through the `git log` history of it, etc.

NOTE: sadly I did not go too much into model initialization in the video lecture, but it is quite important for good performance. The current code will train and work fine, but its convergence is slower because it starts off in a not great spot in the weight space. Please see [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) for `# init all weights` comment, and especially how it calls the `_init_weights` function. Even more sadly, the code in this repo is a bit different in how it names and stores the various modules, so it's not possible to directly copy paste this code here. My current plan is to publish a supplementary video lecture and cover these parts, then I will also push the exact code changes to this repo. For now I'm keeping it as is so it is almost exactly what we actually covered in the video.

### License

MIT


## Local training and saved-model inference

Run commands from this repository with the virtual environment's Python:

```sh
.venv/bin/python v2.py
.venv/bin/python generate.py --prompt "ROMEO:" --tokens 500
```

Training automatically selects CUDA, Apple MPS, or CPU. It saves `latest.pt`
and the lowest-validation-loss `best.pt` every 500 completed steps, and
`final.pt` after 5,000 steps. Ctrl+C saves current weights to `interrupted.pt`;
force-quitting or losing power can only preserve earlier completed saves.
Checkpoints include model weights, architecture, character vocabulary, optimizer
state and completed-step count. They are ignored by Git. There is no training
resume command yet.

Inference defaults to `checkpoints/best.pt`, disables dropout and gradient
tracking, and does not need `input.txt`. Use `--checkpoint checkpoints/final.pt`
to choose another checkpoint, `--device cpu` to override the device, or `--seed`
to choose a sampling seed. Prompts must use characters from the training corpus.
Here a token is one character. Running inference does not update model weights.

For a short training check, use a separate directory to preserve full-run files:

```sh
.venv/bin/python v2.py --steps 2 --eval-iters 1 --batch-size 2 --checkpoint-dir checkpoints/smoke
.venv/bin/python generate.py --checkpoint checkpoints/smoke/final.pt --prompt "ROMEO:" --tokens 20
```

The original tutorial architecture and attention scaling are retained in `v2.py`.
