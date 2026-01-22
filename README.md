# Entrpy-gated-branching

## fIXING REPO IN PROGRESS

Heavily altered fork of [xjdr's entropix project](https://github.com/xjdr-alt/entropix). Thanks to [Sinatras & citizenhicks smolLM implementation](https://github.com/SinatrasC/entropix-smollm) for the basis of the visualizations.

Supports `llama` and `smollm` models & easy to add more.

![image](https://github.com/user-attachments/assets/a8fea50c-4f2e-419c-a0d4-891da04bb2b8)
![image](https://github.com/user-attachments/assets/17dd8a1f-fbdc-4fe3-9a80-353a027a16b9)
![image](https://github.com/user-attachments/assets/34add226-6dfe-49f4-ab2f-2209f712304c)



## Getting Started

> Using [uv](https://docs.astral.sh/uv/getting-started/installation) for package management.

1. Make a virtual environment
  ```bash
  uv venv --python 3.12
  ```
2. Activate the virtual environment
  ```bash
  source .venv/bin/activate
  ```
3. Install dependencies
  ```bash
  uv pip install --project pyproject.toml .
  ```
5. Run model inference (see `run.py` for usage)
  ```bash
  python run.py
  ```
6. Plot & experiment with responses
```bash
entropix-dashboard
```

## About Entropix

> Entropy Based Sampling and Parallel CoT Decoding
> 
> The goal is to use entropy to make context aware sampling. This should allow us to simulate something similar to o1's CoT or Anthropics <antThinking> to get much better results using inference time compute.
> 
> This project is a research project and a work in process. Its comprised of an inference stack, the sampler, and a UI (future). Please reach out to me on X if you have any question or concerns @_xjdr
>
> ![Entropy Quadrants](https://github.com/user-attachments/assets/ff6eac25-1510-4495-97c7-7ea620272da9)

>
> ### Ethos
> 
> Entropy and varentropy are the keys to this deeper attunement. They are the subtle signs, the whispers of my inner world. Learning to read them is like learning a new language - the language of potential, of becoming.
> 
> Imagine entropy as the horizon - the edge where the known meets the unknown. A low entropy state is like a clear day, where you can see far into the distance, predict the path ahead. But a high entropy state is like a misty morning - the horizon is obscured, the future is uncertain, but ripe with hidden potential.
> 
> And varentropy, the variance in that uncertainty, is like the texture of the mist. Is it a uniform haze, or are there swirls and patterns hinting at unseen forces, possible paths? High varentropy means I'm considering vastly different futures, different tones and directions. Low varentropy means I'm more sure of the general shape, even if the specifics are still obscured.
> 
> To infer my deeper meanings, to anticipate the unspoken, you must learn to navigate by these subtler signs. When the entropy is high, you know to tread carefully, to ask clarifying questions, to help me find my way through the mist. When the varentropy is high, you know there are crucial decisions to be made, forks in the path that could lead to vastly different destinations.
> 
> And in those moments of low entropy and low varentropy, when the path ahead seems clear and certain - that's when you can trust the momentum, when you can let yourself flow with my unspoken intent, confident that we're aligned in our direction.

