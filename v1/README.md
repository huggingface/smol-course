# a smol course v1

<div style="background: linear-gradient(to right, #e0f7fa, #e1bee7, orange); padding: 20px; border-radius: 5px; margin-bottom: 20px; color: purple;">
    <h2>Participation is open, free, and now!</h2>
    <p>This course is open and peer reviewed. To get involved with the course <strong>open a pull request</strong> and submit your work for review. Here are the steps:</p>
    <ol>
        <li>Fork the repo <a href="https://github.com/huggingface/smol-course/fork">here</a></li>
        <li>Read the material, make changes, do the exercises, add your own examples.</li>
        <li>Open a PR on the december_2024 branch</li>
        <li>Get it reviewed and merged</li>
    </ol>
    <p>This should help you learn and to build a community-driven course that is always improving.</p>
</div>

## Installation

We maintain the course as a package so you can install dependencies easily via a package manager. We recommend [uv](https://github.com/astral-sh/uv) for this purpose, but you could use alternatives like `pip` or `pdm`.

### Using `uv`

With `uv` installed, you can install the course like this:

```bash
uv venv --python 3.11.0
uv sync
```

### Using `pip`

All the examples run in the same **python 3.11** environment, so you should create an environment and install dependencies like this:

```bash
# python -m venv .venv
# source .venv/bin/activate
pip install -r requirements.txt
```

### Google Colab

**From Google Colab** you will need to install dependencies flexibly based on the hardware you're using. Like this:

```bash
pip install transformers trl datasets huggingface_hub
```
