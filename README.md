# Amharic Bank Voice Assistant

A low-latency, fully open-source voice assistant that answers customer questions in **Amharic**, using a bank's own documents as its knowledge base.

You speak Amharic, and it answers out loud, only from the bank's documents. If the answer is not in the documents, it says so instead of inventing one.

```
mic → ASR → RAG retrieval → LLM → TTS → spoken answer
```

## How it works

| Stage | Component | Model / library |
|---|---|---|
| Speech to text | Amharic ASR | `snapwre/hohe-asr-amharic` |
| Retrieval | Dense embeddings + cosine search | `BAAI/bge-m3` (numpy, no vector DB) |
| Answer generation | Local LLM, streamed | Gemma 4 E4B, `Q4_K_M` GGUF, served by `llama.cpp` |
| Text to speech | Amharic TTS | `gheero-Leyu/amharic-omnivoice-tts` (fallback: `african-low-resource/omnivoice-amharic`) |
| Demo UI | Browser microphone page | Gradio |

Design choices that keep latency low:

- The LLM runs in `llama.cpp` with a 4-bit model on one GPU, with thinking mode switched off.
- The LLM streams its answer, and the first complete clause (ending in `።`) is sent to TTS without waiting for the rest.
- TTS uses 8 diffusion steps (`NUM_STEP = 8`), chosen after comparing 32, 16 and 8 by ear.
- LLM runs on GPU 0, ASR and TTS on GPU 1, so the stages do not compete.
- Document embeddings are computed once; only the question is embedded at run time. Retrieval is capped at 2 short chunks to keep prompts small.
- TTS uses a fixed reference voice so the speaker does not change between sentences.

## Measured results (Kaggle, 2x T4)

| Measurement | Result |
|---|---|
| LLM generation speed (`llama-bench`, `tg64`) | about 53 tokens/s |
| LLM time to first spoken clause | about 0.3 to 0.5 s |
| ASR, short utterance | about 0.1 to 0.4 s |
| TTS, one short sentence at 32 steps | about 1.4 s (8 steps is faster; see the notebook) |

These numbers are per stage and come from a notebook that runs the stages one after another. The "estimated time to first audio" printed by the demo adds them up as if they were pipelined. It does **not** include the end-of-speech wait, browser upload or playback buffering, and it has not yet been measured in a real streaming pipeline.

## Requirements

- Kaggle notebook with **GPU T4 x2** and **Internet on**
- Kaggle secret `HF_TOKEN` (Add-ons, then Secrets), needed if any model is gated
- Input dataset `bank-docs` containing `.docx` files, ideally in `faq/` and `general/` folders (category is taken from the path; anything with "faq" in it is treated as FAQ)

## Running it

Open `amharic_bank_assistant_clean.ipynb` and run the sections **in order, top to bottom**, once per fresh session:

1. **Section A:** builds `llama.cpp` (about 30 min the first time, skipped if already built), downloads the model, starts the LLM server on port 8080.
2. **Section B:** loads ASR and TTS.
3. **Section C (optional):** compares TTS step counts.
4. **Section D:** extracts the bank `.docx` files and splits them into chunks.
5. **Section E:** embeds the chunks with BGE-M3 and defines `retrieve()`.
6. **Section F:** RAG prompt and answer function, plus a text-only check.
7. **Section G:** launches the Gradio voice demo. Open the printed `gradio.live` link **in a new browser tab** so the microphone permission prompt appears, speak Amharic, and press **Stop**.
8. **Section H (optional):** exposes ASR and TTS as OpenAI-compatible endpoints (`/v1/audio/transcriptions`, `/v1/audio/speech`) for use with Pipecat.

Stop the Gradio cell when you are done: anyone with the public link can use it while it is running.

## Known limitations

- Answers are only as good as the documents. Headings or question-style lines are used to split documents; files without headings become one large chunk.
- Tables inside `.docx` files are only read when a document has no paragraph text.
- The demo has no voice activity detection: you press Stop to end your turn.
- ASR output can include a language tag such as `[AMH]`, which the notebook strips.
- Amharic quality (ASR, answers and voice) has been checked informally, not with a formal test set.

## Roadmap

- [ ] Real streaming pipeline with [Pipecat](https://github.com/pipecat-ai/pipecat) (VAD, interruption handling, stage overlap)
- [ ] Measure end-to-end latency including end-of-speech detection
- [ ] Measure the extra delay from long retrieved context
- [ ] Evaluate answer accuracy on a set of real bank questions
- [ ] Run on a dedicated GPU server and add a web widget

## Repository notes

- Do not commit secrets, the `.gguf` model, `llama_bin.tar.gz` or the embeddings `.npy` file; add them to `.gitignore`.
- Clear notebook outputs before committing, since they can contain public demo links.
- Check that the bank documents are allowed in a public repository.

## Licenses

Each model has its own license; check the model cards before any commercial use. Pipecat is BSD-2-Clause, and `llama.cpp` and BGE-M3 are MIT-licensed. Add your own license for this repository's code here.