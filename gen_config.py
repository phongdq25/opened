"""The one switch for how generation configs are read.

transformers >= 4.50 (4.57.3 is pinned) fills every GenerationConfig field left at its library
default from the model's generation_config.json when generate(generation_config=...) is
called. Qwen3 ships do_sample=True, temperature=0.6, top_p=0.95, top_k=20, so a config built
as greedy samples, and temperature 1.0 / top_p 1.0 come out as 0.6 / 0.95. Every run so far
was made that way, so it stays the default. --strict-generation turns the fill-in off and
every config means what it says (greedy evaluation, the requested SD and DistiLLM sampling)."""


def generation_kwargs(args):
    """Extra keyword arguments for every model.generate(generation_config=...) call."""
    return {"use_model_defaults": False} if getattr(args, "strict_generation", False) else {}
