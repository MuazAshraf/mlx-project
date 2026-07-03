from threading import Lock


class ModelRuntime:
    def __init__(self, models, default_key, load_fn):
        self._models = models
        self._current_key = default_key
        self._load_fn = load_fn
        self._model = None
        self._tokenizer = None
        self._lock = Lock()

    def list_models(self):
        return {
            "current": self._current_key,
            "models": [
                {"key": cfg.key, "name": cfg.name}
                for cfg in self._models.values()
            ],
        }

    def load_current(self):
        if self._models[self._current_key].backend == "mlx":
            self.get()

    def get(self):
        with self._lock:
            if self._models[self._current_key].backend != "mlx":
                raise RuntimeError("Current model is not an MLX model")
            self._ensure_loaded()
            return self._model, self._tokenizer, self._models[self._current_key]

    def current(self):
        return self._models[self._current_key]

    def switch(self, key):
        key = (key or "").strip().lower()
        if key not in self._models:
            available = ", ".join(sorted(self._models))
            raise ValueError(f"Unknown model '{key}'. Available: {available}")
        with self._lock:
            if key != self._current_key:
                self._current_key = key
                self._model = None
                self._tokenizer = None
            if self._models[self._current_key].backend == "mlx":
                self._ensure_loaded()
            return self._models[self._current_key]

    def _ensure_loaded(self):
        if self._model is not None and self._tokenizer is not None:
            return
        cfg = self._models[self._current_key]
        print(f"Loading model: {cfg.name}", flush=True)
        self._model, self._tokenizer = self._load_fn(cfg.path)
        print("Model ready.", flush=True)
